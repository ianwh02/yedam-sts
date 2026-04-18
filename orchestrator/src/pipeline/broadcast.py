from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from ..audio.ogg_opus import OpusFrameEncoder

logger = logging.getLogger(__name__)

# Per-listener buffer: 250 frames × 20ms = 5 seconds of audio.
_QUEUE_SIZE = 250


class _ListenerSlot:
    """Per-listener outbound queue and independent drain loop."""

    __slots__ = ("ws", "queue", "_task", "_dropped")

    def __init__(self, ws: WebSocket, on_dead):
        self.ws = ws
        self.queue: asyncio.Queue[tuple[bytes | str, bool]] = asyncio.Queue(
            maxsize=_QUEUE_SIZE,
        )
        self._dropped = 0
        self._task = asyncio.create_task(self._drain(on_dead))

    def enqueue(self, payload: bytes | str, binary: bool) -> None:
        """Non-blocking enqueue.  Drops the frame if the queue is full."""
        try:
            self.queue.put_nowait((payload, binary))
        except asyncio.QueueFull:
            self._dropped += 1

    async def _drain(self, on_dead):
        ws_send_bytes = self.ws.send_bytes
        ws_send_text = self.ws.send_text
        queue = self.queue
        try:
            while True:
                payload, binary = await queue.get()
                try:
                    await (ws_send_bytes(payload) if binary else ws_send_text(payload))
                    # Batch: drain all immediately available frames without
                    # yielding back to queue.get(), reducing event-loop overhead.
                    while not queue.empty():
                        payload, binary = queue.get_nowait()
                        await (ws_send_bytes(payload) if binary else ws_send_text(payload))
                except Exception:
                    logger.warning(
                        "Listener evicted: send failed (dropped %d frames prior)",
                        self._dropped,
                    )
                    await on_dead(self.ws)
                    return
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        if self._dropped:
            logger.info(
                "Listener disconnected (dropped %d frames total)", self._dropped,
            )


class BroadcastHub:
    """Fan-out a single pipeline's output to N listeners.

    Text messages (transcripts) go as JSON text frames.
    Audio is Opus-encoded once (shared encoder) then each raw Opus frame
    is sent as a binary WebSocket frame to all listeners.

    Each listener gets an independent bounded queue with its own drain task,
    so a slow listener never blocks the pipeline or other listeners.
    """

    def __init__(self):
        self._slots: dict[WebSocket, _ListenerSlot] = {}
        self._lock = asyncio.Lock()
        self._opus_encoder = OpusFrameEncoder()

    @property
    def count(self) -> int:
        return len(self._slots)

    async def add(self, ws: WebSocket):
        async with self._lock:
            if ws not in self._slots:
                self._slots[ws] = _ListenerSlot(ws, self._evict)
        logger.info("WS listener added (total: %d)", self.count)

    async def remove(self, ws: WebSocket):
        async with self._lock:
            slot = self._slots.pop(ws, None)
        if slot:
            await slot.stop()
        logger.info("WS listener removed (total: %d)", self.count)

    async def broadcast_text(self, message: dict):
        """Enqueue a JSON text message to all listeners (non-blocking)."""
        payload = json.dumps(message)
        self._enqueue_all(payload, binary=False)

    async def broadcast_audio(self, pcm_bytes: bytes):
        """Encode PCM to Opus once, enqueue frames to all listeners."""
        self._last_audio_t = time.monotonic()
        for frame in self._opus_encoder.feed_pcm(pcm_bytes):
            self._enqueue_all(frame, binary=True)

    async def signal_silence(self):
        """Flush residual PCM at sentence boundary."""
        now = time.monotonic()
        gap = now - getattr(self, "_last_audio_t", now)
        logger.info("[LATENCY] signal_silence: gap_since_last_audio=%.3fs", gap)
        for frame in self._opus_encoder.flush():
            self._enqueue_all(frame, binary=True)

    async def close_all(self):
        """Close all listener connections (called when session stops)."""
        async with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            await slot.stop()
            try:
                await slot.ws.close(code=1000, reason="Session ended")
            except Exception:
                pass
        if slots:
            logger.info("Closed %d WS listener(s) on session stop", len(slots))

    def _enqueue_all(self, payload: bytes | str, binary: bool):
        """Non-blocking fan-out to every listener queue."""
        for slot in self._slots.values():
            slot.enqueue(payload, binary)

    async def _evict(self, ws: WebSocket):
        """Remove a dead/slow listener (called from its own drain task)."""
        async with self._lock:
            self._slots.pop(ws, None)
        try:
            await ws.close(code=1008, reason="Too slow")
        except Exception:
            pass
        logger.info("Evicted slow listener (remaining: %d)", self.count)
