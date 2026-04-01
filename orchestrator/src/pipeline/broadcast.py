from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from ..audio.ogg_opus import OpusFrameEncoder

logger = logging.getLogger(__name__)

# Sentinel pushed into audio queues after a sentence's audio is fully sent.
# Tells the stream generator: "no more audio coming right now, send silence."
SILENCE_SENTINEL = object()


class BroadcastHub:
    """Fan-out a single pipeline's output to N listeners.

    Text messages (transcripts) go to WebSocket listeners.
    Audio is Opus-encoded once (shared encoder) then distributed as
    pre-encoded frames to asyncio.Queue subscribers. Each HTTP stream
    endpoint wraps frames in OGG pages (cheap, per-connection).
    """

    def __init__(self):
        self.listeners: set[WebSocket] = set()
        self._audio_queues: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._opus_encoder = OpusFrameEncoder()

    @property
    def count(self) -> int:
        return len(self.listeners) + len(self._audio_queues)

    async def add(self, ws: WebSocket):
        async with self._lock:
            self.listeners.add(ws)
        logger.info("WS listener added (total: %d)", self.count)

    async def remove(self, ws: WebSocket):
        async with self._lock:
            self.listeners.discard(ws)
        logger.info("WS listener removed (total: %d)", self.count)

    async def add_audio_queue(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._audio_queues.add(q)
        logger.info("Audio queue added (total: %d)", len(self._audio_queues))

    async def remove_audio_queue(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._audio_queues.discard(q)
        logger.info("Audio queue removed (total: %d)", len(self._audio_queues))

    async def broadcast_text(self, message: dict):
        """Send a JSON text message to all WebSocket listeners."""
        payload = json.dumps(message)
        await self._send_to_all_ws(payload, binary=False)

    @property
    def silence_frame(self) -> bytes:
        """Pre-encoded Opus silence frame for stream padding."""
        return self._opus_encoder.silence_frame

    async def broadcast_audio(self, pcm_bytes: bytes):
        """Encode PCM to Opus once, distribute frames to all queue subscribers."""
        self._last_audio_t = time.monotonic()

        # Encode once (shared encoder)
        opus_frames = self._opus_encoder.feed_pcm(pcm_bytes)

        # Distribute pre-encoded frames to all queues
        async with self._lock:
            for frame in opus_frames:
                for q in self._audio_queues:
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        pass  # drop frame if consumer is slow

    async def signal_silence(self):
        """Tell all audio stream consumers to enter silence mode."""
        now = time.monotonic()
        gap = now - getattr(self, "_last_audio_t", now)
        logger.info("[LATENCY] signal_silence: gap_since_last_audio=%.3fs", gap)

        # Flush any residual PCM in the encoder (prevents stale audio mixing
        # into the next sentence's first frame)
        residual_frames = self._opus_encoder.flush()

        async with self._lock:
            # Send any residual audio frames first
            for frame in residual_frames:
                for q in self._audio_queues:
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        pass

            for q in self._audio_queues:
                try:
                    q.put_nowait(SILENCE_SENTINEL)
                except asyncio.QueueFull:
                    pass

    async def _send_to_all_ws(self, payload: str | bytes, binary: bool):
        async with self._lock:
            if not self.listeners:
                return

            dead: set[WebSocket] = set()
            tasks = [
                self._safe_send(ws, payload, binary, dead)
                for ws in self.listeners
            ]
            await asyncio.gather(*tasks)

            if dead:
                self.listeners -= dead
                logger.info(
                    "Removed %d dead WS listeners (remaining: %d)",
                    len(dead),
                    len(self.listeners),
                )

    async def close_all(self):
        """Close all listener connections (called when session stops)."""
        async with self._lock:
            for ws in list(self.listeners):
                try:
                    await ws.close(code=1000, reason="Session ended")
                except Exception:
                    pass
            count = len(self.listeners)
            self.listeners.clear()
            if count:
                logger.info("Closed %d WS listener(s) on session stop", count)

            # Signal audio queues to stop — clear first to guarantee sentinel delivery
            for q in self._audio_queues:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                q.put_nowait(None)  # sentinel
            self._audio_queues.clear()

    @staticmethod
    async def _safe_send(
        ws: WebSocket,
        payload: str | bytes,
        binary: bool,
        dead: set[WebSocket],
    ):
        try:
            if binary:
                await ws.send_bytes(payload)
            else:
                await ws.send_text(payload)
        except Exception:
            dead.add(ws)
