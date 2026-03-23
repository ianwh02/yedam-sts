from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

import websockets

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class STTSegment:
    """A transcription segment from WhisperLive."""

    text: str
    start: float
    end: float
    completed: bool


class STTClient:
    """WebSocket client for WhisperLive STT server.

    Manages a persistent connection per session. Forwards raw PCM audio
    from the admin and receives transcription segments. Implements the
    clear_buffer protocol for rolling audio window.

    Stable-prefix flushing: instead of waiting for Whisper to mark a segment
    as completed (which requires a silence gap), we detect the stable prefix
    of consecutive partial transcriptions. Once N characters at the start
    remain unchanged across M consecutive updates, that prefix is flushed
    to the pipeline as a confirmed segment — without waiting for a pause.

    Protocol (matches WhisperLive):
      Client → Server: JSON config on connect, then binary PCM frames
      Server → Client: JSON { segments: [{text, start, end, completed}] }
    """

    def __init__(
        self,
        session_id: str,
        language: str = "ko",
        on_partial: Callable[[str], Awaitable[None]] | None = None,
        on_completed: Callable[[str], Awaitable[None]] | None = None,
        initial_prompt: str = "",
    ):
        self.session_id = session_id
        self.language = language
        self.on_partial = on_partial
        self.on_completed = on_completed
        self.initial_prompt = initial_prompt

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._completed_count: int = 0
        self._receive_task: asyncio.Task | None = None
        self._send_seq: int = 0

        # ── Stable-prefix flushing ──
        self._prev_partial: str = ""        # previous partial text
        self._stable_prefix: str = ""       # longest common prefix across updates
        self._stable_count: int = 0         # how many consecutive updates the prefix was stable
        self._flushed_len: int = 0          # how many chars of the current partial we've already flushed
        self._min_flush_chars: int = 20     # minimum prefix length before flushing (avoid tiny fragments)
        self._stable_threshold: int = 3     # consecutive stable updates before flushing prefix

        # ── Silence flush ──
        self._last_partial: str = ""
        self._flush_handle: asyncio.TimerHandle | None = None
        self._flush_timeout: float = 2.0    # seconds of no new partials before flushing remainder

    async def connect(self):
        """Connect to WhisperLive and send config."""
        url = settings.stt_ws_url
        self._ws = await websockets.connect(url)

        config = {
            "uid": self.session_id,
            "language": self.language,
            "task": "transcribe",
            "model": "large-v3",
            "use_vad": True,
        }
        if self.initial_prompt:
            config["initial_prompt"] = self.initial_prompt

        await self._ws.send(json.dumps(config))

        # Wait for SERVER_READY
        response = await self._ws.recv()
        data = json.loads(response)
        if data.get("message") != "SERVER_READY":
            raise ConnectionError(f"STT server not ready: {data}")

        logger.info("STT client %s connected", self.session_id)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, pcm_data: bytes):
        """Forward raw PCM audio to WhisperLive."""
        if self._ws:
            await self._ws.send(pcm_data)
            self._send_seq += 1

    async def clear_buffer(self):
        """Clear WhisperLive's audio buffer after a completed segment."""
        if self._ws:
            await self._ws.send(json.dumps({"type": "clear_buffer", "seq": self._send_seq}))
            logger.debug("Sent clear_buffer for session %s (seq=%d)", self.session_id, self._send_seq)

    def _reset_flush_timer(self):
        """Reset the silence flush timer. Called on every partial."""
        if self._flush_handle:
            self._flush_handle.cancel()
        loop = asyncio.get_event_loop()
        self._flush_handle = loop.call_later(
            self._flush_timeout, lambda: asyncio.ensure_future(self._silence_flush())
        )

    async def _silence_flush(self):
        """Force-flush remaining partial text after silence timeout."""
        if self._last_partial:
            # Flush only the unflushed portion
            text = self._last_partial[self._flushed_len:].strip()
            if text:
                logger.info("Silence flush [%s]: %s", self.session_id, text[:60])
                if self.on_completed:
                    await self.on_completed(text)
            self._reset_state()
            await self.clear_buffer()

    def _reset_state(self):
        """Reset all partial tracking state after a full flush."""
        self._prev_partial = ""
        self._stable_prefix = ""
        self._stable_count = 0
        self._flushed_len = 0
        self._last_partial = ""
        self._completed_count = 0
        if self._flush_handle:
            self._flush_handle.cancel()
            self._flush_handle = None

    async def _check_stable_prefix(self, text: str):
        """Detect stable prefix between consecutive partials and flush it.

        Whisper revises from the end, not the beginning. So if the first N
        characters remain the same across M consecutive updates, that prefix
        is stable and can be sent to the LLM for translation immediately.
        """
        if self._flushed_len >= len(text):
            return

        unflushed = text[self._flushed_len:]
        prev_unflushed = self._prev_partial[self._flushed_len:] if len(self._prev_partial) > self._flushed_len else ""

        if not prev_unflushed:
            self._prev_partial = text
            return

        # Find common prefix between previous and current unflushed text
        common_len = 0
        for a, b in zip(prev_unflushed, unflushed):
            if a == b:
                common_len += 1
            else:
                break

        if common_len >= self._min_flush_chars:
            if common_len == len(self._stable_prefix) - self._flushed_len if self._stable_prefix else False:
                self._stable_count += 1
            else:
                # New stable prefix found
                self._stable_prefix = text[:self._flushed_len + common_len]
                self._stable_count = 1

            if self._stable_count >= self._stable_threshold:
                # Prefix is stable — flush it
                flush_text = self._stable_prefix[self._flushed_len:].strip()
                if flush_text:
                    logger.info(
                        "Stable prefix flush [%s] (%d stable updates): %s",
                        self.session_id, self._stable_count, flush_text[:60],
                    )
                    if self.on_completed:
                        await self.on_completed(flush_text)
                self._flushed_len = len(self._stable_prefix)
                self._stable_count = 0
                self._stable_prefix = ""
        else:
            self._stable_count = 0
            self._stable_prefix = ""

        self._prev_partial = text

    async def disconnect(self):
        if self._flush_handle:
            self._flush_handle.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            await self._ws.close()
        logger.info("STT client %s disconnected", self.session_id)

    async def _receive_loop(self):
        """Process incoming transcription segments from WhisperLive."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                # Handle buffer_cleared ACK
                if data.get("message") == "buffer_cleared":
                    continue

                segments = data.get("segments", [])
                for seg_data in segments:
                    segment = STTSegment(
                        text=seg_data.get("text", "").strip(),
                        start=seg_data.get("start", 0),
                        end=seg_data.get("end", 0),
                        completed=seg_data.get("completed", False),
                    )

                    if not segment.text:
                        continue

                    if segment.completed:
                        completed_idx = segments.index(seg_data)
                        if completed_idx >= self._completed_count:
                            self._completed_count = completed_idx + 1
                            # Flush only the unflushed portion
                            remaining = segment.text[self._flushed_len:].strip()
                            if remaining and self.on_completed:
                                await self.on_completed(remaining)
                            self._reset_state()
                            await self.clear_buffer()
                    else:
                        self._last_partial = segment.text
                        self._reset_flush_timer()

                        # Show full partial to listeners (not just unflushed)
                        if self.on_partial:
                            await self.on_partial(segment.text)

                        # Check for stable prefix to flush early
                        await self._check_stable_prefix(segment.text)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("STT connection closed for session %s", self.session_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in STT receive loop for session %s", self.session_id)
