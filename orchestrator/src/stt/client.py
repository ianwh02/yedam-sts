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
        self._flushed_len: int = 0  # tracks how many chars have been flushed (for partial display)

    async def connect(self):
        """Connect to WhisperLive and send config."""
        url = settings.stt_ws_url
        self._ws = await websockets.connect(url, ping_timeout=None)

        config = {
            "uid": self.session_id,
            "language": self.language,
            "task": "transcribe",
            "model": "large-v3",
            "use_vad": True,
        }
        if self.language == "ko":
            config["flush_mode"] = "korean"
        else:
            config["flush_mode"] = "punctuation"
        if self.initial_prompt:
            config["initial_prompt"] = self.initial_prompt

        await self._ws.send(json.dumps(config))

        # Wait for SERVER_READY (WARNING status is acceptable — backend fallback)
        response = await self._ws.recv()
        data = json.loads(response)
        status = data.get("status", "")
        if data.get("message") == "SERVER_READY":
            pass  # Ideal case
        elif status == "WARNING":
            logger.warning("STT server warning: %s", data.get("message"))
        else:
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

    async def disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws:
            await self._ws.close()
        logger.info("STT client %s disconnected", self.session_id)

    async def _receive_loop(self):
        """Process incoming transcription segments from WhisperLive.

        All flushing is handled server-side (flush_mode=korean or punctuation).
        The client just forwards completed segments to the pipeline and
        displays partials to listeners.
        """
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                # Handle control messages (ACKs, not transcript data)
                msg_type = data.get("type", data.get("message", ""))
                if msg_type in ("buffer_cleared", "buffer_trimmed"):
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
                            remaining = segment.text[self._flushed_len:].strip()
                            if remaining and self.on_completed:
                                await self.on_completed(remaining)
                            self._flushed_len = 0
                            self._completed_count = 0
                            await self.clear_buffer()
                    else:
                        if self.on_partial:
                            await self.on_partial(segment.text)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("STT connection closed for session %s", self.session_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in STT receive loop for session %s", self.session_id)
