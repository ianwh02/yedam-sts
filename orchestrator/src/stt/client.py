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
        self._suppressing: bool = False

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
        if self._ws and not self._suppressing:
            await self._ws.send(pcm_data)

    async def clear_buffer(self):
        """Clear WhisperLive's audio buffer after a completed segment.

        Prevents unbounded buffer growth during long sermons.
        The server will reset its internal audio buffer and start
        fresh from the next audio chunk.
        """
        if self._ws:
            self._suppressing = True
            await self._ws.send(json.dumps({"type": "clear_buffer"}))
            logger.debug("Sent clear_buffer for session %s", self.session_id)

            # Auto-lift suppression after timeout if ACK doesn't arrive
            asyncio.get_event_loop().call_later(3.0, self._lift_suppression)

    def _lift_suppression(self):
        self._suppressing = False

    async def disconnect(self):
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
                    self._suppressing = False
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
                        # Only fire callback for NEW completed segments
                        completed_idx = segments.index(seg_data)
                        if completed_idx >= self._completed_count:
                            self._completed_count = completed_idx + 1
                            if self.on_completed:
                                await self.on_completed(segment.text)
                            # Clear buffer after completed segment
                            await self.clear_buffer()
                    else:
                        if self.on_partial and not self._suppressing:
                            await self.on_partial(segment.text)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("STT connection closed for session %s", self.session_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in STT receive loop for session %s", self.session_id)
