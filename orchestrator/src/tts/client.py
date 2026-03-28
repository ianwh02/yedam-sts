from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSQueueItem:
    """A sentence queued for TTS synthesis."""

    text: str
    session_id: str
    segment_index: int
    sentence_index: int
    language: str = "en"
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class SessionVoiceConfig:
    """Per-session voice configuration, injected into every TTS request."""

    mode: str | None = None  # "preset", "design", "clone"
    speaker: str | None = None
    instruct: str | None = None


class TTSClient:
    """HTTP client for Qwen3-TTS server with per-session queues.

    Each session gets its own queue and consumer task, so one session's
    long sentence doesn't block another session. A global semaphore
    limits total concurrent TTS requests to avoid overwhelming the GPU.

    Staleness dropping: items older than tts_stale_max_age_seconds or
    in a queue longer than tts_stale_threshold are dropped. Listeners
    receive text instead of audio for dropped items.
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._on_audio: dict[str, list] = {}  # session_id -> callbacks
        self._session_queues: dict[str, asyncio.Queue[TTSQueueItem | None]] = {}
        self._session_consumers: dict[str, asyncio.Task] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._session_voice_configs: dict[str, SessionVoiceConfig] = {}

    async def initialize(self):
        self._client = httpx.AsyncClient(
            base_url=settings.tts_api_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._semaphore = asyncio.Semaphore(settings.tts_max_concurrent)

    async def shutdown(self):
        # Stop all session consumers
        for session_id in list(self._session_queues):
            await self._stop_session_consumer(session_id)
        if self._client:
            await self._client.aclose()

    def set_session_voice_config(
        self, session_id: str,
        mode: str | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
    ):
        """Set voice configuration for a session. Applied to all subsequent TTS requests."""
        self._session_voice_configs[session_id] = SessionVoiceConfig(
            mode=mode, speaker=speaker, instruct=instruct,
        )
        logger.info(
            "Voice config set for session %s: mode=%s speaker=%s instruct=%s",
            session_id, mode, speaker, instruct[:50] if instruct else None,
        )

    def register_audio_callback(self, session_id: str, callback):
        """Register a callback and start a per-session consumer."""
        self._on_audio.setdefault(session_id, []).append(callback)
        if session_id not in self._session_queues:
            queue: asyncio.Queue[TTSQueueItem | None] = asyncio.Queue()
            self._session_queues[session_id] = queue
            self._session_consumers[session_id] = asyncio.create_task(
                self._consume_loop(session_id, queue)
            )

    def unregister_audio_callbacks(self, session_id: str):
        self._on_audio.pop(session_id, None)
        self._session_voice_configs.pop(session_id, None)
        asyncio.create_task(self._stop_session_consumer(session_id))

    async def _stop_session_consumer(self, session_id: str):
        queue = self._session_queues.pop(session_id, None)
        task = self._session_consumers.pop(session_id, None)
        if queue:
            await queue.put(None)  # Poison pill
        if task:
            task.cancel()
        # Clean up TTS server session decoder context (fire-and-forget, TTL handles failures)
        try:
            await self._client.delete(f"/sessions/{session_id}")
        except Exception:
            pass

    # Map voice modes to required TTS model names
    _MODE_TO_MODEL = {"preset": "custom", "design": "design", "clone": "base"}

    async def ensure_model(self, voice_mode: str, timeout: float = 60.0):
        """Ensure the TTS server has the right model loaded for the given voice mode.

        Checks the current model and triggers a swap if needed (~30s).
        """
        required = self._MODE_TO_MODEL.get(voice_mode)
        if required is None:
            return

        resp = await self._client.get("/models", timeout=httpx.Timeout(5.0))
        resp.raise_for_status()
        current = resp.json().get("current")

        if current == required:
            return

        logger.info("TTS model swap needed: %s → %s (for mode=%s)", current, required, voice_mode)
        swap_resp = await self._client.post(
            "/swap_model",
            json={"model": required},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        swap_resp.raise_for_status()
        result = swap_resp.json()
        logger.info("TTS model swapped: %s (took %.1fs)", result.get("model"), result.get("swap_time_s", 0))

    async def init_session_voice(
        self,
        session_id: str,
        ref_audio_url: str,
        ref_text: str | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """Initialize a per-session voice clone prompt on the TTS server.

        Downloads the ref audio from the signed URL and creates a voice clone
        prompt that will be used for all subsequent TTS requests in this session.
        """
        response = await self._client.post(
            "/sessions/init_voice",
            json={
                "session_id": session_id,
                "ref_audio_url": ref_audio_url,
                "ref_text": ref_text,
            },
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "Voice clone initialized for session %s (mode=%s)",
            session_id, result.get("mode"),
        )
        return result

    async def enqueue(
        self,
        text: str,
        session_id: str,
        segment_index: int,
        sentence_index: int,
        language: str = "en",
    ):
        """Add a sentence to the session's TTS synthesis queue."""
        queue = self._session_queues.get(session_id)
        if queue is None:
            logger.warning("No TTS queue for session %s, dropping: %s...", session_id, text[:50])
            return
        item = TTSQueueItem(
            text=text,
            session_id=session_id,
            segment_index=segment_index,
            sentence_index=sentence_index,
            language=language,
        )
        await queue.put(item)

    def _make_silence(self, duration_ms: int, sample_rate: int = 48000) -> bytes:
        """Generate s16le silence of the given duration."""
        n_samples = int(sample_rate * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    def _build_request_json(self, item: TTSQueueItem) -> dict:
        """Build the JSON payload for a TTS request, including per-session voice config."""
        payload = {
            "text": item.text,
            "language": item.language,
            "session_id": item.session_id,
        }
        voice_cfg = self._session_voice_configs.get(item.session_id)
        if voice_cfg is not None:
            if voice_cfg.mode is not None:
                payload["mode"] = voice_cfg.mode
            if voice_cfg.speaker is not None:
                payload["speaker"] = voice_cfg.speaker
            if voice_cfg.instruct is not None:
                payload["instruct"] = voice_cfg.instruct
        return payload

    async def _consume_loop(self, session_id: str, queue: asyncio.Queue):
        """Process TTS queue items for a single session."""
        while True:
            item = await queue.get()
            if item is None:
                break

            # Acquire semaphore to limit total concurrent TTS requests
            async with self._semaphore:
                try:
                    if settings.tts_streaming_enabled:
                        await self._synthesize_streaming(item)
                    else:
                        audio_bytes = await self._synthesize(item)
                        callbacks = self._on_audio.get(item.session_id, [])
                        for cb in callbacks:
                            await cb(audio_bytes, item)

                    # Append consistent inter-segment silence
                    if settings.tts_inter_segment_pause_ms > 0:
                        pause = self._make_silence(settings.tts_inter_segment_pause_ms)
                        callbacks = self._on_audio.get(item.session_id, [])
                        for cb in callbacks:
                            await cb(pause, item)
                except Exception:
                    logger.exception("TTS synthesis failed for: %s...", item.text[:50])

    async def _synthesize_streaming(self, item: TTSQueueItem):
        """Stream TTS audio chunks, firing callbacks as chunks arrive."""
        payload = self._build_request_json(item)
        try:
            async with self._client.stream(
                "POST",
                "/synthesize/stream",
                json=payload,
            ) as response:
                response.raise_for_status()
                sample_format = response.headers.get("x-sample-format", "f32le")
                callbacks = self._on_audio.get(item.session_id, [])
                async for raw_chunk in response.aiter_bytes(chunk_size=4096):
                    if not raw_chunk:
                        continue
                    if sample_format == "s16le":
                        audio_bytes = raw_chunk
                    else:
                        # f32le PCM → int16 PCM (backward compat)
                        samples = np.frombuffer(raw_chunk, dtype=np.float32)
                        audio_bytes = (samples.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    for cb in callbacks:
                        await cb(audio_bytes, item)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Streaming not available, fall back to non-streaming
                logger.info("Streaming endpoint not available, falling back")
                audio_bytes = await self._synthesize(item)
                callbacks = self._on_audio.get(item.session_id, [])
                for cb in callbacks:
                    await cb(audio_bytes, item)
            else:
                raise

    async def _synthesize(self, item: TTSQueueItem) -> bytes:
        """Call Qwen3-TTS server to synthesize speech.

        Requests raw PCM to avoid WAV encode/decode round-trip.
        Returns int16 PCM bytes for downstream encoding.
        """
        payload = self._build_request_json(item)
        response = await self._client.post(
            "/synthesize",
            json=payload,
            headers={"Accept": "application/octet-stream"},
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "application/octet-stream" in content_type:
            sample_format = response.headers.get("x-sample-format", "f32le")
            if sample_format == "s16le":
                return response.content
            # f32le PCM → int16 (backward compat)
            samples = np.frombuffer(response.content, dtype=np.float32)
            return (samples.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()
        else:
            # Fallback: WAV response (backwards compat)
            import io
            import soundfile as sf

            audio, _sample_rate = sf.read(io.BytesIO(response.content), dtype="int16")
            return audio.tobytes()
