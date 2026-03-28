from __future__ import annotations

import logging
import uuid

import httpx

from ..audio.opus import OpusEncoder
from ..audio.preprocess import AudioPreprocessor
from ..config import settings
from ..processors.passthrough import PassthroughProcessor
from ..processors.translation import TranslationProcessor
from ..tts.client import TTSClient
from .callbacks import SessionCallbacks
from .orchestrator import SessionOrchestrator
from .session import TranslationSession

logger = logging.getLogger(__name__)


class SessionLimitError(Exception):
    """Raised when max concurrent sessions reached."""


class PipelineManager:
    """Manages the lifecycle of all active translation sessions.

    Each session represents one audio source with its own
    STT→Processor→TTS pipeline. Output is via consumer callbacks.
    """

    def __init__(self):
        self.sessions: dict[str, TranslationSession] = {}
        self._orchestrators: dict[str, SessionOrchestrator] = {}
        self._http_client: httpx.AsyncClient | None = None

        # Shared components (set via initialize)
        self._preprocessor: AudioPreprocessor | None = None
        self._tts_client: TTSClient | None = None
        self._audio_encoder: OpusEncoder | None = None
        self._processors: dict[str, TranslationProcessor | PassthroughProcessor] = {}

    async def initialize(
        self,
        preprocessor: AudioPreprocessor,
        tts_client: TTSClient,
        audio_encoder: OpusEncoder | None = None,
    ):
        self._preprocessor = preprocessor
        self._tts_client = tts_client
        self._audio_encoder = audio_encoder
        self._http_client = httpx.AsyncClient(timeout=10.0)

        # Initialize processors
        translation = TranslationProcessor()
        await translation.initialize()
        self._processors["translation"] = translation

        passthrough = PassthroughProcessor()
        await passthrough.initialize()
        self._processors["passthrough"] = passthrough

        logger.info("PipelineManager initialized")

    async def shutdown(self):
        for session_id in list(self.sessions):
            await self.stop_session(session_id)

        for processor in self._processors.values():
            await processor.shutdown()
        self._processors.clear()

        if self._http_client:
            await self._http_client.aclose()
        logger.info("PipelineManager shut down")

    @property
    def active_session_count(self) -> int:
        return len(self.sessions)

    async def create_session(
        self,
        source_lang: str | None = None,
        target_lang: str | None = None,
        processor_type: str | None = None,
        callbacks: SessionCallbacks | None = None,
        tts_config=None,
    ) -> TranslationSession:
        if self.active_session_count >= settings.max_concurrent_sessions:
            raise SessionLimitError(
                f"Maximum concurrent sessions ({settings.max_concurrent_sessions}) reached"
            )

        session_id = str(uuid.uuid4())[:8]
        proc_type = processor_type or settings.default_processor
        session = TranslationSession(
            session_id=session_id,
            source_lang=source_lang or settings.default_source_lang,
            target_lang=target_lang or settings.default_target_lang,
            processor_type=proc_type,
        )

        # Store voice config from platform
        if tts_config is not None:
            session.voice_mode = getattr(tts_config, "mode", None)
            session.ref_audio_url = getattr(tts_config, "reference_audio_url", None)
            session.ref_text = getattr(tts_config, "reference_text", None)
            session.speaker = getattr(tts_config, "speaker", None)
            session.voice_prompt = getattr(tts_config, "effective_voice_prompt", None) or getattr(tts_config, "voice_prompt", None)

        session.is_active = True
        self.sessions[session_id] = session

        # Wire broadcast callbacks for listener WebSockets
        broadcast = session.broadcast

        async def _broadcast_stt_partial(text: str):
            await broadcast.broadcast_text({"type": "stt_partial", "text": text})
            if callbacks and callbacks.on_stt_partial:
                await callbacks.on_stt_partial(text)

        async def _broadcast_stt_final(text: str, segment_index: int):
            await broadcast.broadcast_text({
                "type": "stt_final", "text": text, "segment_index": segment_index,
            })
            if callbacks and callbacks.on_stt_final:
                await callbacks.on_stt_final(text, segment_index)

        async def _broadcast_translation_partial(token: str, segment_index: int):
            await broadcast.broadcast_text({
                "type": "translation_partial", "token": token, "segment_index": segment_index,
            })
            if callbacks and callbacks.on_processor_partial:
                await callbacks.on_processor_partial(token, segment_index)

        async def _broadcast_translation_final(text: str, segment_index: int):
            await broadcast.broadcast_text({
                "type": "translation_final", "text": text, "segment_index": segment_index,
            })
            if callbacks and callbacks.on_processor_final:
                await callbacks.on_processor_final(text, segment_index)

        async def _broadcast_tts_audio(pcm_bytes: bytes, segment_index: int, sentence_index: int):
            await broadcast.broadcast_binary(pcm_bytes)
            if callbacks and callbacks.on_tts_audio:
                await callbacks.on_tts_audio(pcm_bytes, segment_index, sentence_index)

        broadcast_callbacks = SessionCallbacks(
            on_stt_partial=_broadcast_stt_partial,
            on_stt_final=_broadcast_stt_final,
            on_processor_partial=_broadcast_translation_partial,
            on_processor_final=_broadcast_translation_final,
            on_tts_audio=_broadcast_tts_audio,
        )

        # Ensure the TTS server has the right model loaded BEFORE starting the
        # orchestrator (which connects STT WebSocket). Model swap can take ~30s
        # and would cause the STT connection to timeout if done after.
        if session.voice_mode:
            try:
                await self._tts_client.ensure_model(session.voice_mode)
            except Exception:
                logger.warning(
                    "TTS model swap failed for session %s (mode=%s), using current model",
                    session_id, session.voice_mode, exc_info=True,
                )

            instruct = session.voice_prompt
            self._tts_client.set_session_voice_config(
                session_id=session_id,
                mode=session.voice_mode,
                speaker=session.speaker,
                instruct=instruct,
            )

        # For clone mode, init the voice clone prompt on TTS server
        if session.voice_mode == "clone" and session.ref_audio_url:
            try:
                await self._tts_client.init_session_voice(
                    session_id=session_id,
                    ref_audio_url=session.ref_audio_url,
                    ref_text=session.ref_text,
                    timeout=settings.tts_voice_clone_init_timeout,
                )
            except Exception:
                logger.warning(
                    "Voice clone init failed for session %s, falling back to default voice",
                    session_id, exc_info=True,
                )

        # Create and start the per-session orchestrator (connects STT WebSocket)
        processor = self._processors.get(proc_type, self._processors["translation"])
        orchestrator = SessionOrchestrator(
            session=session,
            preprocessor=self._preprocessor,
            processor=processor,
            tts_client=self._tts_client,
            callbacks=broadcast_callbacks,
            audio_encoder=self._audio_encoder,
        )
        await orchestrator.start()
        self._orchestrators[session_id] = orchestrator

        logger.info(
            "Session %s created (%s → %s, processor=%s, voice=%s)",
            session_id,
            session.source_lang,
            session.target_lang,
            session.processor_type,
            session.voice_mode or "default",
        )
        return session

    async def stop_session(self, session_id: str) -> bool:
        # Stop orchestrator first
        orchestrator = self._orchestrators.pop(session_id, None)
        if orchestrator:
            await orchestrator.stop()

        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        session.cancel()
        session.is_active = False
        logger.info(
            "Session %s stopped (duration=%.0fs, segments=%d)",
            session_id,
            session.duration_seconds,
            session.completed_segment_count,
        )
        return True

    async def update_session_voice(self, session_id: str, tts_config) -> None:
        """Hot-swap voice configuration for an active session."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.voice_mode = getattr(tts_config, "mode", None)
        session.ref_audio_url = getattr(tts_config, "reference_audio_url", None)
        session.ref_text = getattr(tts_config, "reference_text", None)
        session.speaker = getattr(tts_config, "speaker", None)
        session.voice_prompt = getattr(tts_config, "effective_voice_prompt", None) or getattr(tts_config, "voice_prompt", None)

        # Ensure the TTS server has the right model loaded
        if session.voice_mode:
            await self._tts_client.ensure_model(session.voice_mode)

            instruct = session.voice_prompt
            self._tts_client.set_session_voice_config(
                session_id=session_id,
                mode=session.voice_mode,
                speaker=session.speaker,
                instruct=instruct,
            )

        # For clone mode, also re-init voice clone prompt on TTS server
        if session.voice_mode == "clone" and session.ref_audio_url:
            await self._tts_client.init_session_voice(
                session_id=session_id,
                ref_audio_url=session.ref_audio_url,
                ref_text=session.ref_text,
                timeout=settings.tts_voice_clone_init_timeout,
            )

        logger.info("Voice config updated for session %s (mode=%s)", session_id, session.voice_mode)

    def get_session(self, session_id: str) -> TranslationSession | None:
        return self.sessions.get(session_id)

    def get_orchestrator(self, session_id: str) -> SessionOrchestrator | None:
        return self._orchestrators.get(session_id)

    async def check_services(self) -> dict[str, str]:
        """Check health of downstream services."""
        results = {}
        checks = {
            "stt": f"http://{settings.stt_ws_url.replace('ws://', '').split(':')[0]}:9090/health",
            "llm": f"{settings.llm_api_url.rstrip('/v1')}/health",
            "tts": f"{settings.tts_api_url}/health",
        }
        for name, url in checks.items():
            try:
                resp = await self._http_client.get(url, timeout=5.0)
                results[name] = "healthy" if resp.status_code == 200 else f"unhealthy ({resp.status_code})"
            except Exception as e:
                results[name] = f"unreachable ({type(e).__name__})"
        return results
