from __future__ import annotations

import asyncio
import logging
import re

from ..audio.preprocess import AudioPreprocessor
from ..bible.lookup import BibleLookup
from ..config import settings
from ..llm.glossary import get_glossary
from ..processors.base import BaseProcessor
from ..stt.client import STTClient
from ..text.filters import strip_leading_fillers
from ..text.korean_postprocess import KoreanPostProcessor
from ..tts.client import TTSClient, TTSQueueItem
from .callbacks import SessionCallbacks
from .session import TranslationSession

logger = logging.getLogger(__name__)

# Sentence-ending punctuation pattern
_SENTENCE_END = re.compile(r"[.!?;]\s*$")
# Comma followed by space (natural clause boundary)
_COMMA_BOUNDARY = re.compile(r",\s+$")


class SentenceBoundaryDetector:
    """Accumulates streaming LLM tokens and yields complete sentences.

    Enables TTS pipelining: synthesize sentence N while the LLM is
    still generating sentence N+1.

    Splitting rules (in priority order):
    1. Sentence punctuation: [.!?;] — always splits
    2. Comma boundary: ", " when buffer has >= min_words_comma_split words
    3. Max length: force split at max_words_per_chunk at last space
    """

    def __init__(self):
        self._buffer: str = ""
        self._sentence_index: int = 0

    def _word_count(self) -> int:
        return len(self._buffer.split())

    def _emit(self, delimiter: str = "") -> tuple[str, int, str]:
        sentence = self._buffer.strip()
        idx = self._sentence_index
        self._buffer = ""
        self._sentence_index += 1
        return sentence, idx, delimiter

    def feed(self, token: str) -> tuple[str, int, str] | None:
        """Feed a token and return (sentence, index, delimiter) if boundary detected."""
        self._buffer += token

        # Rule 1: sentence-ending punctuation splits (if buffer has enough words
        # for TTS to produce quality audio — short fragments like "Yes." get
        # combined with the next sentence)
        m = _SENTENCE_END.search(self._buffer)
        if m and self._word_count() >= settings.tts_min_words_sentence_split:
            return self._emit(delimiter=m.group().strip())

        # Rule 2: comma boundary when buffer is long enough
        m = _COMMA_BOUNDARY.search(self._buffer)
        if (
            m
            and self._word_count() >= settings.tts_min_words_comma_split
        ):
            return self._emit(delimiter=",")

        # Rule 3: hard max length — split at last space
        if self._word_count() >= settings.tts_max_words_per_chunk:
            last_space = self._buffer.rfind(" ")
            if last_space > 0:
                sentence = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space + 1:]
                idx = self._sentence_index
                self._sentence_index += 1
                return sentence, idx, ""

        return None

    def flush(self) -> tuple[str, int, str] | None:
        """Flush remaining buffer as a final sentence."""
        if self._buffer.strip():
            return self._emit(delimiter=".")
        return None


class SessionOrchestrator:
    """Wires the full pipeline for a single session.

    Data flow:
        Audio input → preprocess → STT → processor → TTS → callbacks

    Outputs via consumer-provided callbacks (SessionCallbacks). Transport
    (WebSocket, HTTP SSE, gRPC) and encoding (Opus, MP3) are the
    consumer's responsibility.

    All shared components (preprocessor, processor, TTS) are passed in;
    only the STTClient is per-session (owns its WhisperLive connection).
    """

    def __init__(
        self,
        session: TranslationSession,
        preprocessor: AudioPreprocessor,
        processor: BaseProcessor,
        tts_client: TTSClient,
        callbacks: SessionCallbacks | None = None,
        korean_postprocessor: KoreanPostProcessor | None = None,
        bible_lookup: BibleLookup | None = None,
    ):
        self._session = session
        self._preprocessor = preprocessor
        self._processor = processor
        self._tts_client = tts_client
        self._callbacks = callbacks or SessionCallbacks()
        self._korean_pp = korean_postprocessor
        self._bible = bible_lookup
        self._stt: STTClient | None = None
        self._tasks: set[asyncio.Task] = set()
        self._previous_chunk: str | None = None  # last flushed Korean text for LLM context
        self._segment_lock = asyncio.Lock()  # serialize segment processing to preserve TTS order

    async def start(self):
        """Create STT client and connect to WhisperLive."""
        session = self._session
        self._stt = STTClient(
            session_id=session.session_id,
            language=session.source_lang,
            on_partial=self._on_stt_partial,
            on_completed=self._on_stt_completed,
            initial_prompt=session.get_stt_initial_prompt(),
        )
        await self._stt.connect()

        # Register TTS audio callback for this session
        self._tts_client.register_audio_callback(
            session.session_id, self._on_tts_audio
        )
        self._tts_client.register_sentence_done_callback(
            session.session_id, self._on_tts_sentence_done
        )

        logger.info("SessionOrchestrator started for %s", session.session_id)

    async def feed_audio(self, pcm_float32: bytes):
        """Preprocess and forward audio to STT."""
        processed = self._preprocessor.process(pcm_float32)
        await self._stt.send_audio(processed)

    async def _on_stt_partial(self, text: str):
        """Forward partial STT transcription to consumer callback."""
        if self._callbacks.on_stt_partial:
            await self._callbacks.on_stt_partial(text)

    async def _on_stt_completed(self, text: str):
        """Handle completed STT segment — fire-and-forget processing task."""
        task = asyncio.create_task(self._process_segment(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process_segment(self, korean_text: str):
        """Full pipeline: record → callback → translate → TTS → callback.

        Serialized via _segment_lock so TTS enqueue order matches segment order.
        Segment N+1's LLM+TTS waits for segment N to finish enqueuing all sentences.
        """
        async with self._segment_lock:
            await self._process_segment_inner(korean_text)

    async def _process_segment_inner(self, korean_text: str):
        session = self._session
        if session.is_cancelled:
            return

        # Strip leading filler words (���, 네, 예, …) to prevent the LLM
        # from producing short "single-word, comma" translations that
        # destabilise TTS voice cloning.
        cleaned = strip_leading_fillers(korean_text)
        if cleaned is None:
            logger.info("Segment is filler-only, skipping: %r", korean_text)
            return
        if cleaned != korean_text:
            logger.info("Stripped fillers: %r → %r", korean_text, cleaned)
            korean_text = cleaned

        # Apply domain corrections and optional spacing fix
        if self._korean_pp:
            corrected = self._korean_pp.process(korean_text)
            if corrected != korean_text:
                logger.info("Post-processed: %r → %r", korean_text, corrected)
                korean_text = corrected

        # Record confirmed Korean segment
        segment = session.add_segment(korean_text)

        # Notify consumer of confirmed STT segment
        if self._callbacks.on_stt_final:
            await self._callbacks.on_stt_final(korean_text, segment.index)

        # Bible verse lookup disabled — too buggy for now.
        # if self._bible:
        #     new_verses = await self._bible.lookup_from_text(korean_text)
        #     for v in new_verses:
        #         if v not in session.bible_verses:
        #             session.bible_verses.append(v)
        #             logger.info("Bible verse added to session: %s", v[:60])

        # Build processor context (glossary resolved from session's glossary_id)
        glossary = get_glossary(session.glossary_id) if session.glossary_id else None
        context = {
            "source_lang": session.source_lang,
            "target_lang": session.target_lang,
            "recent_segments": session.get_llm_context(),
            "segment_index": segment.index,
            "previous_chunk": self._previous_chunk,
            "glossary": glossary,
            "church_name": session.church_name,
            "church_name_native": session.church_name_native,
            "bible_verses": session.bible_verses or None,
        }
        self._previous_chunk = korean_text

        # Stream processor tokens
        detector = SentenceBoundaryDetector()
        full_translation: list[str] = []
        use_continuous = settings.tts_continuous_enabled and self._tts_client.has_continuous_stream(session.session_id)

        try:
            async for token in self._processor.process(korean_text, context):
                if session.is_cancelled:
                    return

                full_translation.append(token)

                # Notify consumer of each streaming token
                if self._callbacks.on_processor_partial:
                    await self._callbacks.on_processor_partial(token, segment.index)

                # Check for sentence boundary → send to TTS
                result = detector.feed(token)
                if result:
                    sentence, sentence_idx, delimiter = result
                    logger.info("[DIAG] sentence_boundary detected (%r): %s", delimiter, sentence[:30])
                    if use_continuous:
                        await self._tts_client.send_text_chunk(session.session_id, sentence, language=session.target_lang)
                    else:
                        await self._tts_client.enqueue(
                            text=sentence,
                            session_id=session.session_id,
                            segment_index=segment.index,
                            sentence_index=sentence_idx,
                            language=session.target_lang,
                            delimiter=delimiter,
                        )
        except Exception:
            logger.exception(
                "Processing failed for segment %d in session %s",
                segment.index,
                session.session_id,
            )
            return

        # Flush any remaining text to TTS
        result = detector.flush()
        if result:
            sentence, sentence_idx, delimiter = result
            logger.info("[DIAG] sentence_flush: %s", sentence[:30])
            if use_continuous:
                await self._tts_client.send_text_chunk(session.session_id, sentence, language=session.target_lang)
            else:
                await self._tts_client.enqueue(
                    text=sentence,
                    session_id=session.session_id,
                    segment_index=segment.index,
                    sentence_index=sentence_idx,
                    language=session.target_lang,
                    delimiter=delimiter,
                )

        # Record translation and notify consumer
        translation = "".join(full_translation)
        segment.english = translation

        if self._callbacks.on_processor_final:
            await self._callbacks.on_processor_final(translation, segment.index)

    async def _on_tts_audio(self, pcm_bytes: bytes, item: TTSQueueItem):
        """Forward raw PCM to consumer callback."""
        if self._callbacks.on_tts_audio:
            await self._callbacks.on_tts_audio(
                pcm_bytes, item.segment_index, item.sentence_index
            )

    async def _on_tts_sentence_done(self):
        """Signal that a TTS sentence is fully sent — audio streams enter silence mode."""
        if self._callbacks.on_tts_sentence_done:
            await self._callbacks.on_tts_sentence_done()

    async def stop(self):
        """Shutdown: cancel tasks, disconnect STT, unregister TTS."""
        # Cancel in-flight processing tasks
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Disconnect STT
        if self._stt:
            await self._stt.disconnect()

        # Unregister TTS callbacks
        self._tts_client.unregister_audio_callbacks(self._session.session_id)

        logger.info("SessionOrchestrator stopped for %s", self._session.session_id)
