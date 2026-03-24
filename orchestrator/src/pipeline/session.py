from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import WebSocket

from ..config import settings
from .broadcast import BroadcastHub

logger = logging.getLogger(__name__)


def _load_domain_vocab() -> str:
    """Load domain vocabulary from file or inline setting."""
    path = settings.stt_initial_prompt_vocab_path
    if path:
        p = Path(path)
        if p.is_file():
            vocab = p.read_text(encoding="utf-8").strip()
            logger.info("Loaded domain vocabulary from %s", p)
            return vocab
        logger.warning("Vocab file not found: %s", p)
    return settings.stt_initial_prompt_vocab


_DOMAIN_VOCAB = _load_domain_vocab()


@dataclass
class TranscriptSegment:
    """A single confirmed STT segment with its translation."""

    index: int
    korean: str
    english: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranslationSession:
    """Represents one active speech-to-speech pipeline instance.

    Each session maps to one audio source. Output is delivered via
    consumer-provided callbacks (see SessionCallbacks).
    """

    session_id: str
    source_lang: str = "ko"
    target_lang: str = "en"
    processor_type: str = "translation"

    # Admin connection (audio source)
    admin_ws: WebSocket | None = None

    # Broadcast hub for listener WebSockets
    broadcast: BroadcastHub = field(default_factory=BroadcastHub)

    # Transcript accumulator (for LLM context + storage)
    transcript: list[TranscriptSegment] = field(default_factory=list)
    completed_segment_count: int = 0

    # State
    is_active: bool = False
    started_at: float = field(default_factory=time.time)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.started_at

    def get_stt_initial_prompt(self) -> str:
        """Build Whisper initial_prompt from domain vocabulary + recent transcript.

        This biases Whisper's decoder toward consistent terminology after
        clear_buffer resets the audio context. No GPU overhead.
        """
        parts = []
        if _DOMAIN_VOCAB:
            parts.append(_DOMAIN_VOCAB)

        # Last N completed segments for linguistic context
        recent = self.transcript[-3:]
        for seg in recent:
            parts.append(seg.korean)

        return ". ".join(parts) if parts else ""

    def get_llm_context(self, window_size: int = 5) -> list[dict]:
        """Build sliding context window of recent segment pairs for LLM."""
        recent = self.transcript[-window_size:]
        return [
            {"korean": seg.korean, "english": seg.english}
            for seg in recent
            if seg.english is not None
        ]

    def add_segment(self, korean: str) -> TranscriptSegment:
        """Record a new confirmed Korean segment."""
        seg = TranscriptSegment(
            index=self.completed_segment_count,
            korean=korean,
        )
        self.transcript.append(seg)
        self.completed_segment_count += 1
        return seg

    def cancel(self):
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()
