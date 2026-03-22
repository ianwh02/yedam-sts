from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from .broadcast import BroadcastHub

logger = logging.getLogger(__name__)


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

    # Default domain vocabulary for Korean church sermon STT.
    # Biases Whisper's decoder toward consistent Korean church terminology.
    KOREAN_CHURCH_VOCAB = (
        "다음은 한국어 기독교 교회 설교 음성입니다. "
        "신학적 용어와 교회 표현이 많이 포함되어 있습니다. "
        "주요 단어: 능히 하신다, 하나님, 예수 그리스도, 예수님, 성령님, 주님, "
        "아버지 하나님, 구원자, 메시아, 임마누엘, "
        "교회, 예담교회, 청년부, 집사님, 목사님, 장로님, 전도사님, 성도, "
        "예배, 설교, 찬양, 기도, 헌금, 소그룹, 셀모임, 큐티, "
        "말씀, 성경, 구약, 신약, 에베소서, 빌립보서, 로마서, 고린도서, 시편, "
        "복음, 십자가, 부활, 죄, 은혜, 구원, 칭의, 성화, 회개, 믿음, 소망, 사랑, "
        "넘치도록, 역사하신다, 인도하신다, 붙드신다, 함께하신다, 채우신다, 회복하신다, "
        "정체성, 사명, 소명, 헌신, 순종, 신앙, 겸손, "
        "고난, 시련, 연약함, 회복, 치유, 성장, 광야, "
        "바울, 베드로, 모세, 다윗, 아브라함, 다니엘."
    )

    def get_stt_initial_prompt(self, vocab: str = "") -> str:
        """Build Whisper initial_prompt from recent transcript + domain vocabulary.

        This biases Whisper's decoder toward consistent terminology after
        clear_buffer resets the audio context. No GPU overhead.
        """
        parts = [self.KOREAN_CHURCH_VOCAB]
        if vocab:
            parts.append(vocab)

        # Last N completed Korean segments for linguistic context
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
