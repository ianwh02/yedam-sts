from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseProcessor(ABC):
    """Pluggable processing step between STT output and TTS input.

    Subclasses define how confirmed STT text is transformed before
    being sent to TTS and broadcast to listeners. The pipeline is:

        STT (confirmed text) → Processor.process() → TTS + Broadcast

    The processor receives the raw STT text and a context dict containing
    session state (recent segments, language, etc.) and yields output
    text chunks for streaming display and TTS synthesis.
    """

    @abstractmethod
    async def process(
        self, text: str, context: dict
    ) -> AsyncGenerator[str, None]:
        """Transform STT text into output text.

        Args:
            text: Confirmed STT segment (source language).
            context: Session context dict with keys:
                - source_lang: str
                - target_lang: str
                - recent_segments: list[dict] with korean/english pairs
                - segment_index: int

        Yields:
            Output text chunks (streaming). For translation, these are
            translated text tokens. For passthrough, yields input as-is.
        """
        ...

    @abstractmethod
    async def initialize(self):
        """Called once when the processor is first used."""
        ...

    async def shutdown(self):
        """Called when the processor is no longer needed."""
        pass
