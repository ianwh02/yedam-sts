from __future__ import annotations

from collections.abc import AsyncGenerator

from .base import BaseProcessor


class PassthroughProcessor(BaseProcessor):
    """No LLM processing — broadcasts STT transcription as-is.

    Useful for same-language captioning or when you only need
    real-time transcription without translation.
    """

    async def initialize(self):
        pass

    async def process(
        self, text: str, context: dict
    ) -> AsyncGenerator[str, None]:
        yield text
