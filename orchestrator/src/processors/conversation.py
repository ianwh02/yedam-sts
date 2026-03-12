from __future__ import annotations

from typing import AsyncGenerator

from .base import BaseProcessor


class ConversationProcessor(BaseProcessor):
    """Speech-to-speech AI chat via vLLM.

    For interactive voice conversations (not translation).
    The LLM receives STT text as user input and generates
    a conversational response that gets synthesized via TTS.

    Placeholder — will be implemented when needed for
    non-translation use cases.
    """

    async def initialize(self):
        pass

    async def process(
        self, text: str, context: dict
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError(
            "ConversationProcessor is a placeholder. "
            "Implement when needed for voice AI chat use cases."
        )
