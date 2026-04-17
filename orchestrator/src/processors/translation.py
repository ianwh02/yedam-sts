from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from ..config import settings
from ..llm.client import LLMClient
from ..llm.prompts import build_translation_prompt
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class TranslationProcessor(BaseProcessor):
    """Translates STT text from source to target language via vLLM.

    Uses streaming token generation for low-latency partial output.
    Includes sliding context window of recent segments for coherent
    multi-sentence translation.
    """

    def __init__(self):
        self._llm: LLMClient | None = None

    async def initialize(self):
        self._llm = LLMClient(base_url=settings.llm_api_url)
        await self._llm.initialize()
        logger.info("TranslationProcessor initialized")

    async def shutdown(self):
        if self._llm:
            await self._llm.shutdown()

    async def process(
        self, text: str, context: dict
    ) -> AsyncGenerator[str, None]:
        messages = build_translation_prompt(
            text=text,
            source_lang=context.get("source_lang", "ko"),
            target_lang=context.get("target_lang", "en"),
            recent_segments=context.get("recent_segments", []),
            previous_chunk=context.get("previous_chunk"),
            glossary=context.get("glossary"),
            church_name=context.get("church_name"),
            church_name_native=context.get("church_name_native"),
            bible_verses=context.get("bible_verses"),
        )

        async for token in self._llm.stream_chat(
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        ):
            yield token
