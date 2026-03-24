from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """/no_think
You are a real-time translator. Translate the following text naturally and fluently.

Rules:
- Output ONLY the translation, no explanations or notes
- Maintain the speaker's style and tone
- The input comes from live speech recognition and may contain errors"""


def _load_system_prompt() -> str:
    """Load system prompt from file if configured, otherwise use default."""
    path = settings.llm_system_prompt_path
    if path:
        p = Path(path)
        if p.is_file():
            prompt = p.read_text(encoding="utf-8").strip()
            logger.info("Loaded system prompt from %s", p)
            return prompt
        logger.warning("System prompt file not found: %s — using default", p)
    return _DEFAULT_SYSTEM_PROMPT


TRANSLATION_SYSTEM_PROMPT = _load_system_prompt()


def build_translation_prompt(
    text: str,
    source_lang: str = "ko",
    target_lang: str = "en",
    recent_segments: list[dict] | None = None,
    previous_chunk: str | None = None,
) -> list[dict]:
    """Build the LLM prompt with sliding context window.

    Args:
        text: The Korean text to translate.
        source_lang: Source language code.
        target_lang: Target language code.
        recent_segments: List of recent {korean, english} segment pairs
            for translation continuity.
        previous_chunk: The immediately preceding Korean text chunk (from
            stable-prefix flushing). Gives the LLM sentence-level context
            when a flush splits mid-sentence.

    Returns:
        OpenAI-format messages list.
    """
    system = TRANSLATION_SYSTEM_PROMPT

    # Add recent context for continuity
    if recent_segments:
        context_lines = []
        for seg in recent_segments[-5:]:
            if seg.get("english"):
                context_lines.append(
                    f"Korean: {seg['korean']}\nEnglish: {seg['english']}"
                )
        if context_lines:
            system += (
                "\n\nRecent translation context for continuity:\n"
                + "\n---\n".join(context_lines)
            )

    if previous_chunk:
        user_content = (
            f"[Previous chunk for context: {previous_chunk}]\n"
            f"Translate ONLY the following text:\n{text}"
        )
    else:
        user_content = text

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
