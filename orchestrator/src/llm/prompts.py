from __future__ import annotations

import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """/no_think
You are a real-time {source_lang_name} to {target_lang_name} translator. Translate the following {source_lang_name} text into {target_lang_name} naturally and fluently.

Rules:
- Output ONLY the {target_lang_name} translation, no explanations or notes
- Maintain the speaker's style and tone
- The input comes from live speech recognition and may contain errors"""

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese", "ja": "Japanese",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
}


def _load_system_prompt(source_lang: str, target_lang: str) -> str:
    """Load system prompt from file if configured, otherwise use default."""
    path = settings.llm_system_prompt_path
    if path:
        p = Path(path)
        if p.is_file():
            prompt = p.read_text(encoding="utf-8").strip()
            logger.info("Loaded system prompt from %s", p)
            return prompt
        logger.warning("System prompt file not found: %s — using default", p)
    return _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        source_lang_name=_LANG_NAMES.get(source_lang, source_lang),
        target_lang_name=_LANG_NAMES.get(target_lang, target_lang),
    )


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
    system = _load_system_prompt(source_lang, target_lang)

    # Add recent Korean context for continuity (no English — including
    # prior translations creates copy targets that cause repetition loops)
    if recent_segments:
        context_lines = []
        for seg in recent_segments[-settings.llm_context_window_segments:]:
            context_lines.append(seg["korean"])
        if context_lines:
            system += (
                "\n\nRecent source text for context:\n"
                + "\n".join(context_lines)
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
