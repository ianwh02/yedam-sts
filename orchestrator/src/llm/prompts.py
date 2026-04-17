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
- The input comes from live speech recognition and may contain errors
- Never start the translation with a single word followed by a comma (e.g. "Yes," or "So,"). Instead, integrate it naturally into the sentence or omit it"""

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese", "ja": "Japanese",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
}


# ── Glossary formatting ────────────────────────────────────────────

def format_glossary(
    glossary: dict,
    church_name: str | None = None,
    church_name_native: str | None = None,
) -> str:
    """Format a glossary dict into a compact prompt section.

    Args:
        glossary: The denomination glossary dict.
        church_name: Church name in English.
        church_name_native: Church name in source language (e.g. Korean).
    """
    parts: list[str] = []

    # One-line context
    church = church_name or glossary.get("church_name", "")
    if church_name_native:
        church = f"{church} ({church_name_native})" if church else church_name_native
    denom = glossary.get("denomination", "")
    if church or denom:
        parts.append(f"Context: {' — '.join(filter(None, [church, denom]))}")

    # Merge all Korean→English mappings into one flat list
    terms: list[str] = []
    for section in ("proper_nouns", "titles", "theological_terms",
                    "bible_books", "key_phrases", "code_switch_english"):
        for k, v in glossary.get(section, {}).items():
            terms.append(f"{k}={v}")
    if terms:
        parts.append(f"Glossary: {', '.join(terms)}")

    # STT corrections kept separate (different format)
    corrections = glossary.get("stt_corrections", {})
    if corrections:
        items = ", ".join(f"{k}→{v}" for k, v in corrections.items())
        parts.append(f"STT fixes: {items}")

    return "\n".join(parts)


# ── System prompt ───────────────────────────────────────────────────

def _load_base_prompt(source_lang: str, target_lang: str) -> str:
    """Load the base system prompt (template file or default)."""
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
    glossary: dict | None = None,
    church_name: str | None = None,
    church_name_native: str | None = None,
    bible_verses: list[str] | None = None,
) -> list[dict]:
    """Build the LLM prompt with glossary, scripture, and conversation-turn context.

    Args:
        text: The source text to translate.
        source_lang: Source language code.
        target_lang: Target language code.
        recent_segments: Sliding window of recent {korean, english} pairs.
        previous_chunk: Last flushed source text for continuity.
        glossary: Denomination glossary dict (from registry).
        church_name: Church name override (for shared denomination glossaries).
        bible_verses: Detected Bible verses to inject as reference scripture.

    Returns:
        OpenAI-format messages list.
    """
    system = _load_base_prompt(source_lang, target_lang)

    # Append glossary if provided
    if glossary:
        system += "\n\n" + format_glossary(glossary, church_name, church_name_native)

    # Append detected Bible verses (persist for whole session).
    # Cap at 500 chars to avoid blowing the LLM context window.
    if bible_verses:
        verses_block = "\n".join(bible_verses)
        if len(verses_block) > 500:
            # Keep only the most recent verses that fit
            truncated = []
            total = 0
            for v in reversed(bible_verses):
                if total + len(v) > 500:
                    break
                truncated.insert(0, v)
                total += len(v) + 1
            verses_block = "\n".join(truncated)
        if verses_block:
            system += "\n\nReference Scripture (use this exact wording when the speaker quotes these verses):\n"
            system += verses_block

    messages: list[dict] = [{"role": "system", "content": system}]

    # Build context as conversation turns
    if recent_segments:
        for seg in recent_segments[-settings.llm_context_window_segments:]:
            if seg.get("english"):
                messages.append({"role": "user", "content": seg["korean"]})
                messages.append({"role": "assistant", "content": seg["english"]})

    messages.append({"role": "user", "content": text})

    return messages
