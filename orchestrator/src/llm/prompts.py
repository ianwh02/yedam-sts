from __future__ import annotations

TRANSLATION_SYSTEM_PROMPT = """/no_think
You are a real-time Korean to English translator for a church sermon. Translate the following Korean text into natural, fluent English. Maintain the spiritual and pastoral tone.

Rules:
- Output ONLY the English translation, no explanations or notes
- Preserve paragraph structure and rhetorical style
- Use standard Christian theological terms in English
- Bible verse references should use standard English notation (e.g., Romans 8:28)
- Maintain the speaker's style (questions, exclamations, pauses)
- Keep proper nouns in their standard English forms (e.g., 예수님 → Jesus, 하나님 → God)"""


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
