"""Korean text filters for STT output preprocessing.

Strips filler words (hesitation sounds, bare agreement responses) from
STT segments before they reach the LLM translator.  This prevents the
LLM from producing short "single-word + comma" translations (e.g. "Yes,")
that cause TTS voice-clone instability.

Design constraints:
- Only match fillers as **standalone space-delimited tokens** so we never
  accidentally strip substrings from real words (e.g. 네 in 네가, 어 in 어디).
- Only strip fillers from the **leading** portion of a segment.  Mid-sentence
  fillers are part of natural speech rhythm and should stay.
- If the entire segment is fillers, return ``None`` so the caller can skip it.
"""

from __future__ import annotations

import re

# ── Filler word list ────────────────────────────────────────────────
# Each entry is a standalone Korean token that, when appearing on its own
# (space-delimited), is a discourse filler rather than content.
#
# Categories:
#   Hesitation:  음 응 어 아 에 으 흠
#   Agreement:   네 예 그래
#   Interjection: 아 오 야 와
#
# Words NOT included (carry semantic weight even standalone):
#   그래서 (so/therefore), 근데/그런데 (but/however), 이제 (now),
#   뭐 (what), 막 (emphatic), 좀 (a bit), 그 (that)

_FILLER_WORDS: set[str] = {
    # Hesitation sounds
    "음", "응", "어", "아", "에", "으", "흠",
    # Bare agreement / backchannel
    "네", "예", "그래",
    # Interjections (only filler when standalone)
    "오", "야", "와",
}

# Regex that matches a single filler token at the **start** of a string,
# optionally followed by a comma and/or whitespace.  We consume the
# trailing separator so the next iteration starts at the next real token.
#
# (?:,\s*|\s+)  →  eat comma+space  OR  plain whitespace after the filler
_FILLER_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in sorted(_FILLER_WORDS, key=len, reverse=True)) + r")"
    r"(?:,\s*|\s+|$)"
)


def strip_leading_fillers(text: str) -> str | None:
    """Remove filler words from the beginning of *text*.

    Returns:
        The cleaned text with leading fillers removed, or ``None`` if the
        entire text consisted of fillers (caller should skip the segment).
    """
    result = text.strip()

    # Iteratively strip one leading filler at a time so we handle chains
    # like "네 음 그래서 오늘은..." → "그래서 오늘은..."
    while result:
        m = _FILLER_PATTERN.match(result)
        if not m:
            break
        result = result[m.end():]

    result = result.strip()
    return result if result else None
