"""Rule-based Korean phrase/sentence ending detection for STT flushing.

Detects grammatical boundaries in streaming Korean transcripts to flush text
at linguistically meaningful points instead of arbitrary VAD pauses.

Two tiers:
- Sentence endings → flush to LLM + clear STT buffer (safe, sentence is complete)
- Phrase endings → flush to LLM only, keep STT buffer (Whisper retains context)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ── Hangul jamo helpers ──────────────────────────────────────────────

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
JONGSEONG_COUNT = 28  # number of final consonant slots (0 = no batchim)

# Final consonant indices for common batchim
BIEUP = 17   # ㅂ
NIEUN = 4    # ㄴ
RIEUL = 8    # ㄹ


def get_jongseong(char: str) -> int:
    """Return the final consonant (batchim) index of a Hangul syllable.
    Returns -1 if not a Hangul syllable block.  0 = no batchim."""
    code = ord(char) - HANGUL_BASE
    if code < 0 or code > (HANGUL_END - HANGUL_BASE):
        return -1
    return code % JONGSEONG_COUNT


def is_hangul(char: str) -> bool:
    return HANGUL_BASE <= ord(char) <= HANGUL_END


# ── Ending patterns ─────────────────────────────────────────────────

# Sentence endings: these indicate a grammatically complete sentence.
# Ordered longest-first so longer matches take priority.
SENTENCE_ENDINGS: list[str] = sorted([
    # Formal declarative / question
    "습니다", "습니까", "읍니다", "읍니까",
    # Polite declarative
    "어요", "아요", "여요", "해요", "에요", "예요",
    "이에요",
    # Polite variants
    "죠", "지요",
    "잖아요", "거든요", "을까요", "까요", "을게요", "을래요",
    "네요", "군요", "구요", "는데요", "은데요", "인데요",
    "나요", "던가요", "을걸요",
    # Plain past/future declarative
    "었다", "았다", "였다", "겠다", "했다", "갔다", "왔다", "났다", "됐다",
    "었죠", "았죠", "였죠", "겠죠", "했죠",
    # Imperative / hortative
    "세요", "십시오", "십시다",
    "합시다", "읍시다", "갑시다",
    "하자", "보자", "가자",
    # Quoted endings
    "래요", "대요", "재요",
    # Casual/spoken declarative (common in spoken Korean)
    "거든요", "잖아요",
    "겠죠", "있죠", "없죠", "하죠",
    "서요", "해서요", "돼서요",  # polite reason as sentence ender
    # Noun + copula endings
    "입니다", "이에요", "예요",
], key=len, reverse=True)

# ㅂ니다 / ㅂ니까 forms: the syllable before 니다/니까 has ㅂ batchim.
# These are checked separately via jamo decomposition.
BIEUP_NIDA_ENDINGS = ["니다", "니까"]

# Phrase endings: these indicate a clause boundary within a sentence.
# Flush text to LLM for translation but keep the STT buffer.
PHRASE_ENDINGS: list[str] = sorted([
    # Causal / reason
    "으니까", "니까", "으니깐", "니깐",  # 니깐 = informal spoken variant
    "때문에",
    "어서", "아서", "여서", "해서",
    "므로", "으므로",
    # Contrast / concession
    "지만", "는데", "은데", "인데", "한데",
    "더라도", "을지라도",
    # Conditional
    "으면", "다면", "라면",
    # Sequential / listing
    "하고", "으며", "고서",
    "면서", "으면서",
    # Purpose
    "려고", "으려고", "도록",
    # Background / explanation
    "거든", "잖아",
    # Through / passing (common in spoken narrative)
    "통해서", "을통해",
    # Quoting
    "라고", "다고", "냐고",
    # While / during
    "으면서", "면서",
    # Listing with emphasis
    "뿐만", "아니라",
], key=len, reverse=True)

# Clause connectives: flush everything BEFORE the connective.
# These are standalone tokens that start a new clause.
CLAUSE_CONNECTIVES = {
    "그래서", "그런데", "그러니까", "그러면", "그러나",
    "하지만", "그리고", "그렇지만", "그러므로",
}

# Punctuation that Whisper may insert (strip before checking endings)
TRAILING_PUNCT = set(".,!?;:…·~\"'""''")

# Short markers that always trigger a sentence flush regardless of min_chars.
# Empty by default — populated via extra_flush_markers in KoreanEndingDetector.
_ALWAYS_FLUSH_MARKERS: set[str] = set()


@dataclass
class FlushDecision:
    """Result from KoreanEndingDetector.check()."""
    flush_type: str    # "sentence", "phrase", or "none"
    text: str          # the text to flush (empty if "none")
    end_pos: int       # position in full text where flush ends
    reason: str        # human-readable reason (e.g., "~습니다")


@dataclass
class KoreanEndingDetector:
    """Detects Korean grammatical endings in streaming partial transcripts.

    Call check() on each Whisper partial update. It returns a FlushDecision
    indicating whether to flush and at what boundary.

    Tracks flushed_len internally — only checks unflushed text.
    """
    min_phrase_chars: int = 15      # min chars since last flush for phrase flush
    min_sentence_chars: int = 6     # min chars since last flush for sentence flush
    stability_count: int = 2        # consecutive stable detections before flushing
    max_no_flush_s: float = 15.0    # emergency fallback: force flush after this many seconds (must be < 25s clip threshold)
    extra_flush_markers: set[str] = field(default_factory=set)  # domain-specific markers (e.g. 아멘, 할렐루야)

    # ── Internal state ──
    flushed_len: int = field(default=0, init=False)
    _prev_ending: str = field(default="", init=False)
    _prev_ending_pos: int = field(default=-1, init=False)
    _stable_count: int = field(default=0, init=False)
    _last_flush_time: float = field(default_factory=time.monotonic, init=False)
    _prev_text: str = field(default="", init=False)
    _sentence_endings: list[str] = field(default_factory=list, init=False)
    _always_flush_markers: set[str] = field(default_factory=set, init=False)
    _standalone_ok: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        # Build instance-level copies with extra markers merged in
        if self.extra_flush_markers:
            self._sentence_endings = sorted(
                list(SENTENCE_ENDINGS) + list(self.extra_flush_markers),
                key=len, reverse=True,
            )
            self._always_flush_markers = _ALWAYS_FLUSH_MARKERS | self.extra_flush_markers
        else:
            self._sentence_endings = SENTENCE_ENDINGS
            self._always_flush_markers = _ALWAYS_FLUSH_MARKERS
        self._standalone_ok = {
            "합시다", "읍시다", "갑시다", "하자", "보자", "가자",
        } | self.extra_flush_markers

    def check(self, text: str) -> FlushDecision:
        """Check streaming partial text for Korean endings.

        Args:
            text: The full partial transcript from Whisper (including already-flushed prefix).

        Returns:
            FlushDecision with flush_type "sentence", "phrase", or "none".
        """
        unflushed = text[self.flushed_len:]
        stripped = unflushed.strip()

        # Emergency fallback
        if (stripped and
                time.monotonic() - self._last_flush_time > self.max_no_flush_s):
            self._reset_stability()
            return FlushDecision("sentence", stripped, len(text), "timeout")

        # Need enough text to check (use the smaller threshold)
        # But always allow domain-specific markers regardless of length
        min_chars = min(self.min_sentence_chars, self.min_phrase_chars)
        if len(stripped) < min_chars:
            # Check for short standalone markers before giving up
            if stripped not in self._always_flush_markers:
                self._prev_text = text
                return FlushDecision("none", "", 0, "")

        # Find complete tokens in the unflushed portion.
        # A token is "complete" if it's followed by a space.
        # The last token (at the very end) might be incomplete.
        tokens = self._extract_complete_tokens(unflushed, text)

        if not tokens:
            self._prev_text = text
            return FlushDecision("none", "", 0, "")

        # Check each complete token from the start forward.
        # Flush at the earliest ending to minimize latency.
        for token_text, token_end_pos in tokens:
            # Strip trailing punctuation from token
            clean = token_text.rstrip("".join(TRAILING_PUNCT))
            if not clean or not is_hangul(clean[-1]):
                continue

            # Check sentence endings first (higher priority)
            sent_ending = self._match_sentence_ending(clean)
            if sent_ending:
                chars_since_flush = token_end_pos  # relative to unflushed start
                is_marker = clean in self._always_flush_markers
                if chars_since_flush >= self.min_sentence_chars or is_marker:
                    abs_pos = self.flushed_len + token_end_pos
                    if self._check_stability(sent_ending, abs_pos):
                        flush_text = unflushed[:token_end_pos].strip()
                        self._prev_text = text
                        return FlushDecision("sentence", flush_text, abs_pos,
                                             f"~{sent_ending}")
                # Sentence ending matched but stability not met yet —
                # skip phrase check on this token to avoid ping-pong
                continue

            # Check phrase endings (only if no sentence ending on this token)
            phrase_ending = self._match_phrase_ending(clean)
            if phrase_ending:
                chars_since_flush = token_end_pos
                if chars_since_flush >= self.min_phrase_chars:
                    abs_pos = self.flushed_len + token_end_pos
                    if self._check_stability(phrase_ending, abs_pos):
                        flush_text = unflushed[:token_end_pos].strip()
                        self._prev_text = text
                        return FlushDecision("phrase", flush_text, abs_pos,
                                             f"~{phrase_ending}")

        # ── Connective check (runs after endings found nothing) ──
        # If a clause connective (그래서, 그런데, etc.) appears, flush
        # everything before it so the connective stays with the next clause.
        for token_text, token_end_pos in tokens:
            if token_text in CLAUSE_CONNECTIVES:
                token_start = token_end_pos - len(token_text)
                if token_start >= self.min_phrase_chars:
                    flush_text = unflushed[:token_start].strip()
                    if flush_text:
                        abs_pos = self.flushed_len + token_start
                        self._prev_text = text
                        return FlushDecision("phrase", flush_text, abs_pos,
                                             f"before:{token_text}")

        self._prev_text = text
        return FlushDecision("none", "", 0, "")

    def on_flushed(self, flush_type: str, end_pos: int):
        """Call after a flush is executed to update internal state."""
        # Always advance flushed_len — server-side doesn't clear buffer on flush
        self.flushed_len = end_pos
        self._reset_stability()
        self._last_flush_time = time.monotonic()

    def reset(self):
        """Full reset (called when buffer is externally cleared)."""
        self.flushed_len = 0
        self._reset_stability()
        self._last_flush_time = time.monotonic()
        self._prev_text = ""

    @property
    def time_since_last_flush(self) -> float:
        return time.monotonic() - self._last_flush_time

    # ── Private methods ──────────────────────────────────────────────

    def _extract_complete_tokens(
        self, unflushed: str, full_text: str
    ) -> list[tuple[str, int]]:
        """Extract tokens that are likely word-complete.

        Returns list of (token_text, end_position_in_unflushed).
        A token is complete if followed by a space in the original text.
        The very last token (no trailing space) is only included if it was
        stable between this and the previous update.
        """
        tokens: list[tuple[str, int]] = []
        parts = unflushed.split()
        if not parts:
            return tokens

        # Check if unflushed has trailing whitespace — if so, even the last
        # split part is followed by a space and is complete.
        has_trailing_space = unflushed != unflushed.rstrip()

        pos = 0
        for i, part in enumerate(parts):
            # Find actual position of this part in unflushed
            idx = unflushed.find(part, pos)
            if idx == -1:
                continue
            end = idx + len(part)

            is_last = (i == len(parts) - 1)

            if not is_last:
                # Not the last token — followed by space, so it's complete
                tokens.append((part, end))
            elif has_trailing_space:
                # Last token but trailing space in original — complete
                tokens.append((part, end))
            else:
                # Last token with no trailing space — only if stable
                if self._is_last_token_stable(part, full_text):
                    tokens.append((part, end))

            pos = end

        return tokens

    def _is_last_token_stable(self, token: str, full_text: str) -> bool:
        """Check if the last token has been unchanged between updates."""
        if not self._prev_text:
            return False
        # If the previous text ended with the same token, it's stable
        prev_stripped = self._prev_text.rstrip()
        return prev_stripped.endswith(token)

    def _match_sentence_ending(self, token: str) -> str | None:
        """Check if a token ends with a sentence-ending suffix."""
        # Check standard suffix list
        # Token must be longer than just the ending (at least 1 stem char),
        # OR the token IS the ending for standalone markers.
        for ending in self._sentence_endings:
            if token.endswith(ending):
                if len(token) > len(ending) or ending in self._standalone_ok:
                    return ending

        # Check ㅂ니다 / ㅂ니까 forms via jamo
        for ending in BIEUP_NIDA_ENDINGS:
            if token.endswith(ending) and len(token) > len(ending):
                # The character before 니다/니까 must have ㅂ batchim
                char_before = token[-(len(ending) + 1)]
                if is_hangul(char_before) and get_jongseong(char_before) == BIEUP:
                    return f"ㅂ{ending}"

        return None

    def _match_phrase_ending(self, token: str) -> str | None:
        """Check if a token ends with a phrase-ending suffix."""
        for ending in PHRASE_ENDINGS:
            if token.endswith(ending):
                # For multi-char endings like 때문에, allow exact match
                # For short endings like 고, require stem char
                if len(token) > len(ending) or len(ending) >= 3:
                    return ending
        return None

    def _check_stability(self, ending: str, pos: int) -> bool:
        """Check if the same ending has been detected across consecutive updates.

        Only checks the ending string, not the exact position — Whisper
        frequently shifts text by adding/removing spaces, which changes
        absolute positions without changing the actual ending.
        """
        if ending == self._prev_ending:
            self._stable_count += 1
        else:
            self._prev_ending = ending
            self._prev_ending_pos = pos
            self._stable_count = 1

        return self._stable_count >= self.stability_count

    def _reset_stability(self):
        self._prev_ending = ""
        self._prev_ending_pos = -1
        self._stable_count = 0


# ── Punctuation-based flush detector (language-agnostic) ──────────────


import re

_SENTENCE_PUNCT = re.compile(r'[.!?](?:\s|$)')
_CLAUSE_PUNCT = re.compile(r'[,;:](?:\s|$)')


@dataclass
class PunctuationFlushDetector:
    """Detects sentence/clause boundaries from Whisper punctuation.

    Works for any language. Uses the same stability pattern as KoreanEndingDetector:
    punctuation must appear in N consecutive partial updates before flushing,
    which prevents false flushes from short pauses where Whisper temporarily
    adds a period then revises it away.

    Higher stability_count than Korean (4 vs 2) because Whisper revises
    punctuation more aggressively than Korean grammar endings.
    """
    min_sentence_chars: int = 10     # min chars since last flush for sentence flush
    min_clause_chars: int = 30       # min chars for clause flush (comma/semicolon)
    stability_count: int = 4         # consecutive stable detections before flushing
    max_no_flush_s: float = 15.0     # emergency fallback: force flush after this many seconds (must be < 25s clip threshold)

    # ── Internal state ──
    flushed_len: int = field(default=0, init=False)
    _prev_text: str = field(default="", init=False)
    _prev_punct_pos: int = field(default=-1, init=False)
    _prev_punct_type: str = field(default="", init=False)
    _stable_count: int = field(default=0, init=False)
    _last_flush_time: float = field(default_factory=time.monotonic, init=False)

    def check(self, text: str) -> FlushDecision:
        """Check streaming partial text for punctuation boundaries."""
        unflushed = text[self.flushed_len:]
        stripped = unflushed.strip()

        # Emergency fallback
        if (stripped and
                time.monotonic() - self._last_flush_time > self.max_no_flush_s):
            self._reset_stability()
            return FlushDecision("sentence", stripped, len(text), "timeout")

        if len(stripped) < self.min_sentence_chars:
            self._prev_text = text
            return FlushDecision("none", "", 0, "")

        # Scan for the latest sentence punctuation (.!?) in the unflushed text
        best_match = None
        best_type = None
        for m in _SENTENCE_PUNCT.finditer(unflushed):
            pos = m.start()
            text_before = unflushed[:pos + 1].strip()
            if len(text_before) >= self.min_sentence_chars:
                best_match = pos + 1  # include the punctuation
                best_type = "sentence"

        # If no sentence punct, check for clause punct with longer minimum
        if best_match is None:
            for m in _CLAUSE_PUNCT.finditer(unflushed):
                pos = m.start()
                text_before = unflushed[:pos + 1].strip()
                if len(text_before) >= self.min_clause_chars:
                    best_match = pos + 1
                    best_type = "phrase"

        if best_match is None:
            self._reset_stability()
            self._prev_text = text
            return FlushDecision("none", "", 0, "")

        # Stability check: same punctuation position across consecutive updates
        abs_pos = self.flushed_len + best_match
        if best_type == self._prev_punct_type and abs_pos == self._prev_punct_pos:
            self._stable_count += 1
        else:
            self._prev_punct_pos = abs_pos
            self._prev_punct_type = best_type
            self._stable_count = 1

        if self._stable_count >= self.stability_count:
            flush_text = unflushed[:best_match].strip()
            self._reset_stability()
            self._prev_text = text
            return FlushDecision(best_type, flush_text, abs_pos,
                                 f"punct:{unflushed[best_match - 1]}")

        self._prev_text = text
        return FlushDecision("none", "", 0, "")

    def on_flushed(self, flush_type: str, end_pos: int):
        self.flushed_len = end_pos
        self._reset_stability()
        self._last_flush_time = time.monotonic()

    def reset(self):
        self.flushed_len = 0
        self._reset_stability()
        self._last_flush_time = time.monotonic()
        self._prev_text = ""

    def _reset_stability(self):
        self._prev_punct_pos = -1
        self._prev_punct_type = ""
        self._stable_count = 0
