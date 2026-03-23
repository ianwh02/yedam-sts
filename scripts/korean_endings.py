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
    # Casual/spoken declarative (common in sermons)
    "거든요", "잖아요",
    "겠죠", "있죠", "없죠", "하죠",
    # Noun + copula endings
    "입니다", "이에요", "예요",
    # Sermon-specific boundaries (always end a thought)
    "아멘", "할렐루야",
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
    "지만", "는데", "은데", "인데",
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
    # Through / passing (common in sermon narrative)
    "통해서", "을통해",
    # Quoting
    "라고", "다고", "냐고",
    # While / during
    "으면서", "면서",
    # Listing with emphasis
    "뿐만", "아니라",
], key=len, reverse=True)

# Punctuation that Whisper may insert (strip before checking endings)
TRAILING_PUNCT = set(".,!?;:…·~\"'""''")

# Short markers that always trigger a sentence flush regardless of min_chars
_ALWAYS_FLUSH_MARKERS = {"아멘", "할렐루야"}


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
    min_phrase_chars: int = 12      # min chars since last flush for phrase flush
    min_sentence_chars: int = 6     # min chars since last flush for sentence flush
    stability_count: int = 2        # consecutive stable detections before flushing
    max_no_flush_s: float = 30.0    # emergency fallback: force flush after this many seconds

    # ── Internal state ──
    flushed_len: int = field(default=0, init=False)
    _prev_ending: str = field(default="", init=False)
    _prev_ending_pos: int = field(default=-1, init=False)
    _stable_count: int = field(default=0, init=False)
    _last_flush_time: float = field(default_factory=time.monotonic, init=False)
    _prev_text: str = field(default="", init=False)

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
        # But always allow sermon-specific markers (아멘, 할렐루야) regardless of length
        min_chars = min(self.min_sentence_chars, self.min_phrase_chars)
        if len(stripped) < min_chars:
            # Check for short standalone markers before giving up
            if stripped not in _ALWAYS_FLUSH_MARKERS:
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
            ending = self._match_sentence_ending(clean)
            if ending:
                chars_since_flush = token_end_pos  # relative to unflushed start
                is_marker = clean in _ALWAYS_FLUSH_MARKERS
                if chars_since_flush >= self.min_sentence_chars or is_marker:
                    abs_pos = self.flushed_len + token_end_pos
                    if self._check_stability(ending, abs_pos):
                        flush_text = unflushed[:token_end_pos].strip()
                        self._prev_text = text
                        return FlushDecision("sentence", flush_text, abs_pos,
                                             f"~{ending}")

            # Check phrase endings
            ending = self._match_phrase_ending(clean)
            if ending:
                chars_since_flush = token_end_pos
                if chars_since_flush >= self.min_phrase_chars:
                    abs_pos = self.flushed_len + token_end_pos
                    if self._check_stability(ending, abs_pos):
                        flush_text = unflushed[:token_end_pos].strip()
                        self._prev_text = text
                        return FlushDecision("phrase", flush_text, abs_pos,
                                             f"~{ending}")

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
        # Standalone endings that can be the entire token
        STANDALONE_OK = {"아멘", "할렐루야", "합시다", "읍시다", "갑시다", "하자", "보자", "가자"}

        for ending in SENTENCE_ENDINGS:
            if token.endswith(ending):
                if len(token) > len(ending) or ending in STANDALONE_OK:
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
        """Check if the same ending has been detected at the same position
        across consecutive updates (stability gate)."""
        if ending == self._prev_ending and pos == self._prev_ending_pos:
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
