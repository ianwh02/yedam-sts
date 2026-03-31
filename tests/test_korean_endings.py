"""Tests for Korean ending detection (adapted from scripts/test_korean_endings.py).

Tests KoreanEndingDetector, PunctuationFlushDetector, and Hangul jamo helpers.
These are CPU-only — no GPU/model dependencies.
"""

import time

import pytest

# Import directly — stt-server is added to path in conftest
from whisper_live.korean_endings import (
    BIEUP,
    BIEUP_NIDA_ENDINGS,
    CLAUSE_CONNECTIVES,
    NIEUN,
    RIEUL,
    FlushDecision,
    KoreanEndingDetector,
    PunctuationFlushDetector,
    get_jongseong,
    is_hangul,
)


# ── Jamo helpers ──────────────────────────────────────────────────────


class TestJamoHelpers:
    def test_bieup_batchim(self):
        """ㅂ batchim detection."""
        assert get_jongseong("합") == BIEUP
        assert get_jongseong("갑") == BIEUP
        assert get_jongseong("답") == BIEUP
        assert get_jongseong("입") == BIEUP

    def test_no_batchim(self):
        assert get_jongseong("하") == 0
        assert get_jongseong("가") == 0

    def test_other_batchim(self):
        assert get_jongseong("한") == NIEUN
        assert get_jongseong("할") == RIEUL

    def test_non_hangul_returns_negative(self):
        assert get_jongseong("A") == -1
        assert get_jongseong("1") == -1
        assert get_jongseong("!") == -1

    def test_is_hangul(self):
        assert is_hangul("가")
        assert is_hangul("힣")
        assert not is_hangul("A")
        assert not is_hangul("1")
        assert not is_hangul(" ")


# ── KoreanEndingDetector ──────────────────────────────────────────────


class TestSentenceEndings:
    """Verify sentence-ending patterns are detected."""

    @pytest.fixture
    def detector(self):
        return KoreanEndingDetector(stability_count=1, min_sentence_chars=3)

    @pytest.mark.parametrize(
        "text,expected_ending",
        [
            ("우리가 능히 하십니다 ", "니다"),
            ("그것이 사실입니까 ", "ㅂ니까"),
            ("정말 감사해요 ", "해요"),
            ("그렇죠 ", "죠"),
            ("이미 지나갔다 ", "갔다"),
            ("그랬었다 ", "었다"),
            ("함께 읽으세요 ", "세요"),
            ("함께 합시다 ", "합시다"),
            ("정말 좋은데요 ", "은데요"),
            ("그렇잖아요 ", "잖아요"),
            ("어디 갈까요 ", "까요"),
            ("정말 그렇군요 ", "군요"),
        ],
    )
    def test_sentence_endings(self, detector, text, expected_ending):
        detector.check(text)
        result = detector.check(text)
        assert result.flush_type == "sentence", f"'{text.strip()}': got {result.flush_type}"
        assert expected_ending in result.reason


class TestPhraseEndings:
    @pytest.fixture
    def detector(self):
        return KoreanEndingDetector(stability_count=1, min_phrase_chars=3)

    @pytest.mark.parametrize(
        "text,expected_ending",
        [
            ("힘들기 때문에 ", "때문에"),
            ("그랬으니까 ", "니까"),
            ("힘들어서 ", "어서"),
            ("힘들지만 ", "지만"),
            ("있다면 ", "다면"),
            ("하려고 ", "려고"),
        ],
    )
    def test_phrase_endings(self, detector, text, expected_ending):
        detector.check(text)
        result = detector.check(text)
        assert result.flush_type == "phrase", f"'{text.strip()}': got {result.flush_type}"
        assert expected_ending in result.reason


class TestExtraFlushMarkers:
    def test_domain_markers_detected(self):
        markers = {"아멘", "할렐루야"}
        d = KoreanEndingDetector(stability_count=1, min_sentence_chars=3, extra_flush_markers=markers)
        for marker in markers:
            d.reset()
            d.check(marker + " ")
            result = d.check(marker + " ")
            assert result.flush_type == "sentence"
            assert marker in result.reason

    def test_without_markers_no_match(self):
        d = KoreanEndingDetector(stability_count=1, min_sentence_chars=3)
        for marker in ("아멘", "할렐루야"):
            d.reset()
            d.check(marker + " ")
            result = d.check(marker + " ")
            assert result.flush_type == "none"


class TestNoFalsePositives:
    @pytest.fixture
    def detector(self):
        return KoreanEndingDetector(stability_count=1)

    @pytest.mark.parametrize(
        "text",
        [
            "가장 큰",
            "습니",
            "우리나라",
            "서울시",
            "대학교",
            "아",
            "hello world",
        ],
    )
    def test_no_match(self, detector, text):
        detector.check(text + " ")
        result = detector.check(text + " ")
        assert result.flush_type == "none", f"'{text}': got {result.flush_type}"


class TestSentenceOverPhrase:
    def test_sentence_priority(self):
        """습니까 should match as sentence (not 니까 as phrase)."""
        d = KoreanEndingDetector(stability_count=1)
        d.check("그것이 사실입니까 ")
        result = d.check("그것이 사실입니까 ")
        assert result.flush_type == "sentence"


class TestStabilityGate:
    def test_first_check_no_flush(self):
        d = KoreanEndingDetector(stability_count=2)
        result = d.check("하나님이 하십니다 ")
        assert result.flush_type == "none"

    def test_second_check_flushes(self):
        d = KoreanEndingDetector(stability_count=2)
        d.check("하나님이 하십니다 ")
        result = d.check("하나님이 하십니다 ")
        assert result.flush_type == "sentence"

    def test_revision_resets_stability(self):
        d = KoreanEndingDetector(stability_count=2)
        d.check("하나님이 하십니다 ")
        result = d.check("하나님이 하십니다만 우리가 ")
        assert result.flush_type == "none"


class TestFlushedLenTracking:
    def test_phrase_then_sentence(self):
        d = KoreanEndingDetector(stability_count=1, min_phrase_chars=3, min_sentence_chars=3)

        # First phrase flush
        d.check("힘들지만 ")
        result = d.check("힘들지만 ")
        assert result.flush_type == "phrase"
        d.on_flushed("phrase", result.end_pos)

        # Sentence flush on continuation
        d.check("힘들지만 감사합니다 ")
        result = d.check("힘들지만 감사합니다 ")
        assert result.flush_type == "sentence"
        assert "감사합니다" in result.text
        d.on_flushed("sentence", result.end_pos)
        assert d.flushed_len == result.end_pos


class TestBieupIrregular:
    @pytest.mark.parametrize(
        "text",
        [
            "정말 감사합니다 ",
            "어디 갑니까 ",
        ],
    )
    def test_bieup_forms(self, text):
        d = KoreanEndingDetector(stability_count=1)
        d.check(text)
        result = d.check(text)
        assert result.flush_type == "sentence"


class TestPunctuationStripping:
    def test_period_does_not_block(self):
        d = KoreanEndingDetector(stability_count=1)
        d.check("하나님이 하십니다. ")
        result = d.check("하나님이 하십니다. ")
        assert result.flush_type == "sentence"


class TestMinChars:
    def test_too_short_for_sentence(self):
        d = KoreanEndingDetector(stability_count=1, min_sentence_chars=6, min_phrase_chars=12)
        d.check("해요 ")
        result = d.check("해요 ")
        assert result.flush_type == "none"

    def test_long_enough_for_sentence(self):
        d = KoreanEndingDetector(stability_count=1, min_sentence_chars=6, min_phrase_chars=12)
        d.check("정말 감사해요 ")
        result = d.check("정말 감사해요 ")
        assert result.flush_type == "sentence"


class TestEmergencyTimeout:
    def test_timeout_forces_flush(self):
        d = KoreanEndingDetector(stability_count=1, max_no_flush_s=0.1)
        d.check("가장 큰 위기 속에서 ")
        time.sleep(0.15)
        result = d.check("가장 큰 위기 속에서 ")
        assert result.flush_type == "sentence"
        assert "timeout" in result.reason


class TestStreamingSimulation:
    def test_phrase_then_sentence(self):
        d = KoreanEndingDetector(stability_count=2, min_phrase_chars=3, min_sentence_chars=3)
        flushes = []

        partials = [
            "힘들",
            "힘들지",
            "힘들지만",
            "힘들지만 ",
            "힘들지만 ",
            "힘들지만 감",
            "힘들지만 감사",
            "힘들지만 감사합니",
            "힘들지만 감사합니다",
            "힘들지만 감사합니다 ",
            "힘들지만 감사합니다 ",
        ]

        for p in partials:
            result = d.check(p)
            if result.flush_type != "none":
                flushes.append((result.flush_type, result.text, result.reason))
                d.on_flushed(result.flush_type, result.end_pos)

        assert len(flushes) >= 1
        assert flushes[0][0] == "phrase"
        assert "지만" in flushes[0][2]
        if len(flushes) >= 2:
            assert flushes[1][0] == "sentence"


class TestClauseConnectives:
    def test_connective_flushes_before(self):
        d = KoreanEndingDetector(stability_count=1, min_phrase_chars=3)
        text = "그것이 중요한 일이다 그래서 "
        d.check(text)
        result = d.check(text)
        # Should flush before 그래서 OR detect the sentence ending 일이다
        assert result.flush_type in ("phrase", "sentence")


class TestReset:
    def test_reset_clears_state(self):
        d = KoreanEndingDetector(stability_count=2)
        d.check("하나님이 하십니다 ")
        d.reset()
        assert d.flushed_len == 0
        # After reset, first check should not flush (stability reset)
        result = d.check("하나님이 하십니다 ")
        assert result.flush_type == "none"


# ── PunctuationFlushDetector ──────────────────────────────────────────


class TestPunctuationFlushDetector:
    @pytest.fixture
    def detector(self):
        return PunctuationFlushDetector(stability_count=1, min_sentence_chars=5)

    def test_period_flush(self, detector):
        text = "Hello world. "
        detector.check(text)
        result = detector.check(text)
        assert result.flush_type == "sentence"

    def test_question_flush(self, detector):
        text = "How are you? "
        detector.check(text)
        result = detector.check(text)
        assert result.flush_type == "sentence"

    def test_too_short_no_flush(self, detector):
        text = "Hi. "
        detector.check(text)
        result = detector.check(text)
        assert result.flush_type == "none"

    def test_comma_clause_flush(self):
        d = PunctuationFlushDetector(stability_count=1, min_clause_chars=10)
        text = "This is a long clause with many words, "
        d.check(text)
        result = d.check(text)
        assert result.flush_type == "phrase"

    def test_stability_required(self):
        d = PunctuationFlushDetector(stability_count=3, min_sentence_chars=5)
        text = "Hello world. "
        assert d.check(text).flush_type == "none"
        assert d.check(text).flush_type == "none"
        assert d.check(text).flush_type == "sentence"

    def test_timeout_fallback(self):
        d = PunctuationFlushDetector(stability_count=1, max_no_flush_s=0.1,
                                     min_sentence_chars=5)
        d.check("abcdef no punct here ")
        time.sleep(0.15)
        result = d.check("abcdef no punct here ")
        assert result.flush_type == "sentence"
        assert "timeout" in result.reason

    def test_reset(self):
        d = PunctuationFlushDetector(stability_count=1, min_sentence_chars=5)
        d.check("Hello world. ")
        d.reset()
        assert d.flushed_len == 0

    def test_on_flushed_advances(self):
        d = PunctuationFlushDetector(stability_count=1, min_sentence_chars=5)
        d.check("Hello world. ")
        result = d.check("Hello world. ")
        assert result.flush_type == "sentence"
        d.on_flushed(result.flush_type, result.end_pos)
        assert d.flushed_len == result.end_pos
