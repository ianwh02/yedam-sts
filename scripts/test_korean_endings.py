#!/usr/bin/env python3
"""Unit tests for Korean ending detection module."""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
from korean_endings import (
    KoreanEndingDetector,
    get_jongseong,
    is_hangul,
    BIEUP,
)


def test_jamo():
    """Test Hangul jamo decomposition."""
    # ㅂ batchim
    assert get_jongseong("합") == BIEUP, f"합 should have ㅂ batchim, got {get_jongseong('합')}"
    assert get_jongseong("갑") == BIEUP
    assert get_jongseong("답") == BIEUP
    assert get_jongseong("입") == BIEUP

    # No batchim
    assert get_jongseong("하") == 0
    assert get_jongseong("가") == 0

    # Other batchim
    assert get_jongseong("한") != BIEUP  # ㄴ batchim
    assert get_jongseong("할") != BIEUP  # ㄹ batchim

    # Non-Hangul
    assert get_jongseong("A") == -1
    assert get_jongseong("1") == -1

    assert is_hangul("가")
    assert is_hangul("힣")
    assert not is_hangul("A")
    assert not is_hangul("1")

    print("  ✓ jamo decomposition")


def test_sentence_endings():
    """Test that sentence endings are correctly detected."""
    d = KoreanEndingDetector(stability_count=1, min_sentence_chars=3)  # low min for unit tests

    cases = [
        # (text, expected_type, expected_ending_substring)
        ("하나님이 능히 하십니다 ", "sentence", "니다"),  # matched as ㅂ니다
        ("그것이 사실입니까 ", "sentence", "ㅂ니까"),
        ("정말 감사해요 ", "sentence", "해요"),
        ("그렇죠 ", "sentence", "죠"),
        ("이미 지나갔다 ", "sentence", "갔다"),
        ("그랬었다 ", "sentence", "었다"),
        ("함께 읽으세요 ", "sentence", "세요"),
        ("함께 합시다 ", "sentence", "합시다"),
        ("아멘 ", "sentence", "아멘"),
        ("할렐루야 ", "sentence", "할렐루야"),
        ("정말 좋은데요 ", "sentence", "은데요"),
        ("그렇잖아요 ", "sentence", "잖아요"),
        ("어디 갈까요 ", "sentence", "까요"),
        ("정말 그렇군요 ", "sentence", "군요"),
    ]

    for text, expected_type, expected_ending in cases:
        d.reset()
        # Feed twice for prev_text stability
        d.check(text)
        result = d.check(text)
        assert result.flush_type == expected_type, \
            f"'{text.strip()}': expected {expected_type}, got {result.flush_type} (reason: {result.reason})"
        if expected_ending:
            assert expected_ending in result.reason, \
                f"'{text.strip()}': expected ending containing '{expected_ending}', got '{result.reason}'"

    print("  ✓ sentence endings")


def test_phrase_endings():
    """Test that phrase endings are correctly detected."""
    d = KoreanEndingDetector(stability_count=1, min_phrase_chars=3)  # low min for unit tests

    cases = [
        ("힘들기 때문에 ", "phrase", "때문에"),
        ("그랬으니까 ", "phrase", "니까"),
        ("힘들어서 ", "phrase", "어서"),
        ("힘들지만 ", "phrase", "지만"),
        ("있다면 ", "phrase", "다면"),
        ("하려고 ", "phrase", "려고"),
    ]

    for text, expected_type, expected_ending in cases:
        d.reset()
        d.check(text)
        result = d.check(text)
        assert result.flush_type == expected_type, \
            f"'{text.strip()}': expected {expected_type}, got {result.flush_type} (reason: {result.reason})"
        if expected_ending:
            assert expected_ending in result.reason, \
                f"'{text.strip()}': expected '{expected_ending}' in reason, got '{result.reason}'"

    print("  ✓ phrase endings")


def test_no_match():
    """Test that non-endings don't trigger false positives."""
    d = KoreanEndingDetector(stability_count=1)

    cases = [
        "가장 큰",          # adjective, not a sentence ending
        "습니",              # incomplete ending
        "하나님",            # noun
        "교회",              # noun
        "예수",              # noun
        "아",                # too short
        "hello world",       # English
    ]

    for text in cases:
        d.reset()
        d.check(text + " ")
        result = d.check(text + " ")
        assert result.flush_type == "none", \
            f"'{text}': expected 'none', got {result.flush_type} (reason: {result.reason})"

    print("  ✓ no false positives")


def test_sentence_over_phrase():
    """Test that sentence endings take priority over phrase endings.
    습니까 should match as sentence (not 니까 as phrase)."""
    d = KoreanEndingDetector(stability_count=1)

    d.check("그것이 사실입니까 ")
    result = d.check("그것이 사실입니까 ")
    assert result.flush_type == "sentence", \
        f"습니까 should be sentence, got {result.flush_type}"
    assert "니까" in result.reason or "ㅂ니까" in result.reason

    print("  ✓ sentence takes priority over phrase")


def test_stability_gate():
    """Test that the stability gate prevents premature flushing."""
    d = KoreanEndingDetector(stability_count=2)

    # First check — ending detected but not stable yet
    result = d.check("하나님이 하십니다 ")
    assert result.flush_type == "none", \
        f"First check should be 'none' (stability gate), got {result.flush_type}"

    # Second check — same text, now stable
    result = d.check("하나님이 하십니다 ")
    assert result.flush_type == "sentence", \
        f"Second check should be 'sentence', got {result.flush_type}"

    print("  ✓ stability gate")


def test_stability_revision():
    """Test that revision resets the stability counter."""
    d = KoreanEndingDetector(stability_count=2)

    # First: ending detected
    d.check("하나님이 하십니다 ")

    # Second: ending revised away (added 만)
    result = d.check("하나님이 하십니다만 우리가 ")
    assert result.flush_type == "none", \
        f"After revision, should be 'none', got {result.flush_type}"

    print("  ✓ stability gate resets on revision")


def test_flushed_len_tracking():
    """Test that flushed_len correctly tracks across phrase and sentence flushes."""
    d = KoreanEndingDetector(stability_count=1, min_phrase_chars=3, min_sentence_chars=3)

    # Simulate streaming: "힘들지만 감사합니다"
    # First partial: phrase ending
    d.check("힘들지만 ")
    result = d.check("힘들지만 ")
    assert result.flush_type == "phrase", f"Expected phrase, got {result.flush_type}"
    assert "지만" in result.reason

    # Mark as flushed
    d.on_flushed("phrase", result.end_pos)

    # Continue — now unflushed part starts after "지만"
    d.check("힘들지만 감사합니다 ")
    result = d.check("힘들지만 감사합니다 ")
    assert result.flush_type == "sentence", f"Expected sentence, got {result.flush_type}"
    assert "감사합니다" in result.text, f"Flush text should contain only new part, got '{result.text}'"

    # Mark as flushed (sentence = reset)
    d.on_flushed("sentence", result.end_pos)
    assert d.flushed_len == 0, "Sentence flush should reset flushed_len"

    print("  ✓ flushed_len tracking")


def test_bieup_irregular():
    """Test ㅂ-irregular verb detection (ㅂ니다/ㅂ니까 forms)."""
    d = KoreanEndingDetector(stability_count=1)

    cases = [
        ("정말 감사합니다 ", "sentence", "ㅂ니다"),
        ("이것이 맞습니까 ", "sentence", "ㅂ니까"),  # 맞 has ㅈ not ㅂ... but 습니까 matches directly
        ("어디 갑니까 ", "sentence", "ㅂ니까"),
    ]

    for text, expected_type, expected_ending in cases:
        d.reset()
        d.check(text)
        result = d.check(text)
        assert result.flush_type == expected_type, \
            f"'{text.strip()}': expected {expected_type}, got {result.flush_type}"

    print("  ✓ ㅂ-irregular verbs")


def test_punctuation_stripping():
    """Test that trailing punctuation doesn't prevent ending detection."""
    d = KoreanEndingDetector(stability_count=1)

    # Whisper sometimes adds periods
    d.check("하나님이 하십니다. ")
    result = d.check("하나님이 하십니다. ")
    assert result.flush_type == "sentence", \
        f"Should detect ending through punctuation, got {result.flush_type}"

    print("  ✓ punctuation stripping")


def test_min_chars():
    """Test minimum character requirements."""
    d = KoreanEndingDetector(stability_count=1, min_sentence_chars=6, min_phrase_chars=12)

    # Too short for sentence flush
    d.check("해요 ")
    result = d.check("해요 ")
    assert result.flush_type == "none", \
        f"'해요' alone is too short, expected 'none', got {result.flush_type}"

    # Long enough
    d.reset()
    d.check("정말 감사해요 ")
    result = d.check("정말 감사해요 ")
    assert result.flush_type == "sentence", \
        f"'정말 감사해요' should be long enough, got {result.flush_type}"

    # Too short for phrase flush
    d.reset()
    d.check("지만 ")
    result = d.check("지만 ")
    assert result.flush_type == "none", \
        f"'지만' alone too short for phrase, got {result.flush_type}"

    print("  ✓ minimum char requirements")


def test_emergency_timeout():
    """Test the emergency fallback timer."""
    d = KoreanEndingDetector(stability_count=1, max_no_flush_s=0.1)  # very short for test

    # Feed text with no endings
    d.check("가장 큰 위기 속에서 ")
    time.sleep(0.15)
    result = d.check("가장 큰 위기 속에서 ")
    assert result.flush_type == "sentence", \
        f"Emergency timeout should force sentence flush, got {result.flush_type}"
    assert "timeout" in result.reason

    print("  ✓ emergency timeout")


def test_streaming_simulation():
    """Simulate realistic streaming partials from Whisper."""
    d = KoreanEndingDetector(stability_count=2, min_phrase_chars=3, min_sentence_chars=3)
    flushes = []

    # Simulate Whisper streaming "힘들지만 감사합니다"
    partials = [
        "힘들",
        "힘들지",
        "힘들지만",
        "힘들지만 ",        # space after — token complete
        "힘들지만 ",        # stable (same as prev)
        "힘들지만 감",
        "힘들지만 감사",
        "힘들지만 감사합니",
        "힘들지만 감사합니다",
        "힘들지만 감사합니다 ",   # space after
        "힘들지만 감사합니다 ",   # stable
    ]

    for p in partials:
        result = d.check(p)
        if result.flush_type != "none":
            flushes.append((result.flush_type, result.text, result.reason))
            d.on_flushed(result.flush_type, result.end_pos)

    assert len(flushes) >= 1, f"Expected at least 1 flush, got {len(flushes)}"

    # First flush should be phrase (지만)
    assert flushes[0][0] == "phrase", f"First flush should be phrase, got {flushes[0]}"
    assert "지만" in flushes[0][2]

    # Second flush should be sentence (습니다)
    if len(flushes) >= 2:
        assert flushes[1][0] == "sentence", f"Second flush should be sentence, got {flushes[1]}"

    print(f"  ✓ streaming simulation ({len(flushes)} flushes)")


def main():
    print("=== Korean Ending Detection Tests ===\n")

    tests = [
        test_jamo,
        test_sentence_endings,
        test_phrase_endings,
        test_no_match,
        test_sentence_over_phrase,
        test_stability_gate,
        test_stability_revision,
        test_flushed_len_tracking,
        test_bieup_irregular,
        test_punctuation_stripping,
        test_min_chars,
        test_emergency_timeout,
        test_streaming_simulation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: EXCEPTION: {e}")
            failed += 1

    print(f"\n{'─' * 40}")
    print(f"  {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
