"""Tests for LLM prompt building (CPU-only)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Setup orchestrator imports ──
_orch_root = Path(__file__).parent.parent / "orchestrator"
_orch_src = _orch_root / "src"

try:
    import types

    if str(_orch_root) not in sys.path:
        sys.path.insert(0, str(_orch_root))

    sys.modules.setdefault("opuslib", MagicMock())
    sys.modules.setdefault("rnnoise_wrapper", MagicMock())

    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_orch_src)]
        sys.modules["src"] = src_pkg

    _mock_settings = MagicMock()
    _mock_settings.llm_system_prompt_path = ""
    _mock_settings.llm_context_window_segments = 3

    src_config = types.ModuleType("src.config")
    src_config.settings = _mock_settings
    sys.modules["src.config"] = src_config

    from src.llm.prompts import (
        _LANG_NAMES,
        _load_system_prompt,
        build_translation_prompt,
    )

    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    _skip_reason = str(e)

needs_prompts = pytest.mark.skipif(
    not HAS_PROMPTS,
    reason=f"Prompts import failed: {_skip_reason if not HAS_PROMPTS else ''}",
)


@needs_prompts
class TestLoadSystemPrompt:
    def test_default_prompt_contains_language_names(self):
        """Default template should include source and target language names."""
        prompt = _load_system_prompt("ko", "en")
        assert "Korean" in prompt
        assert "English" in prompt

    def test_unknown_language_uses_code(self):
        """Unknown language codes should be used as-is."""
        prompt = _load_system_prompt("xx", "yy")
        assert "xx" in prompt
        assert "yy" in prompt

    def test_nonexistent_file_falls_back(self):
        """If configured file doesn't exist, falls back to default."""
        _mock_settings.llm_system_prompt_path = "/nonexistent/path.txt"
        prompt = _load_system_prompt("ko", "en")
        assert "Korean" in prompt
        assert "English" in prompt
        _mock_settings.llm_system_prompt_path = ""  # restore

    def test_empty_path_uses_default(self):
        """Empty path uses default template."""
        _mock_settings.llm_system_prompt_path = ""
        prompt = _load_system_prompt("ko", "en")
        assert "/no_think" in prompt


@needs_prompts
class TestBuildTranslationPrompt:
    def test_basic_structure(self):
        """Should return system + user messages."""
        msgs = build_translation_prompt("안녕하세요", "ko", "en")
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "안녕하세요"

    def test_no_context(self):
        """With no recent_segments, should have system + user only."""
        msgs = build_translation_prompt("테스트", "ko", "en", recent_segments=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_empty_context(self):
        """Empty recent_segments list should behave like no context."""
        msgs = build_translation_prompt("테스트", "ko", "en", recent_segments=[])
        assert len(msgs) == 2

    def test_context_window_truncation(self):
        """Should only include last N segments (llm_context_window_segments=3)."""
        segments = [
            {"korean": f"한국어{i}", "english": f"English{i}"}
            for i in range(10)
        ]
        msgs = build_translation_prompt("현재", "ko", "en", recent_segments=segments)
        # system + 3 context pairs (6 msgs) + current user = 8
        assert len(msgs) == 8
        # The context should be the last 3 segments
        assert msgs[1]["content"] == "한국어7"
        assert msgs[2]["content"] == "English7"

    def test_context_skips_segments_without_english(self):
        """Segments without 'english' key should be skipped."""
        segments = [
            {"korean": "한국어1"},  # no english — skipped
            {"korean": "한국어2", "english": "English2"},
        ]
        msgs = build_translation_prompt("현재", "ko", "en", recent_segments=segments)
        # system + 1 context pair (2 msgs) + current user = 4
        assert len(msgs) == 4

    def test_different_languages(self):
        """Should work with non-Korean/English language pairs."""
        msgs = build_translation_prompt("こんにちは", "ja", "zh")
        system = msgs[0]["content"]
        assert "Japanese" in system
        assert "Chinese" in system

    def test_lang_names_mapping(self):
        """All mapped languages should resolve to full names."""
        for code, name in _LANG_NAMES.items():
            prompt = _load_system_prompt(code, "en")
            assert name in prompt
