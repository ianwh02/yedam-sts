"""Tests for orchestrator pipeline components (CPU-only).

Tests session state, sentence boundary detection, prompt building,
broadcast hub, and processor logic. No running services required.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add orchestrator to path
_orch_root = Path(__file__).parent.parent / "orchestrator"
_orch_src = _orch_root / "src"
if str(_orch_src) not in sys.path:
    sys.path.insert(0, str(_orch_src))


# ── Mock settings before importing orchestrator modules ──────────────
# pydantic-settings reads env vars at import; mock to avoid side effects

_mock_settings = MagicMock()
_mock_settings.stt_initial_prompt_vocab = ""
_mock_settings.stt_initial_prompt_vocab_path = ""
_mock_settings.llm_system_prompt_path = ""
_mock_settings.llm_context_window_segments = 3
_mock_settings.tts_min_words_sentence_split = 4
_mock_settings.tts_min_words_comma_split = 8
_mock_settings.tts_max_words_per_chunk = 35
_mock_settings.tts_continuous_enabled = True
_mock_settings.llm_api_url = "http://localhost:8000/v1"

sys.modules.setdefault("config", MagicMock())
# Patch the settings singleton before importing
with patch.dict(sys.modules, {}):
    pass

# We need to patch at the module level
import importlib

# Create a config module mock
_config_mod = MagicMock()
_config_mod.settings = _mock_settings
sys.modules["config"] = _config_mod

# Now import orchestrator modules with patched settings
# We need to handle the relative imports by setting up the package structure
try:
    # Try direct import if orchestrator is on path
    sys.path.insert(0, str(_orch_root))

    # Mock opuslib since it may not be available
    sys.modules.setdefault("opuslib", MagicMock())
    sys.modules.setdefault("opuslib.api", MagicMock())
    sys.modules.setdefault("opuslib.api.encoder", MagicMock())
    sys.modules.setdefault("rnnoise_wrapper", MagicMock())

    # Set up src as a package
    if "src" not in sys.modules:
        import types

        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_orch_src)]
        sys.modules["src"] = src_pkg

    # Import config into the src package
    src_config = types.ModuleType("src.config")
    src_config.settings = _mock_settings
    sys.modules["src.config"] = src_config

    # Now do the imports
    from src.pipeline.session import TranscriptSegment, TranslationSession
    from src.pipeline.broadcast import BroadcastHub
    from src.pipeline.orchestrator import SentenceBoundaryDetector
    from src.llm.prompts import _DEFAULT_SYSTEM_PROMPT_TEMPLATE, _LANG_NAMES, build_translation_prompt
    from src.processors.passthrough import PassthroughProcessor

    HAS_ORCHESTRATOR = True
except Exception as e:
    HAS_ORCHESTRATOR = False
    _orch_skip_reason = str(e)

needs_orchestrator = pytest.mark.skipif(
    not HAS_ORCHESTRATOR,
    reason=f"Orchestrator import failed: {_orch_skip_reason if not HAS_ORCHESTRATOR else ''}"
)


# ── TranslationSession tests ─────────────────────────────────────────


@needs_orchestrator
class TestTranslationSession:
    def test_defaults(self):
        s = TranslationSession(session_id="test-1")
        assert s.source_lang == "ko"
        assert s.target_lang == "en"
        assert s.processor_type == "translation"
        assert s.is_active is False
        assert s.completed_segment_count == 0
        assert s.transcript == []

    def test_add_segment(self):
        s = TranslationSession(session_id="test-1")
        seg = s.add_segment("안녕하세요")
        assert seg.korean == "안녕하세요"
        assert seg.index == 0
        assert s.completed_segment_count == 1

        seg2 = s.add_segment("감사합니다")
        assert seg2.index == 1
        assert s.completed_segment_count == 2

    def test_get_llm_context(self):
        s = TranslationSession(session_id="test-1")
        # Add segments with translations
        seg1 = s.add_segment("안녕하세요")
        seg1.english = "Hello"
        seg2 = s.add_segment("감사합니다")
        seg2.english = "Thank you"

        context = s.get_llm_context(window_size=5)
        assert len(context) == 2
        assert context[0]["korean"] == "안녕하세요"
        assert context[0]["english"] == "Hello"

    def test_get_llm_context_excludes_untranslated(self):
        s = TranslationSession(session_id="test-1")
        seg1 = s.add_segment("안녕하세요")
        seg1.english = "Hello"
        s.add_segment("아직 번역 안 됨")  # no english

        context = s.get_llm_context()
        assert len(context) == 1

    def test_get_llm_context_window_size(self):
        s = TranslationSession(session_id="test-1")
        for i in range(10):
            seg = s.add_segment(f"문장 {i}")
            seg.english = f"Sentence {i}"

        context = s.get_llm_context(window_size=3)
        assert len(context) == 3
        assert context[0]["korean"] == "문장 7"

    def test_cancel(self):
        s = TranslationSession(session_id="test-1")
        assert s.is_cancelled is False
        s.cancel()
        assert s.is_cancelled is True

    def test_duration(self):
        s = TranslationSession(session_id="test-1")
        assert s.duration_seconds >= 0


@needs_orchestrator
class TestTranscriptSegment:
    def test_creation(self):
        seg = TranscriptSegment(index=0, korean="테스트")
        assert seg.index == 0
        assert seg.korean == "테스트"
        assert seg.english is None
        assert seg.timestamp > 0


# ── SentenceBoundaryDetector tests ────────────────────────────────────


@needs_orchestrator
class TestSentenceBoundaryDetector:
    def test_sentence_punctuation_splits(self):
        d = SentenceBoundaryDetector()
        # Feed enough words to meet min_words_sentence_split
        result = d.feed("This is a sentence. ")
        assert result is not None
        sentence, idx = result
        assert sentence == "This is a sentence."
        assert idx == 0

    def test_short_sentence_no_split(self):
        """Short fragments like 'Yes.' shouldn't split."""
        d = SentenceBoundaryDetector()
        result = d.feed("Yes. ")
        # Too few words (1 < min_words_sentence_split=4)
        assert result is None

    def test_question_mark_splits(self):
        d = SentenceBoundaryDetector()
        result = d.feed("How are you doing? ")
        assert result is not None

    def test_exclamation_splits(self):
        d = SentenceBoundaryDetector()
        result = d.feed("This is really great! ")
        assert result is not None

    def test_comma_split_with_enough_words(self):
        d = SentenceBoundaryDetector()
        # Feed 8+ words with comma
        text = "The quick brown fox jumped over the lazy, "
        result = d.feed(text)
        assert result is not None

    def test_comma_no_split_when_short(self):
        d = SentenceBoundaryDetector()
        result = d.feed("Hello, world ")
        assert result is None

    def test_max_length_force_split(self):
        d = SentenceBoundaryDetector()
        # Feed 35+ words without any punctuation
        words = " ".join([f"word{i}" for i in range(36)])
        result = d.feed(words + " ")
        assert result is not None

    def test_flush_returns_remaining(self):
        d = SentenceBoundaryDetector()
        d.feed("Some remaining text")
        result = d.flush()
        assert result is not None
        sentence, idx = result
        assert sentence == "Some remaining text"

    def test_flush_empty_returns_none(self):
        d = SentenceBoundaryDetector()
        result = d.flush()
        assert result is None

    def test_sentence_index_increments(self):
        d = SentenceBoundaryDetector()
        r1 = d.feed("This is sentence one. ")
        r2 = d.feed("This is sentence two. ")
        assert r1 is not None and r1[1] == 0
        assert r2 is not None and r2[1] == 1

    def test_multi_sentence_stream(self):
        d = SentenceBoundaryDetector()
        results = []

        tokens = "The quick brown fox jumped over the lazy dog. And then the cat ran away quickly. "
        for char in tokens:
            r = d.feed(char)
            if r:
                results.append(r)

        r = d.flush()
        if r:
            results.append(r)

        # Should get at least 1 sentence split
        assert len(results) >= 1


# ── Prompt building tests ─────────────────────────────────────────────


@needs_orchestrator
class TestPromptBuilding:
    def test_default_prompt_structure(self):
        messages = build_translation_prompt("안녕하세요")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "안녕하세요"

    def test_system_prompt_has_no_think(self):
        messages = build_translation_prompt("테스트")
        assert "/no_think" in messages[0]["content"]

    def test_context_as_conversation_turns(self):
        segments = [
            {"korean": "첫 번째", "english": "First"},
            {"korean": "두 번째", "english": "Second"},
        ]
        messages = build_translation_prompt("세 번째", recent_segments=segments)
        # system + 2 user/assistant pairs + current user = 6 messages
        assert len(messages) == 6
        assert messages[0]["role"] == "system"
        # Context turns: user=Korean, assistant=English
        assert messages[1] == {"role": "user", "content": "첫 번째"}
        assert messages[2] == {"role": "assistant", "content": "First"}
        assert messages[3] == {"role": "user", "content": "두 번째"}
        assert messages[4] == {"role": "assistant", "content": "Second"}
        # Current input is last user message
        assert messages[5] == {"role": "user", "content": "세 번째"}
        # System prompt should NOT contain context Korean text
        assert "첫 번째" not in messages[0]["content"]

    def test_previous_chunk_ignored(self):
        """previous_chunk is no longer used — conversation turns provide context."""
        messages = build_translation_prompt(
            "감사합니다",
            previous_chunk="안녕하세요"
        )
        # Only system + user (previous_chunk dropped)
        assert len(messages) == 2
        assert messages[-1]["content"] == "감사합니다"

    def test_language_names(self):
        assert _LANG_NAMES["ko"] == "Korean"
        assert _LANG_NAMES["en"] == "English"
        assert _LANG_NAMES["zh"] == "Chinese"
        assert _LANG_NAMES["ja"] == "Japanese"

    def test_default_template_has_placeholders(self):
        result = _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
            source_lang_name="Korean",
            target_lang_name="English",
        )
        assert "Korean" in result
        assert "English" in result


# ── BroadcastHub tests ────────────────────────────────────────────────


@needs_orchestrator
class TestBroadcastHub:
    @pytest.fixture
    def hub(self):
        return BroadcastHub()

    def test_initial_empty(self, hub):
        assert hub.count == 0

    @pytest.mark.asyncio
    async def test_add_listener(self, hub):
        ws = AsyncMock()
        await hub.add(ws)
        assert hub.count == 1

    @pytest.mark.asyncio
    async def test_remove_listener(self, hub):
        ws = AsyncMock()
        await hub.add(ws)
        await hub.remove(ws)
        assert hub.count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, hub):
        ws = AsyncMock()
        await hub.remove(ws)  # Should not raise
        assert hub.count == 0

    @pytest.mark.asyncio
    async def test_broadcast_text(self, hub):
        ws = AsyncMock()
        await hub.add(ws)
        await hub.broadcast_text({"type": "test", "data": "hello"})
        ws.send_text.assert_called_once()
        payload = ws.send_text.call_args[0][0]
        assert json.loads(payload)["type"] == "test"

    @pytest.mark.asyncio
    async def test_broadcast_binary(self, hub):
        ws = AsyncMock()
        await hub.add(ws)
        data = b"\x00\x01\x02\x03"
        await hub.broadcast_binary(data)
        ws.send_bytes.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_dead_listener_removed(self, hub):
        ws = AsyncMock()
        ws.send_text.side_effect = ConnectionError("closed")
        await hub.add(ws)
        assert hub.count == 1
        await hub.broadcast_text({"type": "test"})
        assert hub.count == 0

    @pytest.mark.asyncio
    async def test_multi_listener(self, hub):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await hub.add(ws1)
        await hub.add(ws2)
        assert hub.count == 2
        await hub.broadcast_text({"type": "test"})
        ws1.send_text.assert_called_once()
        ws2.send_text.assert_called_once()


# ── PassthroughProcessor tests ────────────────────────────────────────


@needs_orchestrator
class TestPassthroughProcessor:
    @pytest.mark.asyncio
    async def test_yields_input_unchanged(self):
        proc = PassthroughProcessor()
        await proc.initialize()
        results = []
        async for token in proc.process("Hello world", {}):
            results.append(token)
        assert results == ["Hello world"]

    @pytest.mark.asyncio
    async def test_korean_text(self):
        proc = PassthroughProcessor()
        await proc.initialize()
        results = []
        async for token in proc.process("안녕하세요", {}):
            results.append(token)
        assert results == ["안녕하세요"]


# ── Settings tests ────────────────────────────────────────────────────


@needs_orchestrator
class TestSettingsDefaults:
    """Verify the mock settings match expected patterns."""

    def test_sentence_split_thresholds(self):
        assert _mock_settings.tts_min_words_sentence_split > 0
        assert _mock_settings.tts_min_words_comma_split > _mock_settings.tts_min_words_sentence_split
        assert _mock_settings.tts_max_words_per_chunk > _mock_settings.tts_min_words_comma_split
