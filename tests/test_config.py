"""Tests for orchestrator configuration defaults (CPU-only)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Setup orchestrator imports (same pattern as test_orchestrator.py) ──
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

    from src.config import Settings

    HAS_CONFIG = True
except Exception as e:
    HAS_CONFIG = False
    _skip_reason = str(e)

needs_config = pytest.mark.skipif(
    not HAS_CONFIG,
    reason=f"Config import failed: {_skip_reason if not HAS_CONFIG else ''}",
)


@needs_config
class TestSettingsDefaults:
    """Verify default settings are sensible types and values."""

    def _make_settings(self, **overrides):
        """Create Settings with env vars suppressed."""
        return Settings(**overrides, _env_file=None)

    def test_port_is_positive_int(self):
        s = self._make_settings()
        assert isinstance(s.orchestrator_port, int)
        assert s.orchestrator_port > 0

    def test_service_urls_have_scheme(self):
        s = self._make_settings()
        assert s.stt_ws_url.startswith("ws://") or s.stt_ws_url.startswith("wss://")
        assert s.llm_api_url.startswith("http://") or s.llm_api_url.startswith("https://")
        assert s.tts_api_url.startswith("http://") or s.tts_api_url.startswith("https://")

    def test_sample_rate_positive(self):
        s = self._make_settings()
        assert s.audio_sample_rate > 0

    def test_session_limits_positive(self):
        s = self._make_settings()
        assert s.max_concurrent_sessions > 0
        assert s.max_session_duration_seconds > 0

    def test_sentence_split_ordering(self):
        """min_words_sentence < min_words_comma < max_words_per_chunk."""
        s = self._make_settings()
        assert s.tts_min_words_sentence_split < s.tts_min_words_comma_split
        assert s.tts_min_words_comma_split < s.tts_max_words_per_chunk

    def test_tts_stale_threshold_positive(self):
        s = self._make_settings()
        assert s.tts_stale_threshold > 0
        assert s.tts_stale_max_age_seconds > 0

    def test_rms_target_in_range(self):
        s = self._make_settings()
        assert 0 < s.audio_rms_target <= 1.0
