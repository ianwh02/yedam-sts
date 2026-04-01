"""Tests for audio preprocessing pipeline (CPU-only, numpy)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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

    # Mock settings for AudioPreprocessor
    _mock_settings = MagicMock()
    _mock_settings.audio_rnnoise_enabled = False
    _mock_settings.audio_rms_normalize = True
    _mock_settings.audio_rms_target = 0.1
    _mock_settings.audio_sample_rate = 16000

    src_config = types.ModuleType("src.config")
    src_config.settings = _mock_settings
    sys.modules["src.config"] = src_config

    from src.audio.preprocess import AudioPreprocessor

    HAS_PREPROCESS = True
except Exception as e:
    HAS_PREPROCESS = False
    _skip_reason = str(e)

needs_preprocess = pytest.mark.skipif(
    not HAS_PREPROCESS,
    reason=f"Preprocess import failed: {_skip_reason if not HAS_PREPROCESS else ''}",
)


def _make_pcm(samples: np.ndarray) -> bytes:
    """Convert float32 numpy array to PCM bytes."""
    return samples.astype(np.float32).tobytes()


@needs_preprocess
class TestNormalizeRms:
    def test_zero_signal_unchanged(self):
        """Zero signal should pass through without division by zero."""
        proc = AudioPreprocessor()
        silence = np.zeros(1600, dtype=np.float32)
        result = np.frombuffer(proc.process(silence.tobytes()), dtype=np.float32)
        assert np.allclose(result, 0.0)

    def test_loud_signal_attenuated(self):
        """A loud signal (RMS >> target) should be reduced."""
        proc = AudioPreprocessor()
        loud = np.full(1600, 0.8, dtype=np.float32)
        result = np.frombuffer(proc.process(loud.tobytes()), dtype=np.float32)
        result_rms = np.sqrt(np.mean(result**2))
        # Should be close to target (0.1), not 0.8
        assert result_rms < 0.5

    def test_quiet_signal_amplified(self):
        """A quiet signal should be amplified toward target RMS."""
        proc = AudioPreprocessor()
        quiet = np.full(1600, 0.001, dtype=np.float32)
        result = np.frombuffer(proc.process(quiet.tobytes()), dtype=np.float32)
        result_rms = np.sqrt(np.mean(result**2))
        assert result_rms > 0.001

    def test_gain_clamped_to_max(self):
        """Very quiet signals should have gain clamped at 10x."""
        proc = AudioPreprocessor()
        # RMS = 0.001, target = 0.1 → ideal gain = 100x, clamped to 10x
        tiny = np.full(1600, 0.001, dtype=np.float32)
        result = np.frombuffer(proc.process(tiny.tobytes()), dtype=np.float32)
        # With gain clamped at 10x: 0.001 * 10 = 0.01
        assert np.allclose(result, 0.01, atol=1e-6)

    def test_output_clipped_to_unit(self):
        """Output values should be clipped to [-1.0, 1.0]."""
        proc = AudioPreprocessor()
        loud = np.full(1600, 0.9, dtype=np.float32)
        result = np.frombuffer(proc.process(loud.tobytes()), dtype=np.float32)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


@needs_preprocess
class TestProcessPassthrough:
    def test_both_disabled_passthrough(self):
        """When both rnnoise and normalize are off, audio passes through unchanged."""
        _mock_settings.audio_rnnoise_enabled = False
        _mock_settings.audio_rms_normalize = False
        proc = AudioPreprocessor()
        original = np.random.randn(1600).astype(np.float32) * 0.5
        result = np.frombuffer(proc.process(original.tobytes()), dtype=np.float32)
        assert np.allclose(result, original)
        # Restore
        _mock_settings.audio_rms_normalize = True

    def test_denoise_is_noop(self):
        """The _denoise method returns audio unchanged (placeholder)."""
        proc = AudioPreprocessor()
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        result = proc._denoise(audio)
        assert np.array_equal(result, audio)
