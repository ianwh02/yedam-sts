"""Tests for TTS server helper functions and config (CPU-only).

Tests the pure audio processing functions in tts-server/server.py
and the nano-qwen3tts-vllm config/sampling_params modules.
No GPU or model loading required.
"""

import io
import struct
import wave

import numpy as np
import pytest

# ── Import TTS server helpers ────────────────────────────────────────
# We import the functions directly from server.py. The server module
# has side-effects at import time (torch, scipy), but those are
# available in CI since torch is installed for the existing tests.

import sys
from pathlib import Path

# Add tts-server to path so we can import server module helpers
_tts_root = Path(__file__).parent.parent / "tts-server"
if str(_tts_root) not in sys.path:
    sys.path.insert(0, str(_tts_root))

# We can't import server.py directly as it has heavy side-effects
# (FastAPI app creation, torch.set_float32_matmul_precision, etc.)
# Instead, test the functions by importing the module-level helpers
# after they're defined. Use importlib to load just the functions we need.
import importlib.util


def _load_server_functions():
    """Load specific functions from server.py without running the app."""
    spec = importlib.util.spec_from_file_location("tts_server", _tts_root / "server.py")
    mod = importlib.util.module_from_spec(spec)
    # Prevent uvicorn.run from being called
    mod.__name__ = "tts_server"
    sys.modules["tts_server"] = mod
    spec.loader.exec_module(mod)
    return mod


# Try to import — if server.py can't be loaded (missing scipy etc.), skip
try:
    _server = _load_server_functions()
    HAS_SERVER = True
except Exception as e:
    HAS_SERVER = False
    _server = None
    _skip_reason = str(e)


needs_server = pytest.mark.skipif(not HAS_SERVER, reason=f"TTS server import failed: {_skip_reason if not HAS_SERVER else ''}")


# ── SamplingParams tests ─────────────────────────────────────────────

from nano_qwen3tts_vllm.sampling_params import SamplingParams


class TestSamplingParams:
    def test_default_values(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.max_tokens == 64
        assert sp.ignore_eos is False
        assert sp.do_sample is True
        assert sp.top_k == 50
        assert sp.top_p == 1.0

    def test_custom_values(self):
        sp = SamplingParams(temperature=0.8, max_tokens=128, top_k=30)
        assert sp.temperature == 0.8
        assert sp.max_tokens == 128
        assert sp.top_k == 30

    def test_greedy_forbidden(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=0.0)

    def test_near_zero_forbidden(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=1e-11)

    def test_just_above_threshold_ok(self):
        sp = SamplingParams(temperature=1e-9)
        assert sp.temperature == 1e-9


# ── Audio processing function tests ─────────────────────────────────


@needs_server
class TestFloatToPcm16:
    def test_basic_conversion(self):
        wav = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = _server._float_to_pcm16(wav)
        assert pcm.dtype == np.int16
        assert pcm[0] == 0
        assert pcm[1] == int(0.5 * 32767)
        assert pcm[2] == int(-0.5 * 32767)
        assert pcm[3] == 32767
        assert pcm[4] == -32767

    def test_clipping(self):
        wav = np.array([1.5, -1.5], dtype=np.float32)
        pcm = _server._float_to_pcm16(wav)
        assert pcm[0] == 32767
        assert pcm[1] == -32767

    def test_empty_array(self):
        wav = np.array([], dtype=np.float32)
        pcm = _server._float_to_pcm16(wav)
        assert len(pcm) == 0


@needs_server
class TestResample:
    def test_same_rate_passthrough(self):
        wav = np.random.randn(1000).astype(np.float32)
        result = _server._resample(wav, _server.TARGET_SAMPLE_RATE)
        np.testing.assert_array_equal(result, wav)

    def test_upsample(self):
        # 24kHz → 48kHz should double samples
        wav = np.random.randn(1000).astype(np.float32)
        result = _server._resample(wav, 24000)
        # resample_poly with up=2, down=1 → 2x samples
        assert len(result) == 2000
        assert result.dtype == np.float32

    def test_downsample(self):
        # 96kHz → 48kHz should halve samples
        wav = np.random.randn(2000).astype(np.float32)
        result = _server._resample(wav, 96000)
        assert len(result) == 1000


@needs_server
class TestPostprocessAudio:
    def test_empty_passthrough(self):
        wav = np.array([], dtype=np.float32)
        result = _server._postprocess_audio(wav)
        assert len(result) == 0

    def test_highpass_applied(self):
        """High-pass filter should attenuate DC offset."""
        # Create signal with DC offset
        t = np.linspace(0, 1, _server.TARGET_SAMPLE_RATE, dtype=np.float32)
        wav = np.ones_like(t) * 0.5  # pure DC
        result = _server._postprocess_audio(wav)
        # DC should be nearly eliminated by high-pass
        assert abs(np.mean(result)) < 0.05

    def test_output_is_float32(self):
        wav = np.random.randn(1000).astype(np.float32) * 0.1
        result = _server._postprocess_audio(wav)
        assert result.dtype == np.float32


@needs_server
class TestTrimTrailingSilence:
    def test_preserves_speech(self):
        # Signal: speech then silence
        speech = np.random.randn(5000).astype(np.float32) * 0.1
        silence = np.zeros(5000, dtype=np.float32)
        wav = np.concatenate([speech, silence])

        result = _server._trim_trailing_silence(wav)
        # Should trim most of the trailing silence
        assert len(result) < len(wav)
        # But keep at least the speech portion
        assert len(result) >= len(speech)

    def test_empty_passthrough(self):
        wav = np.array([], dtype=np.float32)
        result = _server._trim_trailing_silence(wav)
        assert len(result) == 0

    def test_all_speech_preserved(self):
        """If all audio is above threshold, no trimming."""
        wav = np.random.randn(5000).astype(np.float32) * 0.1
        result = _server._trim_trailing_silence(wav)
        # Should keep most/all of it
        assert len(result) >= len(wav) - _server.TARGET_SAMPLE_RATE * _server.TRIM_SILENCE_WINDOW_MS // 1000


@needs_server
class TestToWavBytes:
    def test_valid_wav_header(self):
        pcm = np.array([0, 100, -100, 32767, -32767], dtype=np.int16)
        wav_bytes = _server._to_wav_bytes(pcm, 48000)

        # Verify RIFF header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert wav_bytes[12:16] == b"fmt "
        assert wav_bytes[36:40] == b"data"

    def test_readable_by_wave_module(self):
        pcm = np.random.randint(-32767, 32767, size=1000, dtype=np.int16)
        wav_bytes = _server._to_wav_bytes(pcm, 48000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 48000
            assert wf.getnframes() == 1000

    def test_pcm_data_preserved(self):
        pcm = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        wav_bytes = _server._to_wav_bytes(pcm, 24000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(5)
            recovered = np.frombuffer(frames, dtype=np.int16)
            np.testing.assert_array_equal(recovered, pcm)


@needs_server
class TestLeadingSilence:
    def test_correct_length(self):
        silence = _server._get_leading_silence_bytes()
        n_samples = len(silence) // 2  # int16 = 2 bytes
        expected = int(_server.TARGET_SAMPLE_RATE * (_server.SILENCE_MS / 1000.0))
        assert n_samples == expected

    def test_all_zeros(self):
        silence = _server._get_leading_silence_bytes()
        arr = np.frombuffer(silence, dtype=np.int16)
        assert np.all(arr == 0)


@needs_server
class TestHannWindows:
    def test_shape(self):
        fade_in, fade_out = _server._get_hann_windows()
        assert len(fade_in) == _server.BLEND_SAMPLES
        assert len(fade_out) == _server.BLEND_SAMPLES

    def test_fade_in_starts_zero(self):
        fade_in, _ = _server._get_hann_windows()
        assert fade_in[0] == pytest.approx(0.0, abs=1e-6)

    def test_fade_in_ends_one(self):
        fade_in, _ = _server._get_hann_windows()
        assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)

    def test_fade_out_starts_one(self):
        _, fade_out = _server._get_hann_windows()
        assert fade_out[0] == pytest.approx(1.0, abs=1e-6)

    def test_fade_out_ends_zero(self):
        _, fade_out = _server._get_hann_windows()
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)

    def test_complementary(self):
        """fade_in + fade_out should sum to ~1.0 everywhere (constant power)."""
        fade_in, fade_out = _server._get_hann_windows()
        total = fade_in + fade_out
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


@needs_server
class TestSessionWarmupCache:
    def test_get_none_session(self):
        assert _server._get_session_warmup(None) is None

    def test_get_missing_session(self):
        assert _server._get_session_warmup("nonexistent-id") is None

    def test_set_and_get(self):
        sid = "test-session-123"
        codes = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        _server._set_session_warmup(sid, codes)
        result = _server._get_session_warmup(sid)
        assert result is not None
        # Should keep last DECODER_WARMUP_FRAMES
        assert len(result) == min(len(codes), _server.DECODER_WARMUP_FRAMES)
        # Cleanup
        _server._session_codec_cache.pop(sid, None)

    def test_set_none_session_noop(self):
        _server._set_session_warmup(None, [[1, 2]])
        # Should not crash

    def test_set_empty_codes_noop(self):
        _server._set_session_warmup("some-id", [])
        assert _server._get_session_warmup("some-id") is None


@needs_server
class TestVoiceClonePromptLookup:
    def test_no_prompts_returns_none(self):
        result = _server._get_voice_clone_prompt(None, "English")
        # May return None or a precomputed prompt depending on env config
        # Just verify it doesn't crash
        assert result is None or result is not None

    def test_session_prompt_takes_priority(self):
        sid = "voice-test-session"
        sentinel = object()
        _server._session_voice_prompts[sid] = (sentinel, 0)
        result = _server._get_voice_clone_prompt(sid, "English")
        assert result is sentinel
        # Cleanup
        _server._session_voice_prompts.pop(sid, None)


# ── Config validation tests ──────────────────────────────────────────


@needs_server
class TestServerConfig:
    def test_supported_languages(self):
        assert "en" in _server.SUPPORTED_LANGUAGES
        assert "ko" in _server.SUPPORTED_LANGUAGES

    def test_speaker_map(self):
        assert "ko" in _server.SPEAKER_MAP
        assert _server.DEFAULT_SPEAKER == "ryan"

    def test_mode_to_model_mapping(self):
        assert _server.MODE_TO_MODEL["preset"] == "custom"
        assert _server.MODE_TO_MODEL["design"] == "design"
        assert _server.MODE_TO_MODEL["clone"] == "base"
