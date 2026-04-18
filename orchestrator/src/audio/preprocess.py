from __future__ import annotations

import logging

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing pipeline for cleaning audio input.

    Pipeline: Raw PCM → spectral denoise → RMS normalize

    Runs entirely on CPU with <15ms total latency per chunk.
    WhisperLive's Silero VAD and energy pre-check handle the
    remaining filtering downstream.
    """

    def __init__(self):
        self._denoise_fn = None
        self._denoise_enabled = settings.audio_denoise_enabled
        self._rms_normalize = settings.audio_rms_normalize
        self._rms_target = settings.audio_rms_target
        self._sample_rate = settings.audio_sample_rate
        self._denoise_stationary = settings.audio_denoise_stationary
        self._denoise_prop_decrease = settings.audio_denoise_prop_decrease

    async def initialize(self):
        if self._denoise_enabled:
            try:
                import noisereduce as nr

                self._denoise_fn = nr.reduce_noise
                logger.info(
                    "noisereduce initialized (stationary=%s, prop_decrease=%.2f)",
                    self._denoise_stationary,
                    self._denoise_prop_decrease,
                )
            except ImportError:
                logger.warning(
                    "noisereduce not installed, skipping noise suppression. "
                    "Install with: pip install noisereduce"
                )
                self._denoise_enabled = False

    def process(self, pcm_float32: bytes) -> bytes:
        """Process raw Float32 PCM audio through the preprocessing pipeline.

        Args:
            pcm_float32: Raw PCM audio as Float32 bytes at 16kHz mono.

        Returns:
            Processed PCM audio as Float32 bytes.
        """
        audio = np.frombuffer(pcm_float32, dtype=np.float32)

        if self._denoise_enabled and self._denoise_fn is not None:
            audio = self._denoise(audio)

        if self._rms_normalize:
            audio = self._normalize_rms(audio)

        return audio.tobytes()

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral-gating noise suppression via noisereduce.

        Uses stationary noise reduction by default — fast (~5ms per chunk)
        and effective for constant background noise (HVAC, hum, reverb tail).
        Set audio_denoise_stationary=False for non-stationary mode which
        adapts to changing noise but is ~2x slower.
        """
        try:
            return self._denoise_fn(
                y=audio,
                sr=self._sample_rate,
                stationary=self._denoise_stationary,
                prop_decrease=self._denoise_prop_decrease,
            )
        except Exception:
            logger.exception("Noise reduction failed, passing through")
            return audio

    def _normalize_rms(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume to target RMS level.

        Audio sources have wildly varying output levels.
        This ensures consistent volume reaching the STT model.
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return audio

        gain = self._rms_target / rms
        # Clamp gain to avoid amplifying noise or clipping
        gain = np.clip(gain, 0.1, 10.0)
        normalized = audio * gain

        # Soft clip to prevent distortion
        return np.clip(normalized, -1.0, 1.0)
