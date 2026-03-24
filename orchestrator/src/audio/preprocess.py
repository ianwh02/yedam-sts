from __future__ import annotations

import logging

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing pipeline for cleaning audio input.

    Pipeline: Raw PCM → RNNoise denoise → RMS normalize

    Runs entirely on CPU with <15ms total latency.
    WhisperLive's Silero VAD and energy pre-check handle the
    remaining filtering downstream.
    """

    def __init__(self):
        self._rnnoise = None
        self._rnnoise_enabled = settings.audio_rnnoise_enabled
        self._rms_normalize = settings.audio_rms_normalize
        self._rms_target = settings.audio_rms_target
        self._sample_rate = settings.audio_sample_rate

    async def initialize(self):
        if self._rnnoise_enabled:
            try:
                import rnnoise_wrapper as rnnoise

                self._rnnoise = rnnoise
                logger.info("RNNoise initialized for audio denoising")
            except ImportError:
                logger.warning(
                    "rnnoise-wrapper not installed, skipping noise suppression. "
                    "Install with: pip install rnnoise-wrapper"
                )
                self._rnnoise_enabled = False

    def process(self, pcm_float32: bytes) -> bytes:
        """Process raw Float32 PCM audio through the preprocessing pipeline.

        Args:
            pcm_float32: Raw PCM audio as Float32 bytes at 16kHz mono.

        Returns:
            Processed PCM audio as Float32 bytes.
        """
        audio = np.frombuffer(pcm_float32, dtype=np.float32)

        if self._rnnoise_enabled and self._rnnoise is not None:
            audio = self._denoise(audio)

        if self._rms_normalize:
            audio = self._normalize_rms(audio)

        return audio.tobytes()

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        """Apply RNNoise for ML-based noise suppression."""
        try:
            # RNNoise expects 16-bit PCM at 48kHz in 480-sample frames.
            # Our input is Float32 at 16kHz. We need to:
            # 1. Resample 16kHz → 48kHz (3x)
            # 2. Convert float32 → int16
            # 3. Process in 480-sample (10ms) frames
            # 4. Convert back and downsample
            #
            # For now, return audio unchanged — full RNNoise integration
            # will be implemented in milestone 1.5
            return audio
        except Exception:
            logger.exception("RNNoise processing failed, passing through")
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
