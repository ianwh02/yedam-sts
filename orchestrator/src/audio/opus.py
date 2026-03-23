from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OpusEncoder:
    """Encodes raw PCM audio to Opus for efficient WebSocket broadcast.

    Opus vs MP3 for real-time speech:
    - Opus: ~5ms encode latency, ~32kbps, designed for speech
    - MP3: ~50ms encode latency, ~64kbps, designed for music

    The same Opus frame is broadcast to all listeners on a session
    (encode once, send many).
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._encoder = None

    async def initialize(self):
        try:
            import opuslib

            self._encoder = opuslib.Encoder(
                self._sample_rate,
                self._channels,
                opuslib.APPLICATION_VOIP,
            )
            logger.info(
                "Opus encoder initialized (rate=%d, channels=%d)",
                self._sample_rate,
                self._channels,
            )
        except ImportError:
            logger.warning(
                "opuslib not installed, audio will be sent as raw PCM. "
                "Install with: pip install opuslib"
            )

    def encode(self, pcm_bytes: bytes, frame_size: int = 960) -> bytes:
        """Encode raw PCM to Opus.

        Args:
            pcm_bytes: Raw PCM audio (int16 or float32 depending on TTS output).
            frame_size: Number of samples per frame (960 = 20ms at 48kHz,
                        480 = 20ms at 24kHz).

        Returns:
            Opus-encoded bytes, or original bytes if encoder not available.
        """
        if self._encoder is None:
            return pcm_bytes

        try:
            return self._encoder.encode(pcm_bytes, frame_size)
        except Exception:
            logger.exception("Opus encoding failed, falling back to raw PCM")
            return pcm_bytes
