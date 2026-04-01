from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class SessionCallbacks:
    """Consumer-provided callbacks for receiving pipeline output.

    All callbacks are optional. If not provided, that event is silently
    skipped. Consumers register these when creating a session to receive
    STT transcription, processor output, and TTS audio.

    This is the primary integration point for consumers building on top
    of the STS pipeline. Transport (WebSocket, HTTP SSE, gRPC)
    and encoding (Opus, MP3) are the consumer's responsibility.
    """

    on_stt_partial: Callable[[str], Awaitable[None]] | None = None
    """Partial STT transcription (unstable, may revise). Args: text"""

    on_stt_final: Callable[[str, int], Awaitable[None]] | None = None
    """Confirmed STT segment (stable). Args: text, segment_index"""

    on_processor_partial: Callable[[str, int], Awaitable[None]] | None = None
    """Streaming processor token. Args: token, segment_index"""

    on_processor_final: Callable[[str, int], Awaitable[None]] | None = None
    """Completed processor output. Args: text, segment_index"""

    on_tts_audio: Callable[[bytes, int, int], Awaitable[None]] | None = None
    """Raw PCM audio from TTS. Args: pcm_bytes, segment_index, sentence_index"""

    on_tts_sentence_done: Callable[[], Awaitable[None]] | None = None
    """Fired after a TTS sentence's audio (+ inter-segment pause) is fully sent."""
