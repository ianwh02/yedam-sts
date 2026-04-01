from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..audio.ogg_opus import OggPageWriter
from ..pipeline.broadcast import SILENCE_SENTINEL
from ..pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Max queued Opus frames before dropping (prevents memory buildup on slow clients)
MAX_AUDIO_QUEUE = 500

# One Opus frame = 20ms at 48kHz
FRAME_DURATION_S = 0.02

# iOS kills audio sessions after ~10-30s of no audio output (varies by
# version). Modal's proxy may also timeout idle HTTP connections.
# Send one silence frame every 8s to cover both. Accumulation: 20ms per
# 8s = 0.15s per minute of silence — negligible.
IOS_KEEPALIVE_S = 8.0


async def _audio_generator(
    queue: asyncio.Queue,
    silence_frame: bytes,
):
    """OGG/Opus stream — audio delivered immediately, minimal silence.

    Audio frames are yielded as fast as they arrive. Between sentences,
    nothing is sent — the browser stalls silently and resumes instantly
    when new audio arrives (zero accumulated delay).

    After 25s of no audio, one silence frame is sent to prevent iOS
    from killing the audio session on lock screen. This adds only 20ms
    of buffer per 25 seconds — negligible accumulation.

    None sentinel = session ended, stop generator.
    """
    writer = OggPageWriter()
    yield writer.header_pages()

    try:
        while True:
            try:
                frame = await asyncio.wait_for(
                    queue.get(), timeout=IOS_KEEPALIVE_S,
                )
            except TimeoutError:
                # No audio for 25s — send one keepalive frame for iOS
                yield writer.wrap_frame(silence_frame)
                continue

            if frame is None:
                return
            if frame is SILENCE_SENTINEL:
                continue  # consume sentinel, send nothing

            yield writer.wrap_frame(frame)

    except asyncio.CancelledError:
        pass


@router.get("/{session_id}/audio")
async def listen_audio_stream(session_id: str, request: Request):
    """HTTP audio stream for listeners.

    Returns a chunked OGG/Opus stream playable by an <audio> element.
    Sparse silence keepalive keeps the stream alive during lock screen.

    Opus encoding is shared (one encoder per session in BroadcastHub).
    This endpoint only does cheap OGG page wrapping per connection.
    """
    manager: PipelineManager = request.app.state.pipeline_manager
    session = manager.get_session(session_id)

    if session is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Session not found"},
        )

    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_AUDIO_QUEUE)
    await session.broadcast.add_audio_queue(queue)

    # Get pre-encoded silence frame from shared encoder
    silence_frame = session.broadcast.silence_frame

    async def stream_with_cleanup():
        try:
            async for page in _audio_generator(queue, silence_frame):
                yield page
        finally:
            await session.broadcast.remove_audio_queue(queue)

    return StreamingResponse(
        stream_with_cleanup(),
        media_type="audio/ogg",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff",
        },
    )
