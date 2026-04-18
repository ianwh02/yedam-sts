from __future__ import annotations

import asyncio
import logging
import time

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


# Modal's proxy doesn't propagate client disconnects to the server.
# Zombie generators keep running, each holding a queue in BroadcastHub.
# Cap connection lifetime so zombies self-terminate and the client reconnects.
MAX_STREAM_DURATION_S = 300  # 5 minutes


async def _audio_generator(queue: asyncio.Queue):
    """OGG/Opus stream — real-time paced with empty OGG keepalive.

    Audio frames are yielded as fast as they arrive. When the queue is
    empty, empty OGG pages (no audio, no granule advance) are sent at
    frame rate (~50/s) to keep the HTTP connection alive through
    Modal's proxy and iOS lock screen. Cost: ~1.3 KB/s per listener.

    None sentinel = session ended, stop generator.
    """
    writer = OggPageWriter()
    yield writer.header_pages()
    started = time.monotonic()

    try:
        while True:
            # Kill zombie connections — Modal proxy doesn't propagate
            # client disconnects, so old generators run forever.
            if time.monotonic() - started > MAX_STREAM_DURATION_S:
                logger.info("Audio stream max duration reached, closing")
                return

            try:
                frame = await asyncio.wait_for(
                    queue.get(), timeout=FRAME_DURATION_S,
                )
            except TimeoutError:
                # Queue empty — send empty OGG page at frame rate.
                # No granule advance = no silence in browser buffer.
                yield writer.keepalive_page()
                continue

            if frame is None:
                return
            if frame is SILENCE_SENTINEL:
                continue

            yield writer.wrap_frame(frame)

    except asyncio.CancelledError:
        pass


@router.get("/{session_id}/audio")
async def listen_audio_stream(session_id: str, request: Request):
    """HTTP audio stream for listeners.

    Returns a chunked OGG/Opus stream playable by an <audio> element.
    Empty OGG pages at frame rate keep the connection alive during idle.

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

    async def stream_with_cleanup():
        try:
            async for page in _audio_generator(queue):
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
