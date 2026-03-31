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


async def _audio_generator(
    queue: asyncio.Queue,
    silence_frame: bytes,
):
    """OGG/Opus stream — audio delivered immediately, silence for keepalive.

    Audio frames are yielded as fast as they arrive (no pacing). TTS
    generating faster than real-time is a feature, not a bug — it means
    lower TTFA. The browser buffers excess and plays at 1x. The buffer
    self-corrects during natural speech pauses.

    When SILENCE_SENTINEL arrives:
    - If more audio is already queued (TTS behind): skip silence,
      play sentences back-to-back.
    - If queue is empty (TTS idle): enter silence mode for keepalive.
      Silence is paced at real-time rate (monotonic clock) to prevent
      the browser buffer from growing during idle periods.

    None sentinel = session ended, stop generator.
    """
    writer = OggPageWriter()
    yield writer.header_pages()

    try:
        while True:
            frame = await queue.get()

            if frame is None:
                return

            if frame is SILENCE_SENTINEL:
                # Check if more audio is already queued (TTS behind)
                try:
                    next_frame = queue.get_nowait()
                except asyncio.QueueEmpty:
                    next_frame = None

                if next_frame is None and next_frame is not None:
                    # Unreachable, but keeps the pattern clear
                    pass
                elif next_frame is not None and next_frame is not SILENCE_SENTINEL:
                    # Audio waiting — skip silence, play back-to-back
                    yield writer.wrap_frame(next_frame)
                    continue
                elif next_frame is None:
                    # Queue truly empty — enter silence mode
                    pass
                else:
                    # Another SILENCE_SENTINEL — enter silence mode
                    pass

                # --- Silence mode: paced at real-time rate ---
                pacer = time.monotonic()
                while True:
                    pacer += FRAME_DURATION_S
                    yield writer.wrap_frame(silence_frame)

                    sleep_for = pacer - time.monotonic()
                    if sleep_for > 0:
                        try:
                            frame = await asyncio.wait_for(
                                queue.get(), timeout=sleep_for,
                            )
                        except asyncio.TimeoutError:
                            continue
                    else:
                        try:
                            frame = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            continue

                    if frame is None:
                        return
                    if frame is SILENCE_SENTINEL:
                        continue
                    yield writer.wrap_frame(frame)
                    break  # got real audio — exit silence mode

                continue

            # --- Audio mode: yield immediately (no pacing) ---
            yield writer.wrap_frame(frame)

    except asyncio.CancelledError:
        pass


@router.get("/{session_id}/audio")
async def listen_audio_stream(session_id: str, request: Request):
    """HTTP audio stream for listeners.

    Returns a chunked OGG/Opus stream playable by an <audio> element.
    Continuous silence padding keeps the stream alive during lock screen.

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
