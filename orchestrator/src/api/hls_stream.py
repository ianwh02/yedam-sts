"""LL-HLS streaming endpoints for listener audio.

Serves m3u8 playlists with blocking reload support and fMP4 segments.
All listeners for a session read from the same shared HLSSession — no
per-listener state or queues needed.

Endpoints:
  GET /api/listen/{session_id}/stream.m3u8  — LL-HLS playlist (blocking reload)
  GET /api/listen/{session_id}/init.mp4     — fMP4 init segment
  GET /api/listen/{session_id}/seg_N.m4s    — full segment (all parts concatenated)
  GET /api/listen/{session_id}/seg_N.P.m4s  — partial segment (single part)
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from ..pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Max time to block on playlist reload (seconds)
BLOCKING_RELOAD_TIMEOUT = 5.0

# CORS headers added to all responses (including 404s).
# FastAPI's CORSMiddleware may not cover all edge cases with Modal's proxy.
_CORS = {"Access-Control-Allow-Origin": "*"}


def _get_hls_session(manager: PipelineManager, session_id: str):
    """Get the HLSSession for a session, or None.

    Also checks stopped sessions so the player can fetch #EXT-X-ENDLIST.
    """
    session = manager.get_session(session_id)
    if session is not None:
        encoder = getattr(session, "hls_encoder", None)
        if encoder is not None:
            return encoder.session
    # Fallback: session already stopped but HLS kept alive for ENDLIST
    return manager._stopped_hls.get(session_id)


@router.get("/{session_id}/stream.m3u8")
async def hls_playlist(
    session_id: str,
    request: Request,
    _HLS_msn: int | None = Query(None),
    _HLS_part: int | None = Query(None),
):
    """LL-HLS playlist with blocking reload support.

    When _HLS_msn and _HLS_part are provided, the server holds the
    response until that part is available in the playlist (or timeout).
    """
    manager: PipelineManager = request.app.state.pipeline_manager
    hls = _get_hls_session(manager, session_id)

    if hls is None:
        return Response(status_code=404, content="Session not found", headers=_CORS)

    # Blocking reload: wait until requested part exists (skip if stream ended)
    if _HLS_msn is not None and not hls.ended:
        part = _HLS_part if _HLS_part is not None else 0
        for _ in range(50):  # max ~5s (50 × 100ms)
            if hls.has_part(_HLS_msn, part) or hls.ended:
                break
            await hls.wait_for_update(timeout=0.1)
        # Return whatever we have (even if timeout — client will retry)

    playlist = hls.get_playlist()
    return Response(
        content=playlist,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store",
            **_CORS,
        },
    )


@router.get("/{session_id}/init.mp4")
async def hls_init_segment(session_id: str, request: Request):
    """fMP4 init segment (ftyp + moov). Fetched once by the player."""
    manager: PipelineManager = request.app.state.pipeline_manager
    hls = _get_hls_session(manager, session_id)

    if hls is None:
        return Response(status_code=404, headers=_CORS)

    data = hls.get_init_segment()
    if not data:
        return Response(status_code=404, headers=_CORS)

    return Response(
        content=data,
        media_type="video/mp4",
        headers={
            "Cache-Control": "max-age=31536000",
            **_CORS,
        },
    )


# Pattern: seg_5.m4s (full segment) or seg_5.3.m4s (partial: segment 5, part 3)
_SEG_PATTERN = re.compile(r"^seg_(\d+)(?:\.(\d+))?\.m4s$")


@router.get("/{session_id}/{filename}")
async def hls_segment(session_id: str, filename: str, request: Request):
    """Serve full or partial segments."""
    manager: PipelineManager = request.app.state.pipeline_manager
    hls = _get_hls_session(manager, session_id)

    if hls is None:
        return Response(status_code=404, headers=_CORS)

    match = _SEG_PATTERN.match(filename)
    if not match:
        return Response(status_code=404, headers=_CORS)

    msn = int(match.group(1))
    part_str = match.group(2)

    if part_str is not None:
        # Partial segment: seg_N.P.m4s
        data = hls.get_part_data(msn, int(part_str))
    else:
        # Full segment: seg_N.m4s
        data = hls.get_segment_data(msn)

    if data is None:
        return Response(status_code=404, headers=_CORS)

    return Response(
        content=data,
        media_type="video/iso.segment",
        headers={
            "Cache-Control": "no-cache",
            **_CORS,
        },
    )
