from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)

router = APIRouter()

PING_INTERVAL = 30  # seconds — keeps connection alive through reverse proxies


@router.websocket("/{session_id}")
async def listener_websocket(websocket: WebSocket, session_id: str):
    """Listener WebSocket: receives translation text (transcripts only).

    Audio is delivered separately via HTTP streaming at
    GET /api/listen/{session_id}/audio (MP3, <audio> element compatible).

    Protocol:
      Server → Client (text frames, JSON):
        { type: "stt_partial", text: "..." }
        { type: "stt_final", text: "...", segment_index: N }
        { type: "translation_partial", token: "...", segment_index: N }
        { type: "translation_final", text: "...", segment_index: N }
        { type: "session_started", audio_url: "/api/listen/{id}/audio" }
        { type: "session_ended" }
        { type: "ping" }
    """
    manager: PipelineManager = websocket.app.state.pipeline_manager
    session = manager.get_session(session_id)

    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    await session.broadcast.add(websocket)

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "source_lang": session.source_lang,
        "target_lang": session.target_lang,
        "listener_count": session.broadcast.count,
        "audio_url": f"/api/listen/{session_id}/audio",
    })

    logger.info(
        "Listener joined session %s (total: %d)",
        session_id,
        session.broadcast.count,
    )

    async def _ping_loop():
        """Send periodic pings to keep the connection alive through proxies."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                await websocket.send_json({"type": "ping"})
        except Exception:
            pass  # Connection closed — ping loop exits

    ping_task = asyncio.create_task(_ping_loop())

    try:
        while True:
            data = await websocket.receive()
            if "text" in data and data["text"]:
                pass  # Ignore client messages (pong, etc.)

    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("Error in listener WebSocket for session %s", session_id)
    finally:
        ping_task.cancel()
        await session.broadcast.remove(websocket)
        logger.info(
            "Listener left session %s (remaining: %d)",
            session_id,
            session.broadcast.count,
        )
