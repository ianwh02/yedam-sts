from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/{session_id}")
async def listener_websocket(websocket: WebSocket, session_id: str):
    """Listener WebSocket: receives translation text + TTS audio.

    Protocol:
      Server → Client (text frames, JSON):
        { type: "korean_partial", text: "..." }
        { type: "korean_final", text: "...", segment_id: N }
        { type: "translation_partial", text: "...", segment_id: N }
        { type: "translation_final", text: "...", segment_id: N }
        { type: "session_started" }
        { type: "session_ended" }

      Server → Client (binary frames):
        Opus-encoded TTS audio bytes

      Mute/unmute is handled client-side — all listeners receive
      the same audio stream regardless.
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
    })

    logger.info(
        "Listener joined session %s (total: %d)",
        session_id,
        session.broadcast.count,
    )

    try:
        # Keep connection alive — listeners only receive, never send
        # (except for keepalive pings handled by the WebSocket layer)
        while True:
            data = await websocket.receive()
            # Listeners don't send meaningful data, but we need to
            # keep the receive loop running to detect disconnects
            if "text" in data and data["text"]:
                pass  # Ignore any client messages

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Error in listener WebSocket for session %s", session_id)
    finally:
        await session.broadcast.remove(websocket)
        logger.info(
            "Listener left session %s (remaining: %d)",
            session_id,
            session.broadcast.count,
        )
