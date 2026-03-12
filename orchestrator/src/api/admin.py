from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..pipeline.manager import PipelineManager, SessionLimitError

logger = logging.getLogger(__name__)

rest_router = APIRouter()
ws_router = APIRouter()


class CreateSessionRequest(BaseModel):
    source_lang: str = "ko"
    target_lang: str = "en"
    processor: str = "translation"


class CreateSessionResponse(BaseModel):
    session_id: str
    admin_ws_url: str
    listener_ws_url: str


@rest_router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest, request: Request):
    """Create a new translation session."""
    manager: PipelineManager = request.app.state.pipeline_manager
    try:
        session = await manager.create_session(
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            processor_type=req.processor,
        )
    except SessionLimitError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CreateSessionResponse(
        session_id=session.session_id,
        admin_ws_url=f"/ws/admin/{session.session_id}",
        listener_ws_url=f"/ws/listen/{session.session_id}",
    )


@rest_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Stop and remove a session."""
    manager: PipelineManager = request.app.state.pipeline_manager
    stopped = await manager.stop_session(session_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "stopped", "session_id": session_id}


@ws_router.websocket("/{session_id}")
async def admin_websocket(websocket: WebSocket, session_id: str):
    """Admin WebSocket: receives audio input from church PA.

    Protocol:
      - Client sends binary Float32 PCM frames at 16kHz mono
      - Server sends JSON status updates (listener count, segments)
    """
    manager: PipelineManager = websocket.app.state.pipeline_manager
    session = manager.get_session(session_id)

    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    orchestrator = manager.get_orchestrator(session_id)
    if orchestrator is None:
        await websocket.close(code=4004, reason="Session orchestrator not found")
        return

    await websocket.accept()
    session.admin_ws = websocket
    logger.info("Admin connected to session %s", session_id)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                # Binary PCM audio frame — forward through pipeline
                await orchestrator.feed_audio(data["bytes"])

            elif "text" in data and data["text"]:
                # JSON control messages from admin
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "stop":
                    await manager.stop_session(session_id)
                    break

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Admin disconnected from session %s", session_id)
    except Exception:
        logger.exception("Error in admin WebSocket for session %s", session_id)
    finally:
        session.admin_ws = None
