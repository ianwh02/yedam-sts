from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..pipeline.manager import PipelineManager, SessionLimitError

PING_INTERVAL = 30  # seconds — keeps connection alive through reverse proxies

logger = logging.getLogger(__name__)

rest_router = APIRouter()
ws_router = APIRouter()


class PipelineTTSConfig(BaseModel):
    """Voice configuration from the platform."""
    mode: str | None = None  # "preset" | "design" | "clone"
    model: str | None = None
    speaker: str | None = None
    voice_prompt: str | None = None  # alias: instruct
    instruct: str | None = None  # alias: voice_prompt (platform sends this)
    reference_audio_url: str | None = None  # signed URL for voice cloning
    reference_text: str | None = None  # transcript of the reference audio
    params: dict | None = None

    @property
    def effective_voice_prompt(self) -> str | None:
        """Return whichever field is set (voice_prompt or instruct)."""
        return self.voice_prompt or self.instruct


class CreateSessionRequest(BaseModel):
    source_lang: str = "ko"
    target_lang: str = "en"
    processor: str = "translation"
    tts_config: PipelineTTSConfig | None = None


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
            tts_config=req.tts_config,
        )
    except SessionLimitError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CreateSessionResponse(
        session_id=session.session_id,
        admin_ws_url=f"/ws/admin/{session.session_id}",
        listener_ws_url=f"/ws/listen/{session.session_id}",
    )


@rest_router.patch("/sessions/{session_id}/tts")
async def update_tts_config(session_id: str, req: PipelineTTSConfig, request: Request):
    """Hot-swap voice configuration for an active session."""
    manager: PipelineManager = request.app.state.pipeline_manager
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        await manager.update_session_voice(session_id, req)
    except Exception as e:
        logger.exception("Failed to update TTS config for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "session_id": session_id}


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
    """Admin WebSocket: receives audio input from the source.

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

    if session.admin_ws is not None:
        await websocket.close(code=4009, reason="Another admin is already streaming audio")
        return

    await websocket.accept()
    session.admin_ws = websocket
    logger.info("Admin connected to session %s", session_id)

    async def _ping_loop():
        """Send periodic pings to keep the connection alive through proxies."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                await websocket.send_json({"type": "ping"})
        except Exception:
            pass

    ping_task = asyncio.create_task(_ping_loop())

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
        ping_task.cancel()
        session.admin_ws = None
