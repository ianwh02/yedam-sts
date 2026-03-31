import logging

from fastapi import FastAPI

from .api.admin import rest_router as admin_rest_router
from .api.admin import ws_router as admin_ws_router
from .api.audio_stream import router as audio_stream_router
from .api.listener import router as listener_router
from .audio.preprocess import AudioPreprocessor
from .config import settings
from .pipeline.manager import PipelineManager
from .tts.client import TTSClient

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yedam STS Pipeline", version="0.1.0")

# Global singletons
pipeline_manager = PipelineManager()
preprocessor = AudioPreprocessor()
tts_client = TTSClient()


@app.on_event("startup")
async def startup():
    logger.info("Starting Yedam STS orchestrator")

    # Initialize shared components
    await preprocessor.initialize()
    await tts_client.initialize()

    # Initialize pipeline manager with shared deps
    # (Audio encoding is now handled per-endpoint: MP3 for HTTP listeners,
    # raw PCM for admin WebSocket. Opus encoding is no longer used.)
    await pipeline_manager.initialize(
        preprocessor=preprocessor,
        tts_client=tts_client,
    )
    app.state.pipeline_manager = pipeline_manager


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down Yedam STS orchestrator")
    await pipeline_manager.shutdown()
    await tts_client.shutdown()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": pipeline_manager.active_session_count,
        "services": await pipeline_manager.check_services(),
    }


# REST API (session management)
app.include_router(admin_rest_router, prefix="/api", tags=["admin"])
# WebSocket (admin audio input)
app.include_router(admin_ws_router, prefix="/ws/admin", tags=["admin-ws"])
# WebSocket (listener output: text transcripts only)
app.include_router(listener_router, prefix="/ws/listen", tags=["listener-ws"])
# HTTP (listener audio stream: MP3 via <audio> element)
app.include_router(audio_stream_router, prefix="/api/listen", tags=["listener-audio"])
