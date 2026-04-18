import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.admin import rest_router as admin_rest_router
from .api.admin import ws_router as admin_ws_router
from .api.listener import router as listener_router
from .audio.preprocess import AudioPreprocessor
from .config import settings
from .llm.glossary import load_glossaries, load_glossaries_from_supabase
from .pipeline.manager import PipelineManager
from .tts.client import TTSClient

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yedam STS Pipeline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global singletons
pipeline_manager = PipelineManager()
preprocessor = AudioPreprocessor()
tts_client = TTSClient()


@app.on_event("startup")
async def startup():
    logger.info("Starting Yedam STS orchestrator")

    # Load denomination glossaries: prefer Supabase, fall back to local JSON
    glossary_count = 0
    if settings.bible_supabase_url and settings.bible_supabase_key:
        glossary_count = load_glossaries_from_supabase(
            settings.bible_supabase_url, settings.bible_supabase_key,
        )
        if glossary_count:
            logger.info("Loaded %d glossaries from Supabase", glossary_count)
    if not glossary_count and settings.llm_glossaries_dir:
        glossary_count = load_glossaries(settings.llm_glossaries_dir)
        logger.info("Loaded %d glossaries from %s", glossary_count, settings.llm_glossaries_dir)

    # Initialize shared components
    await preprocessor.initialize()
    await tts_client.initialize()

    # Initialize pipeline manager with shared deps
    # Audio: Opus-encoded once per session, OGG-wrapped per listener, sent as
    # binary WebSocket frames on the same /ws/listen/{id} connection.
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
# WebSocket (listener output: text transcripts + OGG/Opus audio)
app.include_router(listener_router, prefix="/ws/listen", tags=["listener-ws"])
