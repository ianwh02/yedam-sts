import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    orchestrator_port: int = 8080
    log_level: str = "INFO"

    # Service URLs
    stt_ws_url: str = "ws://stt:9090"
    llm_api_url: str = "http://llm:8000/v1"
    tts_api_url: str = "http://tts:7860"

    # Pipeline defaults
    default_source_lang: str = "ko"
    default_target_lang: str = "en"
    default_processor: str = "translation"

    # STT context
    stt_initial_prompt_segments: int = 3
    stt_initial_prompt_vocab: str = ""  # comma-separated domain vocabulary

    # LLM context
    llm_model: str = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
    llm_context_window_segments: int = 5
    llm_summary_interval_segments: int = 10

    # Session limits
    max_concurrent_sessions: int = 5

    # TTS
    tts_default_voice_en: str = "ryan"
    tts_default_voice_ko: str = "sohee"
    tts_stale_threshold: int = 2  # drop if session queue exceeds this depth
    tts_stale_max_age_seconds: float = 10.0  # drop items older than this
    tts_max_concurrent: int = 5  # max concurrent TTS HTTP requests (global semaphore)
    tts_streaming_enabled: bool = True  # use streaming TTS endpoint for lower TTFA
    tts_opus_enabled: bool = False  # encode TTS output to Opus before callbacks
    tts_inter_segment_pause_ms: int = 300  # silence between TTS segments for natural pacing

    # Sentence splitting for TTS pipelining
    tts_min_words_comma_split: int = 8  # min words before splitting on comma
    tts_max_words_per_chunk: int = 20  # force split at this word count

    # Audio preprocessing
    audio_rnnoise_enabled: bool = True
    audio_rms_normalize: bool = True
    audio_rms_target: float = 0.1
    audio_sample_rate: int = 16000

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
