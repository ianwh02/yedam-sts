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
    stt_initial_prompt_vocab: str = ""  # inline domain vocabulary
    stt_initial_prompt_vocab_path: str = ""  # path to domain vocabulary file

    # STT post-processing
    stt_corrections_path: str = ""  # path to domain corrections TSV file (wrong<TAB>correct)
    stt_spacing_enabled: bool = False  # enable pykospacing for Korean spacing correction (requires pip install pykospacing)

    # LLM context
    llm_system_prompt_path: str = ""  # path to custom system prompt file
    llm_glossaries_dir: str = ""  # directory containing denomination glossary JSONs
    llm_default_glossary: str = ""  # default glossary ID (filename without .json)
    llm_model: str = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
    llm_context_window_segments: int = 3
    llm_summary_interval_segments: int = 10

    # Bible verse lookup (Supabase)
    bible_supabase_url: str = ""  # Supabase project URL
    bible_supabase_key: str = ""  # Supabase anon key (public read)
    bible_translation: str = "kjv"  # default Bible translation

    # Session limits
    max_concurrent_sessions: int = 5
    max_session_duration_seconds: int = 3 * 60 * 60  # auto-stop after 3 hours

    # TTS
    tts_default_voice_en: str = "ryan"
    tts_default_voice_ko: str = "sohee"
    tts_stale_threshold: int = 2  # drop if session queue exceeds this depth
    tts_stale_max_age_seconds: float = 10.0  # drop items older than this
    tts_max_concurrent: int = 5  # max concurrent TTS HTTP requests (global semaphore)
    tts_streaming_enabled: bool = True  # use streaming TTS endpoint for lower TTFA
    tts_opus_enabled: bool = False  # encode TTS output to Opus before callbacks
    # Delimiter-aware silence is now handled in TTSClient._consume_loop
    # using _DELIMITER_SILENCE durations, only inserted when back-to-back.
    tts_voice_clone_init_timeout: float = 90.0  # timeout for per-session voice clone init (first call after boot can take ~40s)
    tts_continuous_enabled: bool = True  # use continuous TTS streaming (KV cache across segments)

    # Sentence splitting for TTS pipelining
    tts_min_words_sentence_split: int = 4  # min words before splitting on sentence punctuation (short fragments combine with next)
    tts_min_words_comma_split: int = 8  # min words before splitting on comma
    tts_max_words_per_chunk: int = 35  # force split at this word count

    # Audio preprocessing
    audio_denoise_enabled: bool = True  # spectral noise suppression via noisereduce
    audio_denoise_stationary: bool = True  # True = fast stationary mode; False = adaptive (slower)
    audio_denoise_prop_decrease: float = 0.8  # noise reduction strength (0.0-1.0, higher = more aggressive)
    audio_rms_normalize: bool = True
    audio_rms_target: float = 0.1
    audio_sample_rate: int = 16000

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
