"""
TTS server for yedam-sts pipeline using nano-qwen3tts-vllm.

Uses vLLM-style continuous batching + CUDA graphs for concurrent TTS.
Replaces faster-qwen3-tts (which serialized GPU access via threading.Lock).

Endpoints:
    POST /synthesize         - {"text": "...", "language": "en"|"ko"} -> s16le PCM or WAV
    POST /synthesize/stream  - {"text": "...", "language": "en"|"ko"} -> chunked s16le PCM
    GET  /health             - {"status": "ok", "model": "...", "gpu": true}
"""

import asyncio
import io
import logging
import multiprocessing as mp
import os
import re
import struct
import time
import uuid
import wave
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from scipy.signal import butter, resample_poly, sosfilt

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts-server")

# TF32: use reduced-precision float32 matmuls for ~2x throughput on Ampere+
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
PORT = int(os.environ.get("PORT", "7860"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.15"))

# Available TTS models (baked into image, selected by name)
# Each model supports a different generation method:
#   preset → CustomVoice (generate_custom_voice_async, speaker + optional instruct)
#   design → VoiceDesign (generate_voice_design_async, instruct only)
#   clone  → Base        (generate_voice_clone_async, ref_audio + ref_text)
_models_root = os.path.dirname(MODEL_DIR)  # e.g. /models or /app/models
AVAILABLE_MODELS = {
    "base": os.path.join(_models_root, "Qwen3-TTS-12Hz-1.7B-Base"),
    "custom": os.path.join(_models_root, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "design": os.path.join(_models_root, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
}

# Map mode names (from platform) to model names (internal)
MODE_TO_MODEL = {
    "preset": "custom",
    "design": "design",
    "clone": "base",
}

# Track which model type is currently loaded
_active_model_name: str | None = None  # "base", "custom", or "design"

# Track last KV cache budget for reuse after model swap
_last_kv_budget_mb: int = 0

# Output format
TARGET_SAMPLE_RATE = int(os.environ.get("TARGET_SAMPLE_RATE", "48000"))

# Streaming decode config
STREAMING_CHUNK_SIZE = int(os.environ.get("STREAMING_CHUNK_SIZE", "4"))
STREAMING_CONTEXT_SIZE = int(os.environ.get("STREAMING_CONTEXT_SIZE", "8"))
# First N chunks use smaller size for lower start latency
FIRST_CHUNK_COUNT = int(os.environ.get("FIRST_CHUNK_COUNT", "8"))
FIRST_CHUNK_SIZE = int(os.environ.get("FIRST_CHUNK_SIZE", "4"))
_first_codes_threshold = FIRST_CHUNK_COUNT * FIRST_CHUNK_SIZE

# Leading silence (50ms) sent immediately so client gets audio right away
SILENCE_MS = int(os.environ.get("STREAM_LEADING_SILENCE_MS", "50"))
_SILENCE_PCM16: np.ndarray | None = None

# Hann crossfade between streaming chunks (~21ms at 48kHz)
BLEND_SAMPLES = int(os.environ.get("STREAM_BLEND_SAMPLES", "1024"))
_HANN_FADE_IN: np.ndarray | None = None
_HANN_FADE_OUT: np.ndarray | None = None

# Decoder warmup: prepend N copies of first codec frame as left context on first decode
# to avoid cold-start artifacts. Trimmed from output via left_context_frames.
DECODER_WARMUP_FRAMES = int(os.environ.get("DECODER_WARMUP_FRAMES", "4"))

# Audio post-processing
HIGHPASS_FREQ = int(os.environ.get("TTS_HIGHPASS_HZ", "80"))
_HP_SOS = butter(4, HIGHPASS_FREQ, btype='high', fs=TARGET_SAMPLE_RATE, output='sos') if HIGHPASS_FREQ > 0 else None

# Loudness normalization: normalize TTS output to consistent RMS level
TARGET_RMS = float(os.environ.get("TTS_TARGET_RMS", "0.0"))  # target RMS (0.0 = disable)
MAX_GAIN_DB = float(os.environ.get("TTS_MAX_GAIN_DB", "20"))  # cap gain to prevent amplifying noise

# Trailing silence trimming: remove low-energy tail after speech ends
TRIM_SILENCE_THRESHOLD = float(os.environ.get("TTS_TRIM_SILENCE_THRESHOLD", "0.03"))  # RMS threshold (catches breaths)
TRIM_SILENCE_WINDOW_MS = int(os.environ.get("TTS_TRIM_SILENCE_WINDOW_MS", "20"))  # analysis window
TRIM_SILENCE_MIN_MS = int(os.environ.get("TTS_TRIM_SILENCE_MIN_MS", "50"))  # keep at least this much trailing


SUPPORTED_LANGUAGES = {
    "en": "English",
    "english": "English",
    "ko": "Korean",
    "korean": "Korean",
    "zh": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "japanese": "Japanese",
}

SPEAKER_MAP = {
    "ko": "sohee",
    "korean": "sohee",
}
DEFAULT_SPEAKER = "ryan"

# Voice clone mode: set REF_AUDIO_PATH to a .wav file to use base model with clone mode.
# Per-language overrides: REF_AUDIO_PATH_KO, REF_AUDIO_PATH_EN, REF_TEXT_KO, REF_TEXT_EN, etc.
# Falls back to generic REF_AUDIO_PATH/REF_TEXT if no language-specific one is set.
REF_AUDIO_PATH = os.environ.get("REF_AUDIO_PATH", "").strip() or None
REF_TEXT = os.environ.get("REF_TEXT", "").strip() or None
X_VECTOR_ONLY = os.environ.get("X_VECTOR_ONLY", "0").lower() in ("1", "true", "yes")

# Per-language ref audio config: {canonical_language -> (path, ref_text)}
_REF_AUDIO_LANG_CONFIG: dict[str, tuple[str, str | None]] = {}
for _lang_suffix, _canon in [("EN", "English"), ("KO", "Korean"), ("ZH", "Chinese"), ("JA", "Japanese")]:
    _path = os.environ.get(f"REF_AUDIO_PATH_{_lang_suffix}", "").strip()
    _text = os.environ.get(f"REF_TEXT_{_lang_suffix}", "").strip() or None
    if _path:
        _REF_AUDIO_LANG_CONFIG[_canon] = (_path, _text)

# Precomputed voice clone prompts: {canonical_language -> prompt} + None key for generic fallback
_voice_clone_prompts: dict[str | None, object] = {}

# Debug: save generated audio to WAV files for inspection
DEBUG_SAVE_AUDIO = os.environ.get("DEBUG_SAVE_AUDIO", "1").lower() in ("1", "true", "yes")
DEBUG_AUDIO_DIR = Path("/app/debug_audio")


def _debug_save_wav(pcm16_bytes: bytes, text: str, step_count: int, sample_rate: int = TARGET_SAMPLE_RATE):
    """Save PCM16 bytes to a WAV file for debugging."""
    if not DEBUG_SAVE_AUDIO or not pcm16_bytes:
        return
    try:
        DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        words = text.strip().split()[:6]
        name = "_".join(words) if words else "empty"
        name = re.sub(r"[^\w\-_]", "", name)
        audio_dur_ms = len(pcm16_bytes) // 2 / sample_rate * 1000
        filename = f"{name}_{step_count}steps_{int(audio_dur_ms)}ms.wav"
        path = DEBUG_AUDIO_DIR / filename
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_bytes)
        logger.info(f"[debug] saved {path} ({step_count} steps, {audio_dur_ms:.0f}ms audio)")
    except Exception as e:
        logger.warning(f"[debug] failed to save audio: {e}")

# ---------------------------------------------------------------------------
# Global state (set during lifespan)
# ---------------------------------------------------------------------------

_interface = None
_tokenizer = None
_decode_queue: asyncio.Queue = None
_decode_worker_task: asyncio.Task = None

# Optional multiprocessing decoder worker
_mp_decoder_request_queue: mp.Queue = None
_mp_decoder_result_queue: mp.Queue = None
_mp_decoder_process: mp.Process = None

# Cached warmup codec frames from startup speech generation (last N frames of real speech)
_warmup_codec_frames: list | None = None

# Two-phase startup status: "loading" → "weights_ready" → "ready"
_server_status: str = "loading"

# Per-session decoder context cache: session_id -> (last_codec_frames, last_access_time)
_session_codec_cache: dict[str, tuple[list, float]] = {}
SESSION_CODEC_TTL = float(os.environ.get("SESSION_CODEC_TTL", "300"))  # 5 minutes
_ttl_task: asyncio.Task | None = None

# Per-session voice clone prompts: session_id -> (voice_clone_prompt, last_access_time)
_session_voice_prompts: dict[str, tuple[object, float]] = {}
SESSION_VOICE_PROMPT_TTL = float(os.environ.get("SESSION_VOICE_PROMPT_TTL", "3600"))  # 1 hour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session_warmup(session_id: str | None) -> list | None:
    """Retrieve cached codec frames for a session, refreshing TTL."""
    if session_id is None:
        return None
    entry = _session_codec_cache.get(session_id)
    if entry is None:
        return None
    frames, _ = entry
    _session_codec_cache[session_id] = (frames, time.time())
    return frames


def _set_session_warmup(session_id: str | None, all_codes: list):
    """Cache the last DECODER_WARMUP_FRAMES codec frames for a session."""
    if session_id is None or not all_codes:
        return
    tail = all_codes[-DECODER_WARMUP_FRAMES:] if len(all_codes) >= DECODER_WARMUP_FRAMES else list(all_codes)
    _session_codec_cache[session_id] = (tail, time.time())


def _get_voice_clone_prompt(session_id: str | None, language: str) -> object | None:
    """Look up voice clone prompt: per-session first, then global by language, then global default."""
    # 1. Per-session voice prompt (overrides everything)
    if session_id is not None:
        entry = _session_voice_prompts.get(session_id)
        if entry is not None:
            prompt, _ = entry
            _session_voice_prompts[session_id] = (prompt, time.time())  # refresh TTL
            return prompt

    # 2. Global per-language prompt
    prompt = _voice_clone_prompts.get(language)
    if prompt is not None:
        return prompt

    # 3. Global default prompt
    return _voice_clone_prompts.get(None)


def _get_leading_silence_bytes() -> bytes:
    global _SILENCE_PCM16
    if _SILENCE_PCM16 is None:
        n_samples = int(TARGET_SAMPLE_RATE * (SILENCE_MS / 1000.0))
        _SILENCE_PCM16 = np.zeros(n_samples, dtype=np.int16)
    return _SILENCE_PCM16.tobytes()


def _get_hann_windows() -> tuple[np.ndarray, np.ndarray]:
    """Lazily compute Hann fade-in / fade-out windows for crossfade."""
    global _HANN_FADE_IN, _HANN_FADE_OUT
    if _HANN_FADE_IN is None:
        t = np.linspace(0, 1, BLEND_SAMPLES, dtype=np.float32)
        _HANN_FADE_IN = 0.5 * (1 - np.cos(np.pi * t))
        _HANN_FADE_OUT = 0.5 * (1 + np.cos(np.pi * t))
    return _HANN_FADE_IN, _HANN_FADE_OUT



def _float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def _resample(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Anti-aliased polyphase resampling from orig_sr to TARGET_SAMPLE_RATE."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    up = TARGET_SAMPLE_RATE // gcd(TARGET_SAMPLE_RATE, orig_sr)
    down = orig_sr // gcd(TARGET_SAMPLE_RATE, orig_sr)
    return resample_poly(wav, up, down).astype(np.float32)


def _postprocess_audio(wav: np.ndarray) -> np.ndarray:
    """High-pass filter + loudness normalization on decoded float32 audio."""
    if len(wav) == 0:
        return wav
    if _HP_SOS is not None:
        wav = sosfilt(_HP_SOS, wav).astype(np.float32)
    # RMS loudness normalization
    if TARGET_RMS > 0:
        rms = np.sqrt(np.mean(wav ** 2))
        if rms > 1e-6:
            gain = TARGET_RMS / rms
            max_gain = 10 ** (MAX_GAIN_DB / 20)
            gain = min(gain, max_gain)
            wav = wav * gain
            # Soft clip to prevent harsh distortion
            peak = np.max(np.abs(wav))
            if peak > 0.95:
                wav = wav * (0.95 / peak)
    return wav.astype(np.float32)


def _trim_trailing_silence(wav: np.ndarray) -> np.ndarray:
    """Remove low-energy trailing audio (silence/artifacts after speech)."""
    if len(wav) == 0 or TRIM_SILENCE_THRESHOLD <= 0:
        return wav
    window = int(TARGET_SAMPLE_RATE * TRIM_SILENCE_WINDOW_MS / 1000)
    min_samples = int(TARGET_SAMPLE_RATE * TRIM_SILENCE_MIN_MS / 1000)
    if len(wav) <= min_samples:
        return wav
    # Walk backwards in windows, find last window above threshold
    end = len(wav)
    while end > min_samples:
        start = max(end - window, 0)
        rms = np.sqrt(np.mean(wav[start:end] ** 2))
        if rms >= TRIM_SILENCE_THRESHOLD:
            break
        end = start
    # Keep a small tail for natural fade
    end = min(end + window, len(wav))
    return wav[:end]


def _to_wav_bytes(pcm16: np.ndarray, sr: int) -> bytes:
    raw = pcm16.astype(np.int16).tobytes()
    n_channels = 1
    bits = 16
    byte_rate = sr * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 36 + len(raw)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sr, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(raw)))
    buf.write(raw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def get_interface():
    global _interface
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface

        if os.path.isdir(MODEL_DIR):
            _interface = Qwen3TTSInterface(
                model_path=MODEL_DIR,
                enforce_eager=False,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            )
        else:
            _interface = Qwen3TTSInterface.from_pretrained(
                pretrained_model_name_or_path=MODEL_DIR,
                enforce_eager=False,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            )
    return _interface


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

        num_graph_lengths = int(os.environ.get("DECODER_GRAPH_LENGTHS", "50"))
        _tokenizer = SpeechTokenizerCUDAGraph(
            TOKENIZER_DIR,
            device="cuda:0",
            num_graph_lengths=num_graph_lengths,
        )
    return _tokenizer


# ---------------------------------------------------------------------------
# Multiprocessing decoder worker
# ---------------------------------------------------------------------------


def _decoder_worker_process(request_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Dedicated process for window-batch decode (avoids GIL contention)."""
    from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

    num_graph_lengths = int(os.environ.get("DECODER_GRAPH_LENGTHS", "50"))
    tokenizer = SpeechTokenizerCUDAGraph(
        TOKENIZER_DIR,
        device="cuda:0",
        num_graph_lengths=num_graph_lengths,
    )
    while True:
        job = request_queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        batch_inputs = job["batch"]
        try:
            wav_list, sr = tokenizer.decode_window_batched(batch_inputs)
            result_queue.put({"job_id": job_id, "wav_list": wav_list, "sr": sr, "error": None})
        except Exception as e:
            result_queue.put({"job_id": job_id, "wav_list": None, "sr": None, "error": e})


def _decode_window_fallback(tokenizer, batch_inputs: list) -> tuple[list, int]:
    """Decode full audio_codes then extract new samples from end.

    Fixes Qwen3-TTS #223: the decoder can return fewer samples than
    ``total_frames * total_upsample``, so trimming from the front over-trims.
    Instead we compute expected new-sample count and take from the tail.
    """
    spf = int(tokenizer.tokenizer.model.decoder.total_upsample)
    wavs = []
    sr = None
    for r in batch_inputs:
        codes = r["audio_codes"]
        left = r.get("left_context_frames") or 0
        wav_list, sr = tokenizer.decode([{"audio_codes": codes}])
        wav = np.asarray(wav_list[0], dtype=np.float32)
        if left > 0:
            total_frames = len(codes)
            new_frames = total_frames - left
            expected_new_samples = new_frames * spf
            actual_new = min(expected_new_samples, len(wav))
            wav = wav[-actual_new:] if actual_new > 0 else np.array([], dtype=np.float32)
        wavs.append(wav)
    return wavs, sr


# ---------------------------------------------------------------------------
# Batched decode system (async queue + worker)
# ---------------------------------------------------------------------------


async def _decode_worker_loop():
    """Background task: collects decode requests, micro-batches them, runs decode."""
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    while True:
        item = await _decode_queue.get()
        if item is None:
            break
        batch = [item]

        while not _decode_queue.empty():
            try:
                extra = _decode_queue.get_nowait()
                if extra is None:
                    break
                batch.append(extra)
            except asyncio.QueueEmpty:
                break

        batch_window = [r for r in batch if r.get("left_context_frames") is not None]
        batch_full = [r for r in batch if r.get("left_context_frames") is None]

        try:
            if batch_full:
                # Process each full-decode individually — CUDA graph captured
                # with batch_size=1, so batching causes shape mismatch.
                for req in batch_full:
                    single_input = [{"audio_codes": req["audio_codes"]}]

                    def _do_full(inputs=single_input):
                        return tokenizer.decode(inputs)

                    wav_results, sr = await loop.run_in_executor(None, _do_full)
                    wav_resampled = _resample(wav_results[0], sr)
                    wav_resampled = _postprocess_audio(wav_resampled)
                    pcm16 = _float_to_pcm16(wav_resampled)
                    if not req["future"].done():
                        req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))

            if batch_window:
                batch_inputs = [
                    {"audio_codes": r["audio_codes"], "left_context_frames": r["left_context_frames"]}
                    for r in batch_window
                ]
                decode_window_batched = getattr(tokenizer, "decode_window_batched", None)
                if _mp_decoder_process is not None and _mp_decoder_process.is_alive() and decode_window_batched is not None:
                    job_id = str(uuid.uuid4())
                    _mp_decoder_request_queue.put({"job_id": job_id, "batch": batch_inputs})
                    result = await loop.run_in_executor(None, _mp_decoder_result_queue.get)
                    if result.get("error") is not None:
                        for req in batch_window:
                            if not req["future"].done():
                                req["future"].set_exception(result["error"])
                    else:
                        wav_list, sr = result["wav_list"], result["sr"]
                        for req, wav in zip(batch_window, wav_list):
                            wav_resampled = _resample(wav, sr)
                            wav_resampled = _postprocess_audio(wav_resampled)
                            pcm16 = _float_to_pcm16(wav_resampled)
                            if not req["future"].done():
                                req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
                else:
                    def _do_window(inputs=batch_inputs, tok=tokenizer):
                        if getattr(tok, "decode_window_batched", None) is not None:
                            return tok.decode_window_batched(inputs)
                        return _decode_window_fallback(tok, inputs)

                    wav_list, sr = await loop.run_in_executor(None, _do_window)
                    for req, wav in zip(batch_window, wav_list):
                        wav_resampled = _resample(wav, sr)
                        wav_resampled = _postprocess_audio(wav_resampled)
                        pcm16 = _float_to_pcm16(wav_resampled)
                        if not req["future"].done():
                            req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))

        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)


async def _decode_batched(audio_codes: list, left_context_frames: int = None) -> tuple[np.ndarray, int]:
    """Submit a decode request to the batched worker and await the result."""
    future = asyncio.get_event_loop().create_future()
    payload = {"audio_codes": audio_codes, "future": future}
    if left_context_frames is not None:
        payload["left_context_frames"] = left_context_frames
    await _decode_queue.put(payload)
    return await future


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"
    session_id: str | None = None
    speaker: str | None = None
    instruct: str | None = None
    mode: str | None = None  # "preset", "design", "clone" — None = auto-detect from loaded model


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    gpu: bool = False


# ---------------------------------------------------------------------------
# Mode-based generation routing
# ---------------------------------------------------------------------------


def _resolve_mode(mode: str | None, session_id: str | None, language: str) -> str:
    """Resolve the effective TTS mode for a request.

    If mode is explicitly provided, use it. Otherwise fall back to heuristics:
    - If a per-session or global voice clone prompt exists → "clone"
    - Otherwise infer from the active model type
    """
    if mode is not None:
        return mode

    # Legacy fallback: check if clone prompts exist (backward compat with pre-mode clients)
    clone_prompt = _get_voice_clone_prompt(session_id, language)
    if clone_prompt is not None:
        return "clone"

    # Infer from loaded model
    if _active_model_name == "design":
        return "design"
    return "preset"  # default: CustomVoice


def _create_generator(interface, text: str, language: str, speaker: str,
                      instruct: str | None, session_id: str | None, mode: str | None):
    """Create the appropriate async generator based on the resolved mode."""
    effective_mode = _resolve_mode(mode, session_id, language)

    if effective_mode == "clone":
        clone_prompt = _get_voice_clone_prompt(session_id, language)
        if clone_prompt is None:
            raise ValueError(
                "Clone mode requested but no voice clone prompt found for "
                f"session={session_id}. Call POST /sessions/init_voice first."
            )
        return interface.generate_voice_clone_async(
            text=text, language=language, voice_clone_prompt=clone_prompt,
        )

    if effective_mode == "design":
        if not instruct:
            raise ValueError("Design mode requires an 'instruct' field (voice description)")
        return interface.generate_voice_design_async(
            text=text, instruct=instruct, language=language,
        )

    # Default: preset (CustomVoice)
    return interface.generate_custom_voice_async(
        text=text, language=language, speaker=speaker, instruct=instruct,
    )


# ---------------------------------------------------------------------------
# Streaming generation core
# ---------------------------------------------------------------------------


async def _generate_speech_stream(
    text: str, language: str, speaker: str,
    session_id: str | None = None, instruct: str | None = None,
    mode: str | None = None,
):
    """
    Streaming decode: producer feeds code chunks to a queue; consumer decodes
    (window + context) and yields PCM audio chunks.

    Cancellation-safe: cancels producer and closes generator on disconnect.
    """
    gen = None
    producer_task = None
    try:
        # Leading silence so client gets audio right away
        yield _get_leading_silence_bytes()
        await asyncio.sleep(0)

        interface = get_interface()
        start_time = time.time()

        gen_text = text
        logger.info("[stream] text=%d chars speaker=%s lang=%s session=%s mode=%s", len(text), speaker, language, session_id, mode)

        gen = _create_generator(interface, gen_text, language, speaker, instruct, session_id, mode)

        # Producer feeds code chunks; consumer decodes and yields audio
        codes_queue: asyncio.Queue[list | None] = asyncio.Queue()
        prev_code_pos = 0

        async def producer():
            audio_codes = []
            last_chunk_time = start_time
            try:
                async for audio_code in gen:
                    current_time = time.time()
                    inner_latency = current_time - last_chunk_time
                    logger.debug("[producer] inner chunk latency: %.2fms", inner_latency * 1000)
                    last_chunk_time = current_time

                    audio_codes.append(audio_code)
                    n = len(audio_codes)
                    if n <= _first_codes_threshold:
                        if n % FIRST_CHUNK_SIZE == 0:
                            await codes_queue.put(list(audio_codes))
                    else:
                        if n % STREAMING_CHUNK_SIZE == 0:
                            await codes_queue.put(list(audio_codes))
                # Final partial chunk
                if audio_codes:
                    n = len(audio_codes)
                    if n <= _first_codes_threshold and n % FIRST_CHUNK_SIZE != 0:
                        await codes_queue.put(list(audio_codes))
                    elif n > _first_codes_threshold and n % STREAMING_CHUNK_SIZE != 0:
                        await codes_queue.put(list(audio_codes))
            except asyncio.CancelledError:
                logger.warning("[producer] cancelled (client likely disconnected)")
                raise
            except Exception as e:
                logger.exception("[producer] exception: %s", e)
                raise
            finally:
                logger.info("[producer] done, total_codes=%d", len(audio_codes))
                _set_session_warmup(session_id, audio_codes)
                await codes_queue.put(None)

        producer_task = asyncio.create_task(producer())
        chunk_index = 0
        total_bytes = 0
        prev_tail: np.ndarray | None = None  # float32 tail held back for crossfade

        try:
            while True:
                item = await codes_queue.get()
                if item is None:
                    break
                if len(item) <= prev_code_pos:
                    continue

                start_ctx = max(0, prev_code_pos - STREAMING_CONTEXT_SIZE)
                window = item[start_ctx:]
                context_frames = prev_code_pos - start_ctx

                # First decode: prepend warmup frames so the decoder doesn't cold-start
                if chunk_index == 0 and DECODER_WARMUP_FRAMES > 0 and len(window) > 0:
                    session_warmup = _get_session_warmup(session_id)
                    if session_warmup is not None:
                        warmup = session_warmup
                    elif _warmup_codec_frames is not None:
                        warmup = _warmup_codec_frames[:DECODER_WARMUP_FRAMES]
                    else:
                        warmup = [window[0]] * DECODER_WARMUP_FRAMES
                    window = warmup + window
                    context_frames += len(warmup)

                pcm16, _ = await _decode_batched(window, left_context_frames=context_frames)
                prev_code_pos = len(item)

                if len(pcm16) == 0:
                    continue

                audio_f32 = pcm16.astype(np.float32)

                # Hann crossfade with previous chunk's tail
                if prev_tail is not None and len(audio_f32) > BLEND_SAMPLES:
                    fade_in, fade_out = _get_hann_windows()
                    blended = prev_tail * fade_out + audio_f32[:BLEND_SAMPLES] * fade_in
                    audio_f32 = np.concatenate([blended, audio_f32[BLEND_SAMPLES:]])
                elif chunk_index == 0 and len(audio_f32) > BLEND_SAMPLES:
                    # Short Hann fade-in on first chunk to prevent startup pop
                    fade_in, _ = _get_hann_windows()
                    audio_f32[:BLEND_SAMPLES] *= fade_in

                # Hold back tail for next crossfade
                if len(audio_f32) > BLEND_SAMPLES:
                    prev_tail = audio_f32[-BLEND_SAMPLES:].copy()
                    emit = audio_f32[:-BLEND_SAMPLES]
                else:
                    prev_tail = None
                    emit = audio_f32

                chunk_index += 1
                out = np.clip(emit, -32768, 32767).astype(np.int16)
                chunk_bytes = out.tobytes()
                total_bytes += len(chunk_bytes)
                yield chunk_bytes
                await asyncio.sleep(0)

            # Emit held-back tail as final chunk (trim trailing silence)
            if prev_tail is not None:
                trimmed = _trim_trailing_silence(prev_tail / 32768.0) * 32768.0
                out = np.clip(trimmed, -32768, 32767).astype(np.int16)
                chunk_bytes = out.tobytes()
                total_bytes += len(chunk_bytes)
                yield chunk_bytes
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass

        gen_time = time.time() - start_time
        audio_s = total_bytes / 2 / TARGET_SAMPLE_RATE  # s16le: 2 bytes per sample
        rtf = gen_time / audio_s if audio_s > 0 else 0
        logger.info(
            "[stream] done: %d chunks, audio=%.1fs, gen=%.2fs, rtf=%.2f",
            chunk_index, audio_s, gen_time, rtf,
        )

    except asyncio.CancelledError:
        logger.warning("[stream] cancelled (client disconnected)")
        raise
    except Exception as e:
        logger.error("[stream] error: %s", e)
        raise
    finally:
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _decode_queue, _decode_worker_task
    global _mp_decoder_request_queue, _mp_decoder_result_queue, _mp_decoder_process

    global _active_model_name
    logger.info("Loading model: %s (GPU_MEMORY_UTILIZATION=%.2f)", MODEL_DIR, GPU_MEMORY_UTILIZATION)
    interface = get_interface()
    tokenizer = get_tokenizer()
    # Determine which model type is loaded
    for name, path in AVAILABLE_MODELS.items():
        if path == MODEL_DIR:
            _active_model_name = name
            break
    logger.info("Active model type: %s", _active_model_name)
    # Share the CUDAGraph tokenizer with the interface to avoid loading a duplicate
    # on GPU. The CUDAGraph tokenizer has the same encoder — only the decoder differs.
    logger.info("Injecting shared CUDAGraph tokenizer into interface (saving ~600 MB VRAM)")
    interface.speech_tokenizer = tokenizer
    await interface.start_zmq_tasks()

    # Precompute voice clone prompts
    import soundfile as sf

    def _load_clone_prompt(path: str, ref_text: str | None, label: str):
        logger.info("[clone] Loading reference audio for %s: %s", label, path)
        ref_audio, ref_sr = sf.read(path)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1)
        prompt = interface.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_sr),
            ref_text=ref_text,
            x_vector_only_mode=X_VECTOR_ONLY,
        )
        mode = "x_vector_only" if X_VECTOR_ONLY else ("ICL" if ref_text else "x_vector_only")
        logger.info("[clone] %s prompt ready (mode=%s, ref_text=%s)", label, mode, repr(ref_text)[:60])
        return prompt

    # Detect if model supports voice cloning (has speaker_encoder)
    _has_speaker_encoder = getattr(interface, '_speaker_encoder_available', None) is not False
    try:
        interface._load_speaker_encoder()
        _has_speaker_encoder = True
    except Exception:
        _has_speaker_encoder = False

    if _has_speaker_encoder:
        # Per-language ref audio
        for lang, (path, ref_text) in _REF_AUDIO_LANG_CONFIG.items():
            _voice_clone_prompts[lang] = _load_clone_prompt(path, ref_text, lang)

        # Generic fallback ref audio
        if REF_AUDIO_PATH is not None:
            _voice_clone_prompts[None] = _load_clone_prompt(REF_AUDIO_PATH, REF_TEXT, "default")

        if _voice_clone_prompts:
            langs = [k or "default" for k in _voice_clone_prompts]
            logger.info("[clone] Voice clone prompts loaded: %s", ", ".join(langs))
    else:
        logger.info("[clone] CustomVoice model detected (no speaker_encoder) — skipping voice clone prompt loading")

    # Optional multiprocessing decoder worker
    if os.environ.get("DECODER_MP_WORKER", "0").lower() in ("1", "true", "yes"):
        ctx = mp.get_context("spawn")
        _mp_decoder_request_queue = ctx.Queue()
        _mp_decoder_result_queue = ctx.Queue()
        _mp_decoder_process = ctx.Process(
            target=_decoder_worker_process,
            args=(_mp_decoder_request_queue, _mp_decoder_result_queue),
        )
        _mp_decoder_process.start()
        logger.info("[decoder] started MP worker process")

    # Start batched decode worker
    _decode_queue = asyncio.Queue()
    _decode_worker_task = asyncio.create_task(_decode_worker_loop())

    # Phase 1 complete: weights loaded, workers spawned, waiting for KV cache allocation.
    # Warmup inference is deferred to after /allocate_kv_cache is called.
    global _server_status
    _server_status = "weights_ready"
    logger.info("Phase 1 complete: weights loaded, status=weights_ready (waiting for /allocate_kv_cache)")

    # Start TTL eviction for session codec cache + voice prompts
    async def _ttl_eviction_loop():
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [k for k, (_, t) in _session_codec_cache.items() if now - t > SESSION_CODEC_TTL]
            for k in stale:
                _session_codec_cache.pop(k, None)
            if stale:
                logger.info("[session_cache] evicted %d stale codec caches", len(stale))
            stale_voice = [k for k, (_, t) in _session_voice_prompts.items() if now - t > SESSION_VOICE_PROMPT_TTL]
            for k in stale_voice:
                _session_voice_prompts.pop(k, None)
            if stale_voice:
                logger.info("[session_cache] evicted %d stale voice prompts", len(stale_voice))

    _ttl_task = asyncio.create_task(_ttl_eviction_loop())

    logger.info("Server ready on 0.0.0.0:%d", PORT)
    yield

    # Shutdown
    _ttl_task.cancel()
    if _decode_queue is not None:
        await _decode_queue.put(None)
    if _decode_worker_task is not None:
        await _decode_worker_task
    if _mp_decoder_request_queue is not None and _mp_decoder_process is not None and _mp_decoder_process.is_alive():
        _mp_decoder_request_queue.put(None)
        _mp_decoder_process.join(timeout=10.0)
        if _mp_decoder_process.is_alive():
            _mp_decoder_process.terminate()
            _mp_decoder_process.join(timeout=5.0)
    if _interface is not None:
        await _interface.stop_zmq_tasks()


app = FastAPI(title="yedam-tts", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    if _server_status != "ready":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": _server_status, "model": MODEL_DIR, "gpu": False},
        )
    return HealthResponse(
        status="ok",
        model=MODEL_DIR,
        gpu=True,
    )


@app.get("/startup")
async def startup_status():
    """Two-phase startup status: loading → weights_ready → ready.
    Used by the VRAM coordinator to poll for weight loading completion."""
    return {"status": _server_status}


@app.post("/allocate_kv_cache")
async def allocate_kv_cache(budget_mb: int = 0):
    """Phase 2: allocate KV cache in talker + predictor workers with explicit budget.

    Called by the VRAM coordinator after all services have loaded weights.
    budget_mb: total MB for TTS KV cache (split between talker and predictor).
    If 0, workers use free-VRAM-based calculation (backward compat).
    """
    global _server_status, _warmup_codec_frames, _last_kv_budget_mb

    if _server_status == "ready":
        return {"status": "already_ready"}

    if _server_status not in ("weights_ready", "swapping"):
        raise HTTPException(status_code=409, detail=f"Cannot allocate: status={_server_status}")

    _last_kv_budget_mb = budget_mb

    interface = get_interface()
    holder = getattr(interface, "_mp_holder", None)
    if holder is None:
        raise HTTPException(status_code=500, detail="Multiprocess engines not started")

    # Compute per-worker budget split (talker gets ~75%, predictor gets ~25%)
    budget_bytes = budget_mb * 1024 * 1024 if budget_mb > 0 else None
    if budget_bytes is not None:
        talker_budget = int(budget_bytes * 0.75)
        predictor_budget = budget_bytes - talker_budget
    else:
        talker_budget = None
        predictor_budget = None

    logger.info(
        "Allocating KV cache: total=%s MB, talker=%s MB, predictor=%s MB",
        budget_mb or "auto",
        talker_budget // (1024 * 1024) if talker_budget else "auto",
        predictor_budget // (1024 * 1024) if predictor_budget else "auto",
    )

    # Send allocate commands to both workers
    talker_fut = holder.talker_client.send_allocate_kv_cache(talker_budget or 0)
    predictor_fut = holder.predictor_client.send_allocate_kv_cache(predictor_budget or 0)

    # Wait for both acks (timeout 240s for CUDA graph capture — 48kHz decoder needs longer)
    try:
        talker_ack, predictor_ack = await asyncio.wait_for(
            asyncio.gather(talker_fut, predictor_fut),
            timeout=240.0,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="KV cache allocation timed out (240s)")

    talker_ok = talker_ack.get("success", False)
    predictor_ok = predictor_ack.get("success", False)
    if not (talker_ok and predictor_ok):
        raise HTTPException(
            status_code=500,
            detail=f"KV cache allocation failed: talker={talker_ok}, predictor={predictor_ok}",
        )

    # Signal engine loops that KV cache is ready — they've been waiting on this event.
    kv_event = getattr(interface, "_kv_ready_event", None)
    if kv_event is not None:
        kv_event.set()
        logger.info("KV ready event set — engine loops unblocked")

    # Run warmup inference now that engine loops are active and scheduler is ready.
    logger.info("[warmup] batch=1 (capturing codec frames)...")

    # Determine warmup mode based on loaded model
    if _active_model_name == "base":
        # Base model: use clone if prompts exist, otherwise voice design
        if _voice_clone_prompts:
            _warmup_mode = "clone"
        else:
            _warmup_mode = "design"
        _warmup_instruct = "A natural English speaking voice"
    elif _active_model_name == "design":
        _warmup_mode = "design"
        _warmup_instruct = "A natural English speaking voice"
    else:
        _warmup_mode = "preset"
        _warmup_instruct = None

    logger.info("[warmup] Using mode=%s for warmup (model=%s)", _warmup_mode, _active_model_name)

    async def _warmup_one(capture_codes: bool = False) -> list | None:
        codes = [] if capture_codes else None
        try:
            warmup_text = "The quick brown fox jumps over the lazy dog near the river bank."
            gen = _create_generator(
                interface, warmup_text, "English", DEFAULT_SPEAKER,
                instruct=_warmup_instruct,
                session_id=None, mode=_warmup_mode,
            )
            async for audio_code in gen:
                if capture_codes:
                    codes.append(audio_code)
        except Exception as e:
            logger.warning("[warmup] error (non-fatal): %s", e)
        return codes if capture_codes else None

    warmup_codes = await _warmup_one(capture_codes=True)
    if warmup_codes and len(warmup_codes) > DECODER_WARMUP_FRAMES:
        _warmup_codec_frames = warmup_codes[-DECODER_WARMUP_FRAMES:]
        logger.info("[warmup] cached %d codec frames for decoder warmup", len(_warmup_codec_frames))

    logger.info("[warmup] batch=2...")
    await asyncio.gather(*[_warmup_one() for _ in range(2)])
    logger.info("[warmup] done.")

    _server_status = "ready"
    logger.info("Phase 2 complete: KV cache allocated, status=ready")

    return {
        "status": "ready",
        "talker_blocks": talker_ack.get("num_blocks", 0),
        "predictor_blocks": predictor_ack.get("num_blocks", 0),
    }


class SwapModelRequest(BaseModel):
    model: str  # "base", "custom", or "design"


@app.post("/swap_model")
async def swap_model(req: SwapModelRequest):
    """Hot-swap the TTS model without container restart.

    Tears down current model workers, loads the new model, re-allocates KV cache,
    and runs warmup. Takes ~20-30s. Rejects requests during the swap.

    model: "base" (voice cloning), "custom" (preset speakers), or "design" (voice design)
    """
    global _interface, _server_status, _warmup_codec_frames, MODEL_DIR, _voice_clone_prompts, _active_model_name

    new_path = AVAILABLE_MODELS.get(req.model)
    if new_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    if not os.path.isdir(new_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {new_path}")

    if new_path == MODEL_DIR and _server_status == "ready":
        return {"status": "already_loaded", "model": req.model, "model_dir": MODEL_DIR}

    if _server_status == "swapping":
        raise HTTPException(status_code=409, detail="A swap is already in progress")

    _server_status = "swapping"
    swap_start = time.time()
    logger.info("[swap] Starting model swap: %s → %s", MODEL_DIR, new_path)

    try:
        # 1. Stop ZMQ tasks
        if _interface is not None:
            await _interface.stop_zmq_tasks()

        # 2. Shutdown old interface (kills workers, frees VRAM)
        if _interface is not None:
            _interface.shutdown()
            _interface = None

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[swap] Old model released, VRAM freed")

        # 3. Load new model
        MODEL_DIR = new_path
        _active_model_name = req.model
        interface = get_interface()
        tokenizer = get_tokenizer()
        interface.speech_tokenizer = tokenizer
        await interface.start_zmq_tasks()
        logger.info("[swap] New model loaded: %s", MODEL_DIR)

        # 4. Re-precompute voice clone prompts
        _voice_clone_prompts.clear()
        _has_speaker_encoder = False
        try:
            interface._load_speaker_encoder()
            _has_speaker_encoder = True
        except Exception:
            pass

        if _has_speaker_encoder:
            import soundfile as sf
            for lang, (path, ref_text) in _REF_AUDIO_LANG_CONFIG.items():
                logger.info("[swap] Loading ref audio for %s: %s", lang, path)
                ref_audio, ref_sr = sf.read(path)
                if ref_audio.ndim > 1:
                    ref_audio = np.mean(ref_audio, axis=-1)
                _voice_clone_prompts[lang] = interface.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_sr),
                    ref_text=ref_text,
                    x_vector_only_mode=X_VECTOR_ONLY,
                )
            if REF_AUDIO_PATH is not None:
                ref_audio, ref_sr = sf.read(REF_AUDIO_PATH)
                if ref_audio.ndim > 1:
                    ref_audio = np.mean(ref_audio, axis=-1)
                _voice_clone_prompts[None] = interface.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_sr),
                    ref_text=REF_TEXT,
                    x_vector_only_mode=X_VECTOR_ONLY,
                )

        # 5. Re-allocate KV cache (reuse last budget)
        _server_status = "weights_ready"
        if _last_kv_budget_mb > 0:
            logger.info("[swap] Re-allocating KV cache with budget=%d MB", _last_kv_budget_mb)
            result = await allocate_kv_cache(budget_mb=_last_kv_budget_mb)
            logger.info("[swap] KV cache allocated: %s", result)
        else:
            logger.warning("[swap] No saved KV budget — calling allocate with auto")
            result = await allocate_kv_cache(budget_mb=0)

        elapsed = time.time() - swap_start
        logger.info("[swap] Complete in %.1fs: %s", elapsed, MODEL_DIR)

        return {
            "status": "ready",
            "model": req.model,
            "model_dir": MODEL_DIR,
            "swap_time_s": round(elapsed, 1),
        }

    except Exception as e:
        logger.error("[swap] FAILED: %s", e, exc_info=True)
        _server_status = "error"
        raise HTTPException(status_code=500, detail=f"Model swap failed: {e}")


@app.get("/models")
async def list_models():
    """List available TTS models and which is currently loaded."""
    current_name = None
    for name, path in AVAILABLE_MODELS.items():
        if path == MODEL_DIR:
            current_name = name
            break
    return {
        "available": list(AVAILABLE_MODELS.keys()),
        "current": current_name,
        "current_dir": MODEL_DIR,
        "status": _server_status,
    }


@app.delete("/sessions/{session_id}")
async def delete_session_context(session_id: str):
    """Remove all cached state for a session (decoder context + voice prompt)."""
    codec_removed = _session_codec_cache.pop(session_id, None) is not None
    voice_removed = _session_voice_prompts.pop(session_id, None) is not None
    if codec_removed or voice_removed:
        logger.info(
            "[session_cache] cleared session %s (codec=%s, voice=%s)",
            session_id, codec_removed, voice_removed,
        )
    return {"session_id": session_id, "codec_removed": codec_removed, "voice_removed": voice_removed}


def _download_ref_audio(url: str) -> tuple[np.ndarray, int]:
    """Download reference audio from a signed URL or base64 data URI. Returns (waveform, sample_rate).

    Supports any format ffmpeg can decode (WAV, WebM, MP3, OGG, M4A, etc.)
    via soundfile with pydub/ffmpeg fallback.
    """
    import requests
    import soundfile as sf

    if url.startswith("data:"):
        # Handle base64 data URI: data:audio/wav;base64,<data>
        import base64 as b64mod
        header, encoded = url.split(",", 1)
        raw = b64mod.b64decode(encoded)
        buf = io.BytesIO(raw)
    else:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
    try:
        audio, sr = sf.read(buf)
    except Exception:
        # Fallback: use pydub (ffmpeg) for formats soundfile can't handle (WebM, M4A, etc.)
        import subprocess
        import tempfile

        raw_bytes = buf.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
            tmp_in.write(raw_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
                capture_output=True, check=True, timeout=30,
            )
            audio, sr = sf.read(tmp_out_path)
        finally:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    audio = audio.astype(np.float32)

    # Resample to 16kHz — the speech tokenizer expects 16kHz input
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


class InitVoiceRequest(BaseModel):
    session_id: str
    ref_audio_url: str
    ref_text: str | None = None
    x_vector_only: bool = False


@app.post("/sessions/init_voice")
async def init_session_voice(req: InitVoiceRequest):
    """Download reference audio and create a per-session voice clone prompt.

    Requires the base model (with speaker_encoder) to be loaded.
    """
    if _server_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {_server_status}")

    interface = get_interface()

    # Check if model supports voice cloning
    has_encoder = getattr(interface, '_speaker_encoder_loaded', False)
    if not has_encoder:
        try:
            interface._load_speaker_encoder()
            has_encoder = True
        except Exception:
            has_encoder = False

    if not has_encoder:
        raise HTTPException(
            status_code=400,
            detail="Current model does not support voice cloning (no speaker_encoder). "
                   "Call ensure_model('base') or POST /swap_model first.",
        )

    # Download reference audio from signed URL
    loop = asyncio.get_event_loop()
    try:
        ref_audio, ref_sr = await loop.run_in_executor(
            None, _download_ref_audio, req.ref_audio_url,
        )
    except Exception as e:
        logger.error("[voice_clone] Failed to download ref audio for session %s: %s", req.session_id, e)
        raise HTTPException(status_code=400, detail=f"Failed to download ref audio: {e}")

    logger.info(
        "[voice_clone] Session %s: downloaded ref audio (%.1fs, %dHz)",
        req.session_id, len(ref_audio) / ref_sr, ref_sr,
    )

    # Create voice clone prompt (CPU/GPU-bound, run in executor)
    x_vector_only = req.x_vector_only or (req.ref_text is None)
    try:
        prompt = await loop.run_in_executor(
            None,
            lambda: interface.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_sr),
                ref_text=req.ref_text,
                x_vector_only_mode=x_vector_only,
            ),
        )
    except Exception as e:
        logger.error("[voice_clone] Failed to create prompt for session %s: %s", req.session_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to create voice clone prompt: {e}")

    # Cache for this session
    _session_voice_prompts[req.session_id] = (prompt, time.time())

    mode = "x_vector_only" if x_vector_only else "ICL"
    logger.info(
        "[voice_clone] Session %s: prompt created (mode=%s, ref_text=%s)",
        req.session_id, mode, repr(req.ref_text)[:60] if req.ref_text else "None",
    )

    return {"status": "ok", "session_id": req.session_id, "mode": mode}


@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesizeRequest):
    """Streaming TTS: yields s16le PCM chunks as they're generated."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    lang_key = req.language.lower()
    language = SUPPORTED_LANGUAGES.get(lang_key)
    if language is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {req.language}. Use 'en' or 'ko'.",
        )

    speaker = req.speaker or SPEAKER_MAP.get(lang_key, DEFAULT_SPEAKER)

    return StreamingResponse(
        _generate_speech_stream(text, language, speaker, session_id=req.session_id, instruct=req.instruct, mode=req.mode),
        media_type="application/octet-stream",
        headers={
            "x-sample-rate": str(TARGET_SAMPLE_RATE),
            "x-sample-format": "s16le",
        },
    )


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request):
    """Non-streaming TTS: collects all codes, decodes in one pass (no chunk artifacts)."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    lang_key = req.language.lower()
    language = SUPPORTED_LANGUAGES.get(lang_key)
    if language is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {req.language}. Use 'en' or 'ko'.",
        )

    speaker = req.speaker or SPEAKER_MAP.get(lang_key, DEFAULT_SPEAKER)
    start = time.time()

    # Collect all audio codes from the model (no windowed decode)
    interface = get_interface()
    gen_text = text
    logger.info("[synthesize] text=%d chars speaker=%s lang=%s session=%s mode=%s", len(text), speaker, language, req.session_id, req.mode)

    audio_codes = []
    gen = _create_generator(interface, gen_text, language, speaker, req.instruct, req.session_id, req.mode)
    async for audio_code in gen:
        audio_codes.append(audio_code)

    logger.info("[synthesize] collected %d codes, decoding full sequence", len(audio_codes))

    # Decode all codes in one pass — no chunk boundaries, no pops.
    if audio_codes:
        if DECODER_WARMUP_FRAMES > 0:
            session_warmup = _get_session_warmup(req.session_id)
            if session_warmup is not None:
                warmup = session_warmup
            elif _warmup_codec_frames is not None:
                warmup = _warmup_codec_frames[:DECODER_WARMUP_FRAMES]
            else:
                warmup = [audio_codes[0]] * DECODER_WARMUP_FRAMES
            pcm16, _ = await _decode_batched(warmup + audio_codes, left_context_frames=len(warmup))
        else:
            pcm16, _ = await _decode_batched(audio_codes)
        # Short Hann fade-in to prevent startup pop
        if len(pcm16) > BLEND_SAMPLES:
            pcm16_f32 = pcm16.astype(np.float32)
            fade_in, _ = _get_hann_windows()
            pcm16_f32[:BLEND_SAMPLES] *= fade_in
            pcm16 = np.clip(pcm16_f32, -32768, 32767).astype(np.int16)
        # Trim trailing silence/artifacts
        pcm16_f32 = pcm16.astype(np.float32) / 32768.0
        pcm16_f32 = _trim_trailing_silence(pcm16_f32)
        pcm16 = (np.clip(pcm16_f32, -1.0, 1.0) * 32767).astype(np.int16)
        raw_pcm = pcm16.tobytes()
        _set_session_warmup(req.session_id, audio_codes)
    else:
        raw_pcm = b""

    gen_time = time.time() - start
    step_count = len(audio_codes)

    # Calculate metrics
    n_samples = len(raw_pcm) // 2  # s16le: 2 bytes per sample
    audio_duration_s = n_samples / TARGET_SAMPLE_RATE
    audio_duration_ms = int(audio_duration_s * 1000)
    gen_time_ms = int(gen_time * 1000)
    rtf = gen_time / audio_duration_s if audio_duration_s > 0 else 0

    # Expected audio: ~83ms per step at 12Hz → flag if >2x expected
    expected_audio_ms = step_count * (1000 / 12)
    if audio_duration_ms > expected_audio_ms * 2.5 and step_count > 20:
        logger.warning(
            f"[synthesize] OVER-GENERATION? text={text!r} steps={step_count} "
            f"audio={audio_duration_ms}ms expected~{int(expected_audio_ms)}ms "
            f"ratio={audio_duration_ms / expected_audio_ms:.1f}x"
        )

    _debug_save_wav(raw_pcm, text, step_count)

    accept = request.headers.get("accept", "")
    wants_raw = "application/octet-stream" in accept

    if wants_raw:
        logger.info(
            "Synthesized: lang=%s speaker=%s audio=%.1fs gen=%.2fs rtf=%.2f pcm=%dB",
            req.language, speaker, audio_duration_s, gen_time, rtf, len(raw_pcm),
        )
        return Response(
            content=raw_pcm,
            media_type="application/octet-stream",
            headers={
                "x-audio-duration-ms": str(audio_duration_ms),
                "x-generation-time-ms": str(gen_time_ms),
                "x-sample-rate": str(TARGET_SAMPLE_RATE),
                "x-sample-format": "s16le",
                "x-rtf": f"{rtf:.2f}",
            },
        )
    else:
        pcm16_array = np.frombuffer(raw_pcm, dtype=np.int16)
        wav_bytes = _to_wav_bytes(pcm16_array, TARGET_SAMPLE_RATE)
        logger.info(
            "Synthesized: lang=%s speaker=%s audio=%.1fs gen=%.2fs rtf=%.2f wav=%dB",
            req.language, speaker, audio_duration_s, gen_time, rtf, len(wav_bytes),
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "x-audio-duration-ms": str(audio_duration_ms),
                "x-generation-time-ms": str(gen_time_ms),
                "x-sample-rate": str(TARGET_SAMPLE_RATE),
                "x-rtf": f"{rtf:.2f}",
            },
        )


# ---------------------------------------------------------------------------
# Voice design endpoint (experimental)
# ---------------------------------------------------------------------------


class VoiceDesignRequest(BaseModel):
    text: str
    instruct: str  # e.g. "Male, 30 years old, calm and professional"
    language: str = "en"


@app.post("/synthesize/voice-design")
async def synthesize_voice_design(req: VoiceDesignRequest, request: Request):
    """Synthesize speech with voice design — describe the voice in natural language."""
    if _server_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {_server_status}")

    interface = get_interface()
    lang_key = req.language.lower()
    language = SUPPORTED_LANGUAGES.get(lang_key, "Auto")

    start = time.time()
    audio_codes = []
    async for audio_code in interface.generate_voice_design_async(
        text=req.text, instruct=req.instruct, language=language,
    ):
        audio_codes.append(audio_code)

    if not audio_codes:
        raise HTTPException(status_code=500, detail="No audio generated")

    pcm16, _ = await _decode_batched(audio_codes)
    gen_time = time.time() - start
    raw_pcm = pcm16.tobytes()
    audio_duration_s = len(pcm16) / TARGET_SAMPLE_RATE

    accept = request.headers.get("accept", "")
    if "application/octet-stream" in accept:
        return Response(
            content=raw_pcm,
            media_type="application/octet-stream",
            headers={
                "x-audio-duration-ms": str(int(audio_duration_s * 1000)),
                "x-generation-time-ms": str(int(gen_time * 1000)),
                "x-sample-rate": str(TARGET_SAMPLE_RATE),
                "x-sample-format": "s16le",
            },
        )
    else:
        pcm16_array = np.frombuffer(raw_pcm, dtype=np.int16)
        wav_bytes = _to_wav_bytes(pcm16_array, TARGET_SAMPLE_RATE)
        return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Continuous TTS WebSocket — persistent stream with KV cache across segments
# ---------------------------------------------------------------------------

# Active continuous sessions: session_id -> text_channel queue
_continuous_sessions: dict[str, asyncio.Queue] = {}


@app.websocket("/synthesize/continuous/{session_id}")
async def ws_continuous_synthesize(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for continuous TTS streaming.

    Maintains talker KV cache across sentence boundaries for prosodic continuity.

    Protocol:
      Client → Server (JSON):
        {"type": "text", "text": "Hello world.", "language": "en"}
        {"type": "end"}
      Server → Client (binary): s16le PCM audio chunks at TARGET_SAMPLE_RATE
      Server → Client (JSON):
        {"type": "segment_boundary"}
        {"type": "ready"}
        {"type": "error", "detail": "..."}
    """
    if _server_status != "ready":
        await websocket.close(code=1013, reason=f"Server not ready: {_server_status}")
        return

    await websocket.accept()
    interface = get_interface()

    text_channel: asyncio.Queue[str | None] = asyncio.Queue()
    _continuous_sessions[session_id] = text_channel

    # Resolve voice mode/config from session
    clone_prompt = _get_voice_clone_prompt(session_id, "English")
    effective_mode = _active_model_name or "preset"
    if clone_prompt is not None:
        effective_mode = "clone"

    logger.info("[ws_continuous] session %s connected (mode=%s)", session_id, effective_mode)

    # Wait for initial text message to start generation
    try:
        initial_msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
    except (TimeoutError, WebSocketDisconnect):
        logger.warning("[ws_continuous] session %s: no initial text within 30s", session_id)
        _continuous_sessions.pop(session_id, None)
        return

    if initial_msg.get("type") != "text" or not initial_msg.get("text"):
        await websocket.send_json({"type": "error", "detail": "First message must be {type: 'text', text: '...'}"})
        _continuous_sessions.pop(session_id, None)
        return

    initial_text = initial_msg["text"]
    raw_lang = initial_msg.get("language", "English")
    language = SUPPORTED_LANGUAGES.get(raw_lang.lower(), raw_lang) if raw_lang else "English"
    speaker = initial_msg.get("speaker", DEFAULT_SPEAKER)
    instruct = initial_msg.get("instruct")
    mode = initial_msg.get("mode", effective_mode)

    await websocket.send_json({"type": "ready"})

    # Background task: read text messages from client → push to text_channel
    async def text_reader():
        try:
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")
                if msg_type == "text" and msg.get("text"):
                    await text_channel.put(msg["text"])
                elif msg_type == "end":
                    await text_channel.put(None)
                    break
        except WebSocketDisconnect:
            await text_channel.put(None)
        except Exception as e:
            logger.warning("[ws_continuous] text_reader error: %s", e)
            await text_channel.put(None)

    reader_task = asyncio.create_task(text_reader())

    # Generate audio using continuous generator
    try:
        gen = interface.generate_continuous_async(
            text_channel,
            initial_text=initial_text,
            language=language,
            mode=mode,
            speaker=speaker,
            instruct=instruct,
            voice_clone_prompt=clone_prompt,
        )

        # Decode pipeline (reuses existing streaming decode logic)
        audio_codes = []
        prev_code_pos = 0
        chunk_index = 0
        prev_tail = None

        async for item in gen:
            if isinstance(item, str):
                if item == "segment_boundary":
                    # Flush remaining undecoded codes (partial chunk at sentence end)
                    if audio_codes and len(audio_codes) > prev_code_pos:
                        start_ctx = max(0, prev_code_pos - STREAMING_CONTEXT_SIZE)
                        window = audio_codes[start_ctx:]
                        context_frames = prev_code_pos - start_ctx
                        pcm16, _ = await _decode_batched(window, left_context_frames=context_frames)
                        if len(pcm16) > 0:
                            audio_f32 = pcm16.astype(np.float32)
                            if prev_tail is not None and len(audio_f32) > BLEND_SAMPLES:
                                fade_in, fade_out = _get_hann_windows()
                                blended = prev_tail * fade_out + audio_f32[:BLEND_SAMPLES] * fade_in
                                audio_f32 = np.concatenate([blended, audio_f32[BLEND_SAMPLES:]])
                            if len(audio_f32) > BLEND_SAMPLES:
                                prev_tail = audio_f32[-BLEND_SAMPLES:].copy()
                                emit = audio_f32[:-BLEND_SAMPLES]
                            else:
                                prev_tail = None
                                emit = audio_f32
                            if len(emit) > 0:
                                out = np.clip(emit, -32768, 32767).astype(np.int16)
                                await websocket.send_bytes(out.tobytes())

                    # Flush crossfade tail
                    if prev_tail is not None:
                        trimmed = _trim_trailing_silence(prev_tail / 32768.0) * 32768.0
                        out = np.clip(trimmed, -32768, 32767).astype(np.int16)
                        await websocket.send_bytes(out.tobytes())
                        prev_tail = None
                    await websocket.send_json({"type": "segment_boundary"})
                    _set_session_warmup(session_id, audio_codes)
                    audio_codes = []
                    prev_code_pos = 0
                    chunk_index = 0
                    continue
                elif item == "fallback":
                    await websocket.send_json({"type": "error", "detail": "Continuous mode failed, use per-sentence fallback"})
                    break
                continue

            # item is codebook_ids (list[int])
            audio_codes.append(item)
            n = len(audio_codes)

            # Chunk accumulation (same thresholds as HTTP streaming)
            should_decode = False
            if n <= _first_codes_threshold:
                if n % FIRST_CHUNK_SIZE == 0:
                    should_decode = True
            else:
                if n % STREAMING_CHUNK_SIZE == 0:
                    should_decode = True

            if not should_decode:
                continue

            # Decode to PCM
            start_ctx = max(0, prev_code_pos - STREAMING_CONTEXT_SIZE)
            window = audio_codes[start_ctx:]
            context_frames = prev_code_pos - start_ctx

            if chunk_index == 0 and DECODER_WARMUP_FRAMES > 0 and len(window) > 0:
                session_warmup = _get_session_warmup(session_id)
                if session_warmup is not None:
                    warmup = session_warmup
                elif _warmup_codec_frames is not None:
                    warmup = _warmup_codec_frames[:DECODER_WARMUP_FRAMES]
                else:
                    warmup = [window[0]] * DECODER_WARMUP_FRAMES
                window = warmup + window
                context_frames += len(warmup)

            pcm16, _ = await _decode_batched(window, left_context_frames=context_frames)
            prev_code_pos = len(audio_codes)

            if len(pcm16) == 0:
                continue

            audio_f32 = pcm16.astype(np.float32)

            # Hann crossfade
            if prev_tail is not None and len(audio_f32) > BLEND_SAMPLES:
                fade_in, fade_out = _get_hann_windows()
                blended = prev_tail * fade_out + audio_f32[:BLEND_SAMPLES] * fade_in
                audio_f32 = np.concatenate([blended, audio_f32[BLEND_SAMPLES:]])
            elif chunk_index == 0 and len(audio_f32) > BLEND_SAMPLES:
                fade_in, _ = _get_hann_windows()
                audio_f32[:BLEND_SAMPLES] *= fade_in

            if len(audio_f32) > BLEND_SAMPLES:
                prev_tail = audio_f32[-BLEND_SAMPLES:].copy()
                emit = audio_f32[:-BLEND_SAMPLES]
            else:
                prev_tail = None
                emit = audio_f32

            chunk_index += 1
            if chunk_index == 1:
                logger.info("[DIAG] E tts_first_send_bytes len=%d", len(emit) * 2)
            out = np.clip(emit, -32768, 32767).astype(np.int16)
            await websocket.send_bytes(out.tobytes())

        # Final tail
        if prev_tail is not None:
            trimmed = _trim_trailing_silence(prev_tail / 32768.0) * 32768.0
            out = np.clip(trimmed, -32768, 32767).astype(np.int16)
            await websocket.send_bytes(out.tobytes())

        _set_session_warmup(session_id, audio_codes)

    except WebSocketDisconnect:
        logger.info("[ws_continuous] session %s disconnected", session_id)
    except Exception as e:
        logger.exception("[ws_continuous] session %s error: %s", session_id, e)
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        reader_task.cancel()
        _continuous_sessions.pop(session_id, None)
        logger.info("[ws_continuous] session %s cleaned up", session_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
