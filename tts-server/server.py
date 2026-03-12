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
import os
import re
import struct
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import multiprocessing as mp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts-server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
PORT = int(os.environ.get("PORT", "7860"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.15"))

# Output format
TARGET_SAMPLE_RATE = 24000

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

# Hann crossfade between streaming chunks (~21ms at 24kHz)
BLEND_SAMPLES = int(os.environ.get("STREAM_BLEND_SAMPLES", "512"))
_HANN_FADE_IN: np.ndarray | None = None
_HANN_FADE_OUT: np.ndarray | None = None

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
# When unset, uses custom voice mode (preset speakers like ryan/sohee).
REF_AUDIO_PATH = os.environ.get("REF_AUDIO_PATH", "").strip() or None
REF_TEXT = os.environ.get("REF_TEXT", "").strip() or None
X_VECTOR_ONLY = os.environ.get("X_VECTOR_ONLY", "0").lower() in ("1", "true", "yes")

# Precomputed voice clone prompt (set during startup if clone mode)
_voice_clone_prompt = None

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


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
                    wav_24k = _resample_to_24k(wav_results[0], sr)
                    pcm16 = _float_to_pcm16(wav_24k)
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
                            wav_24k = _resample_to_24k(wav, sr)
                            pcm16 = _float_to_pcm16(wav_24k)
                            if not req["future"].done():
                                req["future"].set_result((pcm16, TARGET_SAMPLE_RATE))
                else:
                    def _do_window(inputs=batch_inputs, tok=tokenizer):
                        if getattr(tok, "decode_window_batched", None) is not None:
                            return tok.decode_window_batched(inputs)
                        return _decode_window_fallback(tok, inputs)

                    wav_list, sr = await loop.run_in_executor(None, _do_window)
                    for req, wav in zip(batch_window, wav_list):
                        wav_24k = _resample_to_24k(wav, sr)
                        pcm16 = _float_to_pcm16(wav_24k)
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


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    gpu: bool = False


# ---------------------------------------------------------------------------
# Streaming generation core
# ---------------------------------------------------------------------------


async def _generate_speech_stream(text: str, language: str, speaker: str):
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

        # Prepend "..." for generation quality (nano-qwen3tts-vllm pattern)
        gen_text = "..." + text
        logger.info("[stream] text=%d chars speaker=%s lang=%s", len(text), speaker, language)

        if _voice_clone_prompt is not None:
            gen = interface.generate_voice_clone_async(
                text=gen_text,
                language=language,
                voice_clone_prompt=_voice_clone_prompt,
            )
        else:
            gen = interface.generate_custom_voice_async(
                text=gen_text,
                language=language,
                speaker=speaker,
            )

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
                    # Hann fade-in on first chunk to prevent startup pop
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

            # Emit held-back tail as final chunk
            if prev_tail is not None:
                out = np.clip(prev_tail, -32768, 32767).astype(np.int16)
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

    global _voice_clone_prompt

    logger.info("Loading model: %s (GPU_MEMORY_UTILIZATION=%.2f)", MODEL_DIR, GPU_MEMORY_UTILIZATION)
    interface = get_interface()
    get_tokenizer()
    await interface.start_zmq_tasks()

    # Precompute voice clone prompt if reference audio is provided
    if REF_AUDIO_PATH is not None:
        import soundfile as sf
        logger.info("[clone] Loading reference audio: %s", REF_AUDIO_PATH)
        ref_audio, ref_sr = sf.read(REF_AUDIO_PATH)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1)
        _voice_clone_prompt = interface.create_voice_clone_prompt(
            ref_audio=(ref_audio, ref_sr),
            ref_text=REF_TEXT,
            x_vector_only_mode=X_VECTOR_ONLY,
        )
        mode = "x_vector_only" if X_VECTOR_ONLY else ("ICL" if REF_TEXT else "x_vector_only")
        logger.info("[clone] Voice clone prompt ready (mode=%s, ref_text=%s)", mode, repr(REF_TEXT)[:60])

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

    # Warmup: batch=1 then batch=8 to compile CUDA kernels for all batch sizes
    async def _warmup_one():
        try:
            async for _ in _generate_speech_stream("Hello, warmup.", "English", DEFAULT_SPEAKER):
                pass
        except Exception as e:
            logger.warning("[warmup] error (non-fatal): %s", e)

    logger.info("[warmup] batch=1...")
    await _warmup_one()
    logger.info("[warmup] batch=8...")
    await asyncio.gather(*[_warmup_one() for _ in range(16)])
    logger.info("[warmup] done.")

    logger.info("Server ready on 0.0.0.0:%d", PORT)
    yield

    # Shutdown
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
    return HealthResponse(
        status="ok",
        model=MODEL_DIR,
        gpu=True,
    )


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

    speaker = SPEAKER_MAP.get(lang_key, DEFAULT_SPEAKER)

    return StreamingResponse(
        _generate_speech_stream(text, language, speaker),
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

    speaker = SPEAKER_MAP.get(lang_key, DEFAULT_SPEAKER)
    start = time.time()

    # Collect all audio codes from the model (no windowed decode)
    interface = get_interface()
    gen_text = "..." + text
    logger.info("[synthesize] text=%d chars speaker=%s lang=%s", len(text), speaker, language)

    audio_codes = []
    if _voice_clone_prompt is not None:
        gen = interface.generate_voice_clone_async(
            text=gen_text, language=language, voice_clone_prompt=_voice_clone_prompt,
        )
    else:
        gen = interface.generate_custom_voice_async(
            text=gen_text, language=language, speaker=speaker,
        )
    async for audio_code in gen:
        audio_codes.append(audio_code)

    logger.info("[synthesize] collected %d codes, decoding full sequence", len(audio_codes))

    # Decode all codes in one pass — no chunk boundaries, no pops
    if audio_codes:
        pcm16, _ = await _decode_batched(audio_codes)
        raw_pcm = pcm16.tobytes()
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
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
