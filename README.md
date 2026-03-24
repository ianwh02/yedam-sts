# yedam-sts

A GPU-optimised real-time Speech-to-Speech pipeline designed to maximise concurrent sessions on a single consumer GPU. Built as general-purpose building blocks for anyone to build STS applications on top of — translation services, conference interpretation, meeting tools, voice AI, or anything that needs concurrent real-time speech-to-speech.

## Motivation

I needed a real-time speech-to-speech pipeline for an AI startup where STT, LLM, and TTS API costs were adding up fast. The application handles hundreds of sessions spread throughout the day with peak concurrency around 5 — a single self-hosted GPU instance is far cheaper than per-request API pricing.

The problem: fitting STT, LLM, and TTS models on one GPU and running them concurrently without running out of memory. yedam-sts is the result — 5+ concurrent STS sessions on a single 16GB GPU.

## How It Works

### Pipeline Flow

```
Audio In → [STT] → confirmed text → [Processor] → translated text → [TTS] → Audio Out
              │                          │                              │
         Whisper batch              vLLM streaming              CUDA graph decode
         inference                  token generation            + windowed audio
              │                          │                              │
         buffer clear              sentence boundary            Hann crossfade
         after segment             triggers TTS(N)              between chunks
                                   while LLM handles
                                   segment N+1
```

### Key Mechanisms

**STT Flushing:** Only completed segments are sent to the processor. For Korean, a grammar-based detector identifies phrase and sentence boundaries in real-time. For punctuated languages, standard punctuation triggers flushing.

**Rolling Audio Window:** After each completed segment, the STT buffer is cleared to prevent unbounded VRAM growth. Whisper's `initial_prompt` carries linguistic context forward so transcription quality isn't lost after the clear.

**Sentence-Level Pipelining:** TTS starts generating audio for sentence N while the LLM is still processing segment N+1. This overlap hides latency — the user hears continuous audio instead of waiting for the full translation.

**VRAM Budgeting:** At startup, a coordinator measures available GPU memory after all model weights are loaded, then distributes the remaining VRAM as KV cache budgets to each service. No manual tuning required.

**CUDA MPS:** All three GPU services (STT, LLM, TTS) run as separate processes sharing a single GPU via CUDA Multi-Process Service. This enables concurrent kernel execution without context-switching overhead.

## Tech Stack

### Services

| Service | Model | VRAM (weights + KV cache + CUDA overhead) | Role |
|---------|-------|------|------|
| **STT** | Whisper large-v3-turbo (int8_float16) | ~2.0 GB | Speech recognition |
| **LLM** | Qwen3-4B-AWQ (4-bit) | ~3.3 GB | Translation / processing |
| **TTS** | Qwen3-TTS 0.6B (FP16) + tokenizer + CUDA graphs | ~5.9 GB | Speech synthesis |
| **Orchestrator** | — (CPU only) | 0 | Session management |
| **Total** | | **~11.2 GB** | Fits 16GB with ~4 GB headroom |

### Source Repos and Modifications

**STT Server** — forked from [WhisperLive](https://github.com/collabora/WhisperLive) by Collabora

Changes made:
- Added `BatchInferenceWorker` for multi-session batched Whisper forward passes
- Implemented `clear_buffer` / `trim_buffer` protocol for rolling audio window with zero context loss
- Added server-side Korean grammar-based flush detection (phrase vs sentence boundaries)
- Added `initial_prompt` support for domain vocabulary injection
- Energy-based pre-check to skip silence frames before batching
- TensorRT backend support with batched inference

**LLM Server** — [vLLM](https://github.com/vllm-project/vllm) (unmodified)

Used as-is via Docker with Qwen3-4B-AWQ. The OpenAI-compatible API means swapping models is a one-line env var change. Continuous batching, prefix caching, and fp8 KV cache are enabled out of the box.

**TTS Server** — built on [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) by tsdocode, with [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) models by Alibaba

Changes made:
- Custom FastAPI server wrapping the inference engine with streaming and non-streaming endpoints
- `SpeechTokenizerCUDAGraph`: 50 pre-captured CUDA graphs for codec decode (T=1..50)
- 48kHz decoder integration (community [Qwen3-TTS-Tokenizer-12Hz-48kHz](https://huggingface.co/tsdocode/Qwen3-TTS-Tokenizer-12Hz-48kHz))
- Shared tokenizer injection between interface and decoder (saves ~600MB VRAM)
- Windowed streaming decode with Hann crossfade between chunks
- Session-persistent codec warmup (carries decoder state across sentences)
- Two-phase startup: weight loading then coordinator-triggered KV cache allocation
- Voice cloning via ICL mode with reference audio encoding + x-vector speaker embedding
- Voice design endpoint (natural language voice description)
- Audio post-processing: high-pass filter, optional RMS normalisation, trailing silence trim
- Polyphase anti-aliased resampling (replacing naive linear interpolation)
- Loop detection with progressive EOS boosting and forced termination
- Repetition penalty with configurable sliding window
- Per-session TTS queues with age-based staleness dropping
- Decoder warmup frames to prevent cold-start artifacts

**Orchestrator** — written from scratch

Python FastAPI service that wires STT → Processor → TTS via consumer callbacks. Handles session lifecycle (create/destroy), audio routing, WebSocket management, and the pluggable `BaseProcessor` interface. CPU-only — no GPU resources consumed.

## Usage Guide

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM (tested on RTX 5060 Ti)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `nvidia-smi` accessible from host

### Quick Start

```bash
git clone https://github.com/ianchm/yedam-sts.git
cd yedam-sts
cp .env.example .env
./scripts/start.sh
```

The startup script runs a three-phase process:

1. **Budget** — `vram_budget.py` reads `vram_budget.yml`, queries the GPU, and computes per-service VRAM allocations → writes `.env.vram`
2. **Load** — starts LLM first (vLLM profiles free VRAM at boot), then STT + TTS in parallel. All services load model weights and reach `weights_ready` status.
3. **Allocate** — `vram_coordinator.py` measures remaining free VRAM and distributes KV cache budgets to TTS and STT via their `/allocate_kv_cache` endpoints.

```bash
./scripts/start.sh --trt    # TensorRT STT backend (faster inference, ~15min first build)
./scripts/start.sh --dev    # Expose all ports + debug logging + hot-reload
./scripts/start.sh --dry    # Show VRAM budget calculation only
```

### Verify

```bash
curl http://localhost:8080/health
```

### Create a Session

```bash
curl -X POST http://localhost:8080/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"source_lang": "ko", "target_lang": "en", "processor": "translation"}'
```

Returns `session_id`, `admin_ws_url`, and `listener_ws_url`.

### Feed Audio (Admin WebSocket)

Connect to `/ws/admin/{session_id}` and send binary Float32 PCM frames (16kHz mono, 100ms chunks). The pipeline processes audio through STT → LLM → TTS automatically.

### Receive Output (Listener WebSocket)

Connect to `/ws/listen/{session_id}` to receive:

| Frame Type | Format | Content |
|-----------|--------|---------|
| Text (JSON) | `{type: "korean_partial", text}` | Real-time STT partial |
| Text (JSON) | `{type: "korean_final", text}` | Confirmed STT segment |
| Text (JSON) | `{type: "translation_partial", text}` | LLM streaming tokens |
| Text (JSON) | `{type: "translation_final", text}` | Complete translation |
| Binary | Raw PCM or Opus | TTS audio output |

### Stop a Session

```bash
curl -X DELETE http://localhost:8080/api/sessions/{session_id}
```

### E2E Testing

Test the full pipeline with desktop audio capture:

```bash
# Capture system audio (e.g. YouTube video)
python scripts/test_e2e.py --desktop

# Capture specific app (e.g. Firefox only)
pactl load-module module-null-sink sink_name=firefox_capture
pactl move-sink-input $(pactl list sink-inputs short | grep Firefox | cut -f1) firefox_capture
python scripts/test_e2e.py --desktop --monitor-source firefox_capture.monitor

# Text-only mode (no TTS output)
python scripts/test_e2e.py --desktop --no-tts
```

### Voice Cloning

Use your own voice for TTS output by providing reference audio:

```env
REF_AUDIO_PATH_EN=/app/ref_audio/en.wav
REF_TEXT_EN=The exact text spoken in the reference audio
```

Reference audio requirements:
- **Format:** WAV (PCM, uncompressed — no m4a/mp3/ogg)
- **Duration:** 7-10 seconds
- **Recording:** Headset mic recommended. Phone mics apply AGC and noise cancellation that produce spectral artifacts the voice cloning encoder amplifies.
- **Text match:** The reference text must exactly match what's spoken in the audio. Mismatches cause the model to enter token loops.

Place the WAV file in `tts-server/ref_audio/` and update the env vars in `docker-compose.yml` or `.env`.

### Pluggable Processors

The `BaseProcessor` ABC lets you swap the logic between STT and TTS:

- **translation** — Korean → English via LLM with rolling context window
- **passthrough** — forwards STT text directly to TTS (no LLM)

Implement your own for summarisation, conversation, multi-language routing, or any custom processing.

### Configuration

Copy `.env.example` to `.env` and customise. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `Qwen/Qwen3-4B-AWQ` | vLLM model (any OpenAI-compatible) |
| `WHISPER_MODEL_REPO` | turbo | Whisper model variant |
| `MAX_CONCURRENT_SESSIONS` | 5 | Session limit (503 when exceeded) |
| `TTS_TALKER_TEMPERATURE` | 0.7 | TTS generation temperature |
| `STT_MPS_THREAD_PCT` | 30 | CUDA MPS thread % for STT |
| `LLM_MPS_THREAD_PCT` | 25 | CUDA MPS thread % for LLM |
| `TTS_MPS_THREAD_PCT` | 45 | CUDA MPS thread % for TTS |

VRAM allocation is configured in `vram_budget.yml`:

```yaml
gpu:
  reserved_mb: 800    # OS/desktop overhead (400 for headless)

services:
  llm:
    fixed_mb: 3300        # Model weights + CUDA context
    variable_priority: 1  # KV cache share (relative weight)
  tts:
    fixed_mb: 5900
    variable_priority: 3  # TTS gets most remaining VRAM
  stt:
    fixed_mb: 2000
    variable_priority: 1
```

The startup script auto-computes `LLM_GPU_MEMORY_UTILIZATION`, `TTS_VRAM_BUDGET_MB`, and STT cache fractions.

## Customising the Pipeline

### Swapping the LLM

The LLM runs via vLLM with an OpenAI-compatible API — any model vLLM supports works as a drop-in replacement. Change the model in `.env`:

```env
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
```

Then update the VRAM budget in `vram_budget.yml` to match the new model's weight size:

```yaml
services:
  llm:
    fixed_mb: 5000  # Qwen2.5-7B-AWQ uses ~4-5 GB
```

For larger models, you may also need to adjust `LLM_MAX_MODEL_LEN` and `LLM_MAX_NUM_SEQS` in `docker-compose.yml` to fit within the budget.

### TensorRT STT Backend

The STT server has an optional TensorRT-LLM backend that replaces the default CTranslate2 (faster-whisper) backend. TensorRT builds GPU-specific optimised engines for the Whisper model — inference is faster but requires a one-time engine build step.

**Prerequisites:**

Build the slimmed NGC base image once (this is a large download, ~15 GB):

```bash
cd stt-server
docker build -f Dockerfile.ngc-slim -t ngc-trtllm-slim:1.2.0 .
```

**Running:**

```bash
./scripts/start.sh --trt
```

On first startup, the container builds TRT encoder + decoder engines for your GPU (~10-15 minutes). Engines are cached in a Docker named volume (`trt_engines`) so subsequent starts are fast.

**Rebuilding engines** (e.g. after changing the Whisper model):

```bash
docker volume rm yedam-sts_trt_engines
./scripts/start.sh --trt
```

**Environment variables** (set in `.env` or `docker-compose.tensorrt.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_TRT_MODEL` | `large-v3-turbo` | Whisper model to build engines for |
| `STT_BEAM_SIZE` | `1` | Beam width (1 = greedy, faster) |

Note: TRT engines are tied to your exact GPU architecture and TensorRT version. They are not portable between different GPUs.

### Swapping the STT Model

Change the Whisper variant in `.env`:

```env
WHISPER_MODEL_REPO=large-v3       # or: medium, small, turbo
WHISPER_COMPUTE_TYPE=int8_float16  # or: float16 for higher accuracy
```

Update VRAM budget accordingly — `small` uses ~0.5 GB, `medium` ~1 GB, `large-v3` ~1.5 GB, `large-v3-turbo` ~1.5 GB.

The STT image downloads the model at build time, so rebuild after changing:

```bash
docker compose build stt
```

### TTS: CustomVoice vs Base (Voice Cloning)

The TTS server supports two modes:

**CustomVoice (preset speakers):** Uses built-in speakers like `ryan` (male EN) and `sohee` (female KO). No reference audio needed. Clean output, no artifacts.

```env
TTS_MODEL_DIR=/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
REF_AUDIO_PATH_EN=   # leave empty to use presets
```

Download the CustomVoice model to `tts-server/models/` and add a volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ./tts-server/models/Qwen3-TTS-12Hz-0.6B-CustomVoice:/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice:ro
```

**Base + Voice Cloning:** Clone any voice from a reference audio recording. Uses the `Qwen3-TTS-12Hz-0.6B-Base` model (default).

```env
REF_AUDIO_PATH_EN=/app/ref_audio/en.wav
REF_TEXT_EN=The exact text spoken in the reference audio
```

Reference audio tips:
- WAV format, 7-10 seconds, headset mic
- Text must exactly match the spoken audio (mismatches cause token loops)
- Phone mics produce spectral artifacts — use a headset

### TTS: Voice Design (Experimental)

The `/synthesize/voice-design` endpoint generates speech from a natural language voice description:

```bash
curl -X POST http://localhost:7860/synthesize/voice-design \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "instruct": "Male, deep voice, calm", "language": "en"}'
```

Note: the 0.6B model has limited instruction-following — it responds to pitch/pace cues but gender control is unreliable. The 1.7B model follows instructions much better but requires more VRAM.

### VRAM Budget for Different GPUs

The `vram_budget.yml` file defines how VRAM is split between services. Adjust `fixed_mb` values when changing models, and `reserved_mb` based on your environment:

```yaml
gpu:
  reserved_mb: 800   # Desktop with display (set to 400 for headless servers)
```

For a 24GB GPU (e.g. RTX 4090), you have ~12 GB extra headroom. Options:
- Use a larger LLM (7B-AWQ at ~5 GB instead of 4B-AWQ at ~3.3 GB)
- Use the 1.7B TTS model (~3.9 GB instead of 0.6B at ~5.9 GB total with tokenizer)
- Increase KV cache budgets for more concurrent sessions

Run `./scripts/start.sh --dry` to preview the allocation without starting services.

## Benchmarks

All measurements on **NVIDIA RTX 5060 Ti 16GB**, Docker Compose, CUDA MPS enabled.

### VRAM Usage

| Service | Model | Measured VRAM |
|---------|-------|--------------|
| STT | Whisper large-v3-turbo (int8_float16) | ~1.5 GB |
| LLM | Qwen3-4B-AWQ (4-bit) | ~3.3 GB |
| TTS | Qwen3-TTS 0.6B (FP16) + tokenizer + CUDA graphs | ~5.9 GB |
| **Total (5 sessions)** | | **~15.1 GB** (875 MB free) |

### TTS Latency and RTF

RTF (Real-Time Factor) = generation time / audio duration. RTF < 1.0 means faster than real-time.

| Concurrent | Avg RTF | Avg TTFB | P95 Latency | vs 1 session |
|-----------|---------|----------|-------------|-------------|
| 1 | 0.46 | 173ms | 1.65s | 1.00x |
| 2 | 0.68 | 252ms | 5.51s | 1.93x |
| 4 | 0.76 | 311ms | 5.38s | 2.26x |
| 8 | 1.13 | 439ms | 6.07s | 2.84x |

RTF stays under 1.0 (real-time) for up to 5 concurrent sessions. STT scales sub-linearly via batch inference. LLM scales well via vLLM continuous batching. TTS is the bottleneck.

### Running Benchmarks

```bash
python scripts/benchmark_tts_concurrent.py --max-concurrent 8
python scripts/benchmark_stt.py
python scripts/profile_vram.py
```

## Known Issues

- **CUDA MPS not working with Docker containers** — requires UID matching between host and container processes. Currently using per-process thread percentage as a workaround.
- **Voice cloning sensitive to reference audio quality** — phone microphones produce spectral artifacts that the voice cloning encoder amplifies. Headset recordings work well.
- **TTS streaming has minor audio artifacts** — windowed decode with Hann crossfade produces subtle discontinuities at chunk boundaries. Non-streaming decode is artifact-free.

## TODO

- [ ] Fork vLLM for coordinator support — currently vLLM allocates KV cache at boot via `--gpu-memory-utilization`, so the coordinator can't measure true free VRAM after all weights load. Fork needs: `/startup` endpoint returning `weights_ready`, `/allocate_kv_cache` endpoint accepting `budget_mb`, and deferred KV cache allocation (matching the TTS/STT pattern)
- [ ] Audio preprocessing (RNNoise noise suppression)
- [ ] Legacy GPU compatibility (SM75 / T4 — TTS needs CUDA compute cap rebuild, vLLM needs `--dtype float16`)
- [ ] Punctuation-based flush detector for non-Korean languages

## Acknowledgements

- [WhisperLive](https://github.com/collabora/WhisperLive) by Collabora — STT server foundation
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) by tsdocode — TTS inference engine
- [vLLM](https://github.com/vllm-project/vllm) — LLM serving with continuous batching
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) by Alibaba — TTS model
- [Qwen3-TTS-Tokenizer-12Hz-48kHz](https://huggingface.co/tsdocode/Qwen3-TTS-Tokenizer-12Hz-48kHz) by tsdocode — 48kHz decoder
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN — CTranslate2 Whisper backend

## License

MIT
