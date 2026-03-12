# Yedam STS - Implementation Plan

## Context

This project builds an open-source, GPU-optimized real-time Speech-to-Speech (STS) pipeline designed to **maximise concurrent sessions on a single consumer GPU**. Named after Yedam Manchester Korean Church, but the core pipeline is general-purpose building blocks — not church-specific.

The key insight: existing open-source STS projects (HF speech-to-speech, Pipecat, Vocalis) are designed for 1:1 conversational AI and don't optimise for GPU efficiency or concurrency. No one has built a production-ready STS pipeline that runs 3-5+ concurrent sessions on a single 16GB GPU. yedam-sts fills that gap.

**Target users:** Developers building STS applications — translation services, conference interpretation, meeting translation, voice AI agents, accessibility tools, etc. They get a Docker-deployable pipeline and build their own application layer on top.

**Stage 1** is the open-source pipeline (this repo). **Stage 2** is a church translation platform built on top of it (separate concern). **Stage 3** is fine-tuning for Korean sermon domain.

The developer has already built the core components in the ego-app project: WhisperLive STT with batch inference (15 concurrent users on 5060 Ti), Qwen3 TTS (Python + Rust servers), and a full STT→LLM→TTS pipeline with producer/consumer streaming.

---

## Stage 1: Open-Source GPU-Optimised STS Pipeline

**Goal:** General-purpose STS building blocks that maximise concurrent sessions on a single consumer GPU. Developers build their own application layer (transport, UI, business logic) on top.

### Repo: `yedam-sts` (open source)

```
yedam-sts/
├── docker-compose.yml                # One-command startup for all services
├── docker-compose.dev.yml            # Local dev overrides
├── .env.example
├── README.md                         # Architecture, quickstart, benchmarks
│
├── orchestrator/                     # Central coordination service (Python FastAPI)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── src/
│   │   ├── main.py                   # FastAPI + WebSocket server
│   │   ├── config.py                 # Pydantic settings
│   │   │
│   │   ├── pipeline/
│   │   │   ├── manager.py            # PipelineManager: session lifecycle
│   │   │   ├── session.py            # Session: one active pipeline instance
│   │   │   └── processor.py          # Abstract Processor interface (pluggable)
│   │   │
│   │   ├── processors/              # Pluggable processing modes
│   │   │   ├── base.py               # BaseProcessor ABC
│   │   │   ├── translation.py        # Language translation (LLM)
│   │   │   ├── passthrough.py        # STT-only (transcription)
│   │   │   └── conversation.py       # Speech-to-speech AI chat
│   │   │
│   │   ├── audio/
│   │   │   ├── preprocess.py         # RNNoise + RMS normalization pipeline
│   │   │   └── opus.py               # Opus encoding for TTS output
│   │   │
│   │   ├── stt/
│   │   │   └── client.py             # WhisperLive WebSocket client
│   │   ├── llm/
│   │   │   ├── client.py             # vLLM OpenAI-compatible client
│   │   │   └── prompts.py            # Prompt templates per processor
│   │   ├── tts/
│   │   │   └── client.py             # Qwen3-TTS HTTP client + priority queue
│   │   └── api/
│   │       └── routes.py             # Session start/stop/health + callbacks
│   │
│   └── tests/
│
├── stt-server/                       # Fork of WhisperLive (large-v3 upgrade)
├── llm-server/                       # vLLM config (Qwen2.5-7B-Instruct-AWQ)
├── tts-server/                       # Fork of ego-app Qwen3-TTS server
│
├── scripts/
│   ├── benchmark.sh                  # Latency/throughput benchmarks
│   └── warmup.py                     # GPU warm-up for serverless
│
└── deploy/
    ├── runpod/                       # RunPod serverless template
    ├── modal/                        # Modal.com deployment
    └── docker/                       # Self-hosted deployment
```

### Key Design: Pluggable Processor Interface

The pipeline is **not** hardcoded for any specific use case. Processing is abstracted:

```python
class BaseProcessor(ABC):
    """Pluggable processing step between STT output and TTS input."""

    @abstractmethod
    async def process(self, text: str, context: dict) -> AsyncGenerator[str, None]:
        """Transform STT text into output text. Yields streaming chunks."""
        ...

class TranslationProcessor(BaseProcessor):
    """Language translation via vLLM. Source/target language configurable."""

class PassthroughProcessor(BaseProcessor):
    """No LLM - STT transcription passes through directly."""

class ConversationProcessor(BaseProcessor):
    """Speech-to-speech AI chat via vLLM."""
```

Consumers implement their own processor or use the built-in ones. The pipeline handles all the GPU-optimised orchestration.

### VRAM Budget (16GB GPU - 5060 Ti / T4)

| Service | Model | VRAM |
|---------|-------|------|
| STT | Whisper large-v3-turbo (int8_float16) | ~1.5 GB |
| LLM | Qwen2.5-7B-Instruct-AWQ (4-bit) | ~5.2 GB model + ~2.8 GB KV cache |
| TTS | nano-qwen3tts-vllm 0.6B (FP16 + vLLM batching) | ~2 GB model + ~5.6 GB KV cache (0.35) |
| CUDA overhead | - | ~1.2 GB |
| **Total** | | **~12-14 GB** |

All 3 services share one GPU via CUDA MPS. LLM capped at `GPU_MEMORY_UTILIZATION=0.50` (auto-scales KV cache). TTS at `GPU_MEMORY_UTILIZATION=0.35` for vLLM-style continuous batching KV cache.

### Audio Format Strategy

**Input:** Raw Float32 PCM at 16kHz mono — WhisperLive expects this natively, no conversion needed.

**Output:** Raw PCM from TTS. Consumers handle their own encoding/transport:
- Opus encoding (recommended for real-time speech: ~5ms latency vs ~50ms for MP3)
- WebSocket binary frames, HTTP streaming, gRPC, etc.
- The pipeline provides PCM audio via callbacks — encoding and delivery are the consumer's responsibility

### Audio Input Preprocessing

Church PA audio needs cleaning before STT. Add a preprocessing stage in the orchestrator:

1. **RNNoise** - ML-based noise suppression, runs on CPU (<10ms latency), handles HVAC, reverb, background chatter
2. **RMS Normalization** - normalize volume to consistent level (church PA volumes vary wildly)
3. **Silero VAD** - already in WhisperLive, detects speech vs silence
4. **Energy pre-check** - already implemented (RMS 0.003 threshold), skips silent frames

Pipeline: `Raw PA Audio → RNNoise denoise → RMS normalize → WhisperLive (with VAD + energy check)`

### API Design (pipeline control)

**REST:**
- `POST /sessions` → `{ session_id }` — create a new pipeline session with processor config
- `DELETE /sessions/{id}` → stop and clean up session
- `GET /health` → service health + GPU utilization + active session count

**Pipeline Callbacks:**
Consumers register callbacks to receive pipeline outputs:
- `on_stt_partial(text)` — partial STT transcription (unstable, for display only)
- `on_stt_final(text)` — confirmed STT segment (stable)
- `on_processor_partial(text)` — streaming processor output (e.g., partial translation)
- `on_processor_final(text)` — completed processor output
- `on_tts_audio(pcm_data)` — raw PCM audio from TTS

Consumers build their own transport (WebSocket, HTTP SSE, gRPC, etc.) on top of these callbacks.

### Korean STT Accuracy

Current: whisper-small at ~12% Korean CER. Upgrade path:
- **Immediate:** whisper-large-v3 (~4% Korean CER) - fits VRAM budget since only 1 stream per church
- **Explore:** Korean fine-tuned models (`seastar105/whisper-medium-ko`, `SungBeom/whisper-small-ko`)
- **Stage 3:** LoRA fine-tune on sermon vocabulary

### Reusable Components from ego-app

| Component | Source | Adaptation |
|-----------|--------|------------|
| WhisperLive batch inference | `ego-app/stt-server/` + WhisperLive repo | Upgrade model to large-v3 |
| Qwen3 TTS server | `ego-app/tts-server/server.py` | Fork as-is, minimal changes |
| AudioQueue (iOS background audio) | `ego-app/frontend/.../useRealtimeConversation.ts` lines 66-314 | Reuse in Stage 2 listener TTS playback |
| Buffer clearing protocol | `ego-app/frontend/.../useWhisperSTT.ts` | Reuse for admin audio input |
| Docker GPU orchestration | `ego-app/stt-server/docker-compose.yml` | Template for new compose |

### Concurrency Optimizations (Core Differentiator)

**1. CUDA MPS (Multi-Process Service)** — Most impactful
- Without MPS: GPU context-switches between STT/LLM/TTS processes (~1-2ms per switch, adds up)
- With MPS: concurrent kernel execution from different processes on the same GPU
- Enable via `nvidia-cuda-mps-control` daemon in the Docker entrypoint
- Allocate thread percentages: STT 30%, LLM 50%, TTS 20% (tunable)

**2. vLLM Continuous Batching** — Handles multi-session LLM concurrency natively
- `MAX_NUM_SEQS=16`: up to 16 concurrent sequences batched into single forward passes
- `PREFIX_CACHING=true`: shared system prompts cached in KV cache, not recomputed
- `MAX_MODEL_LEN=2048`: short prompts save KV cache memory for more concurrent sequences
- `GPU_MEMORY_UTILIZATION=0.50`: caps vLLM's VRAM appetite, leaving room for STT+TTS

**vLLM GPU Memory Strategy (CUDA MPS compatibility):**

Under CUDA MPS, all GPU processes share physical memory. vLLM's `--gpu-memory-utilization` calculates KV cache as `total * utilization - model_weights - overhead`, but with MPS, the profiling sees other processes' allocations as overhead, producing incorrect (often negative) KV cache calculations.

**Option A: Startup ordering + percentage (Stage 1 — local/dev)**
- LLM starts first (STT/TTS `depends_on: llm: condition: service_healthy`)
- `--gpu-memory-utilization 0.50` — vLLM profiles on a clean GPU, auto-scales KV cache
- Auto-scales across GPU sizes: 16GB → ~2.8G KV, 24GB → ~6.8G KV, 48GB → ~18.8G KV
- Trade-off: ~90s startup delay while LLM boots before STT/TTS begin

**Option B: Absolute KV cache + entrypoint auto-detect (Stage 2 — RunPod/Modal)**
- All services start in parallel (faster cold start on fresh GPU pod instances)
- `--gpu-memory-utilization 0.70` + `--kv-cache-memory-bytes ${LLM_KV_CACHE_MEMORY}` — bypasses broken MPS profiling
- Entrypoint script detects GPU VRAM and computes KV budget dynamically:
  ```bash
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  KV_MB=$(( TOTAL - 10000 ))  # subtract ~10GB for model weights + other services
  exec vllm serve ... --kv-cache-memory-bytes "${KV_MB}M"
  ```
- Trade-off: more moving parts, but enables parallel startup on fresh instances

**3. STT Batch Inference** — Already proven in ego-app
- WhisperLive's `BatchInferenceWorker` queues audio from multiple sessions into single batched forward passes
- `--batch_max_size` configurable (e.g., 8 concurrent sessions)
- Combined with energy pre-check: silent sessions don't consume batch slots

**4. TTS Priority Queue with Staleness Dropping**
- TTS is the bottleneck (sequential GPU access via Mutex in current Rust server)
- Implement a priority queue: newer sentences get priority over older ones
- If TTS falls behind by >2 sentences, drop oldest queued sentence (consumer falls back to text)
- Consider TTS batching if Qwen3 supports it (batch multiple short sentences)

**5. Dynamic Model Loading**
- If no active session needs TTS, don't keep the TTS model in VRAM
- Lazy-load on first TTS request, unload after 5-min idle timeout
- This frees ~2GB VRAM for more LLM KV cache slots when TTS isn't needed

### Streaming Architecture (STT → Processor → TTS)

**Core challenge:** Live STT produces unstable partial transcriptions that keep revising as more audio arrives. We can't send unstable text to the processor. But waiting too long adds latency.

**Solution: Confirmed-Segment Pipeline with Rolling Window**

```
Audio (continuous) → Preprocess → STT (rolling 30s window)
                                      │
                                 ┌────┴────┐
                            partial     completed
                               │           │
                               ▼           ▼
                        on_stt_partial  Flush to Processor → stream tokens
                        callback            │
                        (consumer       sentence boundary
                         decides)           │
                                      TTS synthesize → on_tts_audio callback
                                      (while processor handles next segment)
```

**STT flushing strategy:**
- Only send `completed: true` segments to the processor (guaranteed stable text)
- WhisperLive marks segments completed when: new segment starts, `same_output_threshold` hit (7 reps), or silence detected
- Partial text is available via callback — consumers decide what to do with it

**Rolling audio window (prevents unbounded buffer growth):**
- After each completed segment → send `clear_buffer` to WhisperLive
- Keep last 3s of audio overlap for acoustic context continuity
- Next inference pass starts from ~3s instead of re-transcribing everything
- This keeps STT inference time constant regardless of session duration
- Already implemented: `clear_buffer` protocol in ego-app `useWhisperSTT.ts`

**STT context via `initial_prompt` (free, no GPU overhead):**
- After `clear_buffer`, Whisper loses linguistic context
- Use Whisper's `initial_prompt` parameter to bias the decoder on reconnection
- Set to: domain-specific vocabulary + last 2-3 transcribed segments
- No encoder overhead — only affects decoder token predictions

**Processor context (configurable per processor):**
- Store full transcript in memory as the session progresses
- Include sliding window of last 5-10 segment pairs as direct context in each LLM request (~500-800 tokens)
- vLLM prefix caching means repeated system prompt portions are cached across requests
- Each processor implementation manages its own context strategy

**Future optimization: Stability-based progressive flushing**
- Track longest common prefix across last 3 consecutive STT partials
- If prefix is stable for 3+ chunks (~1.5s), flush it to processor before segment completion
- Reduces latency from ~3-5s to ~1.5-3s at the cost of slight complexity
- Implement as v2 optimization after the basic pipeline is proven

### Latency Optimizations (End-to-End Speed)

**Target: <4s from speech to processed text, <6s to synthesised audio**

**1. Streaming at Every Stage** (don't batch-and-forward)
```
STT partial segments → on_stt_partial callback immediately
STT completed segment → send to processor immediately
Processor streaming tokens → on_processor_partial callback in real-time
Processor sentence boundary → send to TTS immediately (don't wait for full output)
TTS audio ready → on_tts_audio callback immediately
```

**2. Sentence-Level Pipelining** (overlap processor + TTS)
- While TTS synthesizes sentence N, processor is already handling sentence N+1
- Producer/consumer AsyncQueue pattern from ego-app `realtimeSessionService.ts`
- Net effect: TTS latency is hidden behind processor time for next sentence

**3. Processor Context Window**
- Keep a sliding window of last 3-5 segments as context
- This gives continuity without growing the prompt unboundedly
- Short prompts = faster time-to-first-token (TTFT)

**4. Connection Pooling**
- Persistent WebSocket from orchestrator → WhisperLive (don't reconnect per session)
- HTTP/2 connection pool to vLLM (multiplexed streaming requests)
- Persistent HTTP connection to TTS server

**5. Audio Preprocessing Pipeline** (< 15ms total added latency)
- RNNoise: ~5ms per 20ms frame on CPU
- RMS normalization: ~1ms
- Total preprocessing: ~6ms — negligible compared to STT inference time

**6. Text between stages is fine** (tokenization overhead is ~1ms)
- Audio token / shared embedding approaches exist (SeamlessM4T, Moshi) but aren't production-ready for many languages
- Text boundary is the right abstraction for now — simple, debuggable, swappable models

### Concurrency Capacity Estimates (16GB GPU)

| Sessions | STT VRAM | LLM VRAM (KV cache) | TTS VRAM | Total | Feasible? |
|----------|----------|---------------------|----------|-------|-----------|
| 1 | 3 GB | 4.5 GB (small KV) | 2 GB | 9.5 GB | Yes |
| 3 | 3 GB | 5.5 GB (larger KV) | 2 GB | 10.5 GB | Yes |
| 5 | 3 GB | 7 GB (5 concurrent seqs) | 2 GB | 12 GB | Tight |
| 8+ | 3 GB | 9+ GB | 2 GB | 14+ GB | Needs 24GB GPU |

STT and TTS VRAM is constant (model weights only). LLM VRAM grows with concurrent KV caches. On a 24GB GPU (L4/A10G), 8-10 concurrent sessions are feasible.

### Stage 1 Milestones

| # | Task | Duration | Dependencies |
|---|------|----------|--------------|
| 1.1 | Repo scaffold + Docker Compose skeleton + CUDA MPS setup | 1 day | - |
| 1.2 | STT server: fork WhisperLive, upgrade to large-v3 | 2 days | 1.1 |
| 1.3 | vLLM container: setup with Qwen2.5-7B-AWQ, prefix caching, memory caps | 2 days | 1.1 |
| 1.4 | TTS server: fork from ego-app, add priority queue | 1 day | 1.1 |
| 1.5 | Audio preprocessing: RNNoise + RMS normalization | 2 days | 1.1 |
| 1.6 | Orchestrator: session manager + STT client + audio preprocessing | 3 days | 1.2, 1.5 |
| 1.7 | Orchestrator: processor interface + translation processor (streaming) | 2 days | 1.3, 1.6 |
| 1.8 | Orchestrator: TTS integration + sentence-level pipelining | 2 days | 1.4, 1.7 |
| 1.9 | Pipeline API: session lifecycle + callback hooks for consumers | 2 days | 1.8 |
| 1.10 | End-to-end integration testing on local GPU | 2 days | 1.9 |
| 1.11 | Concurrency testing: multi-session VRAM profiling + latency benchmarks | 2 days | 1.10 |
| 1.12 | Optimization pass: CUDA MPS tuning, TTS staleness dropping, connection pooling | 2 days | 1.11 |

**1.2, 1.3, 1.4, 1.5 can run in parallel.** Total: ~21 days.

---

## Stage 2: Church Translation Platform (Full Stack)

**Goal:** Web platform where churches sign up, run translation sessions, and congregation accesses via QR code. This is where the 1:N broadcast architecture lives — built on top of the yedam-sts pipeline.

### 1:N Broadcast Architecture (Stage 2-specific)
Unlike typical N:N voice pipelines, this is 1:N — one audio source (church PA) → yedam-sts pipeline → broadcast to N listeners via WebSocket. Only 1 GPU pipeline per church regardless of listener count.

- **BroadcastHub:** Fan-out from pipeline callbacks to N listener WebSocket connections
- **Opus encoding:** Pipeline outputs raw PCM → encode to Opus once → broadcast binary frames to all listeners
- **Listener WebSocket:** Binary Opus frames + JSON text (partial Korean, final Korean, partial English, final English)
- **Listener fan-out is CPU-only:** Zero GPU cost regardless of listener count. `asyncio.gather()` for concurrent sends.

### Tech Stack

- **Frontend:** Next.js 14+ (App Router) on Vercel
- **Database:** Supabase (Postgres + Auth + Realtime)
- **Payments:** Stripe (donation-based, not subscription)
- **GPU:** RunPod Pod via REST API (on-demand create/destroy)
- **Subdomains:** Wildcard DNS (`*.churchtranslate.com`) + Next.js middleware routing

### Architecture

```
Admin Dashboard (Next.js on Vercel)
    │
    ├── "Start Sermon" → POST /api/pod/start
    │   → RunPod REST API: create pod + attach network volume
    │   → Poll until healthy (~2-3 min)
    │   → Store pod IP in Supabase session record
    │
Phone (QR) → gracechurch.churchtranslate.com (Vercel)
    │
    ├── WebSocket → RunPod GPU Pod (single container, supervisord)
    │                    │
    │              [yedam-sts pipeline]
    │              STT → LLM → TTS
    │                    │
    │              BroadcastHub → N listeners
    │
    ├── Supabase (auth, church data, transcripts)
    └── Stripe (donations)

Post-sermon:
    Admin clicks "Stop" → DELETE /v1/pods/{id}
    OR auto-idle timeout → container self-terminates via API
```

### Key Pages

- `/` - Landing page + church signup form
- `/admin` - Church admin dashboard (start/stop sessions, monitor)
- `/church/[slug]/listen` - Listener page (QR destination, text + optional audio)
- `/donate/[slug]` - Stripe donation page

### Locked Phone Audio (Critical UX)

Congregation members lock phones during service. Solution:
- **Media Session API** - keeps browser process alive, shows metadata on lock screen
- **Silent audio loop** - prevents OS from suspending the tab
- Reuse `AudioQueue.unlockiOSAudio()` pattern from ego-app

### Database Schema (Supabase)

Core tables:
- `churches` (id, name, slug, admin_email, settings, branding)
- `church_admins` (church_id, user_id, role)
- `sessions` (church_id, started_at, ended_at, stats, gpu_instance_id)
- `transcripts` (session_id, segment_index, korean_text, english_text)
- `donations` (church_id, stripe_payment_intent_id, amount, donor_info)
- `usage_logs` (church_id, date, gpu_seconds, segments_translated)

### Church Signup Flow

Contact-based (intentionally manual for launch):
1. Church fills out signup form on landing page
2. Platform admin reviews and approves via admin panel
3. On approval: create church record, generate slug, send welcome email
4. Church admin logs in, sets up profile, gets QR code for congregation

### Stripe Donations

- One-time donations, not subscriptions
- Each church gets a `/donate/[slug]` page
- Start with platform-held funds + manual payouts
- Add Stripe Connect later for direct-to-church payments

### GPU Deployment

RunPod Serverless is NOT a fit — stateless request-response, doesn't support long-lived WebSocket sessions. Use on-demand GPU pods managed via RunPod REST API instead.

**Provider:** RunPod (best API + reliability) or Vast.ai (cheaper, less reliable)

**Architecture: Single Container + Network Volume**
- RunPod pods run ONE container — bundle all 4 services (STT, LLM, TTS, orchestrator) into a single Docker image using `supervisord` to manage processes
- **Network Volume** (~50GB, ~$3.50/month) persists model weights across pod creates/destroys — no re-downloading 10+GB of models on each cold start
- Cold start with network volume: ~2-3 min (image pull + volume mount + VRAM load)

**Pod Lifecycle via RunPod REST API** (`https://rest.runpod.io/v1`):
```
Admin clicks "Start Sermon"
  → Next.js API route calls GET /v1/pods (startup guard — check for existing pod)
  → POST /v1/pods (create pod with template + network volume)
  → Poll GET /v1/pods/{id} until healthy (~2-3 min)
  → Return pod IP + WebSocket URL to admin dashboard

Admin clicks "Stop Sermon" (or auto-idle timeout)
  → DELETE /v1/pods/{id} (terminate pod — GPU + container destroyed)
  → Only network volume storage remains (~$3.50/month)
```

**Key API operations:**
| Operation | Method | Endpoint |
|---|---|---|
| Create pod | `POST` | `/v1/pods` |
| List pods (startup guard) | `GET` | `/v1/pods` |
| Get pod status | `GET` | `/v1/pods/{id}` |
| Stop pod (keep disk) | `POST` | `/v1/pods/{id}/stop` |
| Terminate pod (destroy) | `DELETE` | `/v1/pods/{id}` |

**Auto-shutdown:** Container runs an idle monitor — if no active WebSocket connections for 10 min, calls RunPod API to self-terminate.

Each pod handles 1-3 churches depending on GPU tier (T4 16GB or A10G 24GB).

### Stage 2 Milestones

| # | Task | Duration | Dependencies |
|---|------|----------|--------------|
| 2.1 | Domain + Vercel + Supabase project setup | 1 day | - |
| 2.2 | Database schema + migrations | 1 day | 2.1 |
| 2.3 | Next.js scaffold + Supabase Auth | 2 days | 2.2 |
| 2.4 | Landing page + signup form | 2 days | 2.3 |
| 2.5 | Admin dashboard: church management | 3 days | 2.3 |
| 2.6 | Admin dashboard: session start/stop/monitor | 3 days | 2.5, Stage 1 |
| 2.7 | Listener page: WebSocket + text display | 3 days | Stage 1 |
| 2.8 | Listener page: TTS audio + locked-phone support | 3 days | 2.7 |
| 2.9 | QR code generation + church branding | 1 day | 2.5 |
| 2.10 | Stripe donation integration | 2 days | 2.3 |
| 2.11 | Wildcard subdomain routing | 1 day | 2.3 |
| 2.12 | Single-container image (supervisord) + network volume setup on RunPod | 3 days | Stage 1 |
| 2.13 | RunPod API integration: pod create/destroy from Next.js + startup guard + auto-idle shutdown | 3 days | 2.12 |
| 2.14 | End-to-end testing | 3 days | all above |

Total: ~30 days. Can overlap with Stage 1 tail (2.1-2.5 don't need GPU pipeline).

---

## Stage 3: Fine-tuning & Training

**Goal:** Improve STT accuracy on Korean sermons, LLM translation quality on theological vocabulary, and offer custom TTS voices.

### 3A: Korean Sermon STT Fine-tuning

**Data source:** YouTube Korean sermon channels with human-written subtitles (NOT auto-generated)
- Use `yt-dlp` with `--write-sub --no-write-auto-sub --sub-lang ko` to filter
- Target channels: 사랑의교회, 여의도순복음교회, 온누리교회
- Target: 100-500 hours of paired audio + transcription

**Method:** LoRA fine-tune Whisper large-v3 using PEFT
- Target modules: `q_proj`, `v_proj` (attention layers)
- r=32, alpha=64, dropout=0.05
- Goal: reduce Korean sermon CER from ~4% to <2%

### 3B: LLM Translation Fine-tuning

**Data source:** Parallel Korean↔English sermon translations from church Google Drive folders
- Many Korean-American churches maintain bilingual bulletins/scripts
- Even 100-200 high-quality pairs make a meaningful difference

**Method:** LoRA fine-tune Qwen2.5-7B-Instruct using unsloth or axolotl
- Training data format: conversation pairs (Korean input → English output)
- Include Bible verse references, theological terms, church-specific vocabulary
- Evaluate with BLEU/COMET scores against held-out test set

### 3C: TTS Custom Voice

- Qwen3-TTS CustomVoice variant already supports voice cloning
- Churches provide 5-10 min of preferred voice → custom TTS voice
- Nice-to-have feature, not critical for launch

### Stage 3 Milestones

| # | Task | Duration | Dependencies |
|---|------|----------|--------------|
| 3.1 | YouTube scraping pipeline + subtitle filtering | 3 days | - (start early) |
| 3.2 | Data cleaning + audio↔subtitle alignment | 5 days | 3.1 |
| 3.3 | Whisper LoRA fine-tuning setup + training | 5 days | 3.2, Stage 1 |
| 3.4 | Parallel sermon data collection from churches | ongoing | - |
| 3.5 | LLM translation LoRA fine-tuning | 5 days | 3.4, Stage 1 |
| 3.6 | Evaluation + A/B testing (base vs fine-tuned) | 3 days | 3.3, 3.5 |
| 3.7 | TTS custom voice pipeline | 3 days | Stage 1 |
| 3.8 | Integration of fine-tuned models | 2 days | 3.6 |

Total: ~26 days. Data collection (3.1, 3.4) can start in parallel with Stage 1/2.

---

## Cost Projections

### Per-Church GPU Cost (On-Demand T4 16GB @ ~$0.44/hr)

| Usage | Calculation | Monthly |
|-------|-------------|---------|
| Sunday service (90 min) | 1.5hr x $0.44 x 4 weeks | $2.64 |
| Cold start boot (~3 min) | 0.05hr x $0.44 x 4 weeks | $0.09 |
| Network volume (50GB models) | 50GB x $0.07/GB/month | $3.50 |
| **Total per church** | | **~$6/month** |
| **Idle/between sermons** | Pod destroyed, only volume billed | **$3.50** |

With multi-church sharing (3 churches per GPU, shared network volume): **~$3/month per church**

### Platform Fixed Costs

| Item | Monthly |
|------|---------|
| Supabase Pro | $25 |
| Vercel Pro | $20 |
| Domain | ~$1 |
| **Total** | **~$46/month** |

### Break-even

Platform costs covered by ~1 church donating $46/month. At 10 churches with average $30/month donation: $300 revenue vs ~$106 costs = comfortable margin.

---

## Verification Plan

### Stage 1 Testing
1. Run `docker compose up` on local 5060 Ti — all 4 services start, pass health checks, CUDA MPS active
2. Audio preprocessing: feed noisy audio → verify RNNoise cleans it, RMS normalizes
3. Feed audio via pipeline API → verify STT callbacks fire with transcription
4. Verify processor output streams via callback within 4 seconds of speech
5. Verify TTS audio arrives as raw PCM via callback
6. Connect 3 simultaneous sessions → verify no cross-contamination, VRAM stays under 14GB
7. Connect 5 sessions → verify TTS priority queue drops stale sentences gracefully
8. Benchmark per-component latency: STT segment time, LLM TTFT, TTS synthesis time
9. Benchmark concurrency: measure max concurrent sessions before VRAM OOM on 16GB / 24GB
10. Verify sentence-level pipelining: TTS(N) overlaps with processor(N+1) — no serial blocking

### Stage 2 Testing
1. Create church account → verify subdomain routing works
2. Admin starts session → verify GPU pod boots and passes health checks within 10 minutes
3. Open listener page on phone → scan QR → see live translation text
4. Lock phone → verify audio continues playing (AudioQueue + Media Session API)
5. Test with 50+ simultaneous listeners on one session (1:N broadcast)
6. Process a Stripe test donation
7. Verify Opus encoding + binary WebSocket broadcast to all listeners

### Stage 3 Testing
1. Compare base vs fine-tuned Whisper CER on held-out Korean sermon test set
2. Compare base vs fine-tuned LLM BLEU score on held-out translation test set
3. Blind evaluation: have Korean-English bilingual speakers rate translation quality

---

## Overall Timeline

```
Weeks 1-3:   Stage 1 (yedam-sts, open source)
Weeks 2-6:   Stage 2 (church platform, overlaps with Stage 1 tail)
Weeks 1+:    Stage 3 data collection (runs in background)
Weeks 7-9:   Stage 3 fine-tuning + integration
Weeks 9-10:  Beta with 2-3 pilot churches
Week 11:     Public launch
```

**Critical path:** Stage 1 completion → Stage 2 listener page → RunPod deployment → beta testing

**Ship MVP without Stage 3** - base models are good enough to launch. Fine-tuning improves quality but isn't blocking.
