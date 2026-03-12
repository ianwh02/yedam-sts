# Yedam STS - Project Context

## What This Is
An open-source, GPU-optimized real-time Speech-to-Speech (STS) pipeline designed to **maximise concurrent sessions on a single consumer GPU**. Named after Yedam Manchester Korean Church, but built as general-purpose building blocks for anyone to build STS applications on top of (translation services, conference interpretation, meeting translation, voice AI, etc.).

Stage 1 is the open-source pipeline. Stage 2 builds a church translation platform on top of it. But the repo itself is not church-specific.

## Core Objective: Maximise Concurrency
The primary goal is squeezing the most concurrent STS sessions out of a single GPU by minimising latency and VRAM usage at every stage. This is what differentiates yedam-sts from every other open-source STS project:

- **CUDA MPS** — concurrent kernel execution across STT/LLM/TTS processes (no context-switching overhead)
- **VRAM-budgeted models** — quantized models (Whisper int8, LLM AWQ 4-bit, TTS FP16) chosen to coexist on 16GB
- **vLLM continuous batching + prefix caching** — multiple concurrent requests batched into single forward passes
- **STT batch inference** — WhisperLive BatchInferenceWorker queues multiple sessions into single batched passes
- **TTS priority queue + staleness dropping** — graceful degradation under load instead of OOM
- **Rolling buffer clear** — constant VRAM usage regardless of session duration
- **Sentence-level pipelining** — TTS(N) overlaps with LLM(N+1), no serial blocking

## Architecture
```
Audio Input → [Preprocess] → [WhisperLive STT] → confirmed segments → [Processor (pluggable)] → stream tokens
                                                                              │
                                                                        sentence boundary
                                                                              │
                                                                    [Qwen3 TTS] → audio output
                                                                    (while processor handles next segment)
```

The pipeline is modular — each stage is a separate service, connected via the orchestrator. Consumers of this pipeline provide their own input/output transport (WebSocket, HTTP, gRPC, etc.) and their own application logic on top.

## Key Design Decisions
- **STT flushing:** Only send `completed: true` segments to the processor. Partial text is available as a callback but not processed by the LLM.
- **Rolling audio window:** `clear_buffer` after each completed segment prevents unbounded VRAM growth. Whisper's `initial_prompt` maintains linguistic context after buffer clear.
- **Pluggable processors:** `BaseProcessor` ABC allows translation, passthrough, conversation, or any custom processing mode. This is what makes the pipeline general-purpose.
- **LLM swappable:** vLLM OpenAI-compatible API means changing models is one env var change.
- **Audio format:** Input = Float32 PCM 16kHz. Output = raw PCM (consumers handle encoding/transport).

## VRAM Budget (16GB GPU target - RTX 5060 Ti)
- Whisper large-v3-turbo (int8_float16): ~1.5 GB
- Qwen2.5-7B-Instruct-AWQ (4-bit): ~4-5 GB
- Qwen3-TTS 0.6B (FP16): ~2 GB
- Overhead/KV cache: ~2-3 GB
- Total: ~10-12 GB (fits, with ~4-6 GB headroom)

## Existing Code to Reuse (from ego-app project)
The developer has a prior project at `C:\Users\ianch\Documents\Ian\Shannon\repos\ego-app` with:
- **WhisperLive STT** with batch inference, energy pre-check, buffer clearing (stt-server/)
- **Qwen3 TTS** Python FastAPI server with Korean/English voice mapping (tts-server/server.py)
- **WhisperLive client** with clear_buffer protocol (frontend/.../useWhisperSTT.ts)
- **Docker GPU orchestration** patterns (stt-server/docker-compose.yml)

## Tech Stack
- **Orchestrator:** Python FastAPI + asyncio (CPU-only coordination)
- **STT:** WhisperLive (faster-whisper backend, large-v3-turbo model)
- **LLM:** vLLM with Qwen2.5-7B-Instruct-AWQ (swappable via env var)
- **TTS:** Qwen3-TTS 0.6B CustomVoice via nano-qwen3tts-vllm (vLLM-style continuous batching + CUDA graphs)
- **GPU sharing:** CUDA MPS for concurrent kernel execution across services
- **Deployment:** Docker Compose locally, RunPod/Modal serverless for production

## Implementation Progress

### Completed
- [x] 1.1 Repo scaffold + Docker Compose skeleton + CUDA MPS setup
- [x] 1.2 STT server: forked WhisperLive into stt-server/ (faster_whisper backend, large-v3 model, batch inference, int8_float16)
- [x] 1.3 vLLM container: Qwen2.5-7B-AWQ verified working
- [x] 1.4 TTS server: nano-qwen3tts-vllm (vLLM-style continuous batching + CUDA graphs, Python FastAPI)
- [x] 1.6-1.9 Pipeline wiring: SessionOrchestrator wires STT→Processor→TTS via consumer callbacks (SessionCallbacks). REST API (POST/DELETE /api/sessions), admin WS feeds audio, sentence boundary detection enables TTS pipelining. Output is raw PCM + text via callbacks — transport/encoding is consumer's responsibility.
- [x] 1.10 E2E integration test: all services healthy, LLM/TTS/STT individual tests pass, full pipeline session lifecycle works

- [x] 1.11 VRAM profiling + latency benchmarks: 5 sessions fit on 16GB (875 MiB free). TTS identified as bottleneck (8.3x at 5 sessions). LLM scales well (1.4x via vLLM batching). Profiling script at scripts/profile_vram.py
- [x] 1.12 Optimization pass: Removed TTS Mutex (concurrent GPU inference), per-session TTS queues with semaphore, smarter sentence splitting (comma + max-length), raw PCM output (skip WAV round-trip), age-based staleness dropping, session concurrency limit, CUDA MPS wired into docker-compose

### Next
- [ ] 1.5 Audio preprocessing: implement RNNoise integration in orchestrator/src/audio/
- [ ] 1.13 Legacy GPU compatibility (SM75 / T4 — TTS needs `CUDA_COMPUTE_CAP` rebuild, vLLM needs `--dtype float16`)
- [ ] 1.14 Example fork repos (demonstrate callback API usage)
- [ ] README.md: quickstart, architecture, GPU compatibility matrix, benchmarks, callback API docs

## Competitive Positioning
No existing open-source STS project optimises for concurrent sessions on a single GPU. Existing projects are either:
- **Research models** (StreamSpeech, Hibiki) — single end-to-end models, not deployable pipelines
- **Voice assistant frameworks** (Pipecat, HF speech-to-speech, Vocalis) — designed for 1:1 conversation, not concurrent multi-session
- **Component libraries** (RealtimeSTT, RealtimeTTS) — building blocks without orchestration

yedam-sts fills the gap: a production-ready, Docker-deployable STS pipeline that maximises concurrent sessions on consumer hardware. Consumers build their own application layer (translation SaaS, meeting tools, conference interpretation, etc.) on top.

## Full Plan
See [PLAN.md](PLAN.md) for the complete 3-stage implementation plan including:
- Stage 2: Church translation web platform (built on top of this pipeline)
- Stage 3: Fine-tuning (Korean sermon STT, LLM translation, TTS custom voice)
- Cost projections, deployment architecture
