# Planned Features

## 1. Accurate VRAM Budget Allocation

**Problem:** Current `vram_budget.yml` uses hand-estimated `fixed_mb` values for model weights that don't match reality. On larger GPUs (A5000 24GB), this over-reserves for weights and under-allocates KV cache, leaving ~5GB unused.

**Goal:** Measure actual VRAM usage after model loading instead of relying on YAML estimates. Scale KV cache allocation to fully utilize whatever GPU is available.

**Approach:**
- Measure actual VRAM after each service loads weights (phase 1)
- Allocate remaining VRAM as KV cache in phase 2 based on measured free space
- Remove or reduce reliance on `fixed_mb` estimates in `vram_budget.yml`

---

## 2. Fork vLLM Base Image for Slim Builds + Parallel Loading

**Problem:** The Docker image is ~48GB (uncompressed) because it pulls the full `vllm/vllm-openai` base image with all model architectures, attention backends, and dev tools. Image pull + extract is the main bottleneck for pod startup (~3-5 min).

**Goals:**
1. **Slim image:** Strip unused vLLM components (model architectures we don't use, dev files, benchmark tools). Target ~15-20GB uncompressed (~4-5GB download).
2. **Parallel model loading:** Currently LLM loads first, then TTS+STT. With a forked vLLM, we can load LLM model weights + allocate KV cache in parallel with TTS and STT loading, reducing total startup time.

**Approach:**
- Fork vLLM, keep only: OpenAI entrypoint, AWQ/Marlin quantization, flash-attn backend
- Start from a slimmer CUDA runtime base image instead of vLLM's full dev image
- Modify `start_unified.sh` to launch all services in parallel with a barrier before KV cache allocation

---

## 3. Continuous TTS Streaming (LLM → TTS Word-by-Word Buffer)

**Problem:** The current pipeline fragments LLM translation output into sentences/clauses and sends each independently to TTS. Because TTS is autoregressive, each fragment starts with a fresh context — causing:
- **Tonal discontinuity** between segments (each segment may have different pitch/energy)
- **Single-word fragments** sound unnatural or use a different voice character
- **Questions** may lose interrogative intonation when fragmented
- The in-session decoder context (`_session_codec_cache`) helps but doesn't fully solve it since each generation starts fresh

**Current flow:**
```
LLM streams tokens → SentenceBoundaryDetector splits at .!?,
→ Each sentence enqueued independently to TTS
→ TTS generates each sentence from scratch (only x-vector/ICL prompt carries over)
→ Tonal mismatch between segments
```

**Desired flow:**
```
LLM streams tokens → Buffer feeds words to TTS continuously
→ TTS generates one long continuous stream
→ Consistent tone across the entire segment
```

**Challenges:**
- TTS needs the full text upfront to plan prosody — word-by-word feeding may not be possible with the current model architecture (Qwen3-TTS tokenizes the full text before generation)
- If the speaker is faster than TTS generation (RTF > 1), the buffer underflows and we need silence padding
- If we wait for the full LLM output before starting TTS, we lose the latency benefit of streaming

**Quick wins (no model changes):**
1. **Full segment accumulation:** Don't split at commas/periods. Wait for the full LLM output per STT segment (~200-500ms) and send as one TTS request. Trades ~300ms latency for perfect tonal consistency within a segment. Simplest fix.
2. **Hybrid buffering:** Wait for at least N words or the full LLM output, whichever comes first. Short segments get the full text; long segments start after a reasonable buffer fills.

**Recommended approach: TTS KV cache continuation (no model retraining needed)**

The core insight: the talker is a causal transformer. Its KV cache contains the full generation history including prosodic state. If we don't clear it between sentences and instead append new text, the model continues generating in the same voice/tone.

```
Current (tonal reset per sentence):
  Sentence 1: [prompt + text1] → generate → clear KV → done
  Sentence 2: [prompt + text2] → generate → clear KV → done

Proposed (continuous KV cache):
  Sentence 1: [prompt + text1] → generate → KEEP KV → EOS
  Sentence 2: [append text2 embeddings] → continue from same KV state
  Sentence 3: [append text3 embeddings] → continue...
  (talker "remembers" prosody — no tonal reset)
```

**What changes in nano-qwen3tts-vllm:**

1. **Talker scheduler** — support "extend" operation: append new input embeddings to an active/paused request without clearing its KV cache
2. **interface.py** — new `continue_generation(request_id, new_text)` method that computes new `trailing_text_hiddens` for the chunk and feeds them into the existing request
3. **server.py** — session-persistent TTS stream that stays alive across sentences, receiving new text chunks as they arrive from the LLM
4. **Orchestrator** — instead of enqueueing independent TTS jobs per sentence, maintain a single persistent TTS stream per session that receives text incrementally

**What stays the same:**
- Model weights/architecture — no retraining
- Predictor/decoder pipeline — only talker input changes
- Voice clone prompt — applied once at session start
- Codec decoding/streaming — still produces chunks as codec tokens arrive

**Open challenges:**
- `trailing_text_hiddens` is pre-computed for a fixed text length. With continuation, need to compute it per-chunk and extend the sequence. The LLM text encoder may need to re-encode with the new text appended (or use incremental KV caching on the text encoder side too).
- KV cache grows unboundedly over a long session — need a sliding window or periodic compaction (similar to how LLM context windows work).
- If LLM output is slower than TTS generation, the talker must "pause" and wait for more text. Options: hold the last hidden state, generate silence tokens, or use a special "wait" token.
- EOS handling changes — the talker shouldn't emit EOS between sentences, only at the true end of the session/segment. Need a mechanism to signal "more text coming" vs "segment done."
- Error recovery — if one sentence's generation fails, the KV cache state may be corrupted for subsequent sentences.
