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

---
---

# Latency / Compute Optimizations from KoljaB's Repos

**Source repos**: [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS), [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)

Identified 2026-03-28. Ranked by impact for the yedam-sts pipeline.

---

## High Impact — Latency Reduction

### K1. Speculative LLM Pre-generation

**Source**: RealtimeVoiceChat (`speech_pipeline_manager.py`)

**Current behavior**: Orchestrator waits for final STT segment → fires LLM translation request.

**Proposed**: Start LLM inference *before* the user finishes speaking, using sentence-end detection heuristics.

**How KoljaB does it**:
- `detect_potential_sentence_end()` monitors realtime STT partials
- Keeps a cache of recent normalized texts
- If the same punctuation-terminated text appears 3+ times within 200ms (STT has stabilized), triggers `prepare_generation()` which queues LLM inference
- If the user continues speaking and text changes significantly, `check_abort()` compares text similarity (weighted SequenceMatcher focusing on last 5 words, threshold 0.95) and aborts the speculative generation
- Uses `TextSimilarity` class to avoid re-generating for nearly identical partial transcriptions

**Integration points in our codebase**:
- `orchestrator/src/pipeline/orchestrator.py` — add speculative LLM trigger on partial STT updates
- `orchestrator/src/stt/client.py` — expose partial text stability metrics to orchestrator
- Need abort mechanism for in-flight LLM requests (cancel httpx streaming)

**Estimated savings**: ~200-500ms of LLM TTFT hidden behind user speech

**Risks**: Wasted GPU compute on aborted speculative requests. Mitigated by text similarity check before aborting.

---

### K2. Quick Answer / First Fragment Split

**Source**: RealtimeVoiceChat (`speech_pipeline_manager.py`) + RealtimeTTS (`stream.py`)

**Current behavior**: `SentenceBoundaryDetector` accumulates LLM tokens, splits on punctuation but waits for `min_words_comma_split=8` words before splitting on commas. First TTS request fires after first full sentence detected.

**Proposed**: Yield the first sub-sentence fragment as soon as any delimiter is found, regardless of length.

**How KoljaB does it**:
- **RealtimeVoiceChat**: Splits response into two phases:
  - *Quick answer*: First sentence extracted via `TextContext.get_context()` which scans for first natural sentence boundary (period, comma, exclamation, etc. with minimum length/alphanumeric constraints). Sent to TTS immediately on a dedicated thread.
  - *Final answer*: Remaining LLM tokens stream to TTS via a separate generator-based synthesis path on another thread. Both go into the same audio queue for seamless playback.
- **RealtimeTTS** (`stream2sentence`): `fast_sentence_fragment=True` yields on the *first* delimiter found. `force_first_fragment_after_words=30` forces yield even without a delimiter for long openings. `minimum_first_fragment_length=10` chars prevents trivially small fragments.

**Integration points in our codebase**:
- `orchestrator/src/pipeline/orchestrator.py` — modify `SentenceBoundaryDetector` to have a `first_fragment` mode with lower thresholds
- `orchestrator/src/tts/client.py` — no change needed (already has per-session queue)
- Consider: reduce `min_words_comma_split` from 8 to 3-4 for the first fragment only, then revert to 8 for subsequent fragments

**Estimated savings**: ~200-500ms to first audio chunk

**Tradeoffs**: Very short fragments (<5 words) may produce slightly lower TTS quality. Tunable via minimum_first_fragment_length.

---

### K3. ML-Based Turn Detection

**Source**: RealtimeVoiceChat (`turndetect.py`)

**Current behavior**: Fixed silence thresholds + Korean grammar endings (`korean_endings.py`) determine when user is done speaking.

**Proposed**: Use a fine-tuned DistilBERT model to predict whether partial STT text is a complete utterance, dynamically adjusting silence duration.

**How KoljaB does it**:
- Model: `KoljaB/SentenceFinishedClassification` (DistilBERT fine-tuned for sentence completion prediction)
- Probability interpolated to a pause duration via configurable anchor points
- Combined with punctuation-based heuristics in a weighted average (65% punctuation weight, 35% model weight)
- Final pause dynamically sets `post_speech_silence_duration` on the RealtimeSTT recorder
- Speed factor (0.0-1.0) adjustable from UI — interpolates between "fast" settings (0.33s question pause) and "slow" (0.8s)
- Dynamic behavior: questions ending with "?" get short pause, ambiguous fragments get longer pause

**Integration points in our codebase**:
- `stt-server/whisper_live/` — add turn detection module alongside `korean_endings.py`
- Would need a Korean-trained equivalent of `SentenceFinishedClassification` (the existing model is English)
- Could use Korean grammar endings as a strong signal (equivalent to punctuation heuristics) combined with a Korean sentence completion model
- `orchestrator/src/pipeline/orchestrator.py` — consume dynamic silence duration from STT

**Estimated savings**: 200-400ms average reduction in turn gap

**Challenges**: Requires Korean-language sentence completion model (fine-tune DistilBERT on Korean corpus, or use existing Korean grammar endings as proxy). English model won't work for Korean speech.

---

### K4. Early Transcription on Silence

**Source**: RealtimeSTT (`audio_recorder.py`, lines 2178-2192)

**Current behavior**: STT waits for silence duration to fully expire, then sends audio to Whisper for final transcription.

**Proposed**: When silence is first detected (before threshold expires), speculatively send audio to the main Whisper model. If speech resumes, discard the result. If silence continues, use the pre-computed result instantly.

**How KoljaB does it**:
- `early_transcription_on_silence` flag enables this behavior
- When `speech_end_silence_start` is first set (voice stops), immediately sends current audio to the main model over the transcription pipe
- `transcribe_count` tracks pending speculative transcriptions
- If voice resumes before silence threshold, increments `transcribe_count` — when the speculative result arrives, it's discarded because count doesn't match
- If silence continues past threshold, the already-completed transcription is used immediately — zero additional wait

**Integration points in our codebase**:
- `stt-server/whisper_live/backend/trt_backend.py` or `faster_whisper_backend.py` — add speculative transcription on silence onset
- `stt-server/whisper_live/batch_inference.py` — may need priority flag for speculative requests (lower priority than confirmed final transcriptions)
- Need epoch/counter mechanism to discard stale speculative results

**Estimated savings**: Hides entire Whisper final transcription latency (~100-300ms depending on model size)

**Tradeoffs**: Wasted GPU compute on false alarms (user pauses then continues). With batch inference, speculative requests compete for batch slots. Could be mitigated with a low-priority queue.

---

## Medium Impact — Perceived Latency

### K5. Dual Whisper Model

**Source**: RealtimeSTT (`audio_recorder.py`)

**Current behavior**: Single Whisper model (large-v3-turbo or whisper-medium-komixv2 TRT) handles both partial and final transcription.

**Proposed**: Use a smaller model for real-time partials (fast, lower quality) and a larger model for final transcription (slower, higher quality).

**How KoljaB does it**:
- **Main model** (`model` param, default "tiny"): Runs in a separate process via `multiprocessing.Process` (Windows/Mac) or `threading.Thread` (Linux). Used only when a complete utterance is ready. Communicates via `SafePipe` (thread-safe `multiprocessing.Pipe` wrapper).
- **Realtime model** (`realtime_model_type` param, default "tiny"): Loaded in the main process. Runs transcription in `_realtime_worker()` on a daemon thread. Transcribes the *entire accumulated frames buffer* on each pass (every 20ms in server config).
- `use_main_model_for_realtime` flag allows sharing one model for both (saves GPU memory).
- Main model uses `beam_size=5` for accuracy, realtime uses `beam_size_realtime=3` for speed.

**Integration points in our codebase**:
- Would require loading two Whisper models in the STT server
- VRAM impact: second small model (whisper-tiny) adds ~75 MiB; whisper-base adds ~150 MiB
- Batch inference would only apply to the main model; realtime model runs independently
- For Korean: both models would need Korean fine-tuning for acceptable quality

**Estimated benefit**: Smoother real-time display, lower perceived latency. Partials update faster with tiny model.

**Tradeoffs**: Additional VRAM for second model. May not be worth it if current single-model partials are already fast enough (batch inference with beam_size=1 is already near-instant).

---

### K6. Realtime Text Stabilization

**Source**: RealtimeSTT (`audio_recorder.py`, lines 2435-2493)

**Current behavior**: Korean flushing uses `stability_count=2` (consecutive stable detections before flushing). Partial text sent directly to UI.

**Proposed**: Compute longest common prefix between consecutive transcriptions. Stable prefix never regresses; only the unstable tail updates.

**How KoljaB does it**:
- Keeps `text_storage` (list of all recent transcriptions)
- Computes `os.path.commonprefix([last_two_texts[0], last_two_texts[1]])`
- `realtime_stabilized_safetext` grows monotonically — only updates if new prefix is longer
- `_find_tail_match_in_text` finds where the stable prefix ends in the fresh transcription, appends only the new unstable tail
- Two callback streams: `on_realtime_transcription_update` (raw, jittery) and `on_realtime_transcription_stabilized` (left portion stable, right portion updates)

**Integration points in our codebase**:
- `orchestrator/src/stt/client.py` — add stabilization layer for partial text before broadcasting to listeners
- Or implement server-side in `stt-server/whisper_live/backend/base.py` alongside existing stability mechanism
- Could combine with Korean grammar endings: stable prefix uses grammar-based validation, unstable tail uses common-prefix approach

**Estimated benefit**: Less visual jitter in partial transcription display. Better UX for listeners.

**Tradeoffs**: Minimal compute cost. Slight delay in partial updates (waits for two consecutive transcriptions to compute prefix).

---

### K7. Buffer Threshold Flow Control

**Source**: RealtimeTTS (`stream.py`, `_synthesis_chunk_generator()`)

**Current behavior**: TTS client has staleness dropping (items older than 10s or queue > 2 items are dropped). No awareness of actual client-side audio buffer state.

**Proposed**: Pause sending text to TTS when `buffered_audio_seconds > threshold`. Resume when buffer drops below threshold.

**How KoljaB does it**:
- `buffer_threshold_seconds` parameter in `play()`
- `_synthesis_chunk_generator()` checks `player.get_buffered_seconds()` before yielding each text chunk
- If buffered audio exceeds threshold, pauses text yielding (blocks the generator)
- `AudioBufferManager` tracks total buffered samples: `total_samples / sample_rate = buffered_seconds`
- When set to 0.0, disables flow control (chunks sent immediately)

**Integration points in our codebase**:
- `orchestrator/src/tts/client.py` — add buffer-aware flow control
- Need client-side feedback: how much audio is buffered/playing on the listener
- WebSocket listener protocol would need a `buffer_status` message from client → server
- Alternative: estimate server-side based on audio sent vs. time elapsed

**Estimated benefit**: Reduces wasted TTS compute on segments that will be dropped. More efficient GPU utilization under load.

**Tradeoffs**: Requires client cooperation (buffer reporting) or server-side estimation. Adds complexity to the WebSocket protocol.

---

## Medium Impact — Compute Optimization

### K8. Speaker Embedding Caching with Sentinel Key

**Source**: RealtimeTTS (`faster_qwen_engine.py`)

**Current behavior**: Per-session voice prompt caching via `/sessions/init_voice` endpoint (TTL=1hr). Speaker embedding extraction runs on first TTS request per session.

**Proposed**: Pre-extract speaker embeddings at server startup and inject into model cache under a sentinel key. All synthesis calls hit the cached embedding with zero encoder overhead.

**How KoljaB does it**:
- `_CACHE_SENTINEL = "__preloaded_spk_emb__"` as a fake ref_audio path
- `_prime_cache()` at init: loads or extracts speaker embedding, injects directly into `self._model._voice_prompt_cache` under the sentinel key for both `append_silence=True` and `append_silence=False`
- All synthesis calls use `ref_audio=_CACHE_SENTINEL`, which hits the pre-primed cache entry
- `FasterQwenVoice` class supports `speaker_pt` path — if `.pt` file exists, loads pre-extracted embedding; otherwise extracts from ref_audio/ref_text and optionally saves to disk
- Embeddings are ~4KB tensors, trivial to cache

**Integration points in our codebase**:
- `tts-server/server.py` — pre-extract embeddings for all configured voices at startup
- `tts-server/nano-qwen3tts-vllm/nano-qwen3tts-vllm/interface.py` — inject pre-computed prompts into cache
- Could save extracted embeddings as `.pt` files in `tts-server/ref_audio/` alongside WAV files
- Session init would load from `.pt` instead of re-extracting

**Estimated savings**: ~50-100ms per new session's first TTS request

**Tradeoffs**: Minimal. Small disk footprint. One-time extraction at startup or build time.

---

### K9. Silence Insertion by Delimiter Type

**Source**: RealtimeTTS (`stream.py`)

**Current behavior**: Orchestrator appends fixed inter-segment silence between TTS segments. No variation by punctuation type.

**Proposed**: Insert different silence durations based on the delimiter that ended each sentence/clause.

**How KoljaB does it**:
- End delimiters (`.!?...`): `sentence_silence_duration` seconds of silence (e.g., 300ms)
- Mid delimiters (`;:,\n()-""`): `comma_silence_duration` seconds (e.g., 100ms)
- Other/default: `default_silence_duration` seconds (e.g., 50ms)
- Silence is PCM zeros injected into the audio stream between synthesized segments
- Configurable per-play() call

**Integration points in our codebase**:
- `orchestrator/src/pipeline/orchestrator.py` — `SentenceBoundaryDetector` already tracks which delimiter caused the split. Pass delimiter type to TTS client.
- `orchestrator/src/tts/client.py` — vary inter-segment silence based on delimiter type
- Simple lookup table: `{'.': 0.3, ',': 0.1, ';': 0.15, '!': 0.25, '?': 0.25}`

**Estimated benefit**: More natural prosody without extra TTS model compute

**Tradeoffs**: None significant. Trivial to implement.

---

### K10. Background Noise Detection / Anti-Hallucination

**Source**: RealtimeSTT server (`stt_server.py`)

**Current behavior**: No detection of steady background noise. Whisper may hallucinate repeated phrases on ambient noise, generating phantom STT segments → wasted LLM+TTS compute.

**Proposed**: If realtime transcriptions are nearly identical for several seconds, force-stop recording and discard.

**How KoljaB does it**:
- Server monitors realtime transcription text over time
- Uses `SequenceMatcher` ratio > 0.99 across 3+ seconds of transcriptions
- If detected, forces `recorder.stop()` — prevents infinite recording on steady background noise
- Effectively a "Whisper is hallucinating" detector

**Integration points in our codebase**:
- `stt-server/whisper_live/backend/base.py` — add similarity tracking in `handle_transcription_output()`
- Compare consecutive partial transcriptions; if identical for N seconds, discard and reset buffer
- Could also integrate with the existing `same_output_threshold` mechanism (currently promotes stable text to completed — but for noise, we want to *discard* instead)

**Estimated benefit**: Avoids phantom STT segments that waste LLM and TTS compute. Prevents cascading pipeline waste.

**Tradeoffs**: Risk of discarding legitimate repeated speech (e.g., chanting, liturgical responses). Mitigate with higher similarity threshold or domain-specific exclusions.

---

## Lower Impact — Quality of Life

### K11. Interruption Handling

**Source**: RealtimeVoiceChat (`server.py`, `TranscriptionCallbacks`)

**Current behavior**: No interruption handling. If user speaks while TTS is playing on the client, both streams continue independently.

**Proposed**: When user starts speaking during TTS playback, immediately stop TTS and abort in-flight generation.

**How KoljaB does it**:
- Binary WebSocket header includes `isTTSPlaying` flag (bit 0 of 4-byte flags field)
- `on_recording_start` callback fires when VAD detects new speech
- If `tts_client_playing` is True, triggers interruption cascade:
  1. `tts_to_client = False` — stops sending TTS chunks
  2. Sends `stop_tts` message to client — clears AudioWorklet buffer
  3. `abort_generations()` — propagates stop events to all 4 worker threads (LLM inference, TTS quick, TTS final, request processor)
  4. Sends `tts_interruption` message to client
  5. Saves partial assistant answer to conversation history

**Integration points in our codebase**:
- `orchestrator/src/main.py` — WebSocket protocol: add `isTTSPlaying` flag to binary audio frames from admin client
- `orchestrator/src/pipeline/orchestrator.py` — add interruption logic: cancel in-flight LLM (cancel httpx stream), flush TTS queue, send stop to listeners
- `orchestrator/src/tts/client.py` — add abort mechanism for per-session TTS queue
- Listener WebSocket protocol: add `stop_tts` message type
- Client (Proverberate frontend): implement `isTTSPlaying` tracking in AudioWorklet, handle `stop_tts`

**Estimated benefit**: Natural conversation flow. Reduces wasted compute on unwanted audio generation.

**Tradeoffs**: Significant implementation effort across frontend + backend. Requires client-side changes. Risk of false interruptions (background noise during playback).

---

### K12. Adaptive Turn Detection Speed

**Source**: RealtimeVoiceChat (`server.py`, `turndetect.py`)

**Current behavior**: Fixed silence thresholds configured at deploy time.

**Proposed**: UI slider that lets users control conversation pace in real-time.

**How KoljaB does it**:
- Client sends `{ type: "set_speed", speed: 0-100 }` over WebSocket
- Server interpolates between "fast" and "slow" turn detection settings:
  - Fast (speed=1.0): 0.33s question pause, 0.5s statement pause
  - Slow (speed=0.0): 0.8s question pause, 1.2s statement pause
- Dynamically updates `post_speech_silence_duration` on the recorder
- Interpolation formula: `fast_value + (1 - speed_factor) * (slow_value - fast_value)`

**Integration points in our codebase**:
- `orchestrator/src/main.py` — add speed control message to admin WebSocket
- `stt-server/` — support dynamic `post_speech_silence_duration` updates per client
- Frontend (Proverberate): add slider UI element
- Korean grammar flushing already handles most turn detection; this would supplement silence-based thresholds

**Estimated benefit**: Per-user tuning of responsiveness vs. interruption risk

**Tradeoffs**: Low priority. Korean grammar-based flushing already provides good turn detection for Korean speech. Most useful for non-Korean languages where grammar-based detection isn't available.

---

## Implementation Priority (Revised for Real-Time Translation)

The original priority ranking was designed for a voice chatbot. For **real-time Korean→English
translation**, the constraints are different:

- **Korean is SOV** — the verb at the end changes meaning of everything before it. Speculating
  on incomplete Korean sentences produces wrong translations.
- **One-directional** — the speaker talks continuously; there are no "turns" with an AI.
- **Accuracy > speed** — a wrong translation 200ms earlier is worse than a correct one 200ms later.
- **TTS prosody matters** — listeners need natural-sounding English to comprehend the translation.
  Short isolated fragments produce unnatural intonation because the TTS model can't plan prosody
  across the full sentence.

### Recommended — Useful for translation

| Priority | Feature | Rationale |
|----------|---------|-----------|
| 1 | **K10 Background Noise Detection** | Prevents phantom translations — the #1 accuracy failure mode. Whisper hallucinating on ambient noise (congregation, HVAC, music) cascades through LLM+TTS, producing confidently-spoken nonsense. |
| 2 | **K9 Silence Insertion by Delimiter Type** | More natural English TTS output with varied pauses (period=300ms, comma=100ms). Trivial to implement, improves comprehension. |
| 3 | **K8 Speaker Embedding Caching** | Minor startup optimization (~50-100ms per new session). Zero risk, free win. |

### Deprioritized — Not suitable for translation, or marginal value

| Feature | Why deprioritized |
|---------|-------------------|
| **K7 Buffer Threshold Flow Control** | **Harmful for translation.** Pausing TTS when the buffer is full means sentences never get synthesized — the listener misses content. For translation, every sentence matters. Falling 5 seconds behind real-time is better than skipping 70% of the content. The current approach (keep everything in queue, play all of it) is correct. |
| **K1 Speculative LLM Pre-generation** | **Harmful for Korean.** Korean SOV word order means incomplete clauses are untranslatable — "나는 학교에" could become "I went to school" / "I want to go to school" / "I didn't go to school." Korean grammar endings already identify the right flush points with linguistic precision. Speculating wastes GPU and risks wrong translations. |
| **K2 Quick Answer / First Fragment Split** | **Degrades TTS quality.** Sending short English fragments to TTS independently prevents the model from planning prosody across the full sentence. "In the beginning," generated alone gets different (worse) intonation than when the model sees the full sentence. For translation, natural prosody is essential for listener comprehension. This is the same problem identified in Feature #3 (Continuous TTS Streaming). The current approach of waiting for full sentences is correct. |
| **K3 ML-Based Turn Detection** | Korean grammar endings (`korean_endings.py`) are a better "turn detector" for Korean than any generic ML model. 80+ sentence endings + 50+ phrase endings with linguistic knowledge of Korean morphology. A DistilBERT model would need Korean fine-tuning to match, duplicating existing capability. |
| **K4 Early Transcription on Silence** | STT already transcribes continuously via batch inference with rolling buffers. Korean grammar endings flush completed clauses regardless of silence. At best this shaves ~50ms on the rare case where silence is the only flush trigger. |
| **K5 Dual Whisper Model** | Partials are display-only; translation uses final text. A tiny model's lower-quality Korean partials don't improve the pipeline. Not worth the VRAM. |
| **K6 Realtime Text Stabilization** | Nice for display but doesn't affect translation accuracy. Existing `stability_count=2` already prevents jittery flushes. |
| **K11 Interruption Handling** | Not applicable. One-directional translation — the speaker doesn't interact with the output. |
| **K12 Adaptive Turn Detection Speed** | Korean grammar endings handle pace variation. The speaker's pauses are linguistic (between sentences), not turn-based. |
