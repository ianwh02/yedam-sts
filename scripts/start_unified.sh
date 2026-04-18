#!/bin/bash
# Unified startup script for yedam-sts (all services in one container).
#
# Phased startup:
#   1. Compute VRAM budget
#   2. Start LLM (vLLM) → wait healthy
#   3. Start STT + TTS in parallel → wait for weights loaded
#   4. Allocate TTS KV cache
#   5. Start orchestrator
#   6. Monitor all processes
set -euo pipefail

log() { echo "$(date '+%H:%M:%S') [startup] $*"; }

# ── Cleanup on exit ──
PIDS=()
cleanup() {
    log "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    log "All services stopped."
}
trap cleanup EXIT INT TERM

# ── Pre-flight: Ensure models are present (slim image support) ──
if [ -x /app/scripts/download_models.sh ]; then
    /app/scripts/download_models.sh "/models"
fi

# ── NVIDIA MPS (disabled) ──
# MPS would partition GPU compute across services, but RunPod containers
# don't expose the required driver support. GPU compute is unpartitioned;
# client-side audio buffering absorbs TTS RTF spikes instead.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 2>/dev/null || true

# ── Phase 0: Compute VRAM budget ──
log "=== Phase 0: Computing VRAM budget ==="
GPU_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_TOTAL_MB" ]; then
    log "ERROR: nvidia-smi not available or no GPU detected"
    exit 1
fi
log "GPU total: ${GPU_TOTAL_MB} MB"

# Detect GPU compute capability and set arch-specific TRT engine path
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -n "$GPU_ARCH" ]; then
    ARCH_ENGINE_DIR="${TRT_ENGINE_DIR:-/models/trt_engines}/sm_${GPU_ARCH}"
    export TRT_ENGINE_DIR="$ARCH_ENGINE_DIR"
    mkdir -p "$ARCH_ENGINE_DIR"
    log "TRT engine path: $ARCH_ENGINE_DIR (sm_${GPU_ARCH})"
fi

STT_BUDGET_FLAG=""
if [ "${STT_BACKEND:-tensorrt}" = "tensorrt" ]; then
    STT_BUDGET_FLAG="--stt-backend trt"
fi
python3 /app/scripts/vram_budget.py \
    --config /app/vram_budget.yml \
    --total-mb "$GPU_TOTAL_MB" \
    --output /tmp/vram.env \
    $STT_BUDGET_FLAG

# Source computed budget
set -a
source /tmp/vram.env
set +a
log "LLM_GPU_MEMORY_UTILIZATION=${LLM_GPU_MEMORY_UTILIZATION}"
log "TTS_VRAM_BUDGET_MB=${TTS_VRAM_BUDGET_MB:-auto}"

# ── Phase 1a: Start LLM ──
log "=== Phase 1a: Starting LLM (vLLM) ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL}" \
    --quantization "${LLM_QUANTIZATION}" \
    --kv-cache-dtype "${LLM_KV_CACHE_DTYPE}" \
    --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${LLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${LLM_MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee /tmp/llm.log &
LLM_PID=$!
PIDS+=($LLM_PID)

log "Waiting for LLM to be healthy (PID=${LLM_PID})..."
TIMEOUT=300
ELAPSED=0
while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    if ! kill -0 "$LLM_PID" 2>/dev/null; then
        log "ERROR: LLM process died. Last logs:"
        tail -50 /tmp/llm.log || true
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: LLM health timeout (${TIMEOUT}s). Last logs:"
        tail -50 /tmp/llm.log || true
        exit 1
    fi
done
log "LLM healthy (${ELAPSED}s)."

# ── Phase 1b: Start STT + TTS in parallel ──
log "=== Phase 1b: Starting STT + TTS ==="

cd /app/stt-server
if [ "${STT_BACKEND:-tensorrt}" = "tensorrt" ]; then
    # Build/verify TRT engines (convert_model.sh handles skip-if-unchanged)
    ENGINE_DIR="${TRT_ENGINE_DIR:-/models/trt_engines}"
    WHISPER_HF_MODEL="${WHISPER_MODEL:-seastar105/whisper-medium-komixv2}"
    # If HF model was downloaded to /models, use local path
    if [ -d "/models/whisper-medium-komixv2-hf" ]; then
        WHISPER_HF_MODEL="/models/whisper-medium-komixv2-hf"
    fi
    log "Checking TRT engines (model=${WHISPER_HF_MODEL}, beam=${BEAM_SIZE:-3})..."
    MAX_BEAM_WIDTH="${BEAM_SIZE:-3}" \
    PATH="/opt/stt-trt-venv/bin:$PATH" \
        ./convert_model.sh "${WHISPER_HF_MODEL}" "${ENGINE_DIR}"
    log "TRT engines ready at ${ENGINE_DIR}"

    /opt/stt-trt-venv/bin/python3 run_server.py \
        --port 9090 \
        --backend tensorrt \
        --trt_model_path "${ENGINE_DIR}" \
        --trt_multilingual \
        --trt_py_session \
        --omp_num_threads "${OMP_NUM_THREADS:-4}" \
        --max_clients 10 \
        --max_connection_time 7200 \
        --batch_inference \
        --batch_max_size 8 \
        --batch_window_ms 25 \
        --beam_size "${BEAM_SIZE:-3}" \
        2>&1 | tee /tmp/stt.log &
else
    python3 run_server.py \
        --port 9090 \
        --backend faster_whisper \
        --faster_whisper_custom_model_path "${WHISPER_MODEL_DIR}" \
        --omp_num_threads "${OMP_NUM_THREADS:-4}" \
        --max_clients 10 \
        --max_connection_time 7200 \
        --batch_inference \
        --batch_max_size 8 \
        --batch_window_ms 25 \
        --beam_size "${BEAM_SIZE:-3}" \
        2>&1 | tee /tmp/stt.log &
fi
STT_PID=$!
PIDS+=($STT_PID)

cd /app/tts-server
python3 server.py \
    2>&1 | tee /tmp/tts.log &
TTS_PID=$!
PIDS+=($TTS_PID)

# Wait for STT healthy
log "Waiting for STT (PID=${STT_PID})..."
ELAPSED=0
while ! curl -sf http://localhost:9090/health > /dev/null 2>&1; do
    if ! kill -0 "$STT_PID" 2>/dev/null; then
        log "ERROR: STT process died. Last logs:"
        tail -20 /tmp/stt.log || true
        exit 1
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: STT health timeout. Last logs:"
        tail -20 /tmp/stt.log || true
        exit 1
    fi
done
log "STT healthy (${ELAPSED}s)."

# Wait for TTS weights_ready
log "Waiting for TTS weights (PID=${TTS_PID})..."
ELAPSED=0
while true; do
    if ! kill -0 "$TTS_PID" 2>/dev/null; then
        log "ERROR: TTS process died. Last logs:"
        tail -20 /tmp/tts.log || true
        exit 1
    fi
    STATUS=$(curl -sf http://localhost:7860/startup 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null \
        || echo "")
    if [ "$STATUS" = "weights_ready" ] || [ "$STATUS" = "ready" ]; then
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: TTS weights timeout. Last logs:"
        tail -20 /tmp/tts.log || true
        exit 1
    fi
done
log "TTS weights loaded (${ELAPSED}s)."

# ── Phase 2: Allocate TTS KV cache ──
log "=== Phase 2: Allocating TTS KV cache ==="
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "Free VRAM: ${FREE_MB} MB"

# budget_mb=0 triggers auto-detect in workers (torch.cuda.mem_get_info).
# Override with TTS_KV_BUDGET_MB env var if needed.
TTS_KV_BUDGET_MB="${TTS_KV_BUDGET_MB:-0}"
ALLOC_RESULT=$(curl -s --max-time 300 -X POST "http://localhost:7860/allocate_kv_cache?budget_mb=${TTS_KV_BUDGET_MB}" 2>&1) || true
log "TTS KV allocation: ${ALLOC_RESULT}"

# Wait for TTS fully ready (warmup completes)
log "Waiting for TTS ready (warmup)..."
ELAPSED=0
while true; do
    STATUS=$(curl -sf http://localhost:7860/startup 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null \
        || echo "")
    if [ "$STATUS" = "ready" ]; then break; fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [ "$ELAPSED" -ge 300 ]; then
        log "ERROR: TTS ready timeout after KV allocation. Last logs:"
        tail -20 /tmp/tts.log || true
        exit 1
    fi
done
log "TTS ready."

# ── Phase 3: Start orchestrator ──
log "=== Phase 3: Starting Orchestrator ==="
cd /app/orchestrator
python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "${ORCHESTRATOR_PORT:-8080}" \
    2>&1 | tee /tmp/orchestrator.log &
ORCH_PID=$!
PIDS+=($ORCH_PID)

# Wait for orchestrator healthy
ELAPSED=0
while ! curl -sf "http://localhost:${ORCHESTRATOR_PORT:-8080}/health" > /dev/null 2>&1; do
    if ! kill -0 "$ORCH_PID" 2>/dev/null; then
        log "ERROR: Orchestrator died. Last logs:"
        tail -20 /tmp/orchestrator.log || true
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge 30 ]; then
        # Orchestrator is lightweight — if not healthy in 30s, just warn and continue
        log "WARNING: Orchestrator health check not responding, continuing..."
        break
    fi
done

log ""

# ── Optional: Start idle monitor (RunPod cost saving) ──
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    python3 /app/deploy/runpod/idle_monitor.py \
        > /tmp/idle_monitor.log 2>&1 &
    IDLE_PID=$!
    log "Idle monitor started (PID=${IDLE_PID}, timeout=${IDLE_TIMEOUT_MIN:-30} min)"
fi

log "=========================================="
log " All services started successfully!"
log "=========================================="
log "  Orchestrator : http://0.0.0.0:${ORCHESTRATOR_PORT:-8080}"
log "  LLM (vLLM)   : http://0.0.0.0:8000"
log "  TTS           : http://0.0.0.0:7860"
log "  STT           : ws://0.0.0.0:9090"
log "=========================================="

# ── Monitor: restart if any service dies ──
while true; do
    for NAME_PID in "LLM:${LLM_PID}" "STT:${STT_PID}" "TTS:${TTS_PID}" "Orchestrator:${ORCH_PID}"; do
        NAME="${NAME_PID%%:*}"
        PID="${NAME_PID##*:}"
        if ! kill -0 "$PID" 2>/dev/null; then
            log "FATAL: ${NAME} (PID=${PID}) has exited unexpectedly."
            LOGFILE="/tmp/$(echo "$NAME" | tr '[:upper:]' '[:lower:]').log"
            log "Last 30 lines of ${LOGFILE}:"
            tail -30 "$LOGFILE" 2>/dev/null || true
            log "Exiting — container will restart if restart policy is set."
            exit 1
        fi
    done
    sleep 10
done
