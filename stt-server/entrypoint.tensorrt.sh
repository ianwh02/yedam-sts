#!/bin/bash
# Entrypoint for TensorRT WhisperLive server.
# 1. Build TRT engines on first startup (if not cached in volume)
# 2. Start the WhisperLive server with TensorRT backend

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-/app/engines}"
MODEL_NAME="${WHISPER_MODEL:-large-v3-turbo}"
BEAM_SIZE="${BEAM_SIZE:-1}"

echo "=== WhisperLive TensorRT Server ==="
echo "Engine dir: ${ENGINE_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Beam size: ${BEAM_SIZE}"

# Step 1: Build engines if not present
if [ ! -f "${ENGINE_DIR}/encoder/rank0.engine" ] || [ ! -f "${ENGINE_DIR}/decoder/rank0.engine" ]; then
    echo ""
    echo "TRT engines not found. Building (this takes ~10-15 minutes)..."
    MAX_BEAM_WIDTH="${BEAM_SIZE}" ./convert_model.sh "${MODEL_NAME}" "${ENGINE_DIR}"
else
    echo "TRT engines found, skipping conversion."
fi

# Step 2: Pre-create KV cache signal if fraction is known (enables restart without coordinator)
KV_FRACTION="${STT_KV_CACHE_FREE_GPU_FRACTION:-}"
if [ -n "${KV_FRACTION}" ] && [ "${STT_DEFER_KV_CACHE:-0}" = "1" ]; then
    echo "{\"fraction\": ${KV_FRACTION}}" > /tmp/allocate_kv_cache.json
    echo "KV cache signal pre-created (fraction=${KV_FRACTION})"
fi

# Step 3: Start server
echo ""
echo "Starting WhisperLive TensorRT server..."
exec python -u run_server.py \
    --port 9090 \
    --backend tensorrt \
    --trt_model_path "${ENGINE_DIR}" \
    --trt_multilingual \
    --omp_num_threads 4 \
    --max_clients 15 \
    --max_connection_time 7200 \
    --batch_inference \
    --batch_max_size 8 \
    --batch_window_ms 25 \
    --beam_size "${BEAM_SIZE}"
