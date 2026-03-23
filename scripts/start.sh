#!/bin/bash
# Start yedam-sts with two-phase VRAM allocation.
#
# Phase 1: All services start in parallel, load model weights only.
# Phase 2: Coordinator measures free VRAM, distributes KV cache budgets.
#
# Usage:
#   ./scripts/start.sh              # faster-whisper STT
#   ./scripts/start.sh --trt        # TensorRT STT
#   ./scripts/start.sh --dev        # expose all ports + debug logging
#   ./scripts/start.sh --trt --dry  # show budget only, don't start
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

STT_BACKEND="fw"
DRY_RUN=false
DEV_MODE=false
COMPOSE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --trt) STT_BACKEND="trt" ;;
        --dry) DRY_RUN=true ;;
        --dev) DEV_MODE=true ;;
        *) COMPOSE_ARGS+=("$arg") ;;
    esac
done

# Phase 1 env: only LLM fraction is needed before startup (vLLM reads it at boot).
# TTS and STT get their KV cache budgets at runtime from the coordinator.
echo "=== Computing LLM budget ==="
python3 scripts/vram_budget.py \
    --config vram_budget.yml \
    --stt-backend "$STT_BACKEND" \
    --output .env.vram
echo ""

if $DRY_RUN; then
    echo "Dry run — not starting services."
    exit 0
fi

# Build compose command
COMPOSE_CMD=(docker compose --env-file .env --env-file .env.vram)

if [ "$STT_BACKEND" = "trt" ]; then
    COMPOSE_CMD+=(-f docker-compose.yml -f docker-compose.tensorrt.yml)
fi

if $DEV_MODE; then
    COMPOSE_CMD+=(-f docker-compose.dev.yml)
fi

# Stop GPU services whose config/image changed to ensure clean CUDA context release.
# docker compose up -d recreates changed containers, but the old CUDA context may
# not fully release before the new process allocates. Explicit stop avoids this.
# Detect which GPU services need recreating
CHANGED_SERVICES=()
DRY_OUTPUT=$("${COMPOSE_CMD[@]}" up -d --no-deps --dry-run tts stt llm 2>&1 || true)
echo "$DRY_OUTPUT" | grep -q "yedam-tts.*Recreate" && CHANGED_SERVICES+=(tts)
echo "$DRY_OUTPUT" | grep -q "yedam-llm.*Recreate" && CHANGED_SERVICES+=(llm)
echo "$DRY_OUTPUT" | grep -q "yedam-stt.*Recreate" && CHANGED_SERVICES+=(stt)

# LLM must start first (vLLM measures free VRAM at boot). If LLM changed,
# all GPU services must restart so VRAM is allocated cleanly.
LLM_CHANGED=false
for svc in "${CHANGED_SERVICES[@]}"; do
    [ "$svc" = "llm" ] && LLM_CHANGED=true
done
if $LLM_CHANGED; then
    CHANGED_SERVICES=(llm tts stt)
fi

if [ ${#CHANGED_SERVICES[@]} -gt 0 ]; then
    echo "=== Stopping changed services for clean VRAM state: ${CHANGED_SERVICES[*]} ==="
    "${COMPOSE_CMD[@]}" stop "${CHANGED_SERVICES[@]}" 2>/dev/null || true
    sleep 2  # allow CUDA contexts to fully release
fi

# Phase 1a: start LLM first (vLLM checks free VRAM at startup — must run before others)
echo "=== Phase 1a: Starting LLM ==="
"${COMPOSE_CMD[@]}" up -d --no-deps llm
echo "Waiting for LLM to be healthy..."
"${COMPOSE_CMD[@]}" exec -T llm sh -c 'until curl -sf http://localhost:8000/health; do sleep 2; done' 2>/dev/null
echo ""
echo "LLM healthy."

# Phase 1b: start TTS and STT in parallel (they defer KV cache allocation)
echo "=== Phase 1b: Starting TTS + STT (parallel weight loading) ==="
"${COMPOSE_CMD[@]}" up -d --no-deps tts stt

# Phase 2: coordinate KV cache allocation after weights are loaded
echo ""
echo "=== Phase 2: VRAM Coordinator ==="
python3 scripts/vram_coordinator.py \
    --config vram_budget.yml \
    --stt-backend "$STT_BACKEND"

# Phase 3: start orchestrator now that all GPU services are ready
echo ""
echo "=== Starting orchestrator ==="
"${COMPOSE_CMD[@]}" up -d --no-deps orchestrator

echo ""
echo "=== All services ready — following logs ==="
"${COMPOSE_CMD[@]}" logs -f
