#!/bin/bash
# CUDA MPS (Multi-Process Service) setup script.
#
# Enables concurrent GPU kernel execution from multiple processes
# (STT, LLM, TTS) on the same GPU. Without MPS, the GPU
# context-switches between processes, adding ~1-2ms per switch.
#
# Run this on the host before starting Docker containers.
# Note: MPS is NOT supported on Windows — Linux/WSL2 only.
#
# Usage: sudo ./cuda-mps.sh [start|stop|status]

set -e

MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-log"

start_mps() {
    echo "Starting CUDA MPS daemon..."

    # Stop any existing MPS daemon
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1

    # Set GPU for MPS
    export CUDA_VISIBLE_DEVICES=0

    # Set thread percentage (optional, default 100%)
    # Can be tuned per-process via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
    export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR
    export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_DIR

    mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"

    # Start the MPS control daemon
    nvidia-cuda-mps-control -d

    echo "CUDA MPS daemon started"
    echo "  Pipe dir: $MPS_PIPE_DIR"
    echo "  Log dir:  $MPS_LOG_DIR"
    echo ""
    echo "Per-container thread allocation (set in docker-compose.yml):"
    echo "  STT: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30"
    echo "  LLM: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50"
    echo "  TTS: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20"
}

stop_mps() {
    echo "Stopping CUDA MPS daemon..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "CUDA MPS daemon stopped"
}

status_mps() {
    echo "CUDA MPS status:"
    nvidia-smi -q -d COMPUTE 2>/dev/null | grep -A5 "MPS" || echo "  MPS not active"
    echo ""
    echo "Active MPS clients:"
    echo "get_server_list" | nvidia-cuda-mps-control 2>/dev/null || echo "  No MPS server running"
}

case "${1:-start}" in
    start)  start_mps ;;
    stop)   stop_mps ;;
    status) status_mps ;;
    *)      echo "Usage: $0 [start|stop|status]"; exit 1 ;;
esac
