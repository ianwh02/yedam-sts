#!/bin/bash
# Run TTS server locally for testing.
# Usage: bash scripts/run_tts_local.sh [EOS_MIN_STEPS]
#
# After server shows "weights_ready", run in another terminal:
#   curl -X POST http://localhost:7860/allocate_kv_cache \
#     -H "Content-Type: application/json" -d '{"budget_mb": 0}'
#
# Then run the test:
#   python scripts/test_eos_min_steps.py

set -euo pipefail

EOS_MIN_STEPS="${1:-0}"

cd "$(dirname "$0")/.."

export MODEL_DIR="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
export TOKENIZER_DIR="Qwen/Qwen3-TTS-Tokenizer-12Hz"
export GPU_MEMORY_UTILIZATION="0.35"
export PORT="7860"
export TARGET_SAMPLE_RATE="48000"
export SHARE_SPEECH_TOKENIZER="1"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor"
export MAX_MODEL_LEN="2048"

# Voice params
export TTS_TALKER_TEMPERATURE="0.7"
export TTS_PREDICTOR_TEMPERATURE="0.9"
export TTS_HIGHPASS_HZ="80"
export TTS_TARGET_RMS="0.08"
export STREAMING_CONTEXT_SIZE="16"
export DECODE_COLLECT_MS="40"

# EOS params — the variable under test
export TTS_EOS_MIN_STEPS="$EOS_MIN_STEPS"
export TTS_EOS_BOOST_MAX="3.0"
export TTS_EOS_BOOST_START_STEP="35"
export TTS_EOS_BOOST_MAX_STEP="300"
export TTS_EOS_LOOP_DETECT_WINDOW="12"
export TTS_EOS_LOOP_THRESHOLD="0.4"
export TTS_EOS_LOOP_BOOST="5.0"
export TTS_EOS_LOOP_FORCE_AFTER="15"
export TTS_REPETITION_PENALTY="1.1"
export TTS_REPETITION_PENALTY_WINDOW="100"
export TTS_SUPPRESS_MASK="1"

# Debug
export DEBUG_SAVE_AUDIO="1"

# No ref audio for local test (uses CustomVoice preset speakers)
unset REF_AUDIO_PATH_EN REF_TEXT_EN REF_AUDIO_PATH REF_TEXT

echo "=== Starting TTS server (EOS_MIN_STEPS=$EOS_MIN_STEPS) ==="
echo "After 'weights_ready', run in another terminal:"
echo "  curl -X POST http://localhost:7860/allocate_kv_cache -H 'Content-Type: application/json' -d '{\"budget_mb\": 0}'"
echo ""

python tts-server/server.py
