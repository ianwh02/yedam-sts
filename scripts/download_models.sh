#!/bin/bash
# Download all models to /models on first boot.
# Skips if /models/.downloaded sentinel exists (volume already populated).
#
# Requires: HF_TOKEN env var for gated models.
# Called by start_unified.sh before starting services.
set -euo pipefail

MODELS_DIR="${1:-/models}"
SENTINEL="${MODELS_DIR}/.downloaded"

log() { echo "$(date '+%H:%M:%S') [download] $*"; }

if [ -f "$SENTINEL" ]; then
    log "Models already present (${SENTINEL} exists), skipping download."
    exit 0
fi

if [ -z "${HF_TOKEN:-}" ]; then
    log "WARNING: HF_TOKEN not set — downloads may fail for gated models"
fi

log "=== Downloading models to ${MODELS_DIR} ==="
mkdir -p "$MODELS_DIR"

download() {
    local repo="$1"
    local dest="${MODELS_DIR}/$2"
    if [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "  SKIP ${repo} → ${dest} (already exists)"
        return 0
    fi
    log "  Downloading ${repo} → ${dest}..."
    python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${repo}', local_dir='${dest}')"
    log "  OK ${repo}"
}

# STT: Korean Whisper small (seastar105 fine-tuned on Korean datasets)
STT_BACKEND_MODE="${STT_BACKEND:-tensorrt}"
WHISPER_HF_REPO="${WHISPER_MODEL:-seastar105/whisper-small-komixv2}"

if [ "$STT_BACKEND_MODE" = "tensorrt" ]; then
    # TRT: download HF model only (engine conversion happens at startup on the GPU)
    STT_HF_DIR="${MODELS_DIR}/whisper-small-komixv2-hf"
    if [ -d "$STT_HF_DIR" ] && [ "$(ls -A "$STT_HF_DIR" 2>/dev/null)" ]; then
        log "  SKIP STT HF model → ${STT_HF_DIR} (already exists)"
    else
        download "$WHISPER_HF_REPO" "whisper-small-komixv2-hf"
    fi
else
    # CTranslate2: download HF model and convert for faster-whisper
    STT_CT2_DIR="${MODELS_DIR}/whisper-small-komixv2-ct2"
    if [ -d "$STT_CT2_DIR" ] && [ "$(ls -A "$STT_CT2_DIR" 2>/dev/null)" ]; then
        log "  SKIP STT model → ${STT_CT2_DIR} (already exists)"
    else
        download "$WHISPER_HF_REPO" "whisper-small-komixv2-hf"
        if [ ! -f "${MODELS_DIR}/whisper-small-komixv2-hf/tokenizer.json" ]; then
            log "  Generating tokenizer.json from slow tokenizer files..."
            python3 -c "\
from transformers import AutoTokenizer; \
t = AutoTokenizer.from_pretrained('${MODELS_DIR}/whisper-small-komixv2-hf'); \
t.save_pretrained('${MODELS_DIR}/whisper-small-komixv2-hf')"
        fi
        log "  Converting to CTranslate2 format (int8_float16)..."
        ct2-transformers-converter \
            --model "${MODELS_DIR}/whisper-small-komixv2-hf" \
            --output_dir "$STT_CT2_DIR" \
            --quantization int8_float16 \
            --copy_files tokenizer.json preprocessor_config.json
        log "  OK CTranslate2 conversion"
        rm -rf "${MODELS_DIR}/whisper-small-komixv2-hf"
    fi
fi

# TTS: Base model (voice cloning with reference audio, ~3.5 GB)
download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" "Qwen3-TTS-12Hz-1.7B-Base"

# TTS: CustomVoice model (preset speakers + instructions, ~3.5 GB)
download "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" "Qwen3-TTS-12Hz-1.7B-CustomVoice"

# TTS: VoiceDesign model (voice from text descriptions, ~3.5 GB)
download "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" "Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# TTS: 48kHz codec decoder
download "takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz" "Qwen3-TTS-Tokenizer-12Hz-48kHz"

# LLM: Qwen3-8B-AWQ (4-bit quantized, ~4.5 GB)
download "Qwen/Qwen3-8B-AWQ" "Qwen3-8B-AWQ"

# Write sentinel
date -u > "$SENTINEL"
log "=== All models downloaded. Sentinel written to ${SENTINEL} ==="
