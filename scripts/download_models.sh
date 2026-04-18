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

# Check if STT model changed since last download (auto-invalidate stale STT cache)
WHISPER_HF_REPO="${WHISPER_MODEL:-seastar105/whisper-medium-komixv2}"
if [ -f "$SENTINEL" ]; then
    PREV_STT=$(grep '^WHISPER_MODEL=' "$SENTINEL" 2>/dev/null | cut -d= -f2- || echo "")
    if [ "$PREV_STT" = "$WHISPER_HF_REPO" ]; then
        log "Models already present and STT model unchanged (${WHISPER_HF_REPO}), skipping."
        exit 0
    elif [ -n "$PREV_STT" ]; then
        log "STT model changed: ${PREV_STT} → ${WHISPER_HF_REPO}"
        log "Removing stale STT artifacts..."
        rm -rf "${MODELS_DIR}"/whisper-*-hf "${MODELS_DIR}"/whisper-*-ct2
        rm -f "$SENTINEL"
    else
        # Old sentinel format (no model info) — re-download STT only
        log "Sentinel exists but has no model info, checking STT..."
        rm -f "$SENTINEL"
    fi
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

# STT: Korean Whisper medium (seastar105 fine-tuned on Korean datasets)
STT_BACKEND_MODE="${STT_BACKEND:-tensorrt}"

if [ "$STT_BACKEND_MODE" = "tensorrt" ]; then
    # TRT: download HF model only (engine conversion happens at startup on the GPU)
    STT_HF_DIR="${MODELS_DIR}/whisper-medium-komixv2-hf"
    if [ -d "$STT_HF_DIR" ] && [ "$(ls -A "$STT_HF_DIR" 2>/dev/null)" ]; then
        log "  SKIP STT HF model → ${STT_HF_DIR} (already exists)"
    else
        download "$WHISPER_HF_REPO" "whisper-medium-komixv2-hf"
    fi
else
    # CTranslate2: download HF model and convert for faster-whisper
    STT_CT2_DIR="${MODELS_DIR}/whisper-medium-komixv2-ct2"
    if [ -d "$STT_CT2_DIR" ] && [ "$(ls -A "$STT_CT2_DIR" 2>/dev/null)" ]; then
        log "  SKIP STT model → ${STT_CT2_DIR} (already exists)"
    else
        download "$WHISPER_HF_REPO" "whisper-medium-komixv2-hf"
        if [ ! -f "${MODELS_DIR}/whisper-medium-komixv2-hf/tokenizer.json" ]; then
            log "  Generating tokenizer.json from slow tokenizer files..."
            python3 -c "\
from transformers import AutoTokenizer; \
t = AutoTokenizer.from_pretrained('${MODELS_DIR}/whisper-medium-komixv2-hf'); \
t.save_pretrained('${MODELS_DIR}/whisper-medium-komixv2-hf')"
        fi
        log "  Converting to CTranslate2 format (int8_float16)..."
        ct2-transformers-converter \
            --model "${MODELS_DIR}/whisper-medium-komixv2-hf" \
            --output_dir "$STT_CT2_DIR" \
            --quantization int8_float16 \
            --copy_files tokenizer.json preprocessor_config.json
        log "  OK CTranslate2 conversion"
        rm -rf "${MODELS_DIR}/whisper-medium-komixv2-hf"
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

# Write sentinel with model versions for change detection
cat > "$SENTINEL" <<EOF
WHISPER_MODEL=${WHISPER_HF_REPO}
DOWNLOADED=$(date -u)
EOF
log "=== All models downloaded. Sentinel written to ${SENTINEL} ==="
