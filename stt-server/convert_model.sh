#!/bin/bash
# Convert Whisper model to TensorRT-LLM engine format.
# Must run ON the target GPU — engines are architecture-specific.
#
# Usage: ./convert_model.sh [model_name] [output_dir]
#   model_name:  Whisper model (default: large-v3-turbo)
#   output_dir:  Where to save engines (default: /app/engines)
#
# Expects TRT-LLM Whisper examples at /app/trt-whisper-examples/
# (copied from NGC container or cloned by Dockerfile.tensorrt)

set -euo pipefail

MODEL_NAME="${1:-large-v3-turbo}"
OUTPUT_DIR="${2:-/app/engines}"
TRT_EXAMPLES="/app/trt-whisper-examples"

# Engine build parameters — optimized for batch inference + greedy decoding
INFERENCE_PRECISION="float16"
QUANTIZATION="${QUANTIZATION:-none}"  # "int8" (weight-only, decoder only) or "none" (FP16)
MAX_BATCH_SIZE=8
MAX_BEAM_WIDTH=1
# Decoder max_seq_len = prompt_len (14) + max_output_tokens (96) = 110, round up
DECODER_MAX_SEQ_LEN=114
DECODER_MAX_INPUT_LEN=14

echo "============================================"
echo "TensorRT-LLM Whisper Engine Builder"
echo "============================================"
echo "Model:        ${MODEL_NAME}"
echo "Output:       ${OUTPUT_DIR}"
echo "Precision:    ${INFERENCE_PRECISION}"
echo "Quantization: ${QUANTIZATION}"
echo "Batch size:   ${MAX_BATCH_SIZE}"
echo "Beam width:   ${MAX_BEAM_WIDTH}"
echo "============================================"

# Skip if engines already exist
if [ -f "${OUTPUT_DIR}/encoder/rank0.engine" ] && [ -f "${OUTPUT_DIR}/decoder/rank0.engine" ]; then
    echo "Engines already exist at ${OUTPUT_DIR}, skipping conversion."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

cd "${TRT_EXAMPLES}"
ASSETS_DIR="${TRT_EXAMPLES}/assets"
mkdir -p "${ASSETS_DIR}"

# Download mel filters if needed
if [ ! -f "${ASSETS_DIR}/mel_filters.npz" ]; then
    cp /app/assets/mel_filters.npz "${ASSETS_DIR}/" 2>/dev/null || \
    wget -q -P "${ASSETS_DIR}" \
        "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
fi

# Download model checkpoint
case "$MODEL_NAME" in
    "tiny.en")  MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt" ;;
    "tiny")     MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" ;;
    "base.en")  MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt" ;;
    "base")     MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt" ;;
    "small.en") MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt" ;;
    "small")    MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt" ;;
    "medium.en") MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt" ;;
    "medium")   MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt" ;;
    "large-v3") MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt" ;;
    "large-v3-turbo") MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt" ;;
    *) echo "Unsupported model: ${MODEL_NAME}"; exit 1 ;;
esac

if [ -n "${MODEL_URL}" ]; then
    if [ ! -f "${ASSETS_DIR}/${MODEL_NAME}.pt" ]; then
        echo "Downloading ${MODEL_NAME} checkpoint..."
        wget -q --show-progress -P "${ASSETS_DIR}" "${MODEL_URL}"
    fi
fi

# Install conversion dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
    pip install --no-deps -r requirements.txt 2>/dev/null || true
fi

# Step 1: Convert checkpoint to TRT-LLM format
SANITIZED_NAME="${MODEL_NAME//./_}"
SANITIZED_NAME="${SANITIZED_NAME//-/_}"
CHECKPOINT_DIR="${TRT_EXAMPLES}/whisper_${SANITIZED_NAME}_weights_${INFERENCE_PRECISION}"

echo ""
echo "[1/3] Converting model weights..."
QUANT_FLAGS=""
if [ "${QUANTIZATION}" = "int8" ]; then
    QUANT_FLAGS="--use_weight_only --weight_only_precision int8"
    echo "  → INT8 weight-only quantization enabled"
fi
python3 convert_checkpoint.py \
    --output_dir "${CHECKPOINT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --model_dir "${ASSETS_DIR}" \
    ${QUANT_FLAGS}

# Step 2: Build encoder engine
echo ""
echo "[2/3] Building encoder engine..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
    --output_dir "${OUTPUT_DIR}/encoder" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --bert_attention_plugin "${INFERENCE_PRECISION}" \
    --max_input_len 3000 \
    --max_seq_len 3000

# Step 3: Build decoder engine
echo ""
echo "[3/3] Building decoder engine..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
    --output_dir "${OUTPUT_DIR}/decoder" \
    --max_beam_width "${MAX_BEAM_WIDTH}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len "${DECODER_MAX_SEQ_LEN}" \
    --max_input_len "${DECODER_MAX_INPUT_LEN}" \
    --max_encoder_input_len 3000

# Copy mel_filters.npz to output dir (needed at runtime)
cp "${ASSETS_DIR}/mel_filters.npz" "${OUTPUT_DIR}/"

echo ""
echo "============================================"
echo "Engine build complete!"
echo "Encoder: ${OUTPUT_DIR}/encoder/rank0.engine"
echo "Decoder: ${OUTPUT_DIR}/decoder/rank0.engine"
echo "============================================"
