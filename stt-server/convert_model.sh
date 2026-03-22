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
MAX_BEAM_WIDTH=${MAX_BEAM_WIDTH:-1}
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
    *)
        # Try as HuggingFace repo ID (e.g. ghost613/whisper-large-v3-turbo-korean)
        if [[ "${MODEL_NAME}" == *"/"* ]]; then
            HF_MODEL_DIR="${ASSETS_DIR}/hf_${MODEL_NAME//\//_}"
            if [ ! -d "${HF_MODEL_DIR}" ]; then
                echo "Downloading HuggingFace model ${MODEL_NAME}..."
                python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_NAME}', local_dir='${HF_MODEL_DIR}')
print('Download complete.')
"
            fi
            MODEL_URL=""
        else
            echo "Unsupported model: ${MODEL_NAME}"; exit 1
        fi
        ;;
esac

if [ -n "${MODEL_URL:-}" ]; then
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
# Use detected arch for HF models, otherwise sanitize MODEL_NAME
EFFECTIVE_NAME="${WHISPER_ARCH:-${MODEL_NAME}}"
SANITIZED_NAME="${EFFECTIVE_NAME//./_}"
SANITIZED_NAME="${SANITIZED_NAME//-/_}"
SANITIZED_NAME="${SANITIZED_NAME//\//_}"
CHECKPOINT_DIR="${TRT_EXAMPLES}/whisper_${SANITIZED_NAME}_weights_${INFERENCE_PRECISION}"

echo ""
echo "[1/3] Converting model weights..."
QUANT_FLAGS=""
if [ "${QUANTIZATION}" = "int8" ]; then
    QUANT_FLAGS="--use_weight_only --weight_only_precision int8"
    echo "  → INT8 weight-only quantization enabled"
fi

# Use HF model dir if available, otherwise standard .pt checkpoint
if [ -n "${HF_MODEL_DIR:-}" ] && [ -d "${HF_MODEL_DIR}" ]; then
    echo "  → Converting HuggingFace model to .pt format: ${HF_MODEL_DIR}"
    # TRT-LLM convert_checkpoint.py expects OpenAI .pt format.
    # Convert HF safetensors → .pt using whisper's state_dict layout.
    # Detect whisper architecture from HF config
    WHISPER_ARCH=$(python3 -c "
import json, os
cfg = json.load(open(os.path.join('${HF_MODEL_DIR}', 'config.json')))
el, dl = cfg['encoder_layers'], cfg['decoder_layers']
d = cfg['d_model']
# Map architecture dimensions to OpenAI model names
if el == 4 and dl == 4: print('tiny')
elif el == 6 and dl == 6 and d == 512: print('base')
elif el == 12 and dl == 12 and d == 768: print('small')
elif el == 24 and dl == 24 and d == 1024: print('medium')
elif el == 32 and dl == 32: print('large-v3')
elif el == 32 and dl == 4: print('large-v3-turbo')
else: print('large-v3')  # fallback
")
    echo "  → Detected architecture: ${WHISPER_ARCH}"
    PT_PATH="${ASSETS_DIR}/${WHISPER_ARCH}.pt"
    if [ ! -f "${PT_PATH}" ]; then
        python3 -c "
import torch
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained('${HF_MODEL_DIR}')

dims = {
    'n_mels': model.config.num_mel_bins,
    'n_vocab': model.config.vocab_size,
    'n_audio_ctx': model.config.max_source_positions,
    'n_audio_state': model.config.d_model,
    'n_audio_head': model.config.encoder_attention_heads,
    'n_audio_layer': model.config.encoder_layers,
    'n_text_ctx': model.config.max_target_positions,
    'n_text_state': model.config.d_model,
    'n_text_head': model.config.decoder_attention_heads,
    'n_text_layer': model.config.decoder_layers,
}

state_dict = {}
for k, v in model.model.state_dict().items():
    new_k = k.replace('encoder.layers.', 'encoder.blocks.')
    new_k = new_k.replace('decoder.layers.', 'decoder.blocks.')
    new_k = new_k.replace('self_attn.k_proj', 'attn.key')
    new_k = new_k.replace('self_attn.v_proj', 'attn.value')
    new_k = new_k.replace('self_attn.q_proj', 'attn.query')
    new_k = new_k.replace('self_attn.out_proj', 'attn.out')
    new_k = new_k.replace('self_attn_layer_norm', 'attn_ln')
    new_k = new_k.replace('encoder_attn.k_proj', 'cross_attn.key')
    new_k = new_k.replace('encoder_attn.v_proj', 'cross_attn.value')
    new_k = new_k.replace('encoder_attn.q_proj', 'cross_attn.query')
    new_k = new_k.replace('encoder_attn.out_proj', 'cross_attn.out')
    new_k = new_k.replace('encoder_attn_layer_norm', 'cross_attn_ln')
    new_k = new_k.replace('final_layer_norm', 'mlp_ln')
    new_k = new_k.replace('fc1', 'mlp.0')
    new_k = new_k.replace('fc2', 'mlp.2')
    new_k = new_k.replace('encoder.layer_norm', 'encoder.ln_post')
    new_k = new_k.replace('encoder.embed_positions.weight', 'encoder.positional_embedding')
    new_k = new_k.replace('decoder.layer_norm', 'decoder.ln')
    new_k = new_k.replace('decoder.embed_positions.weight', 'decoder.positional_embedding')
    new_k = new_k.replace('decoder.embed_tokens', 'decoder.token_embedding')
    state_dict[new_k] = v

state_dict['decoder.proj_out.weight'] = model.proj_out.weight if hasattr(model, 'proj_out') else model.lm_head.weight

torch.save({'dims': dims, 'model_state_dict': state_dict}, '${PT_PATH}')
print(f'Saved .pt to ${PT_PATH}')
print(f'Architecture: ${WHISPER_ARCH}, dims: {dims}')
"
    fi
    python3 convert_checkpoint.py \
        --output_dir "${CHECKPOINT_DIR}" \
        --model_name "${WHISPER_ARCH}" \
        --model_dir "${ASSETS_DIR}" \
        ${QUANT_FLAGS}
else
    python3 convert_checkpoint.py \
        --output_dir "${CHECKPOINT_DIR}" \
        --model_name "${MODEL_NAME}" \
        --model_dir "${ASSETS_DIR}" \
        ${QUANT_FLAGS}
fi

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
