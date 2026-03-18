# syntax=docker/dockerfile:1
# Qwen3-TTS Python server for yedam-sts pipeline
# Uses nano-qwen3tts-vllm (vLLM-style continuous batching + CUDA graphs)
# Model: Qwen3-TTS-12Hz-0.6B-CustomVoice (~2GB VRAM)
#
# Build: docker compose build tts
# Run:   docker compose up tts

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    curl ca-certificates libsndfile1 sox \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip/setuptools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.8 + torchaudio from CUDA 12.8 index
# cu128 required for sm_120 (Blackwell / RTX 50-series) GPU support
# torchaudio is required by qwen-tts (transitive dep of nano-qwen3tts-vllm)
RUN pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# ── Dependencies (rarely change → cached early) ──
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir --no-deps "qwen-tts==0.0.5" && \
    pip3 install --no-cache-dir "transformers==4.57.3" "accelerate==1.12.0" librosa soundfile einops xxhash>=3.6.0 sox protobuf onnxruntime triton

# Flash Attention (required by nano-qwen3tts-vllm attention layer)
# Prebuilt wheel: torch2.8 + cu12 + cp310 + cxx11abiFALSE (matches pip-installed PyTorch)
RUN pip3 install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Verify torch version hasn't been overwritten by transitive deps
RUN python3 -c "import torch; v=torch.__version__; print(f'torch={v}'); assert v.startswith('2.8'), f'torch was overwritten to {v}!'"

# ── Download TTS models at build time (cached, rarely change) ──
ARG HF_TOKEN=
ENV HF_TOKEN=${HF_TOKEN}

# CustomVoice model (preset speakers: ryan, sohee, etc.)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', \
local_dir='/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice')"

# Base model (voice clone mode with reference audio)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', \
local_dir='/app/models/Qwen3-TTS-12Hz-0.6B-Base')"

# Download speech tokenizer model at build time
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', \
local_dir='/app/models/Qwen3-TTS-Tokenizer-12Hz')"

# ── Source code (changes frequently → last layers for fast rebuilds) ──
# nano-qwen3tts-vllm from local source (editable install so Python reads
# from /app/nano-qwen3tts-vllm/ directly — no stale site-packages)
COPY nano-qwen3tts-vllm/ /app/nano-qwen3tts-vllm/
RUN pip3 install --no-cache-dir --no-deps -e /app/nano-qwen3tts-vllm/

# Verify nano-qwen3tts-vllm imports correctly
RUN python3 -c "from nano_qwen3tts_vllm.interface import Qwen3TTSInterface; print('nano-qwen3tts-vllm OK')"

# Copy server
COPY server.py .

EXPOSE 7860

ENV MODEL_DIR=/app/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
ENV TOKENIZER_DIR=/app/models/Qwen3-TTS-Tokenizer-12Hz
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python3", "server.py"]
