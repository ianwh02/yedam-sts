# syntax=docker/dockerfile:1
# WhisperLive GPU STT Server for yedam-sts
# Forked from WhisperLive with faster-whisper backend + batch inference
#
# Build: docker compose build stt
# Run:   docker compose up stt

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    git curl \
    && apt install -y python-is-python3 \
    && pip install --upgrade pip setuptools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (cached unless requirements.txt changes) ---
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# --- Model download (cached, rarely changes) ---
# WHISPER_MODEL_REPO: HuggingFace repo ID (varies by model)
# Systran repos: large-v3, medium, small, base, tiny
# Turbo: deepdml/faster-whisper-large-v3-turbo-ct2
ARG WHISPER_MODEL_REPO=deepdml/faster-whisper-large-v3-turbo-ct2
ARG WHISPER_MODEL_NAME=faster-whisper-large-v3-turbo
ARG HF_TOKEN=
ENV HF_TOKEN=${HF_TOKEN}
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('${WHISPER_MODEL_REPO}', local_dir='/app/models/${WHISPER_MODEL_NAME}')"

# Store model path so CMD can reference it at runtime
ENV WHISPER_MODEL_DIR=/app/models/${WHISPER_MODEL_NAME}

# --- Phrase classifier model (KoELECTRA ONNX, ~55MB) ---
COPY models/phrase-classifier/ ./models/phrase-classifier/

# --- Source layer (rebuilds when code changes) ---
COPY whisper_live/ ./whisper_live/
COPY run_server.py .

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --retries=10 --start-period=180s \
    CMD curl -f http://localhost:9090/health || exit 1

# Use shell form so $WHISPER_MODEL_DIR expands at runtime
CMD python -u run_server.py \
    --port 9090 \
    --backend faster_whisper \
    --faster_whisper_custom_model_path "$WHISPER_MODEL_DIR" \
    --omp_num_threads 4 \
    --max_clients 10 \
    --max_connection_time 7200 \
    --batch_inference \
    --batch_max_size 8 \
    --batch_window_ms 25 \
    --beam_size "${BEAM_SIZE:-3}"
