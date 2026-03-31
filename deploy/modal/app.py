"""
Modal deployment for yedam-sts.

Runs all 4 services (STT, LLM, TTS, Orchestrator) in a single GPU container,
matching the RunPod architecture. Models are stored on a Modal Volume.

Usage:
  # First time: download models to volume
  python -m modal run deploy/modal/app.py::download_models

  # Deploy the app (creates a persistent web endpoint)
  python -m modal deploy deploy/modal/app.py

  # Test locally (ephemeral)
  python -m modal serve deploy/modal/app.py
"""
import os
import subprocess
import sys
import threading

import modal

APP_NAME = "yedam-sts"

# --- Volume for model weights (persists across container restarts) ---
models_volume = modal.Volume.from_name("yedam-models", create_if_missing=True)
VOLUME_MOUNT = "/models"

# --- Image: use pre-built GHCR image ---
# The image is built via `docker build -f Dockerfile.slim` and pushed to GHCR.
# Modal pulls it directly — no Dockerfile build on Modal's infra.
# This avoids Modal's Python detection issues with the vLLM base image.
image = (
    modal.Image.from_registry(
        "ghcr.io/ianwh02/yedam-sts:latest",
        add_python="3.12",
        force_build=True,  # Force re-pull from GHCR (remove after stabilizing)
    )
    .pip_install("huggingface_hub")
    .env({
        "WHISPER_MODEL_DIR": f"{VOLUME_MOUNT}/whisper-small-komixv2-ct2",
        "MODEL_DIR": f"{VOLUME_MOUNT}/Qwen3-TTS-12Hz-1.7B-Base",
        "TOKENIZER_DIR": f"{VOLUME_MOUNT}/Qwen3-TTS-Tokenizer-12Hz-48kHz",
        "LLM_MODEL": f"{VOLUME_MOUNT}/Qwen3-8B-AWQ",
        "TRT_ENGINE_DIR": f"{VOLUME_MOUNT}/trt_engines",
    })
)

app = modal.App(APP_NAME, image=image)


# --- Model downloader (run once to populate the volume) ---
@app.function(
    volumes={VOLUME_MOUNT: models_volume},
    gpu="A10G",
    timeout=1800,  # 30 min for downloads + TRT engine build
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def download_models():
    """Download models to the volume. Run once: `modal run deploy/modal/app.py::download_models`"""
    print("Downloading models to volume...")
    # Force-replace Modal's Python with the image's Python (has huggingface_hub, torch, etc.)
    if os.path.exists("/usr/local/bin/python3"):
        os.remove("/usr/local/bin/python3")
        os.symlink("/usr/bin/python3", "/usr/local/bin/python3")
    env = os.environ.copy()
    result = subprocess.run(
        ["/bin/bash", "/app/scripts/download_models.sh", VOLUME_MOUNT],
        capture_output=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Model download failed with exit code {result.returncode}")
    models_volume.commit()
    print("Models downloaded and committed to volume.")


# --- Main service (all 4 services in one container) ---
@app.function(
    volumes={VOLUME_MOUNT: models_volume},
    gpu="L4",               # L4 (sm_89, cheapest 24GB option)
    timeout=86400,          # 24 hours max
    scaledown_window=300,   # 5 min idle before scale-to-zero
    min_containers=0,       # scale to zero when idle (set to 1 to keep warm)
    max_containers=1,       # single GPU instance for shared pipeline
    secrets=[modal.Secret.from_dict({"PLACEHOLDER": "1"})],  # add real secrets via Modal dashboard
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8080, startup_timeout=600)
def serve():
    """Run all services via start_unified.sh, expose orchestrator on port 8080."""
    # Modal's add_python puts its Python at /usr/local/bin/python3 which shadows
    # the image's /usr/bin/python3 (that has torch, vllm, etc.).
    # Force-replace Modal's Python symlink with the image's Python.
    if os.path.exists("/usr/local/bin/python3"):
        os.remove("/usr/local/bin/python3")
        os.symlink("/usr/bin/python3", "/usr/local/bin/python3")

    # Pipe subprocess output to Python's stdout/stderr so Modal captures logs
    def _stream_output(proc):
        for line in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    proc = subprocess.Popen(
        ["/app/scripts/start_unified.sh"],
        env={**os.environ, "MODELS_DIR": VOLUME_MOUNT},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    t = threading.Thread(target=_stream_output, args=(proc,), daemon=True)
    t.start()
