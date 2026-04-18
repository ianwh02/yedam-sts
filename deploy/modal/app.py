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
        # force_build=True,  # Uncomment to force re-pull from GHCR
    )
    .pip_install("huggingface_hub")
    .env({
        "WHISPER_MODEL_DIR": f"{VOLUME_MOUNT}/whisper-medium-komixv2-ct2",
        "MODEL_DIR": f"{VOLUME_MOUNT}/Qwen3-TTS-12Hz-1.7B-Base",
        "TOKENIZER_DIR": f"{VOLUME_MOUNT}/Qwen3-TTS-Tokenizer-12Hz-48kHz",
        "LLM_MODEL": f"{VOLUME_MOUNT}/Qwen3-8B-AWQ",
        "TRT_ENGINE_DIR": f"{VOLUME_MOUNT}/trt_engines",
        "LLM_GLOSSARIES_DIR": "/app/orchestrator/config/glossaries",
        "LLM_DEFAULT_GLOSSARY": "nazarene",
        "STT_INITIAL_PROMPT_VOCAB_PATH": "/app/orchestrator/config/stt_vocab_church.txt",
        "STT_CORRECTIONS_PATH": "/app/orchestrator/config/stt_corrections.tsv",
    })
    # Shell scripts need copy=True (baked into image) to preserve exec permissions.
    .add_local_file("stt-server/convert_model.sh", remote_path="/app/stt-server/convert_model.sh", copy=True)
    .add_local_file("stt-server/entrypoint.tensorrt.sh", remote_path="/app/stt-server/entrypoint.tensorrt.sh", copy=True)
    .add_local_file("scripts/start_unified.sh", remote_path="/app/scripts/start_unified.sh", copy=True)
    .add_local_file("scripts/download_models.sh", remote_path="/app/scripts/download_models.sh", copy=True)
    .run_commands(
        "chmod +x /app/stt-server/convert_model.sh /app/stt-server/entrypoint.tensorrt.sh "
        "/app/scripts/start_unified.sh /app/scripts/download_models.sh"
    )
    .pip_install("supabase>=2.0")
    # Overlay local source for fast iteration (seconds, not minutes).
    # Mirrors Dockerfile.slim lines 132-151 ("changes frequently" layers).
    # Only rebuild/re-push GHCR image for dependency changes.
    # Must come AFTER all build steps (.pip_install, .env, .run_commands).
    .add_local_dir("stt-server/whisper_live", remote_path="/app/stt-server/whisper_live")
    .add_local_dir("stt-server/models/phrase-classifier", remote_path="/app/stt-server/models/phrase-classifier")
    .add_local_file("stt-server/run_server.py", remote_path="/app/stt-server/run_server.py")
    .add_local_file("tts-server/server.py", remote_path="/app/tts-server/server.py")
    .add_local_dir("tts-server/ref_audio", remote_path="/app/tts-server/ref_audio")
    .add_local_dir("tts-server/stage-configs", remote_path="/app/tts-server/stage-configs")
    .add_local_dir("orchestrator/src", remote_path="/app/orchestrator/src")
    .add_local_dir("orchestrator/config", remote_path="/app/orchestrator/config")
    .add_local_file("vram_budget.yml", remote_path="/app/vram_budget.yml")
    .add_local_file("scripts/vram_budget.py", remote_path="/app/scripts/vram_budget.py")
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
    gpu="A10G",              # A10G (sm_86, 24GB, 2x memory bandwidth vs L4)
    timeout=86400,          # 24 hours max
    scaledown_window=300,   # 5 min idle before scale-to-zero
    min_containers=0,       # scale to zero when idle (set to 1 to keep warm)
    max_containers=1,       # single GPU instance for shared pipeline
    secrets=[modal.Secret.from_name("yedam-supabase", required_keys=["BIBLE_SUPABASE_URL", "BIBLE_SUPABASE_KEY"])],
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
