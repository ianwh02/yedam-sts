#!/usr/bin/env python3
"""
Phase 2 VRAM Coordinator for yedam-sts.

After all services have loaded model weights (Phase 1), this script:
1. Polls /health on all services until they report "weights_ready" (or "ready" for vLLM)
2. Measures actual free VRAM via nvidia-smi
3. Distributes remaining VRAM to TTS and STT as KV cache budgets
4. Sends POST /allocate_kv_cache to each service
5. Polls until all report "ready"

Usage:
    python scripts/vram_coordinator.py --config vram_budget.yml [--stt-backend trt]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import yaml

try:
    import requests
except ImportError:
    import urllib.error
    import urllib.request

    class _FallbackRequests:
        """Minimal requests-like fallback using urllib."""

        class Response:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

            def json(self):
                return json.loads(self.text)

        @staticmethod
        def get(url, timeout=5):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text = resp.read().decode()
                    return _FallbackRequests.Response(resp.status, text)
            except urllib.error.HTTPError as e:
                return _FallbackRequests.Response(e.code, e.read().decode())
            except Exception:
                return _FallbackRequests.Response(0, "")

        @staticmethod
        def post(url, timeout=5, json=None):
            try:
                data = None
                headers = {}
                if json is not None:
                    import json as _json
                    data = _json.dumps(json).encode()
                    headers["Content-Type"] = "application/json"
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text = resp.read().decode()
                    return _FallbackRequests.Response(resp.status, text)
            except urllib.error.HTTPError as e:
                return _FallbackRequests.Response(e.code, e.read().decode())
            except Exception as ex:
                return _FallbackRequests.Response(0, str(ex))

    requests = _FallbackRequests()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [coordinator] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Service endpoints: container_name -> internal port
# STT container name depends on backend (set in set_stt_container())
SERVICE_CONTAINERS = {
    "llm": ("yedam-llm", 8000),
    "tts": ("yedam-tts", 7860),
    "stt": ("yedam-stt", 9090),
}


def set_stt_container(stt_backend: str):
    """Set the STT container name based on backend."""
    if stt_backend == "trt":
        SERVICE_CONTAINERS["stt"] = ("yedam-stt-trt", 9090)


def _docker_curl(container: str, path: str, method: str = "GET", timeout: int = 5) -> tuple[int, str]:
    """Run curl inside a docker container. Returns (status_code, body)."""
    # Look up port for this container
    port = None
    for name, (cname, p) in SERVICE_CONTAINERS.items():
        if cname == container:
            port = p
            break
    if port is None:
        return 0, f"unknown container: {container}"

    try:
        result = subprocess.run(
            ["docker", "exec", container, "curl", "-sf", "-m", str(timeout),
             "-X", method, f"http://localhost:{port}{path}"],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0:
            return 200, result.stdout.strip()
        return 0, result.stderr.strip() or result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 0, "timeout"
    except Exception as e:
        return 0, str(e)


def get_free_vram_mb() -> int:
    """Query nvidia-smi for free VRAM in MB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().split("\n")[0])




def get_service_status(name: str) -> str | None:
    """Get startup status. Uses /startup for TTS/STT, /health for vLLM."""
    container, port = SERVICE_CONTAINERS[name]
    try:
        if name == "llm":
            # vLLM has no /startup endpoint — empty 200 on /health means ready
            code, body = _docker_curl(container, "/health")
            return "ready" if code == 200 else None
        else:
            code, body = _docker_curl(container, "/startup")
            if code == 200 and body:
                data = json.loads(body)
                return data.get("status")
            return None
    except Exception:
        return None


def wait_for_phase1(services: list[str], timeout: float = 300.0) -> dict[str, str]:
    """Poll services until all report weights_ready or ready. Returns final statuses."""
    log.info("Waiting for services to load weights: %s", services)
    start = time.time()
    statuses = {s: None for s in services}

    while time.time() - start < timeout:
        all_ready = True
        for svc in services:
            if statuses[svc] in ("weights_ready", "ready"):
                continue
            status = get_service_status(svc)
            if status in ("weights_ready", "ready"):
                statuses[svc] = status
                log.info("  %s: %s", svc, status)
            else:
                all_ready = False

        if all_ready:
            log.info("All services loaded weights (%.1fs)", time.time() - start)
            return statuses

        time.sleep(2.0)

    failed = [s for s, v in statuses.items() if v not in ("weights_ready", "ready")]
    log.error("Timeout waiting for: %s (statuses: %s)", failed, statuses)
    sys.exit(1)


def allocate_kv_caches(
    config: dict,
    statuses: dict[str, str],
    stt_backend: str,
) -> None:
    """Measure free VRAM and distribute KV cache budgets to TTS and STT in parallel.

    For STT TRT, from_dir() loads engine weights + allocates KV cache from the same
    VRAM pool, so its share must cover both. We subtract the engine fixed_mb from its
    share to get its actual KV budget. TTS weights are already loaded — its full share
    goes to KV cache.
    """
    free_mb = get_free_vram_mb()
    reserved_mb = config["gpu"].get("reserved_mb", 500)
    available_mb = max(0, free_mb - reserved_mb)

    log.info("Free VRAM: %d MB, reserved: %d MB, available for KV: %d MB",
             free_mb, reserved_mb, available_mb)

    # Determine which services need allocation
    services_to_allocate = {}
    stt_key = "stt_trt" if stt_backend == "trt" else "stt"

    for svc_name, status in statuses.items():
        if status == "weights_ready":
            cfg_key = stt_key if svc_name == "stt" else svc_name
            svc_cfg = config["services"].get(cfg_key, {})
            services_to_allocate[svc_name] = {
                "priority": svc_cfg.get("variable_priority", 1),
                "min_mb": svc_cfg.get("min_variable_mb", 50),
                "fixed_mb": svc_cfg.get("fixed_mb", 0),
                "needs_engine_load": svc_name == "stt" and stt_backend == "trt",
            }

    if not services_to_allocate:
        log.info("No services need KV cache allocation (all already ready)")
        return

    cache_path = os.path.join(os.path.dirname(__file__), "..", ".vram_cache.json")

    # Partial restart = some services already ready, only some need allocation.
    # Full start = all GPU services need allocation.
    is_partial_restart = any(
        s == "ready" for s in statuses.values()
    )

    # Load cache only on partial restarts — full starts always recalculate
    cached = {}
    if is_partial_restart and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            log.info("Partial restart detected — using cached allocations")
        except Exception:
            pass
    elif not is_partial_restart:
        log.info("Full start — recalculating allocations")

    kv_pool_mb = available_mb
    total_priority = sum(s["priority"] for s in services_to_allocate.values())

    allocations = {}
    for svc_name, svc_info in services_to_allocate.items():
        if svc_name in cached:
            kv_mb = cached[svc_name]
            log.info("  %s: kv=%d MB (cached)", svc_name, kv_mb)
        else:
            kv_mb = int(kv_pool_mb * svc_info["priority"] / total_priority)
            kv_mb = max(svc_info["min_mb"], kv_mb)
            log.info("  %s: kv=%d MB (computed)", svc_name, kv_mb)

        allocations[svc_name] = {"total_mb": kv_mb, "kv_mb": kv_mb}

    # Send allocate commands
    for svc_name, alloc in allocations.items():
        container, port = SERVICE_CONTAINERS[svc_name]

        if svc_name == "tts":
            budget_mb = alloc["kv_mb"]
            log.info("  POST %s /allocate_kv_cache (budget_mb=%d)", container, budget_mb)
            code, body = _docker_curl(container, f"/allocate_kv_cache?budget_mb={budget_mb}",
                                       method="POST", timeout=360)
        elif svc_name == "stt":
            # STT TRT: from_dir() loads engines then allocates KV from remaining free VRAM.
            # fraction = total_share / free_vram. This tells TRT how much of the free VRAM
            # (at call time) it can use for engines + KV cache combined.
            total_mb = alloc["total_mb"]
            fraction = min(0.90, total_mb / free_mb) if free_mb > 0 else 0.05
            fraction = max(0.01, fraction)
            log.info("  Signaling %s to allocate (fraction=%.3f, share=%d MB)",
                     container, fraction, total_mb)
            try:
                result = subprocess.run(
                    ["docker", "exec", container, "python3", "-c",
                     f"import json; open('/tmp/allocate_kv_cache.json','w').write(json.dumps({{'fraction':{fraction:.4f}}}))"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    log.error("  Failed to write signal file: %s", result.stderr)
                    sys.exit(1)
                log.info("  Signal written, waiting for STT to allocate...")
                alloc_start = time.time()
                while time.time() - alloc_start < 180:
                    status = get_service_status(svc_name)
                    if status == "ready":
                        code, body = 200, '{"status":"ready"}'
                        break
                    time.sleep(2.0)
                else:
                    code, body = 0, "timeout waiting for STT ready"
            except Exception as e:
                code, body = 0, str(e)
        else:
            continue

        if code == 200:
            log.info("  %s: allocated successfully — %s", svc_name, body)
            # Cache successful allocation for future restarts
            cached[svc_name] = alloc["kv_mb"]
            try:
                with open(cache_path, "w") as f:
                    json.dump(cached, f)
            except Exception:
                pass
        else:
            log.error("  %s: allocation FAILED (code=%d): %s", svc_name, code, body)
            sys.exit(1)


def wait_for_ready(services: list[str], timeout: float = 60.0) -> None:
    """Poll until all services report ready."""
    log.info("Waiting for all services to report 'ready'...")
    start = time.time()
    while time.time() - start < timeout:
        all_ready = True
        for svc in services:
            status = get_service_status(svc)
            if status != "ready":
                all_ready = False
                break
        if all_ready:
            log.info("All services ready (%.1fs)", time.time() - start)
            return
        time.sleep(2.0)

    log.error("Timeout waiting for services to become ready")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="VRAM Phase 2 Coordinator")
    parser.add_argument("--config", default="vram_budget.yml", help="Budget config file")
    parser.add_argument("--stt-backend", choices=["fw", "trt"], default="fw",
                        help="STT backend: fw (faster-whisper) or trt (TensorRT)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Max seconds to wait for Phase 1 (default: 300)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine active services
    active_services = ["llm", "tts"]
    if args.stt_backend == "trt":
        active_services.append("stt")
    else:
        active_services.append("stt")

    set_stt_container(args.stt_backend)

    log.info("=== Phase 2: VRAM Coordinator ===")
    log.info("STT backend: %s", args.stt_backend)
    log.info("Active services: %s", active_services)

    # Phase 1: wait for all services to load weights
    statuses = wait_for_phase1(active_services, timeout=args.timeout)

    # Phase 2: measure and allocate
    allocate_kv_caches(config, statuses, args.stt_backend)

    # Phase 3: verify all ready
    wait_for_ready(active_services, timeout=60.0)

    log.info("=== Coordinator complete ===")


if __name__ == "__main__":
    main()
