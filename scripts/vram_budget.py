#!/usr/bin/env python3
"""
VRAM Budget Calculator for yedam-sts.

Reads vram_budget.yml, queries the GPU for total VRAM, and computes
per-service env vars that docker-compose can source.

Two-tier allocation:
  1. Fixed costs (model weights + CUDA context) are subtracted first.
  2. Remaining VRAM is distributed as variable KV cache headroom,
     weighted by each service's variable_priority.

Usage:
    python scripts/vram_budget.py                          # print to stdout
    python scripts/vram_budget.py --output .env.vram       # write env file
    python scripts/vram_budget.py --stt-backend trt        # use stt_trt instead of stt
    python scripts/vram_budget.py --total-mb 16384         # override GPU detection
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def get_gpu_total_mb() -> int:
    """Query GPU total memory via nvidia-smi (no CUDA context needed)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().split("\n")[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Error querying GPU: {e}", file=sys.stderr)
        sys.exit(1)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_budget(config: dict, total_mb: int, stt_backend: str = "stt") -> dict:
    """Compute per-service VRAM allocations.

    Returns dict with service_name -> {fixed_mb, variable_mb, total_mb} and
    computed env vars.
    """
    gpu_reserved = config["gpu"]["reserved_mb"]
    available = total_mb - gpu_reserved

    # Select active services (stt or stt_trt, not both)
    services = {}
    for name, svc in config["services"].items():
        if name == "stt" and stt_backend == "trt":
            continue
        if name == "stt_trt" and stt_backend != "trt":
            continue
        services[name] = svc

    # Check fixed costs fit
    fixed_total = sum(svc["fixed_mb"] for svc in services.values())
    remaining = available - fixed_total

    if remaining < 0:
        print(
            f"ERROR: Fixed costs ({fixed_total} MB) + reserved ({gpu_reserved} MB) "
            f"exceed GPU total ({total_mb} MB). Reduce model sizes or use a larger GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    min_variable_total = sum(svc["min_variable_mb"] for svc in services.values())
    if remaining < min_variable_total:
        print(
            f"WARNING: Only {remaining} MB remaining for variable allocation "
            f"(minimum needed: {min_variable_total} MB). Using minimum allocations.",
            file=sys.stderr,
        )

    # Distribute remaining by priority weight
    priority_total = sum(svc["variable_priority"] for svc in services.values())

    allocations = {}
    for name, svc in services.items():
        if priority_total > 0 and remaining > 0:
            share = remaining * svc["variable_priority"] / priority_total
            variable_mb = max(svc["min_variable_mb"], int(share))
        else:
            variable_mb = svc["min_variable_mb"]

        allocations[name] = {
            "fixed_mb": svc["fixed_mb"],
            "variable_mb": variable_mb,
            "total_mb": svc["fixed_mb"] + variable_mb,
        }

    # Compute env vars
    env_vars = {}

    # LLM: gpu_memory_utilization = total_budget / gpu_total
    if "llm" in allocations:
        llm_util = round(allocations["llm"]["total_mb"] / total_mb, 2)
        env_vars["LLM_GPU_MEMORY_UTILIZATION"] = str(llm_util)

    # TTS: absolute byte budget
    if "tts" in allocations:
        env_vars["TTS_VRAM_BUDGET_MB"] = str(allocations["tts"]["total_mb"])

    # STT TRT: KV cache fraction
    if "stt_trt" in allocations:
        fraction = round(allocations["stt_trt"]["variable_mb"] / total_mb, 3)
        env_vars["STT_KV_CACHE_FREE_GPU_FRACTION"] = str(fraction)

    return {
        "total_mb": total_mb,
        "gpu_reserved_mb": gpu_reserved,
        "available_mb": available,
        "fixed_total_mb": fixed_total,
        "remaining_mb": remaining,
        "allocations": allocations,
        "env_vars": env_vars,
    }


def print_summary(budget: dict):
    print(f"GPU Total:    {budget['total_mb']:>6} MB")
    print(f"Reserved:     {budget['gpu_reserved_mb']:>6} MB")
    print(f"Available:    {budget['available_mb']:>6} MB")
    print(f"Fixed costs:  {budget['fixed_total_mb']:>6} MB")
    print(f"Variable:     {budget['remaining_mb']:>6} MB")
    print()
    print(f"{'Service':<10} {'Fixed':>8} {'Variable':>10} {'Total':>8}")
    print("-" * 40)
    for name, alloc in budget["allocations"].items():
        print(f"{name:<10} {alloc['fixed_mb']:>7} MB {alloc['variable_mb']:>9} MB {alloc['total_mb']:>7} MB")
    total_allocated = sum(a["total_mb"] for a in budget["allocations"].values())
    print("-" * 40)
    print(f"{'TOTAL':<10} {'':>8} {'':>10} {total_allocated:>7} MB")
    headroom = budget["available_mb"] - total_allocated
    print(f"{'Headroom':<10} {'':>8} {'':>10} {headroom:>7} MB")
    print()
    print("Environment variables:")
    for key, val in budget["env_vars"].items():
        print(f"  {key}={val}")


def write_env_file(env_vars: dict, output_path: str):
    lines = [
        "# Auto-generated by scripts/vram_budget.py",
        "# Do not edit — regenerate with: python scripts/vram_budget.py --output .env.vram",
        "",
    ]
    for key, val in env_vars.items():
        lines.append(f"{key}={val}")
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"Written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute VRAM budget for yedam-sts services")
    parser.add_argument("--config", default="vram_budget.yml", help="Budget config file")
    parser.add_argument("--output", help="Write env vars to file (e.g. .env.vram)")
    parser.add_argument("--stt-backend", choices=["fw", "trt"], default="fw",
                        help="STT backend: fw (faster-whisper) or trt (TensorRT)")
    parser.add_argument("--total-mb", type=int, help="Override GPU total (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    total_mb = args.total_mb or get_gpu_total_mb()
    budget = compute_budget(config, total_mb, stt_backend=args.stt_backend)

    print_summary(budget)

    if args.output:
        write_env_file(budget["env_vars"], args.output)


if __name__ == "__main__":
    main()
