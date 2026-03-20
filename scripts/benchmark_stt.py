#!/usr/bin/env python3
"""STT TTFT + latency benchmark for yedam-sts pipeline.

Measures:
  - TTFT: time from first audio chunk sent → first transcription segment received
  - Full latency: time from first audio sent → last segment received
  - VRAM usage: before, during, and after inference
  - Concurrency scaling: 1, 2, 4 concurrent sessions

Uses real speech audio from debug_audio/ (TTS-generated) or downloads a
sample if none available.

Requires the STT server running with port 9090 exposed to the host:
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up stt

  NOTE: The base docker-compose.yml only uses 'expose' (internal).
  You MUST include docker-compose.dev.yml to map 9090 to the host.

Usage:
  python scripts/benchmark_stt.py
  python scripts/benchmark_stt.py --url ws://localhost:9090 --concurrency 1 2 4
  python scripts/benchmark_stt.py --audio path/to/speech.wav --iterations 5
  python scripts/benchmark_stt.py --burst   # send all audio at once (measures pure processing speed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: websockets package required. pip install websockets")
    sys.exit(1)


STT_WS_URL = "ws://localhost:9090"

# Audio settings (WhisperLive expects float32 PCM 16kHz mono)
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4  # float32
CHUNK_DURATION_S = 0.1  # 100ms chunks for streaming simulation
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_S) * BYTES_PER_SAMPLE


# ============================================================
# Audio Loading
# ============================================================


def load_wav_as_float32(path: str) -> tuple[np.ndarray, float]:
    """Load a WAV file and return (float32 samples at 16kHz, duration_seconds)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Mix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16kHz if needed (simple linear interpolation)
    if sr != SAMPLE_RATE:
        duration = len(samples) / sr
        new_len = int(duration * SAMPLE_RATE)
        indices = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    duration = len(samples) / SAMPLE_RATE
    return samples, duration


def find_test_audio() -> tuple[str, str]:
    """Find a test audio file. Returns (path, description)."""
    # Try debug_audio/ first (TTS-generated speech WAVs)
    debug_dir = Path("debug_audio")
    if debug_dir.exists():
        wavs = sorted(debug_dir.glob("*.wav"))
        # Pick a medium-length one (~2-3 seconds)
        for wav in wavs:
            try:
                with wave.open(str(wav), "rb") as wf:
                    dur = wf.getnframes() / wf.getframerate()
                    if 1.5 < dur < 5.0:
                        return str(wav), f"debug_audio ({dur:.1f}s)"
            except Exception:
                continue
        if wavs:
            return str(wavs[0]), "debug_audio"

    # Try TTS ref audio
    ref = Path("tts-server/ref_audio/en.wav")
    if ref.exists():
        return str(ref), "TTS ref audio (en)"

    raise FileNotFoundError(
        "No test audio found. Place a WAV file in debug_audio/ or specify --audio."
    )


# ============================================================
# VRAM Profiling
# ============================================================


def get_vram_mb() -> Optional[dict]:
    """Query nvidia-smi for current VRAM usage via docker exec."""
    for container in ("yedam-stt", "yedam-stt-trt"):
        try:
            result = subprocess.run(
                [
                    "docker", "exec", container, "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                used, total, free = [int(x.strip()) for x in result.stdout.strip().split(",")]
                return {"used_mb": used, "total_mb": total, "free_mb": free}
        except Exception:
            continue

    # Fallback: try nvidia-smi directly (host-level)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            used, total, free = [int(x.strip()) for x in result.stdout.strip().split(",")]
            return {"used_mb": used, "total_mb": total, "free_mb": free}
    except Exception:
        pass

    return None


# ============================================================
# Benchmark Core
# ============================================================


@dataclass
class STTResult:
    """Results from a single STT benchmark run."""
    ttft_ms: float = 0.0          # Time from first audio sent → first segment received
    processing_ms: float = 0.0    # Time from all audio sent → first segment received (pure server processing)
    full_latency_ms: float = 0.0  # Time from first audio sent → last segment received
    total_time_ms: float = 0.0    # Wall clock from start to finish
    audio_duration_s: float = 0.0
    text: str = ""
    n_segments: int = 0
    error: Optional[str] = None


async def benchmark_single_session(
    url: str,
    audio_f32: np.ndarray,
    audio_duration: float,
    language: str = "en",
    burst: bool = False,
    session_id: str = "bench",
) -> STTResult:
    """Run a single STT benchmark session.

    Args:
        url: WebSocket URL for WhisperLive.
        audio_f32: Audio samples as float32 numpy array.
        audio_duration: Duration of audio in seconds.
        language: Language code for transcription.
        burst: If True, send all audio at once. If False, simulate real-time streaming.
        session_id: Unique ID for this session.
    """
    result = STTResult(audio_duration_s=audio_duration)
    audio_bytes = audio_f32.astype(np.float32).tobytes()

    try:
        async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
            # Send config
            config = {
                "uid": session_id,
                "language": language,
                "task": "transcribe",
                "model": "large-v3",
                "use_vad": True,
            }
            await ws.send(json.dumps(config))

            # Wait for SERVER_READY
            resp = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(resp)
            if data.get("message") != "SERVER_READY":
                result.error = f"Expected SERVER_READY, got: {data}"
                return result

            t_start = time.monotonic()
            t_first_segment = None
            t_last_segment = None
            all_text = []
            n_segments = 0

            # Send audio
            if burst:
                # Send all audio at once (measures pure processing speed)
                await ws.send(audio_bytes)
            else:
                # Simulate real-time streaming (100ms chunks)
                chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)
                for i in range(0, len(audio_f32), chunk_samples):
                    chunk = audio_f32[i:i + chunk_samples]
                    await ws.send(chunk.tobytes())
                    # Pace audio sending at real-time speed
                    await asyncio.sleep(CHUNK_DURATION_S * 0.9)

            t_audio_sent = time.monotonic()

            # Collect transcription responses
            # Wait up to (audio_duration + 10s) for transcription to complete
            deadline = t_start + audio_duration + 10.0
            silence_timeout = 3.0  # stop after 3s of no new segments
            t_last_recv = time.monotonic()

            while time.monotonic() < deadline:
                remaining = min(deadline - time.monotonic(), silence_timeout)
                if remaining <= 0:
                    break
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    data = json.loads(resp)
                    segments = data.get("segments", [])
                    if segments:
                        now = time.monotonic()
                        if t_first_segment is None:
                            t_first_segment = now
                        t_last_segment = now
                        t_last_recv = now
                        n_segments = len(segments)
                        all_text = [s.get("text", "") for s in segments]
                except asyncio.TimeoutError:
                    # No more segments coming
                    break

            t_end = time.monotonic()

            result.total_time_ms = (t_end - t_start) * 1000
            result.n_segments = n_segments
            result.text = " ".join(all_text).strip()

            if t_first_segment is not None:
                result.ttft_ms = (t_first_segment - t_start) * 1000
                result.processing_ms = (t_first_segment - t_audio_sent) * 1000
            if t_last_segment is not None:
                result.full_latency_ms = (t_last_segment - t_start) * 1000

    except Exception as e:
        result.error = str(e)

    return result


async def benchmark_concurrent(
    url: str,
    audio_f32: np.ndarray,
    audio_duration: float,
    concurrency: int,
    language: str = "en",
    burst: bool = False,
) -> List[STTResult]:
    """Run multiple concurrent STT sessions."""
    tasks = [
        benchmark_single_session(
            url, audio_f32, audio_duration, language, burst,
            session_id=f"bench-{i}",
        )
        for i in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


# ============================================================
# Reporting
# ============================================================


def print_results(
    results: List[STTResult],
    concurrency: int,
    iteration: int,
    vram_before: Optional[dict],
    vram_during: Optional[dict],
):
    """Print benchmark results for one run."""
    successful = [r for r in results if r.error is None and r.ttft_ms > 0]
    failed = [r for r in results if r.error is not None]
    no_output = [r for r in results if r.error is None and r.ttft_ms == 0]

    if not successful:
        print(f"    All {len(results)} sessions failed:")
        for r in failed:
            print(f"      Error: {r.error}")
        if no_output:
            print(f"      ({len(no_output)} session(s) connected but received no transcription)")
        return

    ttfts = [r.ttft_ms for r in successful]
    proc_times = [r.processing_ms for r in successful if r.processing_ms > 0]
    latencies = [r.full_latency_ms for r in successful]
    audio_dur = successful[0].audio_duration_s

    avg_ttft = sum(ttfts) / len(ttfts)
    min_ttft = min(ttfts)
    max_ttft = max(ttfts)
    avg_proc = sum(proc_times) / len(proc_times) if proc_times else 0
    min_proc = min(proc_times) if proc_times else 0
    max_proc = max(proc_times) if proc_times else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    min_lat = min(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0

    print(f"    Iteration {iteration}: {len(successful)}/{len(results)} sessions OK")
    print(f"      TTFT:         avg={avg_ttft:7.0f}ms  min={min_ttft:7.0f}ms  max={max_ttft:7.0f}ms")
    if proc_times:
        print(f"      Processing:   avg={avg_proc:7.0f}ms  min={min_proc:7.0f}ms  max={max_proc:7.0f}ms")
    print(f"      Full latency: avg={avg_lat:7.0f}ms  min={min_lat:7.0f}ms  max={max_lat:7.0f}ms")
    print(f"      Audio:        {audio_dur:.1f}s | RTF: {avg_lat / 1000 / audio_dur:.2f}x" if audio_dur > 0 else "")

    if successful[0].text:
        preview = successful[0].text[:80]
        suffix = '...' if len(successful[0].text) > 80 else ''
        try:
            print(f"      Text:         \"{preview}{suffix}\"")
        except UnicodeEncodeError:
            # Windows console can't print non-ASCII (e.g. Korean); use ascii fallback
            safe = preview.encode('ascii', errors='replace').decode('ascii')
            print(f"      Text:         \"{safe}{suffix}\"")

    if vram_during:
        print(f"      VRAM:         {vram_during['used_mb']} MiB / {vram_during['total_mb']} MiB "
              f"(free: {vram_during['free_mb']} MiB)")
        if vram_before:
            delta = vram_during['used_mb'] - vram_before['used_mb']
            print(f"      VRAM delta:   {'+' if delta >= 0 else ''}{delta} MiB vs baseline")

    if failed or no_output:
        n_bad = len(failed) + len(no_output)
        print(f"      Failed:       {n_bad} session(s)")
        for r in failed:
            print(f"        Error: {r.error}")
        if no_output:
            print(f"        ({len(no_output)} connected but got no transcription — likely server full or timeout)")


def print_summary(all_results: dict[int, List[List[STTResult]]]):
    """Print final summary table across all concurrency levels."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Concurrency':>12} {'Avg TTFT':>10} {'Processing':>12} {'Avg Latency':>12} {'RTF':>6} {'Success':>8}")
    print("-" * 80)

    for conc, iterations in sorted(all_results.items()):
        all_successful = [r for it in iterations for r in it if r.error is None and r.ttft_ms > 0]
        if not all_successful:
            print(f"{conc:>12} {'FAILED':>10}")
            continue

        avg_ttft = sum(r.ttft_ms for r in all_successful) / len(all_successful)
        proc_vals = [r.processing_ms for r in all_successful if r.processing_ms > 0]
        avg_proc = sum(proc_vals) / len(proc_vals) if proc_vals else 0
        avg_lat = sum(r.full_latency_ms for r in all_successful) / len(all_successful)
        audio_dur = all_successful[0].audio_duration_s
        rtf = avg_lat / 1000 / audio_dur if audio_dur > 0 else 0
        total = sum(len(it) for it in iterations)
        success = len(all_successful)

        proc_str = f"{avg_proc:>9.0f}ms" if proc_vals else f"{'n/a':>11}"
        print(f"{conc:>12} {avg_ttft:>9.0f}ms {proc_str} {avg_lat:>11.0f}ms {rtf:>5.2f}x {success:>4}/{total}")

    print("=" * 80)


# ============================================================
# Main
# ============================================================


async def main():
    parser = argparse.ArgumentParser(description="STT TTFT + latency benchmark")
    parser.add_argument("--url", default=STT_WS_URL, help="WhisperLive WebSocket URL")
    parser.add_argument("--audio", type=str, default=None, help="Path to test WAV file")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 10],
                        help="Concurrency levels to test (default: 1 2 4 8 10)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per concurrency level (default: 3)")
    parser.add_argument("--burst", action="store_true",
                        help="Send all audio at once instead of streaming (measures pure processing speed)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup iterations (not counted in results)")
    args = parser.parse_args()

    # Load test audio
    if args.audio:
        audio_path = args.audio
        audio_desc = Path(args.audio).name
    else:
        audio_path, audio_desc = find_test_audio()

    print(f"Loading test audio: {audio_path} ({audio_desc})")
    audio_f32, audio_duration = load_wav_as_float32(audio_path)
    print(f"  Duration: {audio_duration:.2f}s, Samples: {len(audio_f32)}, "
          f"Mode: {'burst' if args.burst else 'streaming'}")

    # Baseline VRAM
    print("\nVRAM baseline:")
    vram_baseline = get_vram_mb()
    if vram_baseline:
        print(f"  {vram_baseline['used_mb']} MiB / {vram_baseline['total_mb']} MiB "
              f"(free: {vram_baseline['free_mb']} MiB)")
    else:
        print("  (nvidia-smi not available)")

    # Warmup
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} iteration(s))...")
        for i in range(args.warmup):
            results = await benchmark_concurrent(
                args.url, audio_f32, audio_duration,
                concurrency=1, language=args.language, burst=args.burst,
            )
            r = results[0]
            if r.error:
                print(f"  Warmup {i+1}: ERROR - {r.error}")
            else:
                print(f"  Warmup {i+1}: TTFT={r.ttft_ms:.0f}ms, latency={r.full_latency_ms:.0f}ms")

    # Benchmark
    all_results: dict[int, List[List[STTResult]]] = {}

    for ci, conc in enumerate(args.concurrency):
        # Pause between concurrency levels to let server sessions clean up
        if ci > 0:
            await asyncio.sleep(3.0)

        print(f"\n{'=' * 50}")
        print(f"  Concurrency: {conc}x")
        print(f"{'=' * 50}")

        all_results[conc] = []

        for it in range(args.iterations):
            # Measure VRAM during inference
            results = await benchmark_concurrent(
                args.url, audio_f32, audio_duration,
                concurrency=conc, language=args.language, burst=args.burst,
            )

            # Sample VRAM right after results come back
            vram_during = get_vram_mb()

            all_results[conc].append(results)
            print_results(results, conc, it + 1, vram_baseline, vram_during)

            # Pause between iterations to let server sessions fully clean up
            if it < args.iterations - 1:
                await asyncio.sleep(3.0)

    # Final VRAM
    print("\nVRAM after benchmark:")
    vram_after = get_vram_mb()
    if vram_after:
        print(f"  {vram_after['used_mb']} MiB / {vram_after['total_mb']} MiB "
              f"(free: {vram_after['free_mb']} MiB)")
        if vram_baseline:
            delta = vram_after['used_mb'] - vram_baseline['used_mb']
            print(f"  Delta from baseline: {'+' if delta >= 0 else ''}{delta} MiB")

    # Summary
    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
