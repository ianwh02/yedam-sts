#!/usr/bin/env python3
"""VRAM + E2E latency profiling for yedam-sts pipeline.

Measures GPU memory usage under concurrent session load (VRAM-only),
then benchmarks E2E pipeline latency (LLM→TTS) at increasing concurrency
with no background sessions polluting results.

Requires all services running:
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up

Usage:
  python scripts/profile_vram.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

ORCHESTRATOR_URL = "http://localhost:8080"
TTS_URL = "http://localhost:7860"
LLM_URL = "http://localhost:8000"
STT_WS_URL = "ws://localhost:9090"

LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
# Qwen3 defaults to thinking mode — disable for translation output
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_CONCURRENT_SESSIONS = 5

# Save TTS audio locally for qualitative inspection
SAVE_AUDIO = os.environ.get("SAVE_AUDIO", "1").lower() in ("1", "true", "yes")
AUDIO_DIR = Path("debug_audio")
_audio_counter = 0


def save_tts_wav(pcm_bytes: bytes, text: str, label: str, sample_rate: int = 48000):
    """Save raw s16le PCM response as a WAV file for listening."""
    global _audio_counter
    if not SAVE_AUDIO or not pcm_bytes:
        return
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _audio_counter += 1
    words = text.strip().split()[:5]
    name = "_".join(words) if words else "empty"
    name = re.sub(r"[^\w\-_]", "", name)
    dur_ms = len(pcm_bytes) // 2 / sample_rate * 1000
    filename = f"{_audio_counter:03d}_{label}_{name}_{int(dur_ms)}ms.wav"
    path = AUDIO_DIR / filename
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


# ============================================================
# GPU Memory Helpers
# ============================================================


def get_vram_mb() -> dict:
    """Query nvidia-smi for current VRAM usage (host or container fallback)."""
    # Try host nvidia-smi first (works on Windows/Linux with drivers installed)
    for cmd in [
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
        ["docker", "exec", "yedam-stt", "nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                used, total, free = [int(x.strip()) for x in result.stdout.strip().split(",")]
                return {"used_mb": used, "total_mb": total, "free_mb": free}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("Could not query VRAM: nvidia-smi not found on host or in yedam-stt container")


def print_vram(label: str, vram: dict):
    pct = vram["used_mb"] / vram["total_mb"] * 100
    print(f"  {label}: {vram['used_mb']} / {vram['total_mb']} MiB ({pct:.1f}%) — {vram['free_mb']} MiB free")


def generate_speech_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple tone as Float32 PCM bytes."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return audio.tobytes()


# ============================================================
# Latency Helpers
# ============================================================


@dataclass
class LatencyResult:
    service: str
    latency_ms: float
    details: str = ""


def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


# ============================================================
# Per-Service Latency Profiling
# ============================================================


async def profile_stt_latency(n_runs: int = 3) -> list[LatencyResult]:
    """Measure STT latency: time from sending audio to receiving transcription.

    Sends audio in real-time chunks and measures time to first transcription
    response. Uses white noise (passes VAD better than pure tone).
    """
    import websockets

    results = []
    # White noise passes VAD more reliably than a pure sine tone
    sample_rate = 16000
    duration_s = 3.0
    n_samples = int(sample_rate * duration_s)
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(n_samples) * 0.3).astype(np.float32).tobytes()
    chunk_duration_s = 0.1  # 100ms chunks (real-time streaming)
    chunk_size = int(sample_rate * chunk_duration_s) * 4  # 4 bytes per float32

    for i in range(n_runs):
        try:
            async with websockets.connect(STT_WS_URL) as ws:
                config = {
                    "uid": f"latency-test-{i}",
                    "language": "ko",
                    "task": "transcribe",
                    "model": "large-v3",
                    "use_vad": True,
                }
                await ws.send(json.dumps(config))
                await asyncio.wait_for(ws.recv(), timeout=10.0)  # server ready

                t0 = time.perf_counter()

                # Send audio in real-time chunks
                for offset in range(0, len(audio), chunk_size):
                    await ws.send(audio[offset:offset + chunk_size])
                    await asyncio.sleep(chunk_duration_s)

                # Collect responses until we get a transcription or timeout
                first_text_time = None
                text = ""
                deadline = time.perf_counter() + 10.0
                while time.perf_counter() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        data = json.loads(msg)
                        segments = data.get("segments", [])
                        if segments:
                            seg = segments[-1]
                            text = seg.get("text", "")
                            if first_text_time is None and text.strip():
                                first_text_time = time.perf_counter()
                            if seg.get("completed", False):
                                break
                    except asyncio.TimeoutError:
                        break

                if first_text_time is not None:
                    latency = (first_text_time - t0) * 1000
                    results.append(LatencyResult("STT", latency, f"run {i+1}: \"{text.strip()[:50]}\""))
                else:
                    # Still measure the processing time even without text
                    t1 = time.perf_counter()
                    latency = (t1 - t0) * 1000
                    results.append(LatencyResult("STT", latency, f"run {i+1}: no transcription (VAD filtered)"))

        except Exception as e:
            results.append(LatencyResult("STT", -1, f"run {i+1}: {e}"))

    return results


async def profile_llm_latency(n_runs: int = 3) -> list[LatencyResult]:
    """Measure LLM latency: time to first token (TTFT) and total completion."""
    results = []
    messages = [
        {"role": "system", "content": "/no_think\nYou are a Korean to English translator. Translate the Korean text into natural English. Output ONLY the English translation, nothing else."},
        {"role": "user", "content": "오늘 날씨가 정말 좋습니다. 공원에 가서 산책하고 싶습니다."},
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(n_runs):
            # Non-streaming for total latency
            t0 = time.perf_counter()
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 50,
                    "temperature": 0.3,
                    **LLM_EXTRA_BODY,
                },
            )
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000

            data = resp.json()
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", "?")
            content = data.get("choices", [{}])[0].get("message", {}).get("content", f"(error: {str(data)[:100]})")
            results.append(LatencyResult(
                "LLM (total)", total_ms,
                f"run {i+1}: {tokens} tokens, \"{content[:60]}\"",
            ))

            # Streaming for TTFT
            t0 = time.perf_counter()
            ttft = None
            async with client.stream(
                "POST",
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 50,
                    "temperature": 0.3,
                    "stream": True,
                    **LLM_EXTRA_BODY,
                },
            ) as stream:
                async for line in stream.aiter_lines():
                    if line.startswith("data: ") and ttft is None:
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            continue
                        parsed = json.loads(chunk)
                        delta = parsed["choices"][0].get("delta", {})
                        if delta.get("content"):
                            ttft = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()

            if ttft is not None:
                results.append(LatencyResult("LLM (TTFT)", ttft, f"run {i+1}"))

    return results


async def profile_tts_latency(n_runs: int = 3) -> list[LatencyResult]:
    """Measure TTS latency: time from request to audio response."""
    results = []
    texts = [
        "The weather is really nice today.",
        "I want to go for a walk in the park.",
        "Hello, how are you doing today?",
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(n_runs):
            text = texts[i % len(texts)]
            t0 = time.perf_counter()
            resp = await client.post(
                f"{TTS_URL}/synthesize",
                json={"text": text, "language": "en"},
                headers={"Accept": "application/octet-stream"},
            )
            t1 = time.perf_counter()

            latency_s = t1 - t0
            latency = latency_s * 1000
            audio_dur_s = len(resp.content) / 2 / 48000
            duration_ms = int(audio_dur_s * 1000)
            rtf = latency_s / audio_dur_s if audio_dur_s > 0 else 0
            audio_kb = len(resp.content) / 1024
            save_tts_wav(resp.content, text, f"individual_run{i+1}")
            results.append(LatencyResult(
                "TTS", latency,
                f"run {i+1}: \"{text[:40]}\" → {duration_ms}ms audio, RTF={rtf:.2f}, {audio_kb:.0f}KB",
            ))

    return results


async def profile_service_latencies():
    """Run all per-service latency benchmarks."""
    print("\n" + "=" * 60)
    print("  Per-Service Latency Profiling")
    print("=" * 60)

    # Run STT and TTS latency in parallel (they don't interfere),
    # but LLM sequentially to avoid batching effects
    stt_task = asyncio.create_task(profile_stt_latency())
    tts_task = asyncio.create_task(profile_tts_latency())

    stt_results = await stt_task
    tts_results = await tts_task
    llm_results = await profile_llm_latency()

    all_results = {"STT": stt_results, "LLM": llm_results, "TTS": tts_results}

    for service, results in all_results.items():
        print(f"\n  --- {service} ---")
        for r in results:
            if r.latency_ms < 0:
                print(f"    {r.service}: FAILED ({r.details})")
            else:
                print(f"    {r.service}: {fmt_ms(r.latency_ms)}  ({r.details})")

        valid = [r.latency_ms for r in results if r.latency_ms >= 0 and r.service == results[0].service]
        if valid:
            avg = sum(valid) / len(valid)
            p50 = sorted(valid)[len(valid) // 2]
            print(f"    Average: {fmt_ms(avg)}, Median: {fmt_ms(p50)}")

    return all_results


# ============================================================
# E2E Pipeline Latency
# ============================================================


async def profile_e2e_latency():
    """Measure end-to-end latency through the full pipeline.

    Creates a session, feeds audio via admin WS, and measures time until
    we receive translated text and TTS audio back via callbacks.
    Since callbacks are internal, we measure via the orchestrator's
    individual service hops: audio → STT → LLM → TTS.
    """
    print("\n" + "=" * 60)
    print("  E2E Pipeline Baseline 1x (LLM → TTS)")
    print("=" * 60)

    # Measure the serial chain: STT transcribe → LLM translate → TTS synthesize
    # This simulates what happens for a single utterance through the pipeline

    test_texts_ko = [
        "오늘 날씨가 정말 좋습니다.",
        "서울에서 회의를 했습니다.",
        "감사합니다. 좋은 하루 되세요.",
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, text_ko in enumerate(test_texts_ko):
            print(f"\n  --- E2E Run {i+1}: \"{text_ko}\" ---")
            e2e_start = time.perf_counter()

            # STT latency measured separately above. Here we measure the
            # LLM → TTS serial chain for a known Korean segment.

            # Stage 1: LLM translation
            llm_start = time.perf_counter()
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "/no_think\nYou are a Korean to English translator. Translate the Korean text into natural English. Output ONLY the English translation, nothing else."},
                        {"role": "user", "content": text_ko},
                    ],
                    "max_tokens": 50,
                    "temperature": 0.3,
                    **LLM_EXTRA_BODY,
                },
            )
            llm_end = time.perf_counter()
            llm_ms = (llm_end - llm_start) * 1000

            translated = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            print(f"    LLM: {fmt_ms(llm_ms)} → \"{translated[:60]}\"")

            # Stage 3: TTS synthesis
            tts_start = time.perf_counter()
            resp = await client.post(
                f"{TTS_URL}/synthesize",
                json={"text": translated, "language": "en"},
                headers={"Accept": "application/octet-stream"},
            )
            tts_end = time.perf_counter()
            tts_s = tts_end - tts_start
            tts_ms = tts_s * 1000

            audio_dur_s = len(resp.content) / 2 / 48000
            duration_ms = int(audio_dur_s * 1000)
            rtf = tts_s / audio_dur_s if audio_dur_s > 0 else 0
            save_tts_wav(resp.content, translated, f"e2e_run{i+1}")
            print(f"    TTS: {fmt_ms(tts_ms)} → {duration_ms}ms audio, RTF={rtf:.2f}")

            e2e_end = time.perf_counter()
            e2e_ms = (e2e_end - e2e_start) * 1000
            print(f"    E2E (LLM+TTS): {fmt_ms(e2e_ms)}")

    print("\n  Note: E2E = LLM + TTS latency (serial chain).")
    print("  STT latency is additive but measured separately above.")
    print("  In practice, sentence pipelining overlaps TTS(N) with LLM(N+1).")


# ============================================================
# E2E Concurrent Pipeline (LLM → TTS)
# ============================================================


async def profile_e2e_concurrent(concurrency_levels: list[int] | None = None):
    """Fire N simultaneous LLM→TTS pipelines and measure per-request E2E latency.

    Each pipeline: Korean text → LLM translate → TTS synthesize English audio.
    Runs at increasing concurrency levels with no background sessions polluting GPU.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4]

    print("\n" + "=" * 60)
    print("  E2E Concurrent Pipeline (LLM → TTS)")
    print("=" * 60)

    test_texts_ko = [
        "오늘 날씨가 정말 좋습니다.",
        "서울에서 회의를 했습니다.",
        "감사합니다. 좋은 하루 되세요.",
        "한국어를 영어로 번역해 주세요.",
        "이 프로젝트는 정말 흥미롭습니다.",
    ]

    async def e2e_pipeline(text_ko: str, idx: int, client: httpx.AsyncClient) -> dict:
        """Single LLM→TTS chain: translate Korean, then synthesize English."""
        t0 = time.perf_counter()

        # LLM translate
        llm_t0 = time.perf_counter()
        resp = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "/no_think\nYou are a Korean to English translator. Translate the Korean text into natural English. Output ONLY the English translation, nothing else."},
                    {"role": "user", "content": text_ko},
                ],
                "max_tokens": 50,
                "temperature": 0.3,
                **LLM_EXTRA_BODY,
            },
        )
        translated = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        llm_ms = (time.perf_counter() - llm_t0) * 1000

        # TTS synthesize
        tts_t0 = time.perf_counter()
        resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": translated, "language": "en"},
            headers={"Accept": "application/octet-stream"},
        )
        tts_s = time.perf_counter() - tts_t0
        audio_dur_s = len(resp.content) / 2 / 48000
        rtf = tts_s / audio_dur_s if audio_dur_s > 0 else 0
        save_tts_wav(resp.content, translated, f"e2e_conc_req{idx+1}")

        e2e_s = time.perf_counter() - t0
        return {
            "e2e_ms": e2e_s * 1000,
            "llm_ms": llm_ms,
            "tts_ms": tts_s * 1000,
            "audio_dur_ms": audio_dur_s * 1000,
            "rtf": rtf,
            "translated": translated,
            "text_ko": text_ko,
        }

    all_level_results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for n in concurrency_levels:
            print(f"\n  --- {n}x concurrent ---")

            tasks = [
                e2e_pipeline(test_texts_ko[i % len(test_texts_ko)], i, client)
                for i in range(n)
            ]
            results = await asyncio.gather(*tasks)

            # Print per-request breakdown
            for r in results:
                print(f"    \"{r['text_ko'][:20]}...\" → \"{r['translated'][:30]}...\"")
                print(f"      LLM: {fmt_ms(r['llm_ms'])} | TTS: {fmt_ms(r['tts_ms'])} "
                      f"(RTF {r['rtf']:.2f}) | E2E: {fmt_ms(r['e2e_ms'])}")

            # Averages
            avg_llm = sum(r["llm_ms"] for r in results) / len(results)
            avg_tts = sum(r["tts_ms"] for r in results) / len(results)
            avg_rtf = sum(r["rtf"] for r in results) / len(results)
            avg_e2e = sum(r["e2e_ms"] for r in results) / len(results)

            if n > 1:
                print(f"    avg: LLM {fmt_ms(avg_llm)} | TTS {fmt_ms(avg_tts)} "
                      f"(RTF {avg_rtf:.2f}) | E2E {fmt_ms(avg_e2e)}")

            all_level_results.append({
                "n": n,
                "avg_llm_ms": avg_llm,
                "avg_tts_ms": avg_tts,
                "avg_rtf": avg_rtf,
                "avg_e2e_ms": avg_e2e,
                "results": results,
            })

    return all_level_results


# ============================================================
# VRAM Profiling (existing)
# ============================================================


async def profile_baseline():
    """Measure baseline VRAM with all services loaded but no sessions."""
    print("\n" + "=" * 60)
    print("  Baseline VRAM (all services loaded, 0 sessions)")
    print("=" * 60)
    vram = get_vram_mb()
    print_vram("Baseline", vram)
    return vram


async def profile_individual_services():
    """Measure VRAM impact of individual service calls."""
    print("\n" + "=" * 60)
    print("  Individual Service VRAM Impact")
    print("=" * 60)

    import websockets

    async with httpx.AsyncClient(timeout=180.0) as client:
        # LLM inference
        before = get_vram_mb()
        resp = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "/no_think\nYou are a Korean to English translator. Translate the Korean text into natural English. Output ONLY the English translation, nothing else."},
                    {"role": "user", "content": "오늘 날씨가 정말 좋습니다. 공원에 가서 산책하고 싶습니다."},
                ],
                "max_tokens": 50,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        after = get_vram_mb()
        delta = after["used_mb"] - before["used_mb"]
        resp_json = resp.json()
        if "choices" in resp_json:
            content = resp_json["choices"][0]["message"]["content"]
        else:
            content = f"(error: {resp_json.get('message', resp_json.get('detail', str(resp_json)[:120]))})"
        print(f"\n  LLM completion: +{delta} MiB")
        print(f"    Response: {content[:80]}")
        print_vram("After LLM", after)

        # TTS synthesis
        before = get_vram_mb()
        tts_t0 = time.perf_counter()
        resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": "The weather is really nice today. I want to go for a walk in the park.", "language": "en"},
            headers={"Accept": "application/octet-stream"},
        )
        tts_elapsed = time.perf_counter() - tts_t0
        after = get_vram_mb()
        delta = after["used_mb"] - before["used_mb"]
        audio_dur_s = len(resp.content) / 2 / 48000
        duration_ms = int(audio_dur_s * 1000)
        rtf = tts_elapsed / audio_dur_s if audio_dur_s > 0 else 0
        save_tts_wav(resp.content, "The weather is really nice today I want to go for a walk", "vram_baseline")
        print(f"\n  TTS synthesis: +{delta} MiB")
        print(f"    Audio: {duration_ms}ms, RTF={rtf:.2f}")
        print_vram("After TTS", after)

        # STT (just connection, no real audio)
        before = get_vram_mb()
        async with websockets.connect(STT_WS_URL) as ws:
            config = {"uid": "vram-test", "language": "ko", "task": "transcribe", "model": "large-v3", "use_vad": True}
            await ws.send(json.dumps(config))
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            audio = generate_speech_audio(duration_s=2.0)
            await ws.send(audio)
            await asyncio.sleep(2.0)
        after = get_vram_mb()
        delta = after["used_mb"] - before["used_mb"]
        print(f"\n  STT (2s audio): +{delta} MiB")
        print_vram("After STT", after)


async def profile_concurrent_sessions(n_sessions: int) -> dict:
    """Create N concurrent sessions and measure VRAM impact."""
    print(f"\n" + "=" * 60)
    print(f"  {n_sessions} Concurrent Session(s) — VRAM")
    print("=" * 60)

    before = get_vram_mb()
    print_vram("Before", before)

    sessions = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Create sessions
        for i in range(n_sessions):
            resp = await client.post(
                f"{ORCHESTRATOR_URL}/api/sessions",
                json={"source_lang": "ko", "target_lang": "en", "processor": "translation"},
            )
            if resp.status_code == 200:
                data = resp.json()
                sessions.append(data)
                print(f"  Created session {i+1}: {data['session_id']}")
            else:
                print(f"  Failed to create session {i+1}: {resp.status_code} {resp.text[:100]}")

        after_create = get_vram_mb()
        delta = after_create["used_mb"] - before["used_mb"]
        print(f"\n  After creating {len(sessions)} session(s): +{delta} MiB")
        print_vram("Post-create", after_create)

        # Feed audio to all sessions concurrently
        peak = before
        delta_peak = 0
        if sessions:
            print(f"\n  Feeding audio to {len(sessions)} session(s) concurrently...")
            audio = generate_speech_audio(duration_s=3.0)
            chunk_size = 16000 * 4  # 1 second chunks

            import websockets

            async def feed_session(session_data):
                ws_url = f"ws://localhost:8080{session_data['admin_ws_url']}"
                try:
                    async with websockets.connect(ws_url) as ws:
                        for i in range(0, len(audio), chunk_size):
                            await ws.send(audio[i:i + chunk_size])
                            await asyncio.sleep(0.3)
                        await asyncio.sleep(2.0)  # Let pipeline process
                except Exception as e:
                    print(f"    Session {session_data['session_id']}: {e}")

            # Feed all sessions at once
            await asyncio.gather(*[feed_session(s) for s in sessions])

            # Measure peak VRAM during/after processing
            peak = get_vram_mb()
            delta_peak = peak["used_mb"] - before["used_mb"]
            print(f"\n  After processing audio in {len(sessions)} session(s): +{delta_peak} MiB from baseline")
            print_vram("Peak", peak)

        # Cleanup sessions
        for s in sessions:
            await client.delete(f"{ORCHESTRATOR_URL}/api/sessions/{s['session_id']}")

        await asyncio.sleep(2.0)
        after_cleanup = get_vram_mb()
        delta_cleanup = after_cleanup["used_mb"] - before["used_mb"]
        print(f"\n  After cleanup: +{delta_cleanup} MiB from baseline")
        print_vram("Post-cleanup", after_cleanup)

    return {
        "n_sessions": n_sessions,
        "baseline_mb": before["used_mb"],
        "peak_mb": peak["used_mb"] if sessions else before["used_mb"],
        "delta_mb": delta_peak if sessions else 0,
    }


# ============================================================
# Main
# ============================================================


async def main():
    print("yedam-sts VRAM + Latency Profiling")
    print(f"Orchestrator: {ORCHESTRATOR_URL}")
    if SAVE_AUDIO:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving TTS audio to: {AUDIO_DIR.resolve()}  (set SAVE_AUDIO=0 to disable)")
    else:
        print("Audio saving disabled (set SAVE_AUDIO=1 to enable)")

    # ---- VRAM Profiling ----
    baseline = await profile_baseline()
    await profile_individual_services()

    # Concurrent sessions: 1 through MAX (VRAM only)
    vram_results = []
    for n in range(1, MAX_CONCURRENT_SESSIONS + 1):
        try:
            result = await profile_concurrent_sessions(n)
            vram_results.append(result)
            # Stop if we're within 200 MiB of full
            current_free = result["peak_mb"]
            total = baseline["total_mb"]
            if total - current_free < 200:
                print(f"\n  Stopping: only {total - current_free} MiB free at peak")
                break
        except Exception as e:
            print(f"\n  ERROR with {n} sessions: {e}")
            break
        await asyncio.sleep(3.0)

    # ---- Latency Profiling (no active sessions) ----
    await profile_service_latencies()
    await profile_e2e_latency()
    e2e_results = await profile_e2e_concurrent()

    # ---- VRAM Summary ----
    print("\n" + "=" * 60)
    print("  VRAM Summary")
    print("=" * 60)
    print(f"  GPU: {baseline['total_mb']} MiB total")
    print(f"  Baseline (idle): {baseline['used_mb']} MiB ({baseline['used_mb']/baseline['total_mb']*100:.1f}%)")
    print(f"  Free at idle: {baseline['free_mb']} MiB")
    print()
    for r in vram_results:
        free_at_peak = baseline["total_mb"] - r["peak_mb"]
        print(f"  {r['n_sessions']} session(s): peak {r['peak_mb']} MiB (+{r['delta_mb']} MiB) — {free_at_peak} MiB free")
    print()

    if vram_results:
        max_sessions = vram_results[-1]["n_sessions"]
        max_peak = vram_results[-1]["peak_mb"]
        max_free = baseline["total_mb"] - max_peak
        print(f"  Max tested: {max_sessions} concurrent session(s)")
        print(f"  Peak VRAM: {max_peak} MiB ({max_free} MiB remaining)")

    # ---- E2E Scaling Summary ----
    if e2e_results:
        print("\n" + "=" * 60)
        print("  E2E Scaling Summary (LLM → TTS)")
        print("=" * 60)
        base_e2e = e2e_results[0]["avg_e2e_ms"]
        print(f"\n  {'Conc':>6} | {'LLM avg':>10} | {'TTS avg':>10} | {'TTS RTF':>8} | {'E2E avg':>10} | {'vs 1x':>7}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")
        for d in e2e_results:
            ratio = d["avg_e2e_ms"] / base_e2e if base_e2e else 0
            print(f"  {d['n']:>6} | {fmt_ms(d['avg_llm_ms']):>10} | {fmt_ms(d['avg_tts_ms']):>10} | "
                  f"{d['avg_rtf']:>8.2f} | {fmt_ms(d['avg_e2e_ms']):>10} | {ratio:>6.2f}x")
    print()


if __name__ == "__main__":
    asyncio.run(main())
