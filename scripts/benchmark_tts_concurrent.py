#!/usr/bin/env python3
"""Concurrent TTS benchmark for yedam-sts pipeline.

Measures TTS latency and RTF at various concurrency levels (1, 2, 4, 8).
Uses both streaming and non-streaming endpoints.

Requires the TTS server running with port exposed:
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up tts

Usage:
  python scripts/benchmark_tts_concurrent.py
  python scripts/benchmark_tts_concurrent.py --url http://localhost:7860 --max-concurrent 8
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import struct
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

TTS_URL = "http://localhost:7860"

# Debug audio saving
SAVE_AUDIO = os.environ.get("SAVE_AUDIO", "1").lower() not in ("0", "false", "no")
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "debug_audio"))

def _save_wav(pcm_bytes: bytes, label: str, sample_rate: int = 48000, sample_format: str = "s16le"):
    """Save raw PCM bytes as a WAV file for inspection."""
    if not SAVE_AUDIO or not pcm_bytes:
        return
    try:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        # Convert to s16le if needed
        if sample_format == "f32le":
            f32 = np.frombuffer(pcm_bytes, dtype=np.float32)
            s16 = np.clip(f32 * 32767, -32768, 32767).astype(np.int16)
            raw = s16.tobytes()
        else:
            raw = pcm_bytes
        n_samples = len(raw) // 2
        duration_ms = int(n_samples / sample_rate * 1000)
        safe_label = re.sub(r"[^\w\-]", "_", label)[:60]
        filename = f"{safe_label}_{duration_ms}ms.wav"
        path = AUDIO_DIR / filename
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw)
        print(f"  [audio] saved {path} ({duration_ms}ms)")
    except Exception as e:
        print(f"  [audio] save failed: {e}")

# Mix of English and Korean sentences at varying lengths
TEST_SENTENCES = [
    ("The weather is really nice today.", "en"),
    ("I would like to order a coffee please.", "en"),
    ("Machine learning models are getting faster every year.", "en"),
    ("Can you help me find the nearest subway station?", "en"),
    ("The conference will begin at nine o'clock tomorrow morning.", "en"),
    ("오늘 날씨가 정말 좋습니다.", "ko"),
    ("커피 한 잔 주문하고 싶습니다.", "ko"),
    ("가장 가까운 지하철역을 찾아주시겠습니까?", "ko"),
]

SAMPLE_RATE = 48000  # Qwen3-TTS output sample rate (48kHz decoder)


@dataclass
class RequestResult:
    """Result of a single TTS request."""
    text: str
    language: str
    total_time_s: float = 0.0
    ttfb_s: float = 0.0
    audio_duration_s: float = 0.0
    audio_bytes: int = 0
    rtf: float = 0.0
    error: str | None = None
    streaming: bool = False


@dataclass
class ConcurrencyResult:
    """Aggregated results for a concurrency level."""
    concurrency: int
    streaming: bool
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is None]

    @property
    def avg_total_time(self) -> float:
        s = self.successful
        return sum(r.total_time_s for r in s) / len(s) if s else 0

    @property
    def avg_ttfb(self) -> float:
        s = self.successful
        return sum(r.ttfb_s for r in s) / len(s) if s else 0

    @property
    def avg_rtf(self) -> float:
        s = self.successful
        return sum(r.rtf for r in s) / len(s) if s else 0

    @property
    def p95_total_time(self) -> float:
        s = self.successful
        if not s:
            return 0
        times = sorted(r.total_time_s for r in s)
        idx = int(len(times) * 0.95)
        return times[min(idx, len(times) - 1)]

    @property
    def wall_clock_s(self) -> float:
        """Total wall-clock time for all requests (set externally)."""
        return getattr(self, "_wall_clock", 0.0)

    @wall_clock_s.setter
    def wall_clock_s(self, v: float):
        self._wall_clock = v


def estimate_audio_duration(raw_bytes: bytes, sample_format: str) -> float:
    """Estimate audio duration from raw PCM bytes."""
    if sample_format == "f32le":
        n_samples = len(raw_bytes) // 4  # 4 bytes per float32
    else:  # s16le
        n_samples = len(raw_bytes) // 2  # 2 bytes per int16
    return n_samples / SAMPLE_RATE


async def request_streaming(
    client: httpx.AsyncClient, text: str, language: str,
    concurrency: int = 1, round_idx: int = 0, req_idx: int = 0,
    session_id: str | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure timing."""
    result = RequestResult(text=text, language=language, streaming=True)
    start = time.perf_counter()
    first_chunk_time = None
    audio_data = bytearray()

    payload = {"text": text, "language": language}
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        async with client.stream(
            "POST",
            "/synthesize/stream",
            json=payload,
        ) as response:
            response.raise_for_status()
            sample_format = response.headers.get("x-sample-format", "f32le")

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if not chunk:
                    continue
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                audio_data.extend(chunk)

        end = time.perf_counter()
        result.total_time_s = end - start
        result.ttfb_s = (first_chunk_time - start) if first_chunk_time else result.total_time_s
        result.audio_bytes = len(audio_data)
        result.audio_duration_s = estimate_audio_duration(bytes(audio_data), sample_format)
        result.rtf = result.total_time_s / result.audio_duration_s if result.audio_duration_s > 0 else 0
        if session_id is not None:
            label = f"s{round_idx}_{req_idx:02d}_{session_id}_stream_{language}_{text[:20]}_{result.total_time_s:.1f}s"
        else:
            label = f"c{concurrency}_r{round_idx}_stream_{language}_{text[:20]}_{result.total_time_s:.1f}s"
        _save_wav(bytes(audio_data), label, sample_format=sample_format)

    except Exception as e:
        result.error = str(e)
        result.total_time_s = time.perf_counter() - start

    return result


async def request_non_streaming(
    client: httpx.AsyncClient, text: str, language: str,
    concurrency: int = 1, round_idx: int = 0, req_idx: int = 0,
    session_id: str | None = None,
) -> RequestResult:
    """Send a non-streaming TTS request and measure timing."""
    result = RequestResult(text=text, language=language, streaming=False)
    start = time.perf_counter()

    payload = {"text": text, "language": language}
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        response = await client.post(
            "/synthesize",
            json=payload,
            headers={"Accept": "application/octet-stream"},
        )
        response.raise_for_status()

        end = time.perf_counter()
        sample_format = response.headers.get("x-sample-format", "f32le")
        result.total_time_s = end - start
        result.ttfb_s = result.total_time_s  # non-streaming: TTFB == total
        result.audio_bytes = len(response.content)
        result.audio_duration_s = estimate_audio_duration(response.content, sample_format)
        result.rtf = result.total_time_s / result.audio_duration_s if result.audio_duration_s > 0 else 0
        if session_id is not None:
            label = f"s{round_idx}_{req_idx:02d}_{session_id}_full_{language}_{text[:20]}_{result.total_time_s:.1f}s"
        else:
            label = f"c{concurrency}_r{round_idx}_full_{language}_{text[:20]}_{result.total_time_s:.1f}s"
        _save_wav(response.content, label, sample_format=sample_format)

    except Exception as e:
        result.error = str(e)
        result.total_time_s = time.perf_counter() - start

    return result


async def benchmark_concurrency(
    url: str, concurrency: int, streaming: bool, rounds: int = 2,
) -> ConcurrencyResult:
    """Run benchmark at a specific concurrency level."""
    cr = ConcurrencyResult(concurrency=concurrency, streaming=streaming)

    async with httpx.AsyncClient(
        base_url=url,
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:
        wall_start = time.perf_counter()

        for round_idx in range(rounds):
            # Pick sentences for this round
            sentences = []
            for i in range(concurrency):
                idx = (round_idx * concurrency + i) % len(TEST_SENTENCES)
                sentences.append(TEST_SENTENCES[idx])

            # Launch all requests concurrently
            if streaming:
                tasks = [request_streaming(client, text, lang, concurrency, round_idx, i)
                         for i, (text, lang) in enumerate(sentences)]
            else:
                tasks = [request_non_streaming(client, text, lang, concurrency, round_idx, i)
                         for i, (text, lang) in enumerate(sentences)]

            results = await asyncio.gather(*tasks)
            cr.results.extend(results)

        cr.wall_clock_s = time.perf_counter() - wall_start

    return cr


async def benchmark_sessions(
    url: str, num_sessions: int, sentences_per_session: int, streaming: bool,
) -> ConcurrencyResult:
    """Run session-aware benchmark: N concurrent sessions, each sending M sentences sequentially.

    This simulates the real E2E pipeline where same-session sentences are serial
    but different sessions run concurrently. Tests per-session decoder context caching.
    """
    import uuid

    cr = ConcurrencyResult(concurrency=num_sessions, streaming=streaming)

    async def _run_session(client: httpx.AsyncClient, session_idx: int):
        """One session: send sentences sequentially with same session_id."""
        sid = f"bench-{uuid.uuid4().hex[:8]}"
        results = []
        for sent_idx in range(sentences_per_session):
            text, lang = TEST_SENTENCES[(session_idx * sentences_per_session + sent_idx) % len(TEST_SENTENCES)]
            if streaming:
                r = await request_streaming(
                    client, text, lang, num_sessions, session_idx, sent_idx, session_id=sid,
                )
            else:
                r = await request_non_streaming(
                    client, text, lang, num_sessions, session_idx, sent_idx, session_id=sid,
                )
            label = "1st" if sent_idx == 0 else f"{sent_idx+1}th"
            status = "OK" if r.error is None else f"ERR: {r.error}"
            print(f"    session {session_idx} [{sid}] sent {label}: {r.total_time_s:.2f}s {status}")
            results.append(r)
        # Cleanup session
        try:
            await client.delete(f"/sessions/{sid}")
        except Exception:
            pass
        return results

    async with httpx.AsyncClient(
        base_url=url,
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:
        wall_start = time.perf_counter()
        session_tasks = [_run_session(client, i) for i in range(num_sessions)]
        all_session_results = await asyncio.gather(*session_tasks)
        for session_results in all_session_results:
            cr.results.extend(session_results)
        cr.wall_clock_s = time.perf_counter() - wall_start

    return cr


def print_results(cr: ConcurrencyResult):
    """Print results for a concurrency level."""
    mode = "streaming" if cr.streaming else "non-streaming"
    successful = cr.successful
    failed = [r for r in cr.results if r.error is not None]

    print(f"\n{'='*60}")
    print(f"  Concurrency: {cr.concurrency} ({mode})")
    print(f"  Requests: {len(cr.results)} total, {len(successful)} ok, {len(failed)} failed")
    print(f"  Wall clock: {cr.wall_clock_s:.2f}s")
    print(f"{'='*60}")

    if successful:
        print(f"  Avg total time:  {cr.avg_total_time:.3f}s")
        print(f"  Avg TTFB:        {cr.avg_ttfb:.3f}s")
        print(f"  Avg RTF:         {cr.avg_rtf:.3f}")
        print(f"  P95 total time:  {cr.p95_total_time:.3f}s")
        print(f"  Avg audio dur:   {sum(r.audio_duration_s for r in successful)/len(successful):.2f}s")
        print(f"  Avg audio size:  {sum(r.audio_bytes for r in successful)/len(successful)/1024:.1f} KB")

        # Per-request detail
        print(f"\n  {'Text':<45} {'Lang':>4} {'Time':>6} {'TTFB':>6} {'RTF':>5} {'Audio':>6}")
        print(f"  {'-'*45} {'-'*4} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")
        for r in successful:
            text_short = r.text[:42] + "..." if len(r.text) > 45 else r.text
            print(
                f"  {text_short:<45} {r.language:>4} "
                f"{r.total_time_s:>5.2f}s {r.ttfb_s:>5.2f}s "
                f"{r.rtf:>5.2f} {r.audio_duration_s:>5.2f}s"
            )

    if failed:
        print(f"\n  FAILURES:")
        for r in failed:
            print(f"    {r.text[:50]}: {r.error}")


async def main():
    parser = argparse.ArgumentParser(description="Concurrent TTS benchmark")
    parser.add_argument("--url", default=TTS_URL, help="TTS server URL")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrency to test")
    parser.add_argument("--rounds", type=int, default=2, help="Rounds per concurrency level")
    parser.add_argument("--streaming-only", action="store_true", help="Only test streaming")
    parser.add_argument("--non-streaming-only", action="store_true", help="Only test non-streaming")
    parser.add_argument("--sessions", type=int, default=0,
                        help="Run session benchmark: N concurrent sessions, each sending sentences sequentially")
    parser.add_argument("--sentences-per-session", type=int, default=4,
                        help="Sentences per session in session benchmark (default: 4)")
    args = parser.parse_args()

    if SAVE_AUDIO:
        print(f"Saving TTS audio to: {AUDIO_DIR.resolve()}  (set SAVE_AUDIO=0 to disable)")
    else:
        print("Audio saving disabled (set SAVE_AUDIO=1 to enable)")

    # Health check
    async with httpx.AsyncClient(base_url=args.url, timeout=10.0) as client:
        try:
            resp = await client.get("/health")
            health = resp.json()
            print(f"TTS server: {health}")
        except Exception as e:
            print(f"ERROR: Cannot reach TTS server at {args.url}: {e}")
            return

    # Warmup (single request to ensure CUDA graphs are captured)
    print("\nWarming up...")
    async with httpx.AsyncClient(base_url=args.url, timeout=60.0) as client:
        await request_non_streaming(client, "Hello world.", "en")
    print("Warmup complete.\n")

    # Session benchmark mode
    if args.sessions > 0:
        for streaming in [True, False]:
            if args.non_streaming_only and streaming:
                continue
            if args.streaming_only and not streaming:
                continue
            mode = "streaming" if streaming else "non-streaming"
            print(f"\nSession benchmark: {args.sessions} sessions x {args.sentences_per_session} sentences ({mode})...")
            cr = await benchmark_sessions(
                args.url, args.sessions, args.sentences_per_session, streaming=streaming,
            )
            print_results(cr)
        print("\nSession benchmark complete.")
        return

    concurrency_levels = [1, 2, 4, 8]
    concurrency_levels = [c for c in concurrency_levels if c <= args.max_concurrent]

    all_results: list[ConcurrencyResult] = []

    for c in concurrency_levels:
        if not args.non_streaming_only:
            print(f"\nBenchmarking: {c} concurrent (streaming)...")
            cr = await benchmark_concurrency(args.url, c, streaming=True, rounds=args.rounds)
            print_results(cr)
            all_results.append(cr)

        if not args.streaming_only:
            print(f"\nBenchmarking: {c} concurrent (non-streaming)...")
            cr = await benchmark_concurrency(args.url, c, streaming=False, rounds=args.rounds)
            print_results(cr)
            all_results.append(cr)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Mode':<20} {'Conc':>4} {'Avg Time':>9} {'Avg TTFB':>9} {'Avg RTF':>8} {'P95':>8}")
    print(f"  {'-'*20} {'-'*4} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for cr in all_results:
        mode = "streaming" if cr.streaming else "non-streaming"
        if cr.successful:
            print(
                f"  {mode:<20} {cr.concurrency:>4} "
                f"{cr.avg_total_time:>8.3f}s {cr.avg_ttfb:>8.3f}s "
                f"{cr.avg_rtf:>8.3f} {cr.p95_total_time:>7.3f}s"
            )

    # Scaling analysis
    print(f"\n  Scaling factor (vs 1 concurrent):")
    baseline_streaming = next((cr for cr in all_results if cr.streaming and cr.concurrency == 1), None)
    baseline_non_streaming = next((cr for cr in all_results if not cr.streaming and cr.concurrency == 1), None)

    for cr in all_results:
        baseline = baseline_streaming if cr.streaming else baseline_non_streaming
        if baseline and baseline.avg_total_time > 0 and cr.successful:
            mode = "streaming" if cr.streaming else "non-streaming"
            factor = cr.avg_total_time / baseline.avg_total_time
            print(f"    {mode:<20} {cr.concurrency:>2}x concurrent = {factor:.2f}x latency")


if __name__ == "__main__":
    asyncio.run(main())
