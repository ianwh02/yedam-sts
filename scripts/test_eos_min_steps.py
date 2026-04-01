#!/usr/bin/env python3
"""Test TTS EOS behavior with short sentences.

Run with the TTS server already up at http://localhost:7860.
Tests a range of sentence lengths and reports whether audio was generated,
how many codec steps were used, and the audio duration.

Usage:
    python scripts/test_eos_min_steps.py [--url http://localhost:7860]
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import requests

TEST_SENTENCES = [
    # Very short (1 word) — these are the problem cases
    "Hello.",
    "Yes.",
    "No.",
    "Thanks.",
    "Hi.",
    # Short (2-3 words)
    "Good morning.",
    "Thank you.",
    "See you later.",
    "I agree.",
    # Medium (5-8 words)
    "Please have a seat over there.",
    "The weather is quite nice today.",
    "I hope you are doing well.",
    # Normal (10+ words)
    "The Lord is my shepherd, I shall not want.",
    "We are gathered here today to celebrate this joyful occasion together.",
]

SPEAKERS = ["Ryan"]  # Test with the speaker that has the issue


def test_sentence(url: str, text: str, speaker: str, language: str = "en") -> dict:
    """Send a synthesis request and analyze the result."""
    start = time.time()
    try:
        resp = requests.post(
            f"{url}/synthesize",
            json={"text": text, "language": language, "speaker": speaker},
            headers={"Accept": "application/octet-stream"},
            timeout=60,
        )
        elapsed = time.time() - start
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"text": text, "error": str(e), "elapsed": time.time() - start}

    raw_pcm = resp.content
    sample_rate = int(resp.headers.get("x-sample-rate", "48000"))
    audio_dur_ms = int(resp.headers.get("x-audio-duration-ms", "0"))
    rtf = resp.headers.get("x-rtf", "?")
    gen_time_ms = int(resp.headers.get("x-generation-time-ms", "0"))

    # Calculate from PCM if headers missing
    n_samples = len(raw_pcm) // 2  # s16le
    audio_dur_s = n_samples / sample_rate
    if audio_dur_ms == 0:
        audio_dur_ms = int(audio_dur_s * 1000)

    return {
        "text": text,
        "speaker": speaker,
        "audio_dur_ms": audio_dur_ms,
        "audio_dur_s": round(audio_dur_s, 2),
        "gen_time_ms": gen_time_ms,
        "rtf": rtf,
        "pcm_bytes": len(raw_pcm),
        "elapsed": round(elapsed, 2),
        "error": None,
        "raw_pcm": raw_pcm,
        "sample_rate": sample_rate,
    }


def save_wav(pcm_bytes: bytes, sample_rate: int, path: Path):
    """Save raw s16le PCM to WAV."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def test_sentence_stream(url: str, text: str, speaker: str, language: str = "en") -> dict:
    """Send a streaming synthesis request and collect all chunks."""
    start = time.time()
    try:
        resp = requests.post(
            f"{url}/synthesize/stream",
            json={"text": text, "language": language, "speaker": speaker},
            stream=True,
            timeout=60,
        )
        elapsed_first = None
        raw_pcm = b""
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if elapsed_first is None:
                    elapsed_first = time.time() - start
                raw_pcm += chunk
        elapsed = time.time() - start
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"text": text, "error": str(e), "elapsed": time.time() - start}

    sample_rate = int(resp.headers.get("x-sample-rate", "48000"))
    n_samples = len(raw_pcm) // 2
    audio_dur_s = n_samples / sample_rate

    return {
        "text": text,
        "speaker": speaker,
        "audio_dur_ms": int(audio_dur_s * 1000),
        "audio_dur_s": round(audio_dur_s, 2),
        "gen_time_ms": int(elapsed * 1000),
        "ttfa_ms": int(elapsed_first * 1000) if elapsed_first else 0,
        "rtf": f"{elapsed / audio_dur_s:.2f}" if audio_dur_s > 0 else "?",
        "pcm_bytes": len(raw_pcm),
        "elapsed": round(elapsed, 2),
        "error": None,
        "raw_pcm": raw_pcm,
        "sample_rate": sample_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Test TTS EOS behavior")
    parser.add_argument("--url", default="http://localhost:7860", help="TTS server URL")
    parser.add_argument("--save-dir", default="test_eos_output", help="Directory to save WAV files")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--stream", action="store_true", help="Test /synthesize/stream endpoint")
    args = parser.parse_args()

    # Check server health
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        health_data = health.json()
        if health_data.get("status") != "ok":
            print(f"Server not ready: {health_data}")
            sys.exit(1)
        print(f"Server healthy: {health_data}")
    except Exception as e:
        print(f"Cannot reach server at {args.url}: {e}")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    endpoint = "/synthesize/stream" if args.stream else "/synthesize"
    print(f"\n{'='*80}")
    print(f"Testing EOS behavior with {len(TEST_SENTENCES)} sentences")
    print(f"Server: {args.url}  Endpoint: {endpoint}")
    print(f"{'='*80}\n")

    test_fn = test_sentence_stream if args.stream else test_sentence

    results = []
    for speaker in SPEAKERS:
        for text in TEST_SENTENCES:
            word_count = len(text.split())
            expected_dur_s = word_count * 0.4  # ~2.5 words/sec

            result = test_fn(args.url, text, speaker, args.language)
            results.append(result)

            if result["error"]:
                status = "ERROR"
                detail = result["error"]
            else:
                audio_s = result["audio_dur_s"]
                # Flag suspicious results
                if audio_s < 0.2:
                    status = "PREMATURE_EOS"  # Too short — likely cut off
                elif audio_s > expected_dur_s * 4:
                    status = "RUNAWAY"  # Way too long — EOS missed
                elif audio_s > expected_dur_s * 2.5:
                    status = "LONG"  # Longer than expected
                else:
                    status = "OK"

                ttfa = f" ttfa={result['ttfa_ms']}ms" if result.get('ttfa_ms') else ""
                detail = f"audio={audio_s:.1f}s expected~{expected_dur_s:.1f}s gen={result['elapsed']}s rtf={result['rtf']}{ttfa}"

                # Save WAV
                safe_name = text[:30].replace(" ", "_").replace(".", "").replace(",", "")
                suffix = "_stream" if args.stream else ""
                wav_path = save_dir / f"{speaker}_{safe_name}{suffix}.wav"
                save_wav(result["raw_pcm"], result["sample_rate"], wav_path)

            flag = " <<<" if status in ("PREMATURE_EOS", "RUNAWAY") else ""
            print(f"  [{status:>14}] {speaker:>6} | {text:<55} | {detail}{flag}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    ok = sum(1 for r in results if not r["error"] and r["audio_dur_s"] >= 0.2)
    premature = sum(1 for r in results if not r["error"] and r["audio_dur_s"] < 0.2)
    runaway = sum(1 for r in results if not r["error"] and r["audio_dur_s"] > len(r["text"].split()) * 0.4 * 4)
    errors = sum(1 for r in results if r["error"])

    print(f"  OK:            {ok}")
    print(f"  PREMATURE_EOS: {premature}")
    print(f"  RUNAWAY:       {runaway}")
    print(f"  ERROR:         {errors}")
    print(f"\nWAV files saved to: {save_dir}/")

    if premature > 0:
        print("\n*** PREMATURE EOS DETECTED — eos_min_steps=0 causes early cutoff ***")
        print("    Consider a small min_steps value (e.g., 4) instead of 0.")
    elif runaway > 0:
        print("\n*** RUNAWAY GENERATION DETECTED — EOS not firing for some sentences ***")
    else:
        print("\n*** ALL CLEAN — eos_min_steps=0 appears safe! ***")


if __name__ == "__main__":
    main()
