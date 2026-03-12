#!/usr/bin/env python3
"""End-to-end integration test for yedam-sts pipeline.

Requires all services running:
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up

Usage:
  python scripts/test_e2e.py                    # test everything
  python scripts/test_e2e.py --test health      # just health checks
  python scripts/test_e2e.py --test services    # test individual services
  python scripts/test_e2e.py --test pipeline    # test full pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time

import httpx
import numpy as np

# ============================================================
# Config — matches docker-compose.dev.yml exposed ports
# ============================================================

ORCHESTRATOR_URL = "http://localhost:8080"
STT_WS_URL = "ws://localhost:9090"
LLM_URL = "http://localhost:8000"
TTS_URL = "http://localhost:7860"

# ============================================================
# Helpers
# ============================================================


def generate_sine_wave(
    duration_s: float = 3.0,
    freq_hz: float = 440.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a sine wave as Float32 PCM bytes (matches pipeline input format)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return audio.tobytes()


def generate_silence(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate silence as Float32 PCM bytes."""
    samples = int(sample_rate * duration_s)
    return b"\x00" * (samples * 4)  # 4 bytes per float32


def ok(msg: str):
    print(f"  ✓ {msg}")


def fail(msg: str):
    print(f"  ✗ {msg}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Test: Health Checks
# ============================================================


async def test_health():
    """Verify all services are up and responding to health checks."""
    section("Health Checks")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Orchestrator
        try:
            resp = await client.get(f"{ORCHESTRATOR_URL}/health")
            data = resp.json()
            if resp.status_code == 200:
                ok(f"Orchestrator: {data.get('status', 'unknown')}")
                services = data.get("services", {})
                for name, status in services.items():
                    if "healthy" in status:
                        ok(f"  → {name}: {status}")
                    else:
                        fail(f"  → {name}: {status}")
            else:
                fail(f"Orchestrator: HTTP {resp.status_code}")
        except Exception as e:
            fail(f"Orchestrator: {e}")
            return False

        # STT (direct)
        try:
            resp = await client.get(f"http://localhost:9090/health", timeout=5.0)
            ok(f"STT direct: HTTP {resp.status_code}")
        except Exception as e:
            fail(f"STT direct: {e}")

        # LLM (direct)
        try:
            resp = await client.get(f"{LLM_URL}/health", timeout=5.0)
            ok(f"LLM direct: HTTP {resp.status_code}")
        except Exception as e:
            fail(f"LLM direct: {e}")

        # TTS (direct)
        try:
            resp = await client.get(f"{TTS_URL}/health", timeout=5.0)
            data = resp.json()
            gpu = data.get("gpu", False)
            model = data.get("model", "unknown")
            ok(f"TTS direct: model={model}, gpu={gpu}")
        except Exception as e:
            fail(f"TTS direct: {e}")

    return True


# ============================================================
# Test: Individual Services
# ============================================================


async def test_services():
    """Test each service independently to isolate issues."""
    section("Individual Service Tests")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # --- LLM: simple completion ---
        print("\n  LLM (vLLM) — chat completion:")
        try:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "messages": [
                        {"role": "system", "content": "Translate Korean to English."},
                        {"role": "user", "content": "안녕하세요"},
                    ],
                    "max_tokens": 50,
                    "temperature": 0.3,
                },
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            ok(f"LLM response: {content[:80]}")
        except Exception as e:
            fail(f"LLM: {e}")

        # --- TTS: synthesize ---
        print("\n  TTS (Qwen3-TTS) — synthesis:")
        try:
            start = time.time()
            resp = await client.post(
                f"{TTS_URL}/synthesize",
                json={"text": "Hello, this is a test.", "language": "en"},
            )
            elapsed = time.time() - start
            if resp.status_code == 200:
                wav_size = len(resp.content)
                duration_ms = resp.headers.get("X-Audio-Duration-Ms", "?")
                rtf = resp.headers.get("X-RTF", "?")
                ok(f"TTS synthesized: {wav_size}B, duration={duration_ms}ms, rtf={rtf}, gen={elapsed:.2f}s")
            else:
                fail(f"TTS: HTTP {resp.status_code} — {resp.text[:100]}")
        except Exception as e:
            fail(f"TTS: {e}")

        # --- STT: WebSocket connection test ---
        print("\n  STT (WhisperLive) — WebSocket connection:")
        try:
            import websockets

            async with websockets.connect(STT_WS_URL) as ws:
                config = {
                    "uid": "test-e2e",
                    "language": "ko",
                    "task": "transcribe",
                    "model": "large-v3",
                    "use_vad": True,
                }
                await ws.send(json.dumps(config))
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get("message") == "SERVER_READY":
                    ok("STT connected and SERVER_READY received")

                    # Send a short sine wave to verify audio processing
                    audio = generate_sine_wave(duration_s=2.0)
                    chunk_size = 16000 * 4  # 1 second of float32 at 16kHz
                    for i in range(0, len(audio), chunk_size):
                        await ws.send(audio[i : i + chunk_size])
                        await asyncio.sleep(0.1)

                    # Wait for any transcription response
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        segments = json.loads(resp).get("segments", [])
                        ok(f"STT responded with {len(segments)} segment(s)")
                    except asyncio.TimeoutError:
                        ok("STT connected (no transcription from sine wave, as expected)")
                else:
                    fail(f"STT unexpected response: {data}")
        except ImportError:
            fail("STT: websockets package not installed (pip install websockets)")
        except Exception as e:
            fail(f"STT: {e}")


# ============================================================
# Test: Full Pipeline
# ============================================================


async def test_pipeline():
    """Test the full pipeline: create session → feed audio → verify flow."""
    section("Full Pipeline Test")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Create session
        print("\n  Step 1: Create session")
        try:
            resp = await client.post(
                f"{ORCHESTRATOR_URL}/api/sessions",
                json={
                    "source_lang": "ko",
                    "target_lang": "en",
                    "processor": "translation",
                },
            )
            if resp.status_code != 200:
                fail(f"Create session: HTTP {resp.status_code} — {resp.text[:100]}")
                return
            session_data = resp.json()
            session_id = session_data["session_id"]
            admin_ws_url = session_data["admin_ws_url"]
            ok(f"Session created: {session_id}")
            ok(f"  admin_ws: {admin_ws_url}")
        except Exception as e:
            fail(f"Create session: {e}")
            return

        # 2. Check session exists via health
        resp = await client.get(f"{ORCHESTRATOR_URL}/health")
        active = resp.json().get("active_sessions", 0)
        ok(f"Active sessions: {active}")

        # 3. Connect admin WebSocket and feed audio
        print("\n  Step 2: Connect admin WebSocket and feed audio")
        try:
            import websockets

            ws_url = f"ws://localhost:8080{admin_ws_url}"
            async with websockets.connect(ws_url) as ws:
                ok(f"Admin WebSocket connected to {ws_url}")

                # Send 3 seconds of sine wave audio in chunks
                audio = generate_sine_wave(duration_s=3.0, freq_hz=440.0)
                chunk_size = 16000 * 4  # 1 second chunks
                chunks_sent = 0

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    await ws.send(chunk)
                    chunks_sent += 1
                    await asyncio.sleep(0.5)

                ok(f"Sent {chunks_sent} audio chunks ({len(audio)} bytes total)")

                # Give pipeline time to process
                await asyncio.sleep(2.0)

                # Send stop command
                await ws.send(json.dumps({"type": "stop"}))
                ok("Sent stop command")

        except ImportError:
            fail("websockets package not installed (pip install websockets)")
            return
        except Exception as e:
            fail(f"Admin WebSocket: {e}")
            return

        # 4. Verify session was cleaned up
        await asyncio.sleep(1.0)
        resp = await client.get(f"{ORCHESTRATOR_URL}/health")
        active = resp.json().get("active_sessions", 0)
        ok(f"Active sessions after stop: {active}")

        # 5. Test session deletion (create another and delete via REST)
        print("\n  Step 3: Test REST session lifecycle")
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/api/sessions",
            json={"source_lang": "ko", "target_lang": "en"},
        )
        session_id_2 = resp.json()["session_id"]
        ok(f"Created session: {session_id_2}")

        resp = await client.delete(f"{ORCHESTRATOR_URL}/api/sessions/{session_id_2}")
        if resp.status_code == 200:
            ok(f"Deleted session: {session_id_2}")
        else:
            fail(f"Delete session: HTTP {resp.status_code}")

        # Delete non-existent session
        resp = await client.delete(f"{ORCHESTRATOR_URL}/api/sessions/nonexistent")
        if resp.status_code == 404:
            ok("404 for non-existent session (correct)")
        else:
            fail(f"Expected 404, got {resp.status_code}")

    print("\n" + "=" * 60)
    print("  Pipeline test complete!")
    print("  Note: callback outputs (STT/translation/TTS) are logged")
    print("  server-side. Check orchestrator container logs:")
    print("    docker logs yedam-orchestrator")
    print("=" * 60)


# ============================================================
# Main
# ============================================================


async def main():
    parser = argparse.ArgumentParser(description="E2E test for yedam-sts pipeline")
    parser.add_argument(
        "--test",
        choices=["health", "services", "pipeline", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    print("yedam-sts E2E Integration Test")
    print(f"Orchestrator: {ORCHESTRATOR_URL}")

    if args.test in ("health", "all"):
        healthy = await test_health()
        if args.test == "health":
            return

    if args.test in ("services", "all"):
        await test_services()
        if args.test == "services":
            return

    if args.test in ("pipeline", "all"):
        await test_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
