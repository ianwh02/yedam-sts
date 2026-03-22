#!/usr/bin/env python3
"""Live mic → translation → speaker demo for yedam-sts.

Streams microphone audio through the full STS pipeline:
  Mic (Korean) → STT → LLM translation → TTS → Speaker (English)

Requirements:
  pip install sounddevice numpy websockets httpx

Usage:
  python scripts/live_demo.py
  python scripts/live_demo.py --source en --target ko
  python scripts/live_demo.py --list-devices
  python scripts/live_demo.py --input-device 3 --output-device 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import numpy as np

logger = logging.getLogger("live_demo")

ORCHESTRATOR_URL = "http://localhost:8080"
ORCHESTRATOR_WS = "ws://localhost:8080"
SAMPLE_RATE_IN = 16000       # STT expects 16kHz float32
SAMPLE_RATE_TTS = 24000      # TTS outputs 24kHz s16le
SAMPLE_RATE_OUT = 48000      # Speaker output (universally supported)
CHUNK_DURATION_S = 0.1       # 100ms mic chunks
CHANNELS = 1


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\nDefault input:  {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")


async def run_pipeline(args):
    import sounddevice as sd
    import httpx
    import websockets

    # ── Create session ──
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/api/sessions",
            json={
                "source_lang": args.source,
                "target_lang": args.target,
                "processor": "translation",
            },
        )
        resp.raise_for_status()
        session = resp.json()

    session_id = session["session_id"]
    admin_ws_url = f"{ORCHESTRATOR_WS}{session['admin_ws_url']}"
    listener_ws_url = f"{ORCHESTRATOR_WS}{session['listener_ws_url']}"

    print(f"Session: {session_id}")
    print(f"Pipeline: {args.source} → {args.target}")
    print(f"Mic: {sd.query_devices(args.input_device, 'input')['name']}")
    print(f"Speaker: {sd.query_devices(args.output_device, 'output')['name']}")
    print("─" * 50)
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    audio_out_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # ── Mic input → admin WebSocket ──
    async def feed_audio():
        chunk_samples = int(SAMPLE_RATE_IN * CHUNK_DURATION_S)
        loop = asyncio.get_event_loop()

        try:
            async with websockets.connect(admin_ws_url) as ws:
                mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

                def mic_callback(indata, frames, time_info, status):
                    if status:
                        logger.warning(f"Mic: {status}")
                    loop.call_soon_threadsafe(mic_queue.put_nowait, indata.copy().tobytes())

                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE_IN,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=chunk_samples,
                    device=args.input_device,
                    callback=mic_callback,
                )
                stream.start()

                try:
                    while True:
                        pcm = await mic_queue.get()
                        await ws.send(pcm)
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop()
                    stream.close()
        except Exception as e:
            logger.error(f"feed_audio: {e}")

    # ── Listener WebSocket → text display + speaker ──
    async def receive_output():
        try:
            async with websockets.connect(listener_ws_url) as ws:
                # Accumulate translation tokens per segment
                translation_buf: dict[int, str] = {}

                async for msg in ws:
                    if isinstance(msg, bytes):
                        # TTS audio: raw s16le PCM at 24kHz
                        await audio_out_queue.put(msg)
                    else:
                        try:
                            data = json.loads(msg)
                            msg_type = data.get("type", "")

                            if msg_type == "stt_partial":
                                text = data.get("text", "")
                                print(f"  [{args.source}] {text}...    ", end="\r")

                            elif msg_type == "stt_final":
                                text = data.get("text", "")
                                print(f"  [{args.source}] {text}          ")

                            elif msg_type == "translation_partial":
                                seg_idx = data.get("segment_index", 0)
                                token = data.get("token", "")
                                translation_buf.setdefault(seg_idx, "")
                                translation_buf[seg_idx] += token
                                print(f"  [{args.target}] {translation_buf[seg_idx]}...", end="\r")

                            elif msg_type == "translation_final":
                                text = data.get("text", "")
                                seg_idx = data.get("segment_index", 0)
                                translation_buf.pop(seg_idx, None)
                                print(f"  [{args.target}] {text}          ")
                                print()

                            elif msg_type == "session_started":
                                logger.info("Listener connected to session")

                        except json.JSONDecodeError:
                            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"receive_output: {e}")

    # ── Audio playback ──
    async def play_audio():
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE_OUT,
            channels=CHANNELS,
            dtype="float32",
            device=args.output_device,
        )
        stream.start()

        try:
            while True:
                chunk = await audio_out_queue.get()
                # TTS sends s16le at 24kHz — convert to float32 and resample to 48kHz
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                if len(samples) > 0:
                    # Simple 2x upsample (24kHz → 48kHz) via linear interpolation
                    indices = np.linspace(0, len(samples) - 1, len(samples) * 2)
                    resampled = np.interp(indices, np.arange(len(samples)), samples)
                    stream.write(resampled.reshape(-1, 1).astype(np.float32))
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()

    # ── Run all tasks ──
    tasks = [
        asyncio.create_task(feed_audio()),
        asyncio.create_task(receive_output()),
        asyncio.create_task(play_audio()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.delete(f"{ORCHESTRATOR_URL}/api/sessions/{session_id}")
                print(f"\nSession {session_id} cleaned up.")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Live mic → translation → speaker demo")
    parser.add_argument("--source", default="ko", help="Source language (default: ko)")
    parser.add_argument("--target", default="en", help="Target language (default: en)")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_devices:
        list_devices()
        return

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("Missing dependency: pip install sounddevice")
        sys.exit(1)

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
