#!/usr/bin/env python3
"""E2E test — streams audio through STT → LLM translation pipeline.

Displays Korean STT transcription and English LLM translation side-by-side.

Usage:
  python scripts/test_e2e.py --desktop           # desktop audio, VAD flushing + translation
  python scripts/test_e2e.py                      # mic input
  python scripts/test_e2e.py --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time

import httpx
import numpy as np

logger = logging.getLogger("test_e2e")

STT_WS_URL = "ws://localhost:9090"
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
SAMPLE_RATE = 16000
CHUNK_DURATION_S = 0.1
CHANNELS = 1


def list_devices():
    import sounddevice as sd
    print("=== sounddevice ===")
    print(sd.query_devices())
    print(f"\nDefault input: {sd.default.device[0]}")
    # Also show PipeWire/PulseAudio monitor sources for desktop capture
    print("\n=== PulseAudio monitor sources (for --desktop) ===")
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if "monitor" in line.lower():
                print(f"  {line}")
    except Exception:
        print("  (pactl not available)")


def _create_audio_source(args) -> tuple:
    """Create an audio source — returns (start_fn, stop_fn, queue, description).

    For mic: uses sounddevice InputStream
    For desktop: uses parec to capture system audio monitor
    """
    loop = asyncio.get_event_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)

    if args.desktop:
        # Desktop audio capture via PulseAudio/PipeWire monitor
        source = args.monitor_source
        if not source:
            # Auto-detect: find the running monitor source
            try:
                result = subprocess.run(
                    ["pactl", "list", "short", "sources"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2 and "monitor" in parts[1] and "RUNNING" in line:
                        source = parts[1]
                        break
                if not source:
                    # Fallback: first monitor source
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split("\t")
                        if len(parts) >= 2 and "monitor" in parts[1]:
                            source = parts[1]
                            break
            except Exception:
                pass
            if not source:
                raise RuntimeError("No PulseAudio monitor source found. Use --monitor-source to specify.")

        proc = None

        def start():
            nonlocal proc
            # parec outputs raw audio: s16le, mono, 16kHz
            proc = subprocess.Popen(
                [
                    "parec",
                    "--device", source,
                    "--format=float32le",
                    "--channels=1",
                    "--rate=16000",
                    "--latency-msec=100",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Read in a background thread
            import threading

            def reader():
                chunk_bytes = chunk_samples * 4  # float32 = 4 bytes
                while proc and proc.poll() is None:
                    data = proc.stdout.read(chunk_bytes)
                    if data:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, data)

            t = threading.Thread(target=reader, daemon=True)
            t.start()

        def stop():
            nonlocal proc
            if proc:
                proc.terminate()
                proc = None

        return start, stop, audio_queue, f"Desktop ({source})"

    else:
        # Mic capture via sounddevice
        import sounddevice as sd

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic: {status}")
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy().tobytes())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_samples,
            device=args.input_device,
            callback=mic_callback,
        )

        device_name = sd.query_devices(args.input_device, 'input')['name']
        return stream.start, lambda: (stream.stop(), stream.close()), audio_queue, f"Mic ({device_name})"


async def run_raw_stt(args):
    """Stream audio → STT and display raw partials/completions."""
    import websockets

    start_audio, stop_audio, audio_queue, source_desc = _create_audio_source(args)

    print("=== Raw STT Test ===")
    print(f"STT: {STT_WS_URL}")
    print(f"Source: {source_desc}")
    print(f"Language: {args.language}")
    print("─" * 50)
    print("Press Ctrl+C to stop.\n")

    async with websockets.connect(STT_WS_URL) as ws:
        config = {
            "uid": "stt-test",
            "language": args.language,
            "task": "transcribe",
            "model": "large-v3",
            "use_vad": True,
            "initial_prompt": args.initial_prompt,
        }
        await ws.send(json.dumps(config))
        ready = await ws.recv()
        data = json.loads(ready)
        if data.get("message") != "SERVER_READY":
            print(f"Server not ready: {data}")
            return
        print("Server ready.\n")

        completed_count = 0
        segment_num = 0

        async def send_audio():
            while True:
                pcm = await audio_queue.get()
                await ws.send(pcm)

        async def receive_transcripts():
            nonlocal completed_count, segment_num
            async for message in ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                if data.get("message") == "buffer_cleared":
                    print("  [buffer cleared]")
                    continue

                segments = data.get("segments", [])
                for i, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    completed = seg.get("completed", False)
                    if not text:
                        continue

                    if completed and i >= completed_count:
                        completed_count = i + 1
                        segment_num += 1
                        print(f"\n  ✓ COMPLETED #{segment_num}: {text}")
                        print()
                    elif not completed:
                        elapsed = time.strftime("%H:%M:%S")
                        print(f"  [{elapsed}] partial: {text}    ", end="\r")

        start_audio()
        try:
            await asyncio.gather(
                asyncio.create_task(send_audio()),
                asyncio.create_task(receive_transcripts()),
            )
        except asyncio.CancelledError:
            pass
        finally:
            stop_audio()


async def run_with_flushing(args):
    """Stream audio → STT with client-side VAD flushing."""
    import websockets
    import torch

    start_audio, stop_audio, audio_queue, source_desc = _create_audio_source(args)

    print("=== STT + VAD Flushing Test ===")
    print(f"STT: {STT_WS_URL}")
    print(f"Source: {source_desc}")
    print(f"Language: {args.language}")
    print(f"VAD pause: {args.vad_pause}s, Min speech: {args.min_speech}s, Max speech: {args.max_speech}s")
    print("─" * 50)
    print("Press Ctrl+C to stop.\n")

    # ── Load Silero VAD ──
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', model='silero_vad',
        trust_repo=True, verbose=False,
    )
    vad_model.eval()

    loop = asyncio.get_event_loop()
    last_partial = ""
    flush_num = 0
    ignore_until_cleared = False
    recent_pairs: list[dict] = []  # rolling context: [{ko: ..., en: ...}, ...]
    CONTEXT_WINDOW = 3  # include last N translations as context
    completed_count = 0
    translate_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    http_client = httpx.AsyncClient(timeout=30.0)

    # VAD state
    is_speaking = False
    speech_start_time = 0.0
    silence_start_time = 0.0
    vad_pause_duration = args.vad_pause      # seconds of silence to trigger flush
    min_speech_duration = args.min_speech     # minimum speech duration to flush (avoid noise)
    max_speech_duration = args.max_speech     # window fallback: force flush after this long without pause

    transcript_path = f"stt_transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    transcript_file = open(transcript_path, "w", encoding="utf-8")
    print(f"Transcript: {transcript_path}")

    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 120

    def clear_line():
        print(f"\r{' ' * terminal_width}\r", end="", flush=True)

    def do_flush(text: str, reason: str):
        nonlocal flush_num
        flush_num += 1
        clear_line()
        print(f"  [ko] {text}")
        transcript_file.write(f"[ko] {text}\n")
        transcript_file.flush()
        translate_queue.put_nowait((flush_num, text))

    def _lift_ignore():
        nonlocal ignore_until_cleared
        if ignore_until_cleared:
            ignore_until_cleared = False

    # Queue for VAD-triggered flush signals
    flush_signal: asyncio.Queue[str] = asyncio.Queue()

    ws_ref = [None]

    async with websockets.connect(STT_WS_URL) as ws:
        ws_ref[0] = ws
        config = {
            "uid": "stt-vad-test",
            "language": args.language,
            "task": "transcribe",
            "model": "large-v3",
            "use_vad": True,
            "initial_prompt": args.initial_prompt,
        }
        await ws.send(json.dumps(config))
        ready = await ws.recv()
        data = json.loads(ready)
        if data.get("message") != "SERVER_READY":
            print(f"Server not ready: {data}")
            return
        print("Server ready.\n")

        async def send_audio():
            """Send audio to Whisper + run client-side VAD."""
            nonlocal is_speaking, speech_start_time, silence_start_time

            while True:
                pcm = await audio_queue.get()
                await ws.send(pcm)

                # Run VAD on this chunk
                audio_tensor = torch.frombuffer(pcm, dtype=torch.float32).clone()
                # Silero VAD expects 16kHz, 512 samples (32ms) chunks
                # Our chunks are 1600 samples (100ms) — process in 512-sample windows
                chunk_has_speech = False
                for offset in range(0, len(audio_tensor) - 511, 512):
                    window = audio_tensor[offset:offset + 512]
                    confidence = vad_model(window, SAMPLE_RATE).item()
                    if confidence > 0.5:
                        chunk_has_speech = True
                        break

                now = time.monotonic()

                if chunk_has_speech:
                    if not is_speaking:
                        is_speaking = True
                        speech_start_time = now
                    silence_start_time = 0.0

                    # Window fallback: force flush if speaking too long without pause
                    speech_duration = now - speech_start_time
                    if speech_duration >= max_speech_duration:
                        speech_start_time = now  # reset for next window
                        await flush_signal.put(f"window {speech_duration:.1f}s continuous speech")
                else:
                    if is_speaking and silence_start_time == 0.0:
                        silence_start_time = now
                    elif is_speaking and silence_start_time > 0:
                        silence_duration = now - silence_start_time
                        speech_duration = silence_start_time - speech_start_time

                        if silence_duration >= vad_pause_duration and speech_duration >= min_speech_duration:
                            # Pause detected after sufficient speech → signal flush
                            is_speaking = False
                            silence_start_time = 0.0
                            await flush_signal.put(f"pause {silence_duration:.1f}s after {speech_duration:.1f}s speech")

        async def receive_transcripts():
            nonlocal completed_count, last_partial, ignore_until_cleared
            async for message in ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                msg_type = data.get("type", data.get("message", ""))
                if msg_type in ("buffer_cleared", "buffer_trimmed"):
                    ignore_until_cleared = False
                    clear_line()
                    trimmed = data.get("trimmed_seconds", "")
                    label = f"trimmed {trimmed:.1f}s" if trimmed else "cleared"
                    print(f"  [{label}]")
                    continue

                if ignore_until_cleared:
                    continue

                segments = data.get("segments", [])
                for i, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    completed = seg.get("completed", False)
                    if not text:
                        continue

                    if completed and i >= completed_count:
                        completed_count = i + 1
                        do_flush(text, "whisper completed")
                        last_partial = ""
                        completed_count = 0
                        ignore_until_cleared = True
                        await ws.send(json.dumps({"type": "clear_buffer"}))
                        loop.call_later(2.0, _lift_ignore)
                    else:
                        last_partial = text
                        max_display = terminal_width - 12
                        display = text if len(text) <= max_display else text[:max_display - 3] + "..."
                        clear_line()
                        print(f"  partial: {display}", end="", flush=True)

        async def flush_on_vad():
            """Wait for VAD flush signals and flush the latest transcript."""
            nonlocal last_partial, ignore_until_cleared
            while True:
                reason = await flush_signal.get()
                if last_partial and not ignore_until_cleared:
                    do_flush(last_partial, reason)
                    last_partial = ""
                    ignore_until_cleared = True
                    await ws.send(json.dumps({"type": "clear_buffer"}))
                    loop.call_later(2.0, _lift_ignore)

        async def translate_worker():
            """Consume flushed Korean text and translate via LLM."""
            while True:
                num, korean_text = await translate_queue.get()
                try:
                    system = (
                        "/no_think\n"
                        "You are a real-time Korean to English translator for a church sermon. "
                        "Translate the following Korean text into natural, fluent English. "
                        "Output ONLY the English translation, no explanations.\n\n"
                        "Important: The Korean text comes from live speech recognition which may contain errors. "
                        "Apply these corrections:\n"
                        "- The speaker frequently code-switches to English mid-sentence. "
                        "Garbled Korean that sounds like English phrases should be interpreted as English "
                        "(e.g. '에이블' = 'able', '히미' = 'Him', '나와 투' = 'now to', '후이즈' = 'who is').\n"
                        "- Fix obvious STT mishearings based on sermon context "
                        "(e.g. '쓰레기' in a sermon context likely means something else, "
                        "'구이의 이불' likely means 'who is able').\n"
                        "- '능히 하신다' = 'He is able' (key sermon phrase).\n"
                        "- Maintain consistent terminology: 청년 = young adult/youth, "
                        "집사님 = deacon, 장로님 = elder, 목사님 = pastor."
                    )

                    # Build context from recent translations
                    if recent_pairs:
                        context = "\n".join(
                            f"Korean: {p['ko']}\nEnglish: {p['en']}"
                            for p in recent_pairs[-CONTEXT_WINDOW:]
                        )
                        system += f"\n\nRecent context for continuity:\n{context}"

                    # Estimate input tokens and cap output to fit model context.
                    # Korean: ~1.5 tokens/char, English: ~1.3 tokens/char
                    input_chars = len(system) + len(korean_text)
                    est_input_tokens = int(input_chars * 1.5)
                    max_model_len = 2048
                    max_output = min(256, max_model_len - est_input_tokens - 50)
                    max_output = max(max_output, 64)  # floor

                    t0 = time.monotonic()
                    resp = await http_client.post(
                        f"{LLM_URL}/v1/chat/completions",
                        json={
                            "model": LLM_MODEL,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": korean_text},
                            ],
                            "max_tokens": max_output,
                            "temperature": 0.3,
                            **LLM_EXTRA_BODY,
                        },
                    )
                    elapsed = (time.monotonic() - t0) * 1000

                    data = resp.json()
                    translation = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    translation = _THINK_RE.sub("", translation).strip()

                    if translation:
                        clear_line()
                        print(f"  [en] {translation}  ({elapsed:.0f}ms)")
                        transcript_file.write(f"[en] {translation}\n")
                        transcript_file.flush()
                        # Add to rolling context
                        recent_pairs.append({"ko": korean_text, "en": translation})
                        if len(recent_pairs) > CONTEXT_WINDOW * 2:
                            recent_pairs.pop(0)

                except Exception as e:
                    clear_line()
                    print(f"  [en] ERROR: {e}")

        start_audio()
        try:
            await asyncio.gather(
                asyncio.create_task(send_audio()),
                asyncio.create_task(receive_transcripts()),
                asyncio.create_task(flush_on_vad()),
                asyncio.create_task(translate_worker()),
            )
        except asyncio.CancelledError:
            pass
        finally:
            stop_audio()
            await http_client.aclose()
            transcript_file.close()
            print(f"\nTranscript saved to {transcript_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive STT test")
    parser.add_argument("--language", default="ko", help="STT language (default: ko)")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index (mic mode)")
    parser.add_argument("--desktop", action="store_true", help="Capture desktop/system audio instead of mic")
    parser.add_argument("--monitor-source", type=str, default=None, help="PulseAudio monitor source name (auto-detected if omitted)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--with-flushing", action="store_true", help="Enable VAD + window flushing")
    parser.add_argument("--vad-pause", type=float, default=0.2, help="Seconds of silence to trigger flush (default: 0.2)")
    parser.add_argument("--min-speech", type=float, default=0.5, help="Min speech duration before flush eligible (default: 0.5)")
    parser.add_argument("--max-speech", type=float, default=10.0, help="Max continuous speech before window flush (default: 10.0)")
    parser.add_argument("--initial-prompt", type=str,
                        default="다음은 한국어 기독교 교회 설교 음성입니다. "
                                "주요 단어: 능히 하신다, 하나님, 예수 그리스도, 성령님, 교회, 예담교회, 청년부, "
                                "말씀, 은혜, 구원, 십자가, 부활, 사명, 소명, 넘치도록, 역사하신다, "
                                "에베소서, 빌립보서, 로마서, 시편, 정체성, 헌신, 순종, "
                                "바울, 베드로, 모세, 다윗, 고난, 회복, 광야.",
                        help="Whisper initial prompt for domain vocabulary")
    parser.add_argument("-v", "--verbose", action="store_true")
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
        print("Missing: pip install sounddevice")
        sys.exit(1)

    try:
        if args.with_flushing:
            asyncio.run(run_with_flushing(args))
        else:
            asyncio.run(run_raw_stt(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
