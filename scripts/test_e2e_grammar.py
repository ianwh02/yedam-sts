#!/usr/bin/env python3
"""E2E pipeline test — Audio → STT → LLM → TTS → Speaker.

Uses rule-based Korean grammar detection (phrase/sentence endings) for flushing
instead of VAD-only. VAD is demoted to a fallback for English code-switching.

Phrase endings → flush to LLM, keep STT buffer (Whisper retains context)
Sentence endings → flush to LLM + clear STT buffer (safe, sentence complete)

Usage:
  python scripts/test_e2e_grammar.py --desktop              # desktop audio
  python scripts/test_e2e_grammar.py                         # mic input
  python scripts/test_e2e_grammar.py --desktop --no-tts      # text only
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

logger = logging.getLogger("test_e2e_grammar")

STT_WS_URL = os.environ.get("STT_WS_URL", "ws://localhost:9090")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-AWQ")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:7860")
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
SAMPLE_RATE = 16000
SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_OUT = 48000
CHUNK_DURATION_S = 0.1
CHANNELS = 1
CONTEXT_WINDOW = 3


def list_devices():
    import sounddevice as sd
    print("=== sounddevice ===")
    print(sd.query_devices())
    print(f"\nDefault input: {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
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
    """Create an audio source — returns (start_fn, stop_fn, queue, description)."""
    loop = asyncio.get_event_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)

    if args.desktop:
        source = None
        if not source:
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
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split("\t")
                        if len(parts) >= 2 and "monitor" in parts[1]:
                            source = parts[1]
                            break
            except Exception:
                pass
            if not source:
                raise RuntimeError("No PulseAudio monitor source found.")

        proc = None

        def start():
            nonlocal proc
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
            import threading

            def reader():
                chunk_bytes = chunk_samples * 4
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


async def run_e2e(args):
    """Full E2E: Audio → STT (server-side Korean flush) → LLM → TTS → Speaker."""
    import websockets

    start_audio, stop_audio, audio_queue, source_desc = _create_audio_source(args)

    print("=== E2E Pipeline Test: Server-Side Korean Grammar Flushing ===")
    print(f"STT: {STT_WS_URL}")
    print(f"LLM: {LLM_URL} ({LLM_MODEL})")
    print(f"TTS: {TTS_URL}" if not args.no_tts else "TTS: disabled")
    print(f"Source: {source_desc}")
    print(f"Language: {args.source} → {args.target}")
    print(f"Flush mode: korean_sermon (server-side)")
    print("─" * 50)
    print("Press Ctrl+C to stop.\n")

    loop = asyncio.get_event_loop()
    last_partial = ""
    flush_num = 0
    send_seq = 0
    recent_pairs: list[dict] = []
    completed_count = 0
    translate_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    tts_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    audio_out_queue: asyncio.Queue[bytes] = asyncio.Queue()
    llm_client = httpx.AsyncClient(timeout=30.0)
    tts_client = httpx.AsyncClient(timeout=60.0)

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
        text = text.strip()
        if not text:
            return
        flush_num += 1
        clear_line()
        # Show flush type indicator
        marker = "★ SENTENCE" if "sentence" in reason else "★ PHRASE" if "phrase" in reason else "★"
        print(f"  {marker} {reason}: {text}")
        transcript_file.write(f"[{args.source}] {text}\n")
        transcript_file.flush()
        translate_queue.put_nowait((flush_num, text))

    async with websockets.connect(STT_WS_URL) as ws:
        config = {
            "uid": "e2e-test",
            "language": args.source,
            "task": "transcribe",
            "model": "large-v3",
            "use_vad": True,
            "initial_prompt": args.initial_prompt,
            "flush_mode": "korean_sermon",
        }
        await ws.send(json.dumps(config))
        ready = await ws.recv()
        data = json.loads(ready)
        if data.get("message") != "SERVER_READY":
            print(f"Server not ready: {data}")
            return
        print("Server ready.\n")

        async def send_audio():
            """Send audio to Whisper — no client-side flushing needed."""
            nonlocal send_seq
            while True:
                pcm = await audio_queue.get()
                await ws.send(pcm)
                send_seq += 1

        async def receive_transcripts():
            """Receive segments from server. Server handles phrase/sentence detection.

            completed=true segments are flushed to LLM.
            completed=false segments are displayed as partials.
            """
            nonlocal completed_count, last_partial
            async for message in ws:
                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                msg_type = data.get("type", data.get("message", ""))
                if msg_type in ("buffer_cleared", "buffer_trimmed", "final_before_clear"):
                    completed_count = 0
                    last_partial = ""
                    continue

                segments = data.get("segments", [])
                for seg in segments:
                    text = seg.get("text", "").strip()
                    completed = seg.get("completed", False)
                    flush_type = seg.get("flush_type", "")
                    if not text:
                        continue

                    if completed:
                        reason = f"{flush_type}" if flush_type else "completed"
                        do_flush(text, reason)
                        last_partial = ""
                    else:
                        last_partial = text
                        max_display = terminal_width - 12
                        display = text if len(text) <= max_display else text[:max_display - 3] + "..."
                        clear_line()
                        print(f"  partial: {display}", end="", flush=True)

        # Sentence boundary regex for splitting LLM streaming output
        _SENTENCE_END = re.compile(r'[.!?;:]\s|[.!?]$')

        async def translate_worker():
            """Stream LLM translation tokens, split at sentence boundaries, send to TTS."""
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

                    if recent_pairs:
                        context = "\n".join(
                            f"Korean: {p['ko']}\nEnglish: {p['en']}"
                            for p in recent_pairs[-CONTEXT_WINDOW:]
                        )
                        system += f"\n\nRecent context for continuity:\n{context}"

                    input_chars = len(system) + len(korean_text)
                    est_input_tokens = int(input_chars * 1.5)
                    max_model_len = 2048
                    max_output = min(256, max_model_len - est_input_tokens - 50)
                    max_output = max(max_output, 64)

                    t0 = time.monotonic()
                    full_translation = ""
                    sentence_buf = ""

                    async with llm_client.stream(
                        "POST",
                        f"{LLM_URL}/v1/chat/completions",
                        json={
                            "model": LLM_MODEL,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": korean_text},
                            ],
                            "max_tokens": max_output,
                            "temperature": 0.3,
                            "stream": True,
                            **LLM_EXTRA_BODY,
                        },
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                                delta = chunk["choices"][0].get("delta", {})
                                token = delta.get("content", "")
                                if not token:
                                    continue
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                            # Strip think tags from stream
                            full_translation += token
                            sentence_buf += token

                            # Display streaming tokens
                            clean = _THINK_RE.sub("", full_translation).strip()
                            if clean:
                                clear_line()
                                max_display = terminal_width - 8
                                display = clean if len(clean) <= max_display else "..." + clean[-(max_display - 3):]
                                print(f"  [{args.target}] {display}", end="", flush=True)

                            # Check for sentence boundary in buffer
                            clean_buf = _THINK_RE.sub("", sentence_buf).strip()
                            m = _SENTENCE_END.search(clean_buf)
                            if m and len(clean_buf) >= 15:
                                # Split at sentence boundary
                                split_pos = m.end()
                                sentence = clean_buf[:split_pos].strip()
                                sentence_buf = clean_buf[split_pos:]
                                if sentence and not args.no_tts:
                                    tts_queue.put_nowait((num, sentence))

                    elapsed = (time.monotonic() - t0) * 1000
                    translation = _THINK_RE.sub("", full_translation).strip()

                    # Flush remaining sentence buffer to TTS
                    remaining = _THINK_RE.sub("", sentence_buf).strip()
                    if remaining and not args.no_tts:
                        tts_queue.put_nowait((num, remaining))

                    if translation:
                        clear_line()
                        print(f"  [{args.target}] {translation}  ({elapsed:.0f}ms)")
                        transcript_file.write(f"[{args.target}] {translation}\n")
                        transcript_file.flush()
                        recent_pairs.append({"ko": korean_text, "en": translation})
                        if len(recent_pairs) > CONTEXT_WINDOW * 2:
                            recent_pairs.pop(0)

                except Exception as e:
                    clear_line()
                    print(f"  [{args.target}] ERROR: {e}")

        async def tts_worker():
            """Stream TTS audio and queue chunks for playback."""
            while True:
                num, english_text = await tts_queue.get()
                try:
                    t0 = time.monotonic()
                    total_bytes = 0

                    async with tts_client.stream(
                        "POST",
                        f"{TTS_URL}/synthesize/stream",
                        json={"text": english_text, "language": args.target, "session_id": "e2e-test"},
                    ) as response:
                        if response.status_code == 404:
                            # Streaming not available — fall back to non-streaming
                            pass
                        else:
                            response.raise_for_status()
                            sample_format = response.headers.get("x-sample-format", "f32le")
                            async for raw_chunk in response.aiter_bytes(chunk_size=4096):
                                if not raw_chunk:
                                    continue
                                if sample_format == "s16le":
                                    audio_bytes = raw_chunk
                                else:
                                    samples = np.frombuffer(raw_chunk, dtype=np.float32)
                                    audio_bytes = (samples.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()
                                total_bytes += len(audio_bytes)
                                await audio_out_queue.put(audio_bytes)

                            elapsed = (time.monotonic() - t0) * 1000
                            if total_bytes > 0:
                                clear_line()
                                print(f"  [tts] {total_bytes // 2 / SAMPLE_RATE_TTS:.1f}s audio  ({elapsed:.0f}ms stream)")
                            continue

                    # Fallback: non-streaming
                    resp = await tts_client.post(
                        f"{TTS_URL}/synthesize",
                        json={"text": english_text, "language": args.target, "session_id": "e2e-test"},
                        headers={"Accept": "application/octet-stream"},
                    )
                    resp.raise_for_status()
                    elapsed = (time.monotonic() - t0) * 1000
                    sample_format = resp.headers.get("x-sample-format", "f32le")
                    if sample_format == "s16le":
                        audio_bytes = resp.content
                    else:
                        samples = np.frombuffer(resp.content, dtype=np.float32)
                        audio_bytes = (samples.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    await audio_out_queue.put(audio_bytes)
                    clear_line()
                    print(f"  [tts] {len(audio_bytes) // 2 / SAMPLE_RATE_TTS:.1f}s audio  ({elapsed:.0f}ms)")

                except Exception as e:
                    clear_line()
                    print(f"  [tts] ERROR: {e}")

        async def play_audio():
            """Play TTS audio through speaker (or virtual sink for Discord)."""
            if args.tts_sink:
                # Use paplay to route to specific PipeWire/PulseAudio sink
                proc = await asyncio.create_subprocess_exec(
                    "paplay", "--raw", "--rate=24000", "--channels=1",
                    "--format=s16le", f"--device={args.tts_sink}",
                    stdin=asyncio.subprocess.PIPE,
                )
                try:
                    while True:
                        chunk = await audio_out_queue.get()
                        if proc.stdin and len(chunk) > 0:
                            proc.stdin.write(chunk)
                            await proc.stdin.drain()
                except asyncio.CancelledError:
                    pass
                finally:
                    if proc.stdin:
                        proc.stdin.close()
                    await proc.wait()
            else:
                import sounddevice as sd

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
                        # TTS sends s16le at 24kHz → float32 at 48kHz
                        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if len(samples) > 0:
                            # 2x upsample (24kHz → 48kHz) via linear interpolation
                            indices = np.linspace(0, len(samples) - 1, len(samples) * 2)
                            resampled = np.interp(indices, np.arange(len(samples)), samples)
                            stream.write(resampled.reshape(-1, 1).astype(np.float32))
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop()
                    stream.close()

        # ── Build task list ──
        tasks = [
            asyncio.create_task(send_audio()),
            asyncio.create_task(receive_transcripts()),
            asyncio.create_task(translate_worker()),
        ]
        if not args.no_tts:
            tasks.append(asyncio.create_task(tts_worker()))
            tasks.append(asyncio.create_task(play_audio()))

        start_audio()
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            stop_audio()
            await llm_client.aclose()
            await tts_client.aclose()
            transcript_file.close()
            print(f"\nTranscript saved to {transcript_path}")


def main():
    parser = argparse.ArgumentParser(description="E2E pipeline test: Audio → STT → LLM → TTS → Speaker")
    parser.add_argument("--source", default="ko", help="Source language (default: ko)")
    parser.add_argument("--target", default="en", help="Target language (default: en)")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index (mic mode)")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index (speaker)")
    parser.add_argument("--desktop", action="store_true", help="Capture desktop/system audio instead of mic")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS — text output only")
    parser.add_argument("--tts-sink", type=str, default=None, help="PipeWire/PulseAudio sink name for TTS output (e.g. tts_virtual_mic)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--initial-prompt", type=str,
                        default="다음은 한국어 기독교 교회 설교 음성입니다. "
                                "주요 단어: 능히 하신다, 하나님, 예수 그리스도, 성령님, 주님, 아버지 하나님, "
                                "교회, 예담교회, 청년부, 청년교회, 에베소, 에베소 교회, "
                                "말씀, 은혜, 구원, 십자가, 부활, 사명, 소명, 넘치도록, 역사하신다, "
                                "에베소서, 빌립보서, 로마서, 시편, 고린도서, "
                                "바울, 바울의 기도, 베드로, 모세, 다윗, 아브라함, "
                                "정체성, 헌신, 순종, 고난, 회복, 광야, 성장주기, "
                                "집사님, 목사님, 장로님, 전도사님, 성도, "
                                "교회 성장주기, 청년기, 어린이 시절, 청소년 시절, "
                                "무릎을 꿇다, 간절한 기도, 절박한 기도, "
                                "He is able, who is able, now to Him, amen.",
                        help="Whisper initial prompt for domain vocabulary")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.list_devices:
        list_devices()
        return

    if not args.desktop:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            print("Missing: pip install sounddevice")
            sys.exit(1)

    try:
        asyncio.run(run_e2e(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
