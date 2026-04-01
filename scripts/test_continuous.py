"""Test continuous TTS streaming with voice clone mode.

Simulates realistic LLM output patterns: variable delays between segments,
mixed short/long sentences, and pauses mid-session.
"""
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import httpx
import websockets

TTS_URL = "http://localhost:7860"
OUTPUT_DIR = Path(__file__).parent.parent / "test_continuous_output"
REF_AUDIO_PATH = Path(__file__).parent.parent / "tts-server" / "ref_audio" / "en.wav"
REF_TEXT = (
    "Hello my name is Ian and I'm testing the voice cloning system "
    "the weather is quite nice and I hope to go for a walk later this afternoon "
    "this recording will help the system learn how I speak"
)
SESSION_ID = f"test-continuous-{int(time.time()) % 10000:04d}"

# Simulated LLM output: (text, delay_before_sending_seconds)
# 25 segments — mix of short, medium, long, questions, exclamations, and pauses
SEGMENTS = [
    ("Brothers and sisters, welcome to today's worship service.", 0),
    ("Let us open our hearts to the Lord.", 0.5),
    ("Yes. The passage we will read today comes from Ephesians chapter three, verses twenty through twenty-one.", 3.0),
    # (combined with "Yes." above)
    ("Now to him who is able to do immeasurably more than all we ask or imagine.", 0.3),
    ("According to his power that is at work within us.", 0.2),
    ("To him be glory in the church and in Christ Jesus throughout all generations, forever and ever.", 0.8),
    ("Amen. Let us pray together.", 4.0),
    # (combined with "Amen." above)
    ("Thank you Lord for gathering us here today.", 0.5),
    ("We come before you with humble hearts.", 0.3),
    ("Guide us in your wisdom and fill us with your grace.", 0.4),
    ("Help us to love one another as you have loved us.", 0.5),
    ("Amen. Please turn to page forty-two in your hymn books.", 2.0),
    # (combined with "Amen." above)
    ("Today we will be singing Amazing Grace.", 0.3),
    ("How sweet the sound that saved a wretch like me.", 0.5),
    ("I once was lost, but now I am found.", 0.3),
    ("Was blind, but now I see.", 0.2),
    ("Can I get an amen?", 3.0),                            # question
    ("Hallelujah! The Lord has truly blessed us on this beautiful day.", 0.5),
    # (combined with "Hallelujah!" above)
    ("Before we close, I want to remind everyone about the fellowship dinner this Wednesday evening.", 0.5),
    ("Please bring a dish to share and join us for a time of community and thanksgiving.", 0.3),
    ("May God bless you and keep you until we meet again.", 1.0),
]


async def init_voice_clone():
    audio_bytes = REF_AUDIO_PATH.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:audio/wav;base64,{b64}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{TTS_URL}/sessions/init_voice",
            json={"session_id": SESSION_ID, "ref_audio_url": data_uri, "ref_text": REF_TEXT},
        )
        print(f"init_voice: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()


def save_wav(pcm_bytes: bytes, path: Path, sample_rate: int = 48000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    print(f"  Saved: {path.name} ({len(pcm_bytes) / (sample_rate * 2):.1f}s)")


async def test_continuous():
    OUTPUT_DIR.mkdir(exist_ok=True)
    url = f"ws://localhost:7860/synthesize/continuous/{SESSION_ID}"
    all_audio = bytearray()

    async with websockets.connect(url) as ws:
        text, _ = SEGMENTS[0]
        await ws.send(json.dumps({
            "type": "text", "text": text, "language": "en", "mode": "clone",
        }))
        session_start = time.time()
        print(f"[0.0s] Sent segment 1: {text[:50]}")

        msg = await ws.recv()
        print(f"Server: {msg}")

        segment_idx = 0
        audio_chunks = 0
        segment_audio = bytearray()
        segment_start = time.time()
        send_done = asyncio.Event()

        async def segment_sender():
            for i in range(1, len(SEGMENTS)):
                text, delay = SEGMENTS[i]
                if delay > 0:
                    await asyncio.sleep(delay)
                elapsed = time.time() - session_start
                await ws.send(json.dumps({"type": "text", "text": text}))
                print(f"[{elapsed:.1f}s] Queued segment {i + 1}: "
                      f"{'(after ' + f'{delay:.1f}s pause) ' if delay > 1 else ''}"
                      f"{text[:50]}{'...' if len(text) > 50 else ''}")
            send_done.set()

        sender_task = asyncio.create_task(segment_sender())

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
            except TimeoutError:
                elapsed = time.time() - session_start
                print(f"[{elapsed:.1f}s] Timeout")
                if segment_audio:
                    save_wav(bytes(segment_audio), OUTPUT_DIR / f"segment_{segment_idx + 1}.wav")
                break

            if isinstance(msg, bytes):
                audio_chunks += 1
                segment_audio.extend(msg)
                all_audio.extend(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "segment_boundary":
                    seg_duration = time.time() - segment_start
                    audio_duration = len(segment_audio) / (48000 * 2)
                    elapsed = time.time() - session_start
                    print(f"[{elapsed:.1f}s] Segment {segment_idx + 1}: "
                          f"{audio_chunks} chunks, {audio_duration:.1f}s audio, {seg_duration:.2f}s gen")
                    if segment_audio:
                        save_wav(bytes(segment_audio), OUTPUT_DIR / f"segment_{segment_idx + 1}.wav")

                    gap_samples = int(48000 * 0.4)
                    all_audio.extend(b'\x00' * (gap_samples * 2))

                    segment_idx += 1
                    audio_chunks = 0
                    segment_audio = bytearray()
                    segment_start = time.time()

                    if send_done.is_set() and segment_idx >= len(SEGMENTS):
                        await ws.send(json.dumps({"type": "end"}))
                        print(f"[{elapsed:.1f}s] All segments complete")
                        try:
                            while True:
                                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                                if isinstance(msg, bytes):
                                    all_audio.extend(msg)
                        except (TimeoutError, websockets.exceptions.ConnectionClosed):
                            pass
                        break

                elif data.get("type") == "error":
                    print(f"ERROR: {data.get('detail')}")
                    break

        sender_task.cancel()

        if all_audio:
            save_wav(bytes(all_audio), OUTPUT_DIR / "full_session.wav")
            total_duration = len(all_audio) / (48000 * 2)
            total_time = time.time() - session_start
            print(f"\nSession: {segment_idx} segments, {total_duration:.1f}s audio, "
                  f"{total_time:.1f}s total, RTF {total_time / total_duration:.2f}")


async def main():
    print(f"Session: {SESSION_ID}")
    print(f"Segments: {len(SEGMENTS)} sentences\n")
    await init_voice_clone()
    await test_continuous()


if __name__ == "__main__":
    asyncio.run(main())
