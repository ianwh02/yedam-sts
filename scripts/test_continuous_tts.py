"""
Stress-test continuous TTS WebSocket to reproduce stall/timeout.

Sends 100+ text segments through the continuous WebSocket endpoint in voice clone mode.
Logs timing per segment to spot when/if the talker stalls.

Usage:
    python scripts/test_continuous_tts.py [--host localhost] [--port 7860] [--segments 100]
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import websockets

REF_AUDIO_PATH = Path(__file__).parent.parent / "tts-server" / "ref_audio" / "en.wav"
REF_TEXT = (
    "Hello my name is Ian and I'm testing the voice cloning system "
    "the weather is quite nice and I hope to go for a walk later this afternoon "
    "this recording will help the system learn how I speak"
)

# Sample English sentences (church sermon style, varying lengths)
SENTENCES = [
    "Therefore, I kneel before the Father.",
    "Do you remember how the prayer began?",
    "And Paul prays like this.",
    "It is not a hunting prayer, but a heartfelt one.",
    "If you do not look upon the God who gives you strength, you cannot endure.",
    "You cannot know the desperate prayer contained in these words.",
    "It is a prayer of desperate urgency that comes from the depths of the soul.",
    "Today's prayer is the last part of Paul's letter to the Ephesians.",
    "Looking at that, they seem to have overflowed with passion.",
    "But is that all there is to it?",
    "It's not just the outward appearance; what's inside?",
    "The church was going through a time of crisis and uncertainty.",
    "Young adults ask questions like, Who am I and what is my purpose?",
    "We need to understand the depth of God's love for us.",
    "This is the mystery that has been hidden for ages and generations.",
    "But now it has been revealed to the saints.",
    "To them God chose to make known the riches of this glory.",
    "Which is Christ in you, the hope of glory.",
    "We proclaim Him, admonishing every man and teaching every man.",
    "So that we may present every man complete in Christ.",
    "For this purpose I labor, striving according to His power.",
    "The power that works mightily within me.",
    "I want you to know how great a struggle I have for you.",
    "And for those who are at Laodicea.",
    "That their hearts may be encouraged, being knit together in love.",
    "To reach all the riches of full assurance of understanding.",
    "And the knowledge of God's mystery, that is, Christ Himself.",
    "In whom are hidden all the treasures of wisdom and knowledge.",
    "I say this so that no one will delude you with persuasive argument.",
    "For even though I am absent in body, I am with you in spirit.",
]


async def init_voice_clone(host: str, port: int, session_id: str):
    """Initialize voice clone prompt on the TTS server."""
    audio_bytes = REF_AUDIO_PATH.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:audio/wav;base64,{b64}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"http://{host}:{port}/sessions/init_voice",
            json={"session_id": session_id, "ref_audio_url": data_uri, "ref_text": REF_TEXT},
        )
        print(f"  init_voice: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()


async def run_test(host: str, port: int, num_segments: int, session_id: str):
    uri = f"ws://{host}:{port}/synthesize/continuous/{session_id}"
    print(f"Connecting to {uri}")
    print(f"Sending {num_segments} segments\n")

    # Init voice clone first
    print("Initializing voice clone...")
    await init_voice_clone(host, port, session_id)
    print()

    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        # Send initial text in clone mode
        first_sentence = SENTENCES[0]
        await ws.send(json.dumps({
            "type": "text",
            "text": first_sentence,
            "language": "en",
            "mode": "clone",
        }))
        print(f"  Sent initial: {first_sentence[:50]}")

        # Wait for "ready"
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", f"Expected 'ready', got {ready}"
        print(f"  Server ready\n")

        # Track stats
        segment_times = []
        audio_bytes_total = 0
        segment_start = time.monotonic()
        current_segment = 1
        segments_sent = 1  # initial text counts as segment 1

        async def send_segments():
            """Send remaining segments with a small delay between them."""
            nonlocal segments_sent
            for i in range(1, num_segments):
                sentence = SENTENCES[i % len(SENTENCES)]
                await ws.send(json.dumps({"type": "text", "text": sentence}))
                segments_sent += 1
                await asyncio.sleep(0.1)
            # Signal end
            await ws.send(json.dumps({"type": "end"}))

        sender = asyncio.create_task(send_segments())

        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=45.0)

                if isinstance(msg, bytes):
                    audio_bytes_total += len(msg)
                    continue

                data = json.loads(msg)
                if data["type"] == "segment_boundary":
                    elapsed = time.monotonic() - segment_start
                    segment_times.append(elapsed)
                    print(
                        f"  Segment #{current_segment:3d} done in {elapsed:6.2f}s "
                        f"(audio: {audio_bytes_total / 1024:.0f} KB, "
                        f"sent: {segments_sent}/{num_segments})"
                    )
                    current_segment += 1
                    segment_start = time.monotonic()

                    if current_segment > num_segments:
                        break

                elif data["type"] == "error":
                    print(f"\n  ERROR from server: {data.get('detail', data)}")
                    break

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - segment_start
            print(f"\n  TIMEOUT at segment #{current_segment} after {elapsed:.1f}s!")
        except websockets.ConnectionClosed as e:
            print(f"\n  Connection closed: {e}")
        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  Segments completed: {len(segment_times)} / {num_segments}")
        print(f"  Total audio: {audio_bytes_total / 1024:.0f} KB")
        if segment_times:
            print(f"  Avg segment time: {sum(segment_times) / len(segment_times):.2f}s")
            print(f"  Min: {min(segment_times):.2f}s  Max: {max(segment_times):.2f}s")

            slow = [(i + 1, t) for i, t in enumerate(segment_times) if t > 10.0]
            if slow:
                print(f"\n  SLOW segments (>10s):")
                for seg, t in slow:
                    print(f"    Segment #{seg}: {t:.2f}s")

        if len(segment_times) < num_segments:
            print(f"\n  *** STALL/ERROR at segment #{current_segment} ***")
        else:
            print(f"\n  All segments completed successfully!")
        print(f"{'=' * 60}")


async def run_parallel(host: str, port: int, num_segments: int, concurrency: int):
    """Run multiple concurrent sessions to stress the talker/predictor."""
    print(f"Running {concurrency} concurrent sessions, {num_segments} segments each\n")

    tasks = []
    for i in range(concurrency):
        session_id = f"test-stress-{int(time.time()) % 10000:04d}-{i}"
        tasks.append(run_test(host, port, num_segments, session_id))

    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description="Stress-test continuous TTS WebSocket")
    parser.add_argument("--host", default="localhost", help="TTS server host")
    parser.add_argument("--port", type=int, default=7860, help="TTS server port")
    parser.add_argument("--segments", type=int, default=100, help="Number of segments to send")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent sessions")
    parser.add_argument("--session-id", default=f"test-stress-{int(time.time()) % 10000:04d}",
                        help="Session ID (only used when concurrency=1)")
    args = parser.parse_args()

    if args.concurrency > 1:
        asyncio.run(run_parallel(args.host, args.port, args.segments, args.concurrency))
    else:
        asyncio.run(run_test(args.host, args.port, args.segments, args.session_id))


if __name__ == "__main__":
    main()
