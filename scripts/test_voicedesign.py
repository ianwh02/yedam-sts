"""Quick test: VoiceDesign mode on continuous WebSocket.

Prerequisites:
  1. TTS server running with VoiceDesign model loaded
     curl -X POST http://localhost:7860/swap_model -H "Content-Type: application/json" -d '{"model": "design"}'
  2. No init_voice needed for design mode — just send text
"""
import asyncio
import json
import sys
import wave

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "7860"

INSTRUCT = "A clear, well-paced, christian aged 50 male speaker"
SENTENCES = [
    "This is the first test sentence.",
    "The church was going through a time of growth.",
    "We are meant to be the instruments of God.",
]

async def run():
    session_id = "test-vd"
    uri = f"ws://{HOST}:{PORT}/synthesize/continuous/{session_id}"
    print(f"Connecting to {uri}")
    async with websockets.connect(uri) as ws:
        all_audio = b""
        for i, text in enumerate(SENTENCES):
            print(f"\n  Sending segment {i+1}: {text}")
            msg = {
                "type": "text",
                "text": text,
                "language": "en",
            }
            # First message includes mode + instruct
            if i == 0:
                msg["mode"] = "design"
                msg["instruct"] = INSTRUCT
            await ws.send(json.dumps(msg))
            # Collect audio until segment_boundary or timeout
            seg_audio = b""
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                except TimeoutError:
                    print("  Timeout waiting for audio")
                    break
                if isinstance(msg, bytes):
                    seg_audio += msg
                else:
                    data = json.loads(msg)
                    if data.get("type") == "ready":
                        continue  # skip ready message on first segment
                    print(f"    msg: {data}")
                    if data.get("type") in ("segment_boundary", "segment_done", "error", "done"):
                        break
            dur = len(seg_audio) / (48000 * 2)  # 48kHz, 16-bit (2 bytes/sample)
            print(f"  Segment {i+1}: {len(seg_audio)} bytes ({dur:.1f}s)")
            all_audio += seg_audio

        # Send end signal
        await ws.send(json.dumps({"type": "end"}))

        # Save as WAV for playback
        out = "test_voicedesign_output.wav"
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(48000)
            wf.writeframes(all_audio)
        print(f"\nSaved {out} ({len(all_audio)/(48000*2):.1f}s)")
        print("Listen to the WAV — if you hear the instruct text before each sentence, the bug is confirmed.")

asyncio.run(run())
