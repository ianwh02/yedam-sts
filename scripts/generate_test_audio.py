#!/usr/bin/env python3
"""Generate test audio WAVs for STT benchmarking via Google TTS."""
from gtts import gTTS
from pydub import AudioSegment
import io

def generate_wav(text, lang, output_path):
    tts = gTTS(text=text, lang=lang)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    audio = AudioSegment.from_mp3(mp3_buf)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")
    dur = len(audio) / 1000.0
    print(f"  {output_path}: {dur:.1f}s")

print("Generating test audio...")

# Longer English (~10s)
generate_wav(
    "The annual artificial intelligence conference will be held in San Francisco next month. "
    "Researchers from around the world will present their latest findings on large language models, "
    "computer vision, and robotics.",
    "en",
    "debug_audio/bench_long_en.wav",
)

# Short Korean (~3-5s)
generate_wav(
    "오늘 날씨가 정말 좋습니다. 산책하기 좋은 날이에요.",
    "ko",
    "debug_audio/bench_short_ko.wav",
)

# Longer Korean (~8-10s)
generate_wav(
    "인공지능 기술이 빠르게 발전하고 있습니다. 특히 대규모 언어 모델의 성능이 크게 향상되면서 "
    "다양한 분야에서 활용되고 있습니다. 앞으로 더 많은 혁신이 기대됩니다.",
    "ko",
    "debug_audio/bench_long_ko.wav",
)

print("Done!")
