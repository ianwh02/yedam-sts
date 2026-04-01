#!/usr/bin/env python3
"""Clean reference audio for TTS voice cloning.

Trims silence, applies noise gate to suppress breaths/mic noise, normalizes loudness.
Run once on reference audio files before using them for voice cloning.

Usage:
    python scripts/clean_ref_audio.py tts-server/ref_audio/en.wav
    python scripts/clean_ref_audio.py tts-server/ref_audio/*.wav
    python scripts/clean_ref_audio.py tts-server/ref_audio/en.wav --output tts-server/ref_audio/en_clean.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def clean_ref_audio(
    audio: np.ndarray,
    sr: int,
    trim_threshold: float = 0.01,
    gate_threshold: float = 0.015,
    target_rms: float = 0.06,
) -> tuple[np.ndarray, int]:
    """Clean reference audio: trim silence, noise gate, normalize. Returns (audio, sample_rate)."""
    if len(audio) == 0:
        return audio, sr
    audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Resample to 24kHz (what the speech tokenizer encoder expects)
    target_sr = 24000
    if sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly
        up = target_sr // gcd(target_sr, sr)
        down = sr // gcd(target_sr, sr)
        audio = resample_poly(audio, up, down).astype(np.float32)
        print(f"  Resampled: {sr}Hz -> {target_sr}Hz")
        sr = target_sr

    original_len = len(audio) / sr

    # Trim leading/trailing silence (RMS-based, 20ms frames)
    frame_len = int(sr * 0.02)
    rms = np.array([
        np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, frame_len)
    ])
    above = np.where(rms > trim_threshold)[0]
    if len(above) > 0:
        start = max(0, above[0] * frame_len - frame_len)
        end = min(len(audio), (above[-1] + 2) * frame_len)
        audio = audio[start:end]

    trimmed_len = len(audio) / sr

    # Noise gate: suppress frames below threshold (breaths, mic noise)
    # Uses soft fade instead of hard cut for smooth transitions
    frame_len = int(sr * 0.01)  # 10ms frames
    gated_frames = 0
    total_frames = 0
    for i in range(0, len(audio) - frame_len, frame_len):
        frame_rms = np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        total_frames += 1
        if frame_rms < gate_threshold:
            audio[i:i + frame_len] *= (frame_rms / gate_threshold)
            gated_frames += 1

    # De-esser: compress energy in 4-8kHz sibilant range
    from scipy.fft import irfft, rfft
    from scipy.signal import butter, sosfilt

    frame_len_deess = int(sr * 0.02)  # 20ms frames
    deess_count = 0
    sibilant_lo = int(4000 * frame_len_deess / sr)
    sibilant_hi = int(8000 * frame_len_deess / sr)
    for i in range(0, len(audio) - frame_len_deess, frame_len_deess):
        frame = audio[i:i + frame_len_deess]
        spectrum = rfft(frame)
        sibilant_energy = np.sum(np.abs(spectrum[sibilant_lo:sibilant_hi]) ** 2)
        total_energy = np.sum(np.abs(spectrum) ** 2) + 1e-10
        if sibilant_energy / total_energy > 0.3:  # sibilant-heavy frame
            # Attenuate sibilant band by 6dB
            spectrum[sibilant_lo:sibilant_hi] *= 0.5
            audio[i:i + frame_len_deess] = irfft(spectrum, n=frame_len_deess)
            deess_count += 1

    # Gentle low-pass at 8kHz to smooth mic harshness
    lp_sos = butter(2, 8000, btype='low', fs=sr, output='sos')
    audio = sosfilt(lp_sos, audio).astype(np.float32)

    # Gentle normalization: nudge toward target_rms without aggressive scaling
    rms = np.sqrt(np.mean(audio ** 2))
    if target_rms > 0 and rms > 1e-6:
        gain = target_rms / rms
        gain = max(0.5, min(gain, 3.0))  # limit to 0.5x-3x range
        audio = audio * gain
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)

    print(f"  Original: {original_len:.1f}s")
    print(f"  Trimmed:  {trimmed_len:.1f}s (removed {original_len - trimmed_len:.1f}s silence)")
    print(f"  Gated:    {gated_frames}/{total_frames} frames suppressed")
    print(f"  De-essed: {deess_count} sibilant frames attenuated")
    print("  Low-pass: 8kHz applied")
    print(f"  RMS:      {np.sqrt(np.mean(audio ** 2)):.4f} (target {target_rms})")

    return audio, sr


def main():
    parser = argparse.ArgumentParser(description="Clean reference audio for TTS voice cloning")
    parser.add_argument("files", nargs="+", help="WAV files to clean")
    parser.add_argument("--output", "-o", help="Output path (only for single file, default: overwrite in-place)")
    parser.add_argument("--suffix", default="", help="Add suffix before extension (e.g. '_clean' → en_clean.wav)")
    parser.add_argument("--trim-threshold", type=float, default=0.01, help="RMS threshold for silence trimming")
    parser.add_argument("--gate-threshold", type=float, default=0.015, help="RMS threshold for noise gate")
    parser.add_argument("--target-rms", type=float, default=0.06, help="Target RMS for normalization (0 to disable)")
    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        print("Error: --output can only be used with a single file", file=sys.stderr)
        sys.exit(1)

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Skipping {path}: not found", file=sys.stderr)
            continue

        print(f"Processing: {path}")
        audio, sr = sf.read(str(path))

        cleaned, sr = clean_ref_audio(
            audio, sr,
            trim_threshold=args.trim_threshold,
            gate_threshold=args.gate_threshold,
            target_rms=args.target_rms,
        )

        if args.output:
            out_path = Path(args.output)
        elif args.suffix:
            out_path = path.with_stem(path.stem + args.suffix)
        else:
            out_path = path

        sf.write(str(out_path), cleaned, sr)
        print(f"  Saved:    {out_path}\n")


if __name__ == "__main__":
    main()
