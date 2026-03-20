# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union

import numpy as np
import soundfile
import av
import wave
import torch
import torch.nn.functional as F
from whisper_live.utils import resample


Pathlike = Union[str, Path]

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file, resample it, and read as a mono waveform.

    Parameters
    ----------
    file: str
        The audio file to open.

    sr: int
        The sample rate to resample the audio if necessary.

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    resampled_file = resample(file, sr)

    with wave.open(resampled_file, "rb") as wav_file:
        num_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(num_frames)

        audio_data = np.frombuffer(raw_data, dtype=np.int16)

    audio_data = audio_data.astype(np.float32) / 32768.0

    return audio_data


def load_audio_wav_format(wav_path):
    assert wav_path.endswith(
        '.wav'), f"Only support .wav format, but got {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device,
                n_mels: int,
                mel_filters_dir: str = None) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if mel_filters_dir is None:
        mel_filters_path = os.path.join(os.path.dirname(__file__), "assets",
                                        "mel_filters.npz")
    else:
        mel_filters_path = os.path.join(mel_filters_dir, "mel_filters.npz")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    return_duration: bool = False,
    mel_filters_dir: str = None,
):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            if audio.endswith('.wav'):
                audio, _ = load_audio_wav_format(audio)
            else:
                audio = load_audio(audio)
        assert isinstance(audio,
                          np.ndarray), f"Unsupported audio type: {type(audio)}"
        duration = audio.shape[-1] / SAMPLE_RATE
        audio = pad_or_trim(audio, N_SAMPLES)
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio,
                      N_FFT,
                      HOP_LENGTH,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = mel_filters(audio.device, n_mels, mel_filters_dir)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if return_duration:
        return log_spec, duration
    else:
        return log_spec
