"""
Batch inference scheduler for WhisperLive — TensorRT-LLM backend.

Mirrors the architecture of ``batch_inference.py`` (CTranslate2) but uses
TensorRT-LLM's ``ModelRunnerCpp.generate()`` for batched GPU execution.

Per-session threads submit ``TRTBatchRequest`` objects to a central queue;
a single dedicated worker thread collects pending requests within a time
window and runs them as a GPU batch.

For batch_size=1, falls back to ``WhisperTRTLLM.transcribe()`` for
identical behavior to the non-batched path.

Thread safety:
    - ``queue.Queue`` is stdlib thread-safe.
    - Each ``TRTBatchRequest.future`` (``threading.Event``) is written by the
      batch worker BEFORE ``.set()``, read by the session thread AFTER
      ``.wait()`` — no data race.
    - Only the batch worker thread touches the GPU model — zero lock
      contention between session threads.
"""

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from faster_whisper.vad import (
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

from whisper_live.transcriber.tensorrt_utils import (
    SAMPLE_RATE,
    N_SAMPLES,
    pad_or_trim,
)


@dataclass
class TRTBatchRequest:
    """A single inference request submitted by a session thread."""
    audio: np.ndarray
    language: str = "en"
    task: str = "transcribe"
    use_vad: bool = True
    vad_parameters: Optional[Dict] = None
    text_prefix_override: Optional[str] = None  # custom decoder prompt (e.g. with prev transcript context)
    # Signaling
    future: threading.Event = field(default_factory=threading.Event)
    # Results (filled by batch worker)
    result: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[Exception] = None


class BatchInferenceTRTWorker:
    """Central batch inference scheduler for the TensorRT-LLM backend."""

    def __init__(
        self,
        transcriber,
        max_batch_size: int = 8,
        batch_window_ms: int = 50,
        beam_size: int = 1,  # must match MAX_BEAM_WIDTH used at TRT engine build time
    ):
        self.transcriber = transcriber
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.beam_size = beam_size
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logging.info(
            f"[BatchInferenceTRT] Started (max_batch={self.max_batch_size}, "
            f"window={self.batch_window_ms}ms)"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: TRTBatchRequest):
        self._queue.put(request)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            batch: List[TRTBatchRequest] = []

            try:
                first = self._queue.get(timeout=0.5)
                batch.append(first)
            except queue.Empty:
                continue

            deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            try:
                self._process_batch(batch)
            except Exception as e:
                logging.error(f"[BatchInferenceTRT] Batch processing error: {e}")
                for req in batch:
                    if not req.future.is_set():
                        req.error = e
                        req.future.set()

    def _process_batch(self, batch: List[TRTBatchRequest]):
        if len(batch) == 1:
            self._process_single(batch[0])
            return

        logging.info(f"[BatchInferenceTRT] Processing batch of {len(batch)}")
        self._process_multi(batch)

    def _process_single(self, req: TRTBatchRequest):
        try:
            audio = req.audio

            if req.use_vad:
                vad_params = req.vad_parameters or {}
                vad_opts = VadOptions(**vad_params) if isinstance(vad_params, dict) else vad_params
                speech_chunks = get_speech_timestamps(audio, vad_opts)
                if speech_chunks:
                    audio_chunks, _ = collect_chunks(audio, speech_chunks)
                    audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else audio

            if audio.shape[0] == 0:
                req.result = ""
                req.duration = 0.0
                req.future.set()
                return

            duration = audio.shape[0] / SAMPLE_RATE
            mel, _ = self.transcriber.log_mel_spectrogram(audio)

            text_prefix = (
                f"<|startoftranscript|><|{req.language}|>"
                f"<|{req.task}|><|notimestamps|>"
            )
            result = self.transcriber.transcribe(mel, text_prefix=text_prefix, num_beams=self.beam_size)

            req.result = result or ""
            req.duration = duration
        except Exception as e:
            req.error = e
        finally:
            req.future.set()

    def _process_multi(self, batch: List[TRTBatchRequest]):
        preprocessed = []
        for req in batch:
            try:
                audio = req.audio

                if req.use_vad:
                    vad_params = req.vad_parameters or {}
                    vad_opts = VadOptions(**vad_params) if isinstance(vad_params, dict) else vad_params
                    speech_chunks = get_speech_timestamps(audio, vad_opts)
                    if speech_chunks:
                        audio_chunks, _ = collect_chunks(audio, speech_chunks)
                        audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else audio

                if audio.shape[0] == 0:
                    req.result = ""
                    req.duration = 0.0
                    req.future.set()
                    continue

                duration = audio.shape[0] / SAMPLE_RATE
                mel = self.transcriber.log_mel_spectrogram(
                    audio, return_duration=False
                )
                preprocessed.append((req, mel, duration))
            except Exception as e:
                req.error = e
                req.future.set()

        if not preprocessed:
            return

        try:
            prompt_tensors = []
            for req, mel, duration in preprocessed:
                if req.text_prefix_override:
                    text_prefix = req.text_prefix_override
                else:
                    text_prefix = (
                        f"<|startoftranscript|><|{req.language}|>"
                        f"<|{req.task}|><|notimestamps|>"
                    )
                prompt_id = self.transcriber.tokenizer.encode(
                    text_prefix,
                    allowed_special=set(
                        self.transcriber.tokenizer.special_tokens.keys()
                    ),
                )
                prompt_tensors.append(torch.tensor(prompt_id, dtype=torch.int32))

            mel_list = []
            mel_input_lengths = []
            for req, mel, duration in preprocessed:
                mel_padded = torch.nn.functional.pad(
                    mel, (0, 3000 - mel.shape[-1])
                ) if mel.shape[-1] < 3000 else mel[:, :3000]
                mel_input_lengths.append(min(mel.shape[-1], 3000))
                mel_list.append(
                    mel_padded.transpose(0, 1)
                    .type(torch.float16)
                    .contiguous()
                )

            mel_input_lengths_tensor = torch.tensor(
                mel_input_lengths, dtype=torch.int32, device="cuda"
            )

            if self.transcriber.use_py_session:
                mel_batch = torch.stack(
                    [m.transpose(0, 1) for m in mel_list]
                )
                decoder_input_ids = prompt_tensors[0].unsqueeze(0).repeat(
                    len(preprocessed), 1
                )
                encoder_output, encoder_output_lengths = (
                    self.transcriber.encoder.get_audio_features(
                        mel_batch.type(torch.float16).cuda(),
                        mel_input_lengths_tensor,
                    )
                )
                encoder_max_input_length = torch.max(
                    encoder_output_lengths
                ).item()
                output_ids = self.transcriber.decoder.generate(
                    decoder_input_ids,
                    encoder_output,
                    encoder_max_input_length,
                    encoder_output_lengths,
                    self.transcriber.tokenizer.eot,
                    max_new_tokens=96,
                    num_beams=self.beam_size,
                )
            else:
                with torch.no_grad():
                    outputs = self.transcriber.model_runner_cpp.generate(
                        batch_input_ids=prompt_tensors,
                        encoder_input_features=mel_list,
                        encoder_output_lengths=mel_input_lengths_tensor // 2,
                        max_new_tokens=96,
                        end_id=self.transcriber.tokenizer.eot,
                        pad_id=self.transcriber.tokenizer.eot,
                        num_beams=self.beam_size,
                        output_sequence_lengths=True,
                        return_dict=True,
                    )
                    torch.cuda.synchronize()
                    output_ids = outputs["output_ids"].cpu().numpy().tolist()

            for i, (req, mel, duration) in enumerate(preprocessed):
                try:
                    text = self.transcriber.tokenizer.decode(
                        output_ids[i][0]
                    ).strip()
                    text = re.sub(r"<\|.*?\|>", "", text).strip()

                    req.result = text
                    req.duration = duration
                except Exception as e:
                    req.error = e
                finally:
                    req.future.set()

        except Exception as e:
            logging.error(f"[BatchInferenceTRT] GPU batch error: {e}")
            for req, *_ in preprocessed:
                if not req.future.is_set():
                    req.error = e
                    req.future.set()
