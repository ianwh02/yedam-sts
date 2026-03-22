import json
import logging
import threading
import time

from whisper_live.backend.base import ServeClientBase
from whisper_live.transcriber.transcriber_tensorrt import WhisperTRTLLM


class ServeClientTensorRT(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()
    BATCH_WORKER = None

    def __init__(
        self,
        websocket,
        task="transcribe",
        multilingual=False,
        language=None,
        client_uid=None,
        model=None,
        single_model=False,
        use_py_session=False,
        max_new_tokens=96,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=5,
        max_batch_size=1,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
        )

        self.language = language if multilingual else "en"
        self.task = task
        self.eos = False
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size

        if single_model:
            if ServeClientTensorRT.SINGLE_MODEL is None:
                self.create_model(model, multilingual, use_py_session=use_py_session)
                ServeClientTensorRT.SINGLE_MODEL = self.transcriber
            else:
                self.transcriber = ServeClientTensorRT.SINGLE_MODEL
        else:
            self.create_model(model, multilingual, use_py_session=use_py_session)

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "tensorrt"
        }))

    def create_model(self, model, multilingual, warmup=True, use_py_session=False):
        """
        Instantiates a new model, sets it as the transcriber and does warmup if desired.
        """
        import os
        assets_dir = os.environ.get("ASSETS_DIR", "/app/assets")
        self.assets_dir = assets_dir
        self.transcriber = WhisperTRTLLM(
            model,
            assets_dir=assets_dir,
            device="cuda",
            is_multilingual=multilingual,
            language=self.language,
            task=self.task,
            use_py_session=use_py_session,
            max_output_len=self.max_new_tokens,
            max_batch_size=self.max_batch_size,
        )
        if warmup:
            self.warmup()

    def warmup(self, warmup_steps=10):
        """
        Warmup TensorRT since first few inferences are slow.
        """
        import os
        warmup_audio = os.path.join(self.assets_dir, "jfk.flac")
        logging.info(f"[INFO:] Warming up TensorRT engine ({warmup_steps} steps)...")
        mel, _ = self.transcriber.log_mel_spectrogram(warmup_audio)
        for i in range(warmup_steps):
            self.transcriber.transcribe(mel)

    def set_eos(self, eos):
        self.lock.acquire()
        self.eos = eos
        self.lock.release()

    def handle_transcription_output(self, last_segment, duration):
        segments = self.prepare_segments({"text": last_segment})
        self.send_transcription_to_client(segments)
        if self.eos:
            self.update_timestamp_offset(last_segment, duration)

    def transcribe_audio(self, input_bytes):
        # Batch inference path: submit to central queue and wait
        if ServeClientTensorRT.BATCH_WORKER is not None:
            from whisper_live.batch_inference_trt import TRTBatchRequest
            request = TRTBatchRequest(
                audio=input_bytes,
                language=self.language,
                task=self.task,
                use_vad=False,  # VAD already handled in speech_to_text loop
            )
            ServeClientTensorRT.BATCH_WORKER.submit(request)
            request.future.wait(timeout=30)
            if request.error:
                raise request.error
            if request.result:
                self.handle_transcription_output(request.result, request.duration)
            return

        # Fallback: serial path with mutex
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.acquire()
        logging.info(f"[WhisperTensorRT:] Processing audio with duration: {input_bytes.shape[0] / self.RATE}")
        mel, duration = self.transcriber.log_mel_spectrogram(input_bytes)
        last_segment = self.transcriber.transcribe(
            mel,
            text_prefix=f"<|startoftranscript|><|{self.language}|><|{self.task}|><|notimestamps|>",
        )
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.release()
        if last_segment:
            self.handle_transcription_output(last_segment, duration)

    def update_timestamp_offset(self, last_segment, duration):
        if not len(self.transcript):
            self.transcript.append({"text": last_segment + " "})
        elif self.transcript[-1]["text"].strip() != last_segment:
            self.transcript.append({"text": last_segment + " "})

        with self.lock:
            self.timestamp_offset += duration

    def speech_to_text(self):
        last_transcribed_samples = 0

        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                time.sleep(0.02)
                continue

            self.clip_audio_if_no_valid_segment()

            epoch = self.buffer_epoch

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 0.5:
                time.sleep(0.1)
                continue

            # Skip re-transcription if no new audio has arrived since last pass
            n_samples = len(input_bytes)
            if n_samples == last_transcribed_samples:
                time.sleep(0.1)
                continue

            try:
                input_sample = input_bytes.copy()
                self.transcribe_audio(input_sample)

                # Discard result if buffer was cleared during transcription
                if self.buffer_epoch != epoch:
                    logging.info(f"[{self.client_uid}] Discarding stale TRT transcription")
                    last_transcribed_samples = 0  # reset on buffer clear
                    continue

                last_transcribed_samples = n_samples

            except Exception as e:
                logging.error(f"[ERROR]: {e}")
                time.sleep(0.01)
