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
        use_py_session=True,
        max_new_tokens=96,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=5,
        max_batch_size=1,
        initial_prompt=None,
        flush_mode="default",
        min_phrase_chars=15,
        min_sentence_chars=6,
        stability_count=2,
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
        self.initial_prompt = initial_prompt or ""
        self.prev_transcript = ""  # updated after each clear_buffer for context continuity
        self.eos = False
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size

        # Flush mode: "default" = standard WhisperLive, "korean" = grammar-based,
        # "punctuation" = language-agnostic punctuation detection
        self.flush_mode = flush_mode
        self._detector = None
        if flush_mode in ("korean", "korean_sermon"):  # korean_sermon kept for backward compat
            import os
            from whisper_live.korean_endings import KoreanEndingDetector
            extra_markers_str = os.environ.get("STT_EXTRA_FLUSH_MARKERS", "")
            extra_markers = {m.strip() for m in extra_markers_str.split(",") if m.strip()} if extra_markers_str else set()
            self._detector = KoreanEndingDetector(
                min_phrase_chars=min_phrase_chars,
                min_sentence_chars=min_sentence_chars,
                stability_count=stability_count,
                extra_flush_markers=extra_markers,
            )
            self._last_flushed_text = ""  # dedup: skip re-transcription of just-flushed text
        elif flush_mode == "punctuation":
            from whisper_live.korean_endings import PunctuationFlushDetector
            self._detector = PunctuationFlushDetector()
            self._last_flushed_text = ""
            self._last_flush_was_phrase = False  # track consecutive phrases for trimming
            logging.info(f"[{client_uid}] Korean grammar flush mode enabled (extra markers: {extra_markers or 'none'})")

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

    def _build_text_prefix(self):
        """Build decoder prompt with vocab anchoring + transcript context.

        Whisper's initial_prompt mechanism: previous text is prepended with
        <|startofprev|> token so the decoder has linguistic context after
        a buffer clear. Vocab (initial_prompt) gets a reserved 200-char
        budget for entity anchoring; prev_transcript fills the remainder
        up to ~400 chars total (~200 tokens, within decoder limit of 224).
        """
        task_tokens = f"<|startoftranscript|><|{self.language}|><|{self.task}|><|notimestamps|>"

        # Always include initial_prompt (vocab) for entity anchoring,
        # then append prev_transcript for linguistic continuity.
        # Vocab gets a reserved budget so it's never fully displaced.
        max_chars = 400
        vocab = self.initial_prompt[:200] if self.initial_prompt else ""
        transcript = self.prev_transcript[-(max_chars - len(vocab)):] if self.prev_transcript else ""
        context = f"{vocab} {transcript}".strip() if (vocab or transcript) else ""

        if context:
            return f"<|startofprev|>{context}{task_tokens}"
        return task_tokens

    def clear_buffer(self, seq=None, context=None):
        """Override to save transcript context before clearing.

        If context is provided by the client, use it (more accurate than
        current_out which may already be stale). Otherwise fall back to
        current_out.
        """
        with self.lock:
            if context:
                self.prev_transcript = context
            else:
                self.prev_transcript = self.current_out.strip() if self.current_out else ""
                if not self.prev_transcript and self.text:
                    self.prev_transcript = " ".join(self.text).strip()
        if self._detector:
            self._detector.reset()
        super().clear_buffer(seq=seq)

    def set_eos(self, eos):
        self.lock.acquire()
        self.eos = eos
        self.lock.release()

    def handle_transcription_output(self, last_segment, duration):
        if self._detector is not None:
            self._handle_korean_flush(last_segment, duration)
        elif self.flush_mode == "turn_based":
            self._handle_turn_based(last_segment, duration)
        else:
            segments = self.prepare_segments({"text": last_segment})
            self.send_transcription_to_client(segments)
            if self.eos:
                self.update_timestamp_offset(last_segment, duration)

    def _handle_korean_flush(self, last_segment, duration):
        """Handle transcription output with Korean phrase/sentence detection.

        Uses last_segment directly as the full transcript text (TRT backend
        re-transcribes the entire buffer each pass). The detector tracks
        flushed_len to only flush new text.
        """
        full_text = last_segment.strip() if isinstance(last_segment, str) else ""
        if not full_text:
            return

        # After _internal_trim, Whisper re-transcribes remaining audio which may
        # reproduce the just-flushed text. Skip if unflushed portion
        # is just the old text being re-transcribed.
        # Safety: clear dedup after 3 seconds to prevent permanent blocking.
        if self._last_flushed_text:
            if time.monotonic() - self._detector._last_flush_time > 3.0:
                self._last_flushed_text = ""
            else:
                unflushed_check = full_text[self._detector.flushed_len:].strip()
                last_clean = self._last_flushed_text.rstrip(".!? ")
                if not unflushed_check or unflushed_check.rstrip(".!? ") == last_clean:
                    return
                self._last_flushed_text = ""

        decision = self._detector.check(full_text)

        # Track flushed position for partial computation below.
        # _internal_trim() resets detector.flushed_len to 0 for the next pass,
        # but we need the post-flush value to slice THIS pass's full_text.
        partial_offset = self._detector.flushed_len

        if decision.flush_type == "none" and len(full_text) > 40:
            unflushed = full_text[self._detector.flushed_len:]
            # Log complete tokens found
            tokens = self._detector._extract_complete_tokens(unflushed, full_text)
            token_strs = [(t, p) for t, p in tokens[:5]]
            logging.warning(
                f"[{self.client_uid}] NO FLUSH: flushed_len={self._detector.flushed_len}, "
                f"text_len={len(full_text)}, tokens={token_strs}, "
                f"stability={self._detector._stable_count}/{self._detector.stability_count}, "
                f"prev_ending='{self._detector._prev_ending}', "
                f"dedup={self._last_flushed_text[:20] if self._last_flushed_text else 'none'}"
            )

        if decision.flush_type in ("phrase", "sentence"):
            flush_text = decision.text.strip()
            if flush_text:
                segment = self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    flush_text,
                    completed=True,
                )
                segment["flush_type"] = decision.flush_type
                try:
                    self.websocket.send(json.dumps({
                        "uid": self.client_uid,
                        "segments": [segment],
                    }))
                except Exception as e:
                    logging.error(f"[ERROR]: Sending flushed segment: {e}")

                self._detector.on_flushed(decision.flush_type, decision.end_pos)
                partial_offset = self._detector.flushed_len
                logging.info(
                    f"[{self.client_uid}] {decision.flush_type} flush: "
                    f"{decision.reason} ({len(flush_text)} chars)"
                )

                # Trim processed audio to prevent unbounded buffer growth.
                # Sentence: always trim (sentence is complete)
                # Phrase: trim if previous was also a phrase (consecutive phrases)
                if decision.flush_type == "sentence":
                    self.timestamp_offset += duration
                    self._internal_trim()
                    self._last_flush_was_phrase = False
                    self._last_flushed_text = flush_text
                elif decision.flush_type == "phrase":
                    if self._last_flush_was_phrase:
                        # Consecutive phrase — trim to prevent unbounded growth
                        self.timestamp_offset += duration
                        self._internal_trim()
                        self._last_flushed_text = flush_text
                    self._last_flush_was_phrase = True

        # Send the unflushed portion as a partial
        unflushed = full_text[partial_offset:].strip()
        if unflushed:
            segment = self.format_segment(
                self.timestamp_offset,
                self.timestamp_offset + duration,
                unflushed,
                completed=False,
            )
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "segments": [segment],
                }))
            except Exception as e:
                logging.error(f"[ERROR]: Sending partial segment: {e}")

        if self.eos:
            self.update_timestamp_offset(last_segment, duration)

    def _handle_turn_based(self, last_segment, duration):
        """Handle transcription for turn-based chat mode.

        Sends partials while the speaker talks. When EOS (end of speech)
        is detected via server-side VAD, marks the full text as completed
        and clears the buffer for the next turn.
        """
        full_text = last_segment.strip() if isinstance(last_segment, str) else ""
        if not full_text:
            return

        if self.eos:
            # Speaker stopped — flush entire transcript as completed
            segment = self.format_segment(
                self.timestamp_offset,
                self.timestamp_offset + duration,
                full_text,
                completed=True,
            )
            segment["flush_type"] = "turn"
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "segments": [segment],
                }))
            except Exception as e:
                logging.error(f"[ERROR]: Sending turn segment: {e}")
            self.update_timestamp_offset(last_segment, duration)
            self._internal_trim()
            logging.info(f"[{self.client_uid}] Turn flush ({len(full_text)} chars)")
        else:
            # Still speaking — send as partial
            segment = self.format_segment(
                self.timestamp_offset,
                self.timestamp_offset + duration,
                full_text,
                completed=False,
            )
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "segments": [segment],
                }))
            except Exception as e:
                logging.error(f"[ERROR]: Sending partial segment: {e}")

    def _internal_trim(self):
        """Trim already-processed audio after a sentence flush.

        Zero audio loss — only removes audio before timestamp_offset.
        No epoch increment so in-flight inference results stay valid.
        Dedup in _handle_korean_flush prevents re-flushing the same text.
        """
        with self.lock:
            if self.frames_np is None:
                return
            processed_samples = int((self.timestamp_offset - self.frames_offset) * self.RATE)
            if processed_samples > 0 and processed_samples < self.frames_np.shape[0]:
                kept_duration = (self.frames_np.shape[0] - processed_samples) / self.RATE
                self.frames_np = self.frames_np[processed_samples:]
                self.frames_offset = self.timestamp_offset
                self._chunk_boundaries = [
                    (s, off - processed_samples)
                    for s, off in self._chunk_boundaries
                    if off >= processed_samples
                ]
                logging.info(
                    f"[{self.client_uid}] Internal trim: kept {kept_duration:.1f}s"
                )
            elif processed_samples >= self.frames_np.shape[0]:
                self.frames_np = None
                self.frames_offset = 0.0
                self.timestamp_offset = 0.0
                self._chunk_boundaries = []

            # Reset transcript state — next pass builds fresh text
            self.transcript = []
            self.text = []
            self.current_out = ""
            self.prev_out = ""
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # Reset detector — trimmed buffer produces new text from position 0
        self._detector.flushed_len = 0
        self._detector._reset_stability()

    def transcribe_audio(self, input_bytes):
        # Batch inference path: submit to central queue and wait
        if ServeClientTensorRT.BATCH_WORKER is not None:
            from whisper_live.batch_inference_trt import TRTBatchRequest
            text_prefix = self._build_text_prefix()
            request = TRTBatchRequest(
                audio=input_bytes,
                language=self.language,
                task=self.task,
                use_vad=False,  # VAD already handled in speech_to_text loop
                text_prefix_override=text_prefix,
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
        text_prefix = self._build_text_prefix()
        last_segment = self.transcriber.transcribe(
            mel,
            text_prefix=text_prefix,
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
