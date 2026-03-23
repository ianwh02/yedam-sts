import os
import time
import threading
import queue
import json
import functools
import logging
import shutil
import tempfile
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, JSONResponse
import uvicorn
from faster_whisper import WhisperModel
import torch

from enum import Enum
from typing import List, Optional
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response
from websockets.datastructures import Headers
from whisper_live.vad import VoiceActivityDetector
from whisper_live.backend.base import ServeClientBase

logging.basicConfig(level=logging.INFO)
# Suppress noisy "connection rejected" logs from websockets for HTTP health/startup checks.
# We log these ourselves with path + status for consistency with TTS/LLM uvicorn format.
logging.getLogger("websockets.server").setLevel(logging.WARNING)

class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        if websocket in self.clients:
            return self.clients[websocket]
        return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT
    
    def is_openvino(self) -> bool:
        return self == BackendType.OPENVINO


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = None
        self.use_vad = True
        self.single_model = False
        self.vac_enabled = False

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
    ):
        client: Optional[ServeClientBase] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)
        
        # Create translation queue if translation is enabled
        translation_queue = None
        translation_client = None
        translation_thread = None
        
        if enable_translation:
            target_language = options.get("target_language", "fr")
            translation_queue = queue.Queue()
            from whisper_live.backend.translation_backend import ServeClientTranslation
            translation_client = ServeClientTranslation(
                client_uid=options["uid"],
                websocket=websocket,
                translation_queue=translation_queue,
                target_language=target_language,
                send_last_n_segments=options.get("send_last_n_segments", 10)
            )
            
            # Start translation thread
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text,
                daemon=True
            )
            translation_thread.start()
            
            logging.info(f"Translation enabled for client {options['uid']} with target language: {target_language}")

        if self.backend.is_tensorrt():
            try:
                from whisper_live.backend.trt_backend import ServeClientTensorRT
                trt_max_batch = (
                    self.batch_config.get("max_batch_size", 8)
                    if self.batch_config is not None
                    else 1
                )
                client = ServeClientTensorRT(
                    websocket,
                    multilingual=trt_multilingual,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=whisper_tensorrt_path,
                    single_model=self.single_model,
                    use_py_session=trt_py_session,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 5),
                    max_batch_size=trt_max_batch,
                    initial_prompt=options.get("initial_prompt"),
                    flush_mode=options.get("flush_mode", "default"),
                    min_phrase_chars=options.get("min_phrase_chars", 12),
                    min_sentence_chars=options.get("min_sentence_chars", 6),
                    stability_count=options.get("stability_count", 2),
                )
                logging.info("Running TensorRT backend.")

                # Start batch inference worker on first client (after model is loaded)
                if (self.batch_config is not None
                        and ServeClientTensorRT.BATCH_WORKER is None
                        and ServeClientTensorRT.SINGLE_MODEL is not None):
                    from whisper_live.batch_inference_trt import BatchInferenceTRTWorker
                    worker = BatchInferenceTRTWorker(
                        transcriber=ServeClientTensorRT.SINGLE_MODEL,
                        **self.batch_config,
                    )
                    worker.start()
                    ServeClientTensorRT.BATCH_WORKER = worker
            except Exception as e:
                logging.error(f"TensorRT-LLM not supported: {e}", exc_info=True)
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "TensorRT-LLM not supported on Server yet. "
                               "Reverting to available backend: 'faster_whisper'"
                }))
                self.backend = BackendType.FASTER_WHISPER
        
        if self.backend.is_openvino():
            try:
                from whisper_live.backend.openvino_backend import ServeClientOpenVINO
                client = ServeClientOpenVINO(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 5),
                )
                logging.info("Running OpenVINO backend.")
            except Exception as e:
                logging.error(f"OpenVINO not supported: {e}")
                self.backend = BackendType.FASTER_WHISPER
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "OpenVINO not supported on Server yet. "
                                "Reverting to available backend: 'faster_whisper'"
                }))

        try:
            if self.backend.is_faster_whisper():
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                # model is of the form namespace/repo_name and not a filesystem path
                if faster_whisper_custom_model_path is not None:
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 5),
                    cache_path=self.cache_path,
                    translation_queue=translation_queue
                )

                logging.info("Running faster_whisper backend.")

                # Start batch inference worker on first client (after model is loaded)
                if (self.batch_config is not None
                        and ServeClientFasterWhisper.BATCH_WORKER is None
                        and ServeClientFasterWhisper.SINGLE_MODEL is not None):
                    from whisper_live.batch_inference import BatchInferenceWorker
                    worker = BatchInferenceWorker(
                        transcriber=ServeClientFasterWhisper.SINGLE_MODEL,
                        **self.batch_config,
                    )
                    worker.start()
                    ServeClientFasterWhisper.BATCH_WORKER = worker
        except Exception as e:
            logging.error(e)
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.
        Also handles control messages (JSON text frames).

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio, False for END_OF_AUDIO, or
            "CONTROL" if a control message was handled (caller should continue).
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False

        # Handle JSON control messages (text frames)
        if isinstance(frame_data, str):
            try:
                msg = json.loads(frame_data)
                if msg.get("type") == "clear_buffer":
                    client = self.client_manager.get_client(websocket)
                    if client:
                        seq = msg.get("seq")  # None if not provided (backward compat)
                        client.clear_buffer(seq=seq)
                    return "CONTROL"
                if msg.get("type") == "trim_buffer":
                    client = self.client_manager.get_client(websocket)
                    if client:
                        trim_s = float(msg.get("trim_seconds", 0))
                        client.trim_buffer(trim_s)
                    return "CONTROL"
            except (json.JSONDecodeError, AttributeError):
                pass
            return "CONTROL"  # Ignore unrecognized text frames

        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual, trt_py_session=False):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt():
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}", exc_info=True)
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        if isinstance(frame_np, str):
            return True  # Control message handled, continue receiving

        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            client.set_eos(True)
            return False

        if self.backend.is_tensorrt() and hasattr(self, 'vad_detector') and self.use_vad:
            # Silero VAD requires >= 512 samples at 16kHz; skip VAD for short trailing chunks
            if len(frame_np) >= 512:
                voice_active = self.voice_activity(websocket, frame_np)
                if voice_active:
                    client.no_voice_activity_chunks = 0
                    client.set_eos(False)
                else:
                    return True  # Drop silence frame

        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,   
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False,
                   trt_py_session=False):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            trt_py_session=False,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/",
            rest_port=8000,
            enable_rest=False,
            cors_origins: Optional[str] = None,
            batch_enabled=False,
            batch_max_size=8,
            batch_window_ms=50,
            beam_size=1,
            vac_enabled=False):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            batch_enabled (bool): Enable cross-client GPU batch inference for
                the faster_whisper backend. When enabled, ``single_model`` is
                forced to True and a ``BatchInferenceWorker`` is started after
                the first client connects. Defaults to False.
            batch_max_size (int): Maximum number of requests per GPU batch.
                Defaults to 8.
            batch_window_ms (int): Maximum time in milliseconds to wait for
                the batch to fill after the first request arrives. Defaults
                to 50.
        """
        self.cache_path = cache_path
        self.client_manager = ClientManager(max_clients, max_connection_time)
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            if "/" not in faster_whisper_custom_model_path:
                raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path or HuggingFace model.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")

        # Batch inference config
        if batch_enabled:
            single_model = True  # Batch mode requires shared model
            self.batch_config = {
                'max_batch_size': batch_max_size,
                'batch_window_ms': batch_window_ms,
                'beam_size': beam_size,
            }
            logging.info(f"Batch inference enabled (max_batch={batch_max_size}, window={batch_window_ms}ms, beam={beam_size})")
        else:
            self.batch_config = None

        self.vac_enabled = vac_enabled

        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path:
                logging.info("Custom model option was provided. Switching to single model mode.")
                self.single_model = True
                # Pre-load model at startup to reserve GPU memory before TTS expands.
                # Sets the class-level SINGLE_MODEL so clients reuse it (no lazy load OOM).
                if faster_whisper_custom_model_path:
                    from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
                    logging.info(f"Pre-loading Whisper model: {faster_whisper_custom_model_path} ({compute_type})")
                    self.shared_model = WhisperModel(
                        faster_whisper_custom_model_path,
                        device="cuda",
                        compute_type=compute_type,
                        local_files_only=True,
                    )
                    ServeClientFasterWhisper.SINGLE_MODEL = self.shared_model
                    logging.info("Whisper model pre-loaded and GPU memory reserved.")
            else:
                logging.info("Single model mode currently only works with custom models.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")

        # New OpenAI-compatible REST API (toggleable via enable_rest boolean)
        if enable_rest:
            app = FastAPI(title="WhisperLive OpenAI-Compatible API")
            origins = [o.strip() for o in cors_origins.split(',')] if cors_origins else []
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
                allow_headers=["*"],  # Allows all headers
            )


            @app.post("/v1/audio/transcriptions")
            async def transcribe(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                response_format: str = Form(default="json"),
                temperature: float = Form(default=0.0),
                timestamp_granularities: Optional[List[str]] = Form(default=None),
                # Stubs for unsupported OpenAI params
                chunking_strategy: Optional[str] = Form(default=None),
                include: Optional[List[str]] = Form(default=None),
                known_speaker_names: Optional[List[str]] = Form(default=None),
                known_speaker_references: Optional[List[str]] = Form(default=None),
                stream: bool = Form(default=False)
            ):
                if stream:
                    return JSONResponse({"error": "Streaming not supported in this backend."}, status_code=400)
                if chunking_strategy or known_speaker_names or known_speaker_references:
                    logging.warning("Diarization/chunking params ignored; not supported.")

                supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
                if response_format not in supported_formats:
                    return JSONResponse({"error": f"Unsupported response_format. Supported: {supported_formats}"}, status_code=400)

                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    # Use pre-loaded TRT model if available, else fall back to faster-whisper
                    from whisper_live.backend.trt_backend import ServeClientTensorRT
                    if ServeClientTensorRT.SINGLE_MODEL is not None:
                        trt_model = ServeClientTensorRT.SINGLE_MODEL
                        mel, duration = trt_model.log_mel_spectrogram(tmp_path)
                        lang = language or "en"
                        text_prefix = f"<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"
                        text = trt_model.transcribe(mel, text_prefix=text_prefix) or ""
                    else:
                        model_name = faster_whisper_custom_model_path or "small"
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute_type = "float16" if device == "cuda" else "int8"
                        fw_model = WhisperModel(model_name, device=device, compute_type=compute_type)
                        segments, info = fw_model.transcribe(
                            tmp_path, language=language, initial_prompt=prompt,
                            temperature=temperature, vad_filter=False,
                            word_timestamps=(timestamp_granularities and "word" in timestamp_granularities)
                        )
                        text = " ".join([s.text.strip() for s in segments])

                    os.unlink(tmp_path)

                    if response_format == "text":
                        return PlainTextResponse(text.strip())
                    elif response_format == "json":
                        return {"text": text.strip()}
                    else:
                        return {"text": text.strip()}
                except Exception as e:
                    logging.error(f"REST transcription error: {e}", exc_info=True)
                    return JSONResponse({"error": str(e)}, status_code=500)

            threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={"host": "0.0.0.0", "port": rest_port, "log_level": "info"},
                daemon=True
            ).start()
            logging.info(f"✅ OpenAI-Compatible API started on http://0.0.0.0:{rest_port}")

        # Pre-load model and warmup at startup (before accepting connections)
        if backend == "tensorrt" and whisper_tensorrt_path:
            try:
                from whisper_live.backend.trt_backend import ServeClientTensorRT
                from whisper_live.transcriber.transcriber_tensorrt import WhisperTRTLLM
                trt_max_batch = (
                    self.batch_config.get("max_batch_size", 8)
                    if self.batch_config is not None
                    else 1
                )
                assets_dir = os.environ.get("ASSETS_DIR", "/app/assets")
                defer_kv = os.environ.get("STT_DEFER_KV_CACHE", "0") == "1"
                logging.info(
                    "Pre-loading TensorRT model at startup (defer_kv_cache=%s)...",
                    defer_kv,
                )
                transcriber = WhisperTRTLLM(
                    whisper_tensorrt_path,
                    assets_dir=assets_dir,
                    device="cuda",
                    is_multilingual=trt_multilingual,
                    language="en",
                    task="transcribe",
                    use_py_session=trt_py_session,
                    max_output_len=96,
                    max_batch_size=trt_max_batch,
                    defer_kv_cache=defer_kv,
                )
                ServeClientTensorRT.SINGLE_MODEL = transcriber

                if not defer_kv:
                    # Immediate mode: warmup now
                    warmup_audio = os.path.join(assets_dir, "jfk.flac")
                    logging.info("Warming up TensorRT engine (10 steps)...")
                    mel, _ = transcriber.log_mel_spectrogram(warmup_audio)
                    for _ in range(10):
                        transcriber.transcribe(mel)
                    logging.info("TensorRT warmup complete.")

                    # Start batch worker immediately if batch inference is enabled
                    if self.batch_config is not None:
                        from whisper_live.batch_inference_trt import BatchInferenceTRTWorker
                        worker = BatchInferenceTRTWorker(
                            transcriber=transcriber,
                            **self.batch_config,
                        )
                        worker.start()
                        ServeClientTensorRT.BATCH_WORKER = worker
                        logging.info("TRT batch inference worker started at startup.")
            except Exception as e:
                logging.error(f"Failed to pre-load TensorRT model: {e}", exc_info=True)

        # Track server status for two-phase startup
        stt_status = {"value": "ready"}  # default: ready (faster-whisper or non-deferred TRT)
        if backend == "tensorrt" and os.environ.get("STT_DEFER_KV_CACHE", "0") == "1":
            stt_status["value"] = "weights_ready"

            # Background thread polls for /tmp/allocate_kv_cache.json signal file
            # written by the VRAM coordinator via docker exec.
            import threading

            def _poll_allocate_signal():
                signal_path = "/tmp/allocate_kv_cache.json"
                logging.info("[STT] Polling for KV cache signal at %s", signal_path)
                while stt_status["value"] == "weights_ready":
                    if os.path.exists(signal_path):
                        try:
                            with open(signal_path) as f:
                                sig = json.load(f)
                            os.remove(signal_path)
                            fraction = float(sig.get("fraction", 0.05))
                            logging.info("[STT] Signal received, allocating KV cache (fraction=%.3f)", fraction)

                            from whisper_live.backend.trt_backend import ServeClientTensorRT
                            transcriber = ServeClientTensorRT.SINGLE_MODEL
                            transcriber.allocate_kv_cache(kv_cache_fraction=fraction)

                            _assets = os.environ.get("ASSETS_DIR", "/app/assets")
                            warmup_audio = os.path.join(_assets, "jfk.flac")
                            logging.info("Warming up TensorRT engine (10 steps)...")
                            mel, _ = transcriber.log_mel_spectrogram(warmup_audio)
                            for _ in range(10):
                                transcriber.transcribe(mel)
                            logging.info("TensorRT warmup complete.")

                            if self.batch_config is not None and ServeClientTensorRT.BATCH_WORKER is None:
                                from whisper_live.batch_inference_trt import BatchInferenceTRTWorker
                                worker = BatchInferenceTRTWorker(
                                    transcriber=transcriber,
                                    **self.batch_config,
                                )
                                worker.start()
                                logging.info("TRT batch inference worker started after KV allocation.")

                            stt_status["value"] = "ready"
                            logging.info("[STT] Phase 2 complete: status=ready")
                            return
                        except Exception as e:
                            logging.exception("[STT] KV cache allocation failed: %s", e)
                            return
                    time.sleep(1.0)

            threading.Thread(target=_poll_allocate_signal, daemon=True).start()

        # Health check + allocate_kv_cache handler
        def _respond(request, status_code, body):
            """Return HTTP response and log in uvicorn-like format."""
            reason = "OK" if status_code == 200 else "Service Unavailable" if status_code == 503 else str(status_code)
            # Log like: INFO: 127.0.0.1:PORT - "GET /health HTTP/1.1" 200 OK
            try:
                addr = getattr(connection, "remote_address", ("?", "?"))
                host_port = f"{addr[0]}:{addr[1]}" if addr else "?"
            except Exception:
                host_port = "?"
            method = getattr(request, "method", "GET") or "GET"
            logging.info('INFO:     %s - "%s %s HTTP/1.1" %d %s',
                         host_port, method, request.path, status_code, reason)
            return Response(status_code, reason, Headers(), body if isinstance(body, bytes) else body.encode())

        def process_request(connection, request):
            # request.path may include query string; split it
            raw_path = str(request.path)
            path = raw_path.split("?")[0]
            query_string = raw_path.split("?", 1)[1] if "?" in raw_path else ""

            if path == "/health":
                if stt_status["value"] == "ready":
                    return _respond(request, 200, b'{"status":"ok"}\n')
                else:
                    return _respond(request, 503, json.dumps({"status": stt_status["value"]}).encode() + b"\n")

            if path == "/startup":
                return _respond(request, 200, json.dumps({"status": stt_status["value"]}).encode() + b"\n")

            if path == "/allocate_kv_cache":
                if stt_status["value"] == "ready":
                    return _respond(request, 200, b'{"status":"already_ready"}\n')
                if stt_status["value"] != "weights_ready":
                    return _respond(request, 409, json.dumps({"error": f"status={stt_status['value']}"}).encode() + b"\n")

                try:
                    import urllib.parse
                    fraction = 0.05
                    if query_string:
                        params = urllib.parse.parse_qs(query_string)
                        fraction = float(params.get("fraction", [0.05])[0])

                    logging.info("[STT] allocating KV cache (fraction=%.3f)", fraction)
                    from whisper_live.backend.trt_backend import ServeClientTensorRT
                    transcriber = ServeClientTensorRT.SINGLE_MODEL
                    transcriber.allocate_kv_cache(kv_cache_fraction=fraction)

                    # Now do warmup
                    _assets = os.environ.get("ASSETS_DIR", "/app/assets")
                    warmup_audio = os.path.join(_assets, "jfk.flac")
                    logging.info("Warming up TensorRT engine (10 steps)...")
                    mel, _ = transcriber.log_mel_spectrogram(warmup_audio)
                    for _ in range(10):
                        transcriber.transcribe(mel)
                    logging.info("TensorRT warmup complete.")

                    # Start batch worker if configured
                    if self.batch_config is not None and ServeClientTensorRT.BATCH_WORKER is None:
                        from whisper_live.batch_inference_trt import BatchInferenceTRTWorker
                        worker = BatchInferenceTRTWorker(
                            transcriber=transcriber,
                            **self.batch_config,
                        )
                        worker.start()
                        ServeClientTensorRT.BATCH_WORKER = worker
                        logging.info("TRT batch inference worker started after KV allocation.")

                    stt_status["value"] = "ready"
                    body = b'{"status":"ready"}\n'
                    return Response(200, "OK", Headers(), body)
                except Exception as e:
                    logging.exception("[STT] KV cache allocation failed: %s", e)
                    body = json.dumps({"error": str(e)}).encode() + b"\n"
                    return Response(500, "Internal Server Error", Headers(), body)

            return None  # proceed with WebSocket upgrade

        # Original WebSocket server (always supported)
        with serve(
            functools.partial(
                self.recv_audio,
                backend=BackendType(backend),
                faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                whisper_tensorrt_path=whisper_tensorrt_path,
                trt_multilingual=trt_multilingual,
                trt_py_session=trt_py_session,
            ),
            host,
            port,
            process_request=process_request,
        ) as server:
            server.serve_forever()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        client = self.client_manager.get_client(websocket)
        if not self.vad_detector(frame_np):
            client.no_voice_activity_chunks += 1
            if client.no_voice_activity_chunks > 3:
                if not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100ms; wait for voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()
                
            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            self.client_manager.remove_client(websocket)