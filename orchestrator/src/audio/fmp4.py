"""Fragmented MP4 encoder for WebSocket audio delivery.

Spawns an ffmpeg subprocess: PCM stdin → AAC fMP4 stdout.
A reader thread splits the output at moof boundaries and pushes
each fragment into an asyncio queue. The init segment (ftyp+moov)
is stored separately for new listeners joining mid-stream.

Fragments are pushed to listeners as binary WebSocket messages
and appended directly to MSE SourceBuffers on the client.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import subprocess
import threading

logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLE_RATE = 48000
CHANNELS = 1
AAC_BITRATE = "64k"
# Target fragment duration in microseconds (ffmpeg -frag_duration)
FRAG_DURATION_US = 200_000  # 200ms


class FMP4Encoder:
    """PCM → AAC fMP4 via ffmpeg. Fragments delivered via async queue."""

    # 200ms of silence at 48kHz mono s16le
    _SILENCE_CHUNK = bytes(SAMPLE_RATE * CHANNELS * 2 * 200 // 1000)

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.init_segment: bytes = b""
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._silence_task: asyncio.Task | None = None
        self._has_real_audio = asyncio.Event()
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Queue for init segment (as bytes) and fragments (as bytes).
        # None sentinel = EOF.
        self._fragment_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def start(self):
        """Spawn ffmpeg and start reading fragments."""
        self._loop = asyncio.get_event_loop()

        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-f", "s16le",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "-i", "pipe:0",
                "-c:a", "aac",
                "-b:a", AAC_BITRATE,
                "-f", "mp4",
                "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "-frag_duration", str(FRAG_DURATION_US),
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._reader_thread = threading.Thread(
            target=self._read_fragments,
            daemon=True,
        )
        self._reader_thread.start()
        self._started = True
        # Start silence immediately so ffmpeg produces the init segment
        # right away. Listeners get it on connect and MSE primes its buffer.
        self._has_real_audio.clear()
        self._silence_task = asyncio.create_task(self._silence_generator())
        logger.info("fMP4 encoder started for session %s", self.session_id)

    def feed_pcm(self, pcm_bytes: bytes):
        """Write PCM to ffmpeg's stdin (sync, call via executor)."""
        if self._proc and self._proc.stdin and self._proc.poll() is None:
            try:
                self._proc.stdin.write(pcm_bytes)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.warning("fMP4 encoder pipe broken for session %s", self.session_id)

    async def feed_pcm_async(self, pcm_bytes: bytes):
        """Async wrapper for feed_pcm."""
        self._has_real_audio.set()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.feed_pcm, pcm_bytes)

    async def flush_sentence_async(self):
        """Signal sentence boundary — silence generator flushes ffmpeg buffer."""
        self._has_real_audio.clear()

    async def get_fragment(self) -> bytes | None:
        """Get next fragment from queue. None = stream ended."""
        return await self._fragment_queue.get()

    async def close(self):
        """Close stdin → ffmpeg flushes final fragment → process exits."""
        self._started = False
        if self._silence_task:
            self._silence_task.cancel()
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait(timeout=10)
        if self._reader_thread:
            self._reader_thread.join(timeout=5)
        logger.info("fMP4 encoder closed for session %s", self.session_id)

    async def _silence_generator(self):
        """Feed silence to ffmpeg during TTS gaps.

        Keeps ffmpeg producing fragments (flushes real audio from its
        buffer and keeps the MSE <audio> element active for lock screen).
        """
        while self._started:
            try:
                await asyncio.wait_for(self._has_real_audio.wait(), timeout=0.2)
                await asyncio.sleep(0.05)
                continue
            except TimeoutError:
                pass

            if self._proc and self._proc.poll() is None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self.feed_pcm, self._SILENCE_CHUNK
                    )
                except Exception:
                    break

    def _read_fragments(self):
        """Read fMP4 from ffmpeg stdout, split at moof boundaries.

        Stream structure:
          [ftyp][moov]  — init segment (queued once, also stored)
          [moof][mdat]  — fragment 1 (~200ms)
          [moof][mdat]  — fragment 2 (~200ms)
          ...
        """
        stdout = self._proc.stdout
        init_data = bytearray()
        frag_buf = bytearray()
        seen_first_moof = False
        frag_index = 0

        while True:
            header = self._read_exact(stdout, 8)
            if not header:
                break

            box_size = struct.unpack(">I", header[:4])[0]
            box_type = header[4:8]

            if box_size < 8:
                break

            body = self._read_exact(stdout, box_size - 8)
            if body is None:
                break

            box_data = header + body

            if not seen_first_moof:
                if box_type == b"moof":
                    # Init segment complete
                    self.init_segment = bytes(init_data)
                    logger.info(
                        "fMP4 init segment: %d bytes (session %s)",
                        len(self.init_segment), self.session_id,
                    )
                    seen_first_moof = True
                    frag_buf.extend(box_data)
                else:
                    init_data.extend(box_data)
                    continue

            elif box_type == b"moof":
                # New fragment — flush previous
                if frag_buf:
                    self._emit_fragment(bytes(frag_buf))
                    frag_index += 1
                frag_buf.clear()
                frag_buf.extend(box_data)

            else:
                frag_buf.extend(box_data)

        # Flush last fragment
        if frag_buf:
            self._emit_fragment(bytes(frag_buf))
            frag_index += 1

        # Signal EOF
        if self._loop:
            self._loop.call_soon_threadsafe(self._fragment_queue.put_nowait, None)

        logger.info(
            "fMP4 reader finished: %d fragments (session %s)",
            frag_index, self.session_id,
        )

    def _emit_fragment(self, data: bytes):
        """Push fragment to async queue from reader thread."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._fragment_queue.put_nowait, data)

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly n bytes from stream, or None on EOF."""
        data = bytearray()
        while len(data) < n:
            chunk = stream.read(n - len(data))
            if not chunk:
                return None if not data else bytes(data)
            data.extend(chunk)
        return bytes(data)
