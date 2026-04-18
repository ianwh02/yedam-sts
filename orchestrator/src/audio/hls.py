"""LL-HLS audio encoder and segment manager.

Architecture:
  HLSEncoder  — one per session. Spawns ffmpeg subprocess that encodes
                PCM → AAC and outputs fragmented MP4 to stdout. Python
                splits at moof boundaries into partial segments.
  HLSSession  — in-memory segment storage. Serves m3u8 playlist and
                segment data to all listeners (shared, no per-listener state).

The ffmpeg process produces ~200ms fMP4 fragments (moof+mdat pairs).
Python groups these into full segments (default 20 parts = ~4s) and
generates an LL-HLS playlist with EXT-X-PART tags and blocking reload
support (_HLS_msn / _HLS_part query parameters).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import subprocess
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLE_RATE = 48000
CHANNELS = 1
AAC_BITRATE = "64k"
# Target fragment duration in microseconds (ffmpeg -frag_duration)
FRAG_DURATION_US = 200_000  # 200ms (ffmpeg target)
# Actual part duration is dictated by AAC frame alignment:
# 10 AAC frames × 1024 samples / 48kHz = 0.21333s.
# PART-TARGET must be >= max actual part duration (HLS spec requirement).
# iOS Safari strictly enforces this; Chrome is lenient.
PART_TARGET_S = 0.214
# Number of partial segments per full segment
PARTS_PER_SEGMENT = 10  # 10 × 213ms = ~2.1s full segments
# Rolling window: keep this many full segments in playlist
PLAYLIST_SIZE = 10


@dataclass
class PartialSegment:
    """One fMP4 fragment (moof+mdat), ~200ms of audio."""
    data: bytes
    duration: float  # seconds
    index: int  # part index within its parent segment


@dataclass
class FullSegment:
    """Collection of partial segments forming one HLS segment."""
    parts: list[PartialSegment] = field(default_factory=list)
    sequence: int = 0  # media sequence number

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.parts)

    @property
    def complete(self) -> bool:
        return len(self.parts) >= PARTS_PER_SEGMENT

    @property
    def data(self) -> bytes:
        return b"".join(p.data for p in self.parts)


class HLSSession:
    """In-memory LL-HLS segment storage for one session.

    Thread-safe. All listeners read from the same segments.
    """

    def __init__(self):
        self.init_segment: bytes = b""
        self._segments: list[FullSegment] = []
        self._current_segment: FullSegment | None = None
        self._media_sequence: int = 0
        self._lock = threading.Lock()
        self._update_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_finalize = False
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Store the event loop for cross-thread event signaling."""
        self._loop = loop

    def set_init_segment(self, data: bytes):
        self.init_segment = data

    def request_finalize(self):
        """Request segment finalization after the next part arrives.

        Called from flush_sentence. The actual finalization happens in
        add_part() after the reader thread delivers the next fragment,
        guaranteeing all audio is included.
        """
        with self._lock:
            self._pending_finalize = True

    def add_part(self, part_data: bytes, duration: float):
        """Add a partial segment. Called from the reader thread."""
        with self._lock:
            if self._current_segment is None:
                seq = self._media_sequence + len(self._segments)
                self._current_segment = FullSegment(sequence=seq)

            part = PartialSegment(
                data=part_data,
                duration=duration,
                index=len(self._current_segment.parts),
            )
            self._current_segment.parts.append(part)

            if self._current_segment.complete:
                self._segments.append(self._current_segment)
                self._current_segment = None
                self._pending_finalize = False
                # Trim old segments
                while len(self._segments) > PLAYLIST_SIZE:
                    self._segments.pop(0)
                    self._media_sequence += 1
            elif self._pending_finalize and len(self._current_segment.parts) >= 5:
                # Sentence boundary: finalize now that the last fragment arrived
                self._segments.append(self._current_segment)
                self._current_segment = None
                self._pending_finalize = False
                while len(self._segments) > PLAYLIST_SIZE:
                    self._segments.pop(0)
                    self._media_sequence += 1

        # Signal waiting playlist requests
        if self._loop:
            self._loop.call_soon_threadsafe(self._update_event.set)

    def finalize(self):
        """Finalize the current partial segment as a complete segment (on session stop)."""
        with self._lock:
            if self._current_segment and self._current_segment.parts:
                self._segments.append(self._current_segment)
                self._current_segment = None
            self._ended = True
        if self._loop:
            self._loop.call_soon_threadsafe(self._update_event.set)

    async def wait_for_update(self, timeout: float = 5.0) -> bool:
        """Block until new data is available (for blocking playlist reload)."""
        self._update_event.clear()
        try:
            await asyncio.wait_for(self._update_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def get_playlist(self) -> str:
        """Generate m3u8 playlist with EXT-X-PART tags."""
        with self._lock:
            return self._build_playlist()

    def has_part(self, msn: int, part: int) -> bool:
        """Check if a specific segment/part exists."""
        with self._lock:
            return self._has_part_locked(msn, part)

    def get_init_segment(self) -> bytes:
        return self.init_segment

    def get_segment_data(self, msn: int) -> bytes | None:
        """Get full segment data by media sequence number."""
        with self._lock:
            for seg in self._segments:
                if seg.sequence == msn:
                    return seg.data
            return None

    def get_part_data(self, msn: int, part_index: int) -> bytes | None:
        """Get partial segment data."""
        with self._lock:
            # Check completed segments
            for seg in self._segments:
                if seg.sequence == msn and part_index < len(seg.parts):
                    return seg.parts[part_index].data
            # Check current (in-progress) segment
            if (self._current_segment
                    and self._current_segment.sequence == msn
                    and part_index < len(self._current_segment.parts)):
                return self._current_segment.parts[part_index].data
            return None

    def _has_part_locked(self, msn: int, part: int) -> bool:
        for seg in self._segments:
            if seg.sequence == msn:
                return part < len(seg.parts)
            if seg.sequence > msn:
                return True
        if (self._current_segment
                and self._current_segment.sequence == msn):
            return part < len(self._current_segment.parts)
        if (self._current_segment
                and self._current_segment.sequence > msn):
            return True
        return False

    def _build_playlist(self) -> str:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:10",
            f"#EXT-X-TARGETDURATION:{PARTS_PER_SEGMENT * FRAG_DURATION_US // 1_000_000 + 1}",
            f"#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={PART_TARGET_S * 3:.3f}",
            f"#EXT-X-PART-INF:PART-TARGET={PART_TARGET_S:.3f}",
            '#EXT-X-MAP:URI="init.mp4"',
            f"#EXT-X-MEDIA-SEQUENCE:{self._media_sequence}",
        ]

        # Completed segments with their parts
        for seg in self._segments:
            for p in seg.parts:
                lines.append(
                    f"#EXT-X-PART:DURATION={p.duration:.6f},"
                    f'URI="seg_{seg.sequence}.{p.index}.m4s"'
                )
            lines.append(f"#EXTINF:{seg.duration:.6f},")
            lines.append(f"seg_{seg.sequence}.m4s")

        # Current (in-progress) segment parts
        if self._current_segment:
            for p in self._current_segment.parts:
                lines.append(
                    f"#EXT-X-PART:DURATION={p.duration:.6f},"
                    f'URI="seg_{self._current_segment.sequence}.{p.index}.m4s"'
                )
            # Preload hint for next expected part
            next_part = len(self._current_segment.parts)
            lines.append(
                f"#EXT-X-PRELOAD-HINT:TYPE=PART,"
                f'URI="seg_{self._current_segment.sequence}.{next_part}.m4s"'
            )

        if self._ended:
            lines.append("#EXT-X-ENDLIST")

        return "\n".join(lines) + "\n"


class HLSEncoder:
    """Manages ffmpeg subprocess: PCM stdin → fragmented MP4 stdout.

    Reads fMP4 fragments from ffmpeg's stdout and feeds them into
    an HLSSession as partial segments.
    """

    # 200ms of silence at 48kHz mono s16le (pre-computed, reused)
    _SILENCE_CHUNK = bytes(SAMPLE_RATE * CHANNELS * 2 * 200 // 1000)

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session = HLSSession()
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._silence_task: asyncio.Task | None = None
        self._has_real_audio = asyncio.Event()
        self._started = False

    async def start(self):
        """Spawn ffmpeg and start reading fragments."""
        loop = asyncio.get_event_loop()
        self.session.set_loop(loop)

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
        self._has_real_audio.set()  # suppress silence until first sentence ends
        self._silence_task = asyncio.create_task(self._silence_generator())
        logger.info("HLS encoder started for session %s", self.session_id)

    def feed_pcm(self, pcm_bytes: bytes):
        """Write PCM to ffmpeg's stdin. Called from async context via executor."""
        if self._proc and self._proc.stdin and self._proc.poll() is None:
            try:
                self._proc.stdin.write(pcm_bytes)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.warning("HLS encoder pipe broken for session %s", self.session_id)

    async def feed_pcm_async(self, pcm_bytes: bytes):
        """Async wrapper for feed_pcm."""
        self._has_real_audio.set()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.feed_pcm, pcm_bytes)

    async def flush_sentence_async(self):
        """Signal sentence boundary — start filling silence until next audio.

        Continuous silence keeps ffmpeg producing fragments, which:
        1. Flushes the last real audio out of ffmpeg's buffer immediately
        2. Keeps HLS segments completing naturally (no finalize hacks)
        3. Sounds correct (silence while speaker is silent)
        """
        self.session.request_finalize()
        # Clear the flag so silence generator starts
        self._has_real_audio.clear()

    async def _silence_generator(self):
        """Feed silence to ffmpeg when no real audio is arriving.

        Runs as a background task. Yields 200ms silence chunks at
        real-time rate when idle. Stops when real audio arrives
        (feed_pcm_async sets _has_real_audio).
        """
        while self._started:
            # Wait for real audio to stop
            try:
                await asyncio.wait_for(self._has_real_audio.wait(), timeout=0.2)
                # Real audio is flowing — wait for it to stop
                await asyncio.sleep(0.05)
                continue
            except TimeoutError:
                pass

            # No real audio for 200ms — feed silence
            if self._proc and self._proc.poll() is None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self.feed_pcm, self._SILENCE_CHUNK
                    )
                except Exception:
                    break

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
        # Finalize any remaining partial segment
        self.session.finalize()
        self._started = False
        logger.info("HLS encoder closed for session %s", self.session_id)

    def _read_fragments(self):
        """Read fMP4 from ffmpeg stdout, split at moof boundaries.

        The fMP4 stream structure is:
          [ftyp][moov]  — init segment (emitted once)
          [moof][mdat]  — fragment 1 (~200ms)
          [moof][mdat]  — fragment 2 (~200ms)
          ...

        We parse MP4 box headers to detect boundaries.
        """
        stdout = self._proc.stdout
        init_data = bytearray()
        frag_buf = bytearray()
        seen_first_moof = False
        frag_index = 0

        # Timescale from moov (default 48000 for AAC at 48kHz).
        # Updated after parsing init segment.
        timescale = SAMPLE_RATE

        while True:
            # Read box header (8 bytes: 4 size + 4 type)
            header = self._read_exact(stdout, 8)
            if not header:
                break

            box_size = struct.unpack(">I", header[:4])[0]
            box_type = header[4:8]

            if box_size < 8:
                break

            # Read box body
            body = self._read_exact(stdout, box_size - 8)
            if body is None:
                break

            box_data = header + body

            if not seen_first_moof:
                if box_type == b"moof":
                    # Init segment complete — parse timescale from moov
                    init_bytes = bytes(init_data)
                    ts = self._parse_timescale(init_bytes)
                    if ts:
                        timescale = ts
                    self.session.set_init_segment(init_bytes)
                    logger.info(
                        "HLS init segment: %d bytes, timescale=%d (session %s)",
                        len(init_data), timescale, self.session_id,
                    )
                    seen_first_moof = True
                    frag_buf.extend(box_data)
                else:
                    # Still collecting init segment (ftyp, moov)
                    init_data.extend(box_data)
                    continue

            elif box_type == b"moof":
                # New fragment starting — flush previous fragment
                if frag_buf:
                    duration = self._parse_frag_duration(bytes(frag_buf), timescale)
                    self.session.add_part(bytes(frag_buf), duration)
                    frag_index += 1
                frag_buf.clear()
                frag_buf.extend(box_data)

            else:
                # mdat or other box — append to current fragment
                frag_buf.extend(box_data)

        # Flush last fragment
        if frag_buf:
            duration = self._parse_frag_duration(bytes(frag_buf), timescale)
            self.session.add_part(bytes(frag_buf), duration)

        logger.info(
            "HLS reader finished: %d fragments (session %s)",
            frag_index + 1, self.session_id,
        )

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

    @staticmethod
    def _parse_timescale(init_data: bytes) -> int | None:
        """Extract timescale from moov/trak/mdia/mdhd box."""
        offset = 0
        while offset + 8 <= len(init_data):
            size = struct.unpack(">I", init_data[offset:offset + 4])[0]
            box_type = init_data[offset + 4:offset + 8]
            if size < 8:
                break
            # mdhd is nested: moov > trak > mdia > mdhd
            # Search recursively through container boxes
            if box_type in (b"moov", b"trak", b"mdia"):
                result = HLSEncoder._parse_timescale(init_data[offset + 8:offset + size])
                if result:
                    return result
            elif box_type == b"mdhd":
                body = init_data[offset + 8:offset + size]
                if len(body) >= 20:
                    version = body[0]
                    if version == 0:
                        # v0: 4B creation, 4B modification, 4B timescale, 4B duration
                        return struct.unpack(">I", body[12:16])[0]
                    elif version == 1:
                        # v1: 8B creation, 8B modification, 4B timescale, 8B duration
                        return struct.unpack(">I", body[20:24])[0]
            offset += size
        return None

    @staticmethod
    def _parse_frag_duration(frag_data: bytes, timescale: int) -> float:
        """Parse actual duration from moof's trun box (sample count × default_sample_duration).

        Falls back to trun sample_count × 1024 (AAC frame size) if no
        default_sample_duration is found.
        """
        default_duration = 1024  # AAC-LC frame size in samples
        sample_count = 0

        offset = 0
        while offset + 8 <= len(frag_data):
            size = struct.unpack(">I", frag_data[offset:offset + 4])[0]
            box_type = frag_data[offset + 4:offset + 8]
            if size < 8:
                break

            # Search inside moof and traf containers
            if box_type in (b"moof", b"traf"):
                result = HLSEncoder._parse_frag_duration(
                    frag_data[offset + 8:offset + size], timescale,
                )
                if result > 0:
                    return result

            elif box_type == b"tfhd":
                body = frag_data[offset + 8:offset + size]
                if len(body) >= 8:
                    flags = struct.unpack(">I", body[:4])[0] & 0xFFFFFF
                    pos = 8  # skip version+flags (4B) + track_id (4B)
                    if flags & 0x01:  # base-data-offset-present
                        pos += 8
                    if flags & 0x02:  # sample-description-index-present
                        pos += 4
                    if flags & 0x08:  # default-sample-duration-present
                        if pos + 4 <= len(body):
                            default_duration = struct.unpack(">I", body[pos:pos + 4])[0]

            elif box_type == b"trun":
                body = frag_data[offset + 8:offset + size]
                if len(body) >= 8:
                    flags = struct.unpack(">I", body[:4])[0] & 0xFFFFFF
                    sample_count = struct.unpack(">I", body[4:8])[0]
                    # If per-sample durations exist, sum them
                    if flags & 0x100:  # sample-duration-present
                        total = 0
                        pos = 8
                        if flags & 0x01:  # data-offset-present
                            pos += 4
                        if flags & 0x04:  # first-sample-flags-present
                            pos += 4
                        for _ in range(sample_count):
                            if pos + 4 > len(body):
                                break
                            total += struct.unpack(">I", body[pos:pos + 4])[0]
                            pos += 4
                            if flags & 0x200:  # sample-size-present
                                pos += 4
                            if flags & 0x400:  # sample-flags-present
                                pos += 4
                            if flags & 0x800:  # sample-composition-time-offset
                                pos += 4
                        return total / timescale

            offset += size

        # Fallback: sample_count × default_duration
        if sample_count > 0:
            return (sample_count * default_duration) / timescale
        return FRAG_DURATION_US / 1_000_000  # last resort estimate
