"""OGG/Opus streaming encoder for HTTP audio delivery.

Architecture (encode-once):
  OpusFrameEncoder — one per session, shared. Encodes PCM → Opus frames.
  OggPageWriter    — one per listener connection, cheap. Wraps pre-encoded
                     Opus frames in OGG pages with per-stream sequencing.

OGG page format: https://xiph.org/ogg/doc/framing.html
Opus in OGG: https://tools.ietf.org/html/rfc7845
"""

from __future__ import annotations

import random
import struct

# Opus frame parameters (48kHz mono, 20ms frames)
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_DURATION_MS = 20
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960 samples
FRAME_BYTES = FRAME_SIZE * 2  # int16 = 2 bytes per sample

# Pre-skip: encoder delay in 48kHz samples (standard Opus value)
PRE_SKIP = 312

# ---------------------------------------------------------------------------
# OGG CRC
# ---------------------------------------------------------------------------

_CRC_TABLE: list[int] = []


def _build_crc_table():
    for i in range(256):
        r = i << 24
        for _ in range(8):
            if r & 0x80000000:
                r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                r = (r << 1) & 0xFFFFFFFF
        _CRC_TABLE.append(r)


_build_crc_table()


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ b]) & 0xFFFFFFFF
    return crc


# ---------------------------------------------------------------------------
# OGG page builder
# ---------------------------------------------------------------------------

def _make_ogg_page(
    serial: int,
    page_seq: int,
    granule: int,
    packets: list[bytes],
    bos: bool = False,
    eos: bool = False,
) -> bytes:
    """Build a single OGG page containing one or more packets."""
    header_type = 0
    if bos:
        header_type |= 0x02
    if eos:
        header_type |= 0x04

    # Segment table
    segments = []
    for pkt in packets:
        remaining = len(pkt)
        while remaining >= 255:
            segments.append(255)
            remaining -= 255
        segments.append(remaining)

    n_segments = len(segments)
    body = b"".join(packets)

    header = struct.pack(
        "<4sBBqIIIB",
        b"OggS", 0, header_type, granule,
        serial, page_seq, 0, n_segments,
    )
    header += bytes(segments)

    page_no_crc = header + body
    crc = _ogg_crc(page_no_crc)
    header = header[:22] + struct.pack("<I", crc) + header[26:]

    return header + body


# ---------------------------------------------------------------------------
# OpusFrameEncoder — shared, one per session
# ---------------------------------------------------------------------------

class OpusFrameEncoder:
    """Encodes int16 PCM into Opus frames. One instance shared by all listeners.

    Call feed_pcm() with variable-length PCM. Yields 20ms Opus frames.
    The silence_frame property provides a pre-encoded silence frame for
    stream padding.
    """

    def __init__(self, bitrate: int = 32000):
        import opuslib

        self._encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
        self._encoder.bitrate = bitrate
        self._buffer = bytearray()

        # Pre-encode silence once (reused by all connections)
        self.silence_frame: bytes = self._encoder.encode(bytes(FRAME_BYTES), FRAME_SIZE)

    def feed_pcm(self, pcm_int16: bytes) -> list[bytes]:
        """Buffer PCM and return a list of encoded Opus frames (0 or more)."""
        self._buffer.extend(pcm_int16)
        frames = []
        while len(self._buffer) >= FRAME_BYTES:
            frame_pcm = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            frames.append(self._encoder.encode(frame_pcm, FRAME_SIZE))
        return frames


# ---------------------------------------------------------------------------
# OggPageWriter — per listener connection, cheap
# ---------------------------------------------------------------------------

class OggPageWriter:
    """Wraps pre-encoded Opus frames into OGG pages.

    One instance per listener HTTP connection. No Opus encoder — just
    struct.pack + CRC for OGG framing. Negligible CPU.
    """

    def __init__(self):
        self._serial = random.randint(0, 0xFFFFFFFF)
        self._page_seq = 0
        self._granule = 0

    def header_pages(self) -> bytes:
        """Generate the two required OGG header pages (OpusHead + OpusTags)."""
        opus_head = struct.pack(
            "<8sBBHIhB",
            b"OpusHead", 1, CHANNELS, PRE_SKIP, SAMPLE_RATE, 0, 0,
        )
        page1 = _make_ogg_page(
            self._serial, self._page_seq, 0,
            [opus_head], bos=True,
        )
        self._page_seq += 1

        vendor = b"yedam-sts"
        opus_tags = struct.pack("<8sI", b"OpusTags", len(vendor))
        opus_tags += vendor
        opus_tags += struct.pack("<I", 0)

        page2 = _make_ogg_page(
            self._serial, self._page_seq, 0,
            [opus_tags],
        )
        self._page_seq += 1

        return page1 + page2

    def wrap_frame(self, opus_frame: bytes) -> bytes:
        """Wrap a single pre-encoded Opus frame as an OGG page."""
        self._granule += FRAME_SIZE
        page = _make_ogg_page(
            self._serial, self._page_seq, self._granule,
            [opus_frame],
        )
        self._page_seq += 1
        return page
