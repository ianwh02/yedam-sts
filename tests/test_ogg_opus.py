"""Tests for OGG/Opus page construction (pure Python, no opuslib needed)."""

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Setup orchestrator imports ──
_orch_root = Path(__file__).parent.parent / "orchestrator"
_orch_src = _orch_root / "src"

try:
    import types

    if str(_orch_root) not in sys.path:
        sys.path.insert(0, str(_orch_root))

    sys.modules.setdefault("opuslib", MagicMock())
    sys.modules.setdefault("rnnoise_wrapper", MagicMock())

    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_orch_src)]
        sys.modules["src"] = src_pkg

    # Minimal config mock (ogg_opus doesn't use settings, but config module must exist)
    src_config = types.ModuleType("src.config")
    src_config.settings = MagicMock()
    sys.modules.setdefault("src.config", src_config)

    from src.audio.ogg_opus import (
        FRAME_SIZE,
        OggPageWriter,
        _make_ogg_page,
        _ogg_crc,
    )

    HAS_OGG = True
except Exception as e:
    HAS_OGG = False
    _skip_reason = str(e)

needs_ogg = pytest.mark.skipif(
    not HAS_OGG,
    reason=f"OGG import failed: {_skip_reason if not HAS_OGG else ''}",
)


@needs_ogg
class TestOggCrc:
    def test_empty_bytes(self):
        """CRC of empty data should be 0."""
        assert _ogg_crc(b"") == 0

    def test_deterministic(self):
        """Same input produces same CRC."""
        data = b"OggS test data for CRC"
        assert _ogg_crc(data) == _ogg_crc(data)

    def test_different_input_different_crc(self):
        """Different inputs produce different CRCs."""
        assert _ogg_crc(b"hello") != _ogg_crc(b"world")


@needs_ogg
class TestMakeOggPage:
    def test_magic_bytes(self):
        """OGG page must start with 'OggS' capture pattern."""
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[b"\x00"])
        assert page[:4] == b"OggS"

    def test_bos_flag(self):
        """Beginning-of-stream flag should be set in header_type byte."""
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[b"\x00"], bos=True)
        header_type = page[5]
        assert header_type & 0x02  # BOS bit

    def test_eos_flag(self):
        """End-of-stream flag should be set in header_type byte."""
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[b"\x00"], eos=True)
        header_type = page[5]
        assert header_type & 0x04  # EOS bit

    def test_no_flags_by_default(self):
        """Neither BOS nor EOS set by default."""
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[b"\x00"])
        header_type = page[5]
        assert header_type == 0

    def test_serial_number_encoded(self):
        """Serial number should be at bytes 14-17 (little-endian uint32)."""
        serial = 0xDEADBEEF
        page = _make_ogg_page(serial=serial, page_seq=0, granule=0, packets=[b"\x00"])
        encoded_serial = struct.unpack_from("<I", page, 14)[0]
        assert encoded_serial == serial

    def test_segment_table_single_short_packet(self):
        """A short packet (<255 bytes) has one segment entry."""
        payload = b"\x42" * 100
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[payload])
        n_segments = page[26]
        assert n_segments == 1
        assert page[27] == 100  # segment size

    def test_crc_is_nonzero(self):
        """The CRC field (bytes 22-25) should be filled (non-zero for real data)."""
        page = _make_ogg_page(serial=1, page_seq=0, granule=0, packets=[b"test data"])
        crc = struct.unpack_from("<I", page, 22)[0]
        assert crc != 0


@needs_ogg
class TestOggPageWriter:
    def test_header_pages_starts_with_oggs(self):
        """Header pages should start with OggS magic."""
        writer = OggPageWriter()
        headers = writer.header_pages()
        assert headers[:4] == b"OggS"

    def test_header_pages_has_bos(self):
        """First header page should have BOS flag."""
        writer = OggPageWriter()
        headers = writer.header_pages()
        header_type = headers[5]
        assert header_type & 0x02

    def test_header_contains_opus_head(self):
        """Header pages should contain OpusHead identification."""
        writer = OggPageWriter()
        headers = writer.header_pages()
        assert b"OpusHead" in headers

    def test_header_contains_opus_tags(self):
        """Header pages should contain OpusTags."""
        writer = OggPageWriter()
        headers = writer.header_pages()
        assert b"OpusTags" in headers

    def test_header_contains_vendor(self):
        """OpusTags should include the vendor string."""
        writer = OggPageWriter()
        headers = writer.header_pages()
        assert b"yedam-sts" in headers

    def test_wrap_frame_increments_granule(self):
        """Each wrap_frame call should advance granule by FRAME_SIZE (960)."""
        writer = OggPageWriter()
        writer.header_pages()  # consume header pages

        fake_opus_frame = b"\x00" * 50
        page1 = writer.wrap_frame(fake_opus_frame)
        # Granule is at bytes 6-13 (little-endian int64)
        granule1 = struct.unpack_from("<q", page1, 6)[0]
        assert granule1 == FRAME_SIZE  # 960

        page2 = writer.wrap_frame(fake_opus_frame)
        granule2 = struct.unpack_from("<q", page2, 6)[0]
        assert granule2 == FRAME_SIZE * 2  # 1920

    def test_wrap_frame_increments_page_seq(self):
        """Page sequence numbers should increment."""
        writer = OggPageWriter()
        writer.header_pages()  # pages 0 and 1

        fake_opus_frame = b"\x00" * 50
        page1 = writer.wrap_frame(fake_opus_frame)
        seq1 = struct.unpack_from("<I", page1, 18)[0]
        assert seq1 == 2  # after 2 header pages

        page2 = writer.wrap_frame(fake_opus_frame)
        seq2 = struct.unpack_from("<I", page2, 18)[0]
        assert seq2 == 3
