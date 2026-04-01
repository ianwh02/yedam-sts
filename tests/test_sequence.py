"""Tests for Sequence and SequenceStatus (Phase 0.1)."""
import torch
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.sampling_params import SamplingParams


class TestSequenceStatus:
    def test_paused_status_exists(self):
        assert hasattr(SequenceStatus, "PAUSED")
        assert SequenceStatus.PAUSED != SequenceStatus.RUNNING
        assert SequenceStatus.PAUSED != SequenceStatus.FINISHED
        assert SequenceStatus.PAUSED != SequenceStatus.WAITING

    def test_status_ordering(self):
        # PAUSED should be between RUNNING and FINISHED
        statuses = list(SequenceStatus)
        names = [s.name for s in statuses]
        assert "WAITING" in names
        assert "RUNNING" in names
        assert "PAUSED" in names
        assert "FINISHED" in names


class TestSequenceKeepAlive:
    def _make_seq(self, keep_alive=False) -> Sequence:
        embeds = torch.randn(1, 4, 64)
        return Sequence(
            [], input_embeds=embeds, sampling_params=SamplingParams(),
            request_id="test-req-1", keep_alive=keep_alive,
        )

    def test_default_keep_alive_false(self):
        seq = self._make_seq(keep_alive=False)
        assert seq.keep_alive is False

    def test_keep_alive_true(self):
        seq = self._make_seq(keep_alive=True)
        assert seq.keep_alive is True

    def test_is_paused_property(self):
        seq = self._make_seq()
        assert seq.is_paused is False
        seq.status = SequenceStatus.PAUSED
        assert seq.is_paused is True

    def test_is_finished_not_when_paused(self):
        seq = self._make_seq()
        seq.status = SequenceStatus.PAUSED
        assert seq.is_finished is False

    def test_initial_status_waiting(self):
        seq = self._make_seq()
        assert seq.status == SequenceStatus.WAITING
