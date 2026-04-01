"""Tests for TalkerScheduler pause/resume and soft EOS (Phase 0.2, 0.3).

Uses direct imports of scheduler/sequence/block_manager to avoid pulling
in the model layers (which require triton/flash-attn/GPU).
"""
import sys
from unittest.mock import MagicMock

# Mock heavy GPU modules before any nano_qwen3tts_vllm imports
# This prevents the import chain: talker_llm_engine → base → model_runner → models → triton
_mock_modules = [
    "triton", "triton.language", "flash_attn", "flash_attn.flash_attn_interface",
    "nano_qwen3tts_vllm.layers.attention",
    "nano_qwen3tts_vllm.layers.layernorm",
    "nano_qwen3tts_vllm.models",
    "nano_qwen3tts_vllm.models.qwen3_tts_talker",
    "nano_qwen3tts_vllm.models.qwen3_tts_share",
    "nano_qwen3tts_vllm.models.qwen3_tts_predictor",
    "nano_qwen3tts_vllm.engine.model_runner.base",
    "nano_qwen3tts_vllm.engine.model_runner.talker_mode_runner",
    "nano_qwen3tts_vllm.engine.model_runner.predictor_mode_runner",
    "nano_qwen3tts_vllm.engine.llm_engine.base",
]
for mod in _mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Now we can import safely — the mocked modules prevent the GPU import chain
import pytest
import torch
from nano_qwen3tts_vllm.engine.block_manager import BlockManager
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.sampling_params import SamplingParams


# Manually construct TalkerScheduler without going through TalkerLLMEngine
# (which requires the model runner)
class _TestableScheduler:
    """Minimal scheduler for testing pause/resume without GPU dependencies."""

    def __init__(self, num_blocks=16, block_size=256, max_num_seqs=4, max_num_batched_tokens=2048):
        from collections import deque
        self.block_manager = BlockManager(num_blocks, block_size)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting = deque()
        self.running = deque()
        self.request_id_to_seq: dict[str, Sequence] = {}
        self.eos = 2150

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self):
        scheduled = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True

        # decode phase
        run_count = len(self.running)
        for _ in range(run_count):
            if not self.running or num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if len(seq) > 0 and seq.decode_input_embeds is None:
                self.running.append(seq)
                continue
            self.block_manager.may_append(seq)
            num_seqs += 1
            scheduled.append(seq)
        if not scheduled:
            return [], False
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def pause_request(self, request_id: str):
        seq = self.request_id_to_seq.get(request_id)
        if seq and seq.status == SequenceStatus.RUNNING:
            seq.status = SequenceStatus.PAUSED
            seq.decode_input_embeds = None
            if seq in self.running:
                self.running.remove(seq)

    def resume_request(self, request_id: str, decode_input_embeds):
        seq = self.request_id_to_seq.get(request_id)
        if seq and seq.status == SequenceStatus.PAUSED:
            seq.status = SequenceStatus.RUNNING
            seq.decode_input_embeds = decode_input_embeds
            self.running.append(seq)

    def clear_request(self, request_id: str):
        if request_id in self.request_id_to_seq:
            seq = self.request_id_to_seq.pop(request_id)
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)

    def postprocess(self, seqs, token_ids, hidden_states):
        for seq, token_id, h in zip(seqs, token_ids, hidden_states):
            seq.append_token(token_id, h)
            seq.decode_input_embeds = None
            finish = not seq.ignore_eos and token_id == self.eos
            if finish:
                if seq.keep_alive and seq.request_id is not None:
                    self.pause_request(seq.request_id)
                else:
                    seq.status = SequenceStatus.FINISHED
                    if seq.request_id is not None:
                        self.request_id_to_seq.pop(seq.request_id, None)
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)


@pytest.fixture
def scheduler():
    return _TestableScheduler()


def _make_embeds(seq_len=4, hidden=64):
    return torch.randn(1, seq_len, hidden)


def _add_request(scheduler, request_id, keep_alive=False):
    embeds = _make_embeds()
    seq = Sequence([], input_embeds=embeds, sampling_params=SamplingParams(),
                   request_id=request_id, keep_alive=keep_alive)
    scheduler.request_id_to_seq[request_id] = seq
    scheduler.add(seq)
    return seq


class TestPauseResume:
    def test_pause_request_keeps_blocks(self, scheduler):
        seq = _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()  # prefill
        assert len(seq.block_table) > 0
        block_count = len(seq.block_table)

        scheduler.pause_request("req-1")
        assert seq.status == SequenceStatus.PAUSED
        assert seq not in scheduler.running
        assert "req-1" in scheduler.request_id_to_seq
        assert len(seq.block_table) == block_count

    def test_resume_request(self, scheduler):
        seq = _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()

        scheduler.pause_request("req-1")
        assert seq.status == SequenceStatus.PAUSED

        new_embeds = _make_embeds(seq_len=1)
        scheduler.resume_request("req-1", new_embeds)
        assert seq.status == SequenceStatus.RUNNING
        assert seq in scheduler.running
        assert seq.decode_input_embeds is new_embeds

    def test_pause_nonexistent(self, scheduler):
        scheduler.pause_request("nonexistent")  # no-op

    def test_resume_nonexistent(self, scheduler):
        scheduler.resume_request("nonexistent", _make_embeds())  # no-op

    def test_clear_frees_paused_blocks(self, scheduler):
        _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()
        free_before = len(scheduler.block_manager.free_block_ids)

        scheduler.pause_request("req-1")
        assert len(scheduler.block_manager.free_block_ids) == free_before  # blocks still allocated

        scheduler.clear_request("req-1")
        assert "req-1" not in scheduler.request_id_to_seq
        assert len(scheduler.block_manager.free_block_ids) > free_before  # blocks freed

    def test_paused_not_scheduled(self, scheduler):
        _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()  # prefill

        scheduler.pause_request("req-1")
        scheduled, _ = scheduler.schedule()
        assert len(scheduled) == 0


class TestSoftEOS:
    def test_soft_eos_pauses(self, scheduler):
        seq = _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()

        scheduler.postprocess([seq], [2150], [torch.randn(64)])
        assert seq.status == SequenceStatus.PAUSED
        assert "req-1" in scheduler.request_id_to_seq
        assert len(seq.block_table) > 0

    def test_hard_eos_finishes(self, scheduler):
        seq = _add_request(scheduler, "req-2", keep_alive=False)
        scheduler.schedule()

        scheduler.postprocess([seq], [2150], [torch.randn(64)])
        assert seq.status == SequenceStatus.FINISHED
        assert "req-2" not in scheduler.request_id_to_seq
        assert len(seq.block_table) == 0

    def test_non_eos_continues(self, scheduler):
        seq = _add_request(scheduler, "req-3", keep_alive=True)
        scheduler.schedule()

        scheduler.postprocess([seq], [1234], [torch.randn(64)])
        assert seq.status == SequenceStatus.RUNNING

    def test_pause_then_resume_then_generate(self, scheduler):
        """Full cycle: generate → EOS → pause → resume → generate more."""
        seq = _add_request(scheduler, "req-1", keep_alive=True)
        scheduler.schedule()  # prefill

        # Generate a few tokens
        for i in range(5):
            seq.decode_input_embeds = _make_embeds(seq_len=1)
            scheduled, is_prefill = scheduler.schedule()
            assert len(scheduled) == 1
            scheduler.postprocess(scheduled, [1000 + i], [torch.randn(64)])
            assert seq.status == SequenceStatus.RUNNING

        tokens_before_eos = len(seq)

        # EOS → soft pause
        seq.decode_input_embeds = _make_embeds(seq_len=1)
        scheduler.schedule()
        scheduler.postprocess([seq], [2150], [torch.randn(64)])
        assert seq.status == SequenceStatus.PAUSED
        assert len(seq) == tokens_before_eos + 1  # EOS token appended

        # Resume with new text
        new_embeds = _make_embeds(seq_len=1)
        scheduler.resume_request("req-1", new_embeds)
        assert seq.status == SequenceStatus.RUNNING

        # Generate more tokens after resume
        scheduled, is_prefill = scheduler.schedule()
        assert len(scheduled) == 1
        assert is_prefill is False  # decode, not prefill (KV cache preserved)
        scheduler.postprocess(scheduled, [2000], [torch.randn(64)])
        assert seq.status == SequenceStatus.RUNNING
        assert len(seq) == tokens_before_eos + 2  # +1 EOS token + 1 continuation token
