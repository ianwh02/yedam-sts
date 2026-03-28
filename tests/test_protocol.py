"""Tests for worker protocol serialization (Phase 0.4)."""
import torch
import numpy as np
import pytest

from nano_qwen3tts_vllm.workers.protocol import (
    CMD_PAUSE_REQUEST,
    serialize_pause_request,
    serialize_talker_add_request,
    serialize_talker_result,
    deserialize_command,
    deserialize_talker_result,
)


class TestPauseRequestProtocol:
    def test_cmd_constant_exists(self):
        assert CMD_PAUSE_REQUEST == "pause_request"

    def test_serialize_deserialize(self):
        payload = serialize_pause_request("req-123")
        cmd = deserialize_command(payload)
        assert cmd["cmd"] == CMD_PAUSE_REQUEST
        assert cmd["request_id"] == "req-123"


class TestAddRequestKeepAlive:
    def test_keep_alive_false_default(self):
        embeds = [torch.randn(1, 4, 64)]
        payload = serialize_talker_add_request("req-1", embeds, {"temperature": 0.7})
        cmd = deserialize_command(payload)
        assert cmd["keep_alive"] is False

    def test_keep_alive_true(self):
        embeds = [torch.randn(1, 4, 64)]
        payload = serialize_talker_add_request("req-1", embeds, {"temperature": 0.7}, keep_alive=True)
        cmd = deserialize_command(payload)
        assert cmd["keep_alive"] is True

    def test_embeds_serialized_as_numpy(self):
        embeds = [torch.randn(1, 4, 64)]
        payload = serialize_talker_add_request("req-1", embeds, {})
        cmd = deserialize_command(payload)
        assert isinstance(cmd["inputs_embeds"][0], np.ndarray)
        assert cmd["inputs_embeds"][0].shape == (1, 4, 64)


class TestTalkerResultWithPaused:
    def test_6_tuple_result(self):
        """Results should include is_paused as 6th element."""
        hidden = torch.randn(64)
        outputs_all = [
            ("req-1", 0, [1234], hidden, False, False),  # normal step
        ]
        payload = serialize_talker_result("step-1", outputs_all)
        step_id, results = deserialize_talker_result(payload)
        assert step_id == "step-1"
        assert len(results) == 1
        assert len(results[0]) == 6
        req_id, seq_id, token_ids, h, is_finished, is_paused = results[0]
        assert req_id == "req-1"
        assert is_finished is False
        assert is_paused is False

    def test_paused_result(self):
        """Paused sequence should have is_finished=True, is_paused=True."""
        hidden = torch.randn(64)
        outputs_all = [
            ("req-1", 0, [2150], hidden, True, True),  # EOS + paused
        ]
        payload = serialize_talker_result("step-1", outputs_all)
        _, results = deserialize_talker_result(payload)
        _, _, _, _, is_finished, is_paused = results[0]
        assert is_finished is True
        assert is_paused is True

    def test_finished_result(self):
        """Finished (not paused) sequence."""
        hidden = torch.randn(64)
        outputs_all = [
            ("req-2", 0, [2150], hidden, True, False),  # EOS + finished
        ]
        payload = serialize_talker_result("step-1", outputs_all)
        _, results = deserialize_talker_result(payload)
        _, _, _, _, is_finished, is_paused = results[0]
        assert is_finished is True
        assert is_paused is False

    def test_backward_compat_5_tuple(self):
        """5-tuple (old format without is_paused) should still work."""
        hidden = torch.randn(64)
        outputs_all = [
            ("req-1", 0, [1234], hidden, False),  # 5-tuple
        ]
        payload = serialize_talker_result("step-1", outputs_all)
        _, results = deserialize_talker_result(payload)
        assert len(results[0]) == 6  # serializer always produces 6-tuple
        _, _, _, _, is_finished, is_paused = results[0]
        assert is_finished is False
        assert is_paused is False  # defaults to False
