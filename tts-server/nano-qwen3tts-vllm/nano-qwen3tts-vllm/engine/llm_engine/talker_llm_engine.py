import logging

import torch
from nano_qwen3tts_vllm.engine.llm_engine.base import LLMEngine
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.engine.scheduler import Scheduler
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.engine.model_runner.talker_mode_runner import TalkerModeModelRunner


from nano_qwen3tts_vllm.config import Config

logger = logging.getLogger(__name__)

class TalkerScheduler(Scheduler):
    CODEC_EOS_TOKEN_ID = 2150

    def __init__(self, config: Config):
        super().__init__(config)
        self.request_id_to_seq: dict[str, Sequence] = {}

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill: same as base
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode: only schedule seqs that have decode_input_embeds set (interface has fed next input)
        # Iterate at most once over running to avoid infinite loop when all seqs wait for decode input
        run_count = len(self.running)
        for _ in range(run_count):
            if not self.running or num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if len(seq) > 0 and seq.decode_input_embeds is None:
                self.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            return [], False
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def pause_request(self, request_id: str):
        """Pause sequence: remove from running but keep KV cache blocks allocated."""
        seq = self.request_id_to_seq.get(request_id)
        if seq and seq.status == SequenceStatus.RUNNING:
            seq.status = SequenceStatus.PAUSED
            seq.decode_input_embeds = None
            if seq in self.running:
                self.running.remove(seq)
            logger.info(f"[scheduler] paused request {request_id} (blocks={len(seq.block_table)}, tokens={len(seq)})")

    def resume_request(self, request_id: str, decode_input_embeds):
        """Resume a paused sequence with new input embeddings."""
        seq = self.request_id_to_seq.get(request_id)
        if seq and seq.status == SequenceStatus.PAUSED:
            seq.status = SequenceStatus.RUNNING
            seq.decode_input_embeds = decode_input_embeds
            self.running.append(seq)
            logger.info(f"[scheduler] resumed request {request_id} (tokens={len(seq)})")

    def clear_request(self, request_id: str):
        if request_id in self.request_id_to_seq:
            seq = self.request_id_to_seq.pop(request_id)
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], hidden_states: list[torch.Tensor]):
        idx = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id, hidden_states[idx])
            seq.decode_input_embeds = None
            idx += 1

            # For keep_alive sequences (continuous TTS): check EOS to pause
            # Require at least 4 decode steps before allowing EOS (prevents premature stop)
            if seq.keep_alive and seq.request_id is not None:
                seq.generation_steps += 1
                if not seq.ignore_eos and token_id == self.CODEC_EOS_TOKEN_ID and seq.generation_steps > 4:
                    logger.info(f"[postprocess] keep_alive EOS detected: request={seq.request_id[:8]} token={token_id} gen_step={seq.generation_steps}")
                    self.pause_request(seq.request_id)
                continue  # skip normal finish logic — interface manages the step loop

            # For regular sequences: use original finish logic (eos=-1, so only max_tokens triggers)
            if seq.request_id is not None:
                finish = not seq.ignore_eos and token_id == self.eos
            else:
                finish = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens
            if finish:
                seq.status = SequenceStatus.FINISHED
                if seq.request_id is not None:
                    self.request_id_to_seq.pop(seq.request_id, None)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)



class TalkerLLMEngine(LLMEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model_runner = TalkerModeModelRunner(self.config, 0, self.events)
        self.scheduler = TalkerScheduler(self.config)

    def add_request(
        self,
        inputs_embeds: list[torch.Tensor],
        sampling_params: SamplingParams | list[SamplingParams],
        request_id: str | None = None,
        keep_alive: bool = False,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(inputs_embeds)
        for inp_embeds, sp in zip(inputs_embeds, sampling_params):
            if request_id is not None and request_id in self.scheduler.request_id_to_seq:
                seq = self.scheduler.request_id_to_seq[request_id]
                # Resume if paused (continuation after soft EOS)
                if seq.status == SequenceStatus.PAUSED:
                    self.scheduler.resume_request(request_id, inp_embeds)
                else:
                    seq.decode_input_embeds = inp_embeds
                return
            # Request not found — could mean clear_request already ran, or first add
            if request_id is not None and keep_alive:
                bm = self.scheduler.block_manager
                logger.debug(
                    f"[add_request] new seq request={request_id[:8]} "
                    f"free_blocks={len(bm.free_block_ids)}/{len(bm.blocks)}"
                )
            seq = Sequence([], input_embeds=inp_embeds, sampling_params=sp, request_id=request_id, keep_alive=keep_alive)
            if request_id is not None:
                self.scheduler.request_id_to_seq[request_id] = seq
            self.scheduler.add(seq)

    def clear_request(self, request_id: str):
        self.scheduler.clear_request(request_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def step_with_outputs(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0, []

        token_ids, hidden_states = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, hidden_states)
        outputs = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state) for seq in seqs if seq.is_finished]
        # is_finished includes both FINISHED and PAUSED (both mean "this step produced a terminal token")
        outputs_all = [(seq.request_id, seq.seq_id, seq.completion_token_ids, seq.last_hidden_state, seq.is_finished or seq.is_paused, seq.is_paused) for seq in seqs]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens, outputs_all
            