import os
import json

import torch
from typing import Optional
import time
from tqdm import tqdm
from safetensors.torch import load_file

from nano_qwen3tts_vllm.engine.model_runner.base import ModelRunner
from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from nano_qwen3tts_vllm.engine.sequence import Sequence, SequenceStatus
from nano_qwen3tts_vllm.sampling_params import SamplingParams

from nano_qwen3tts_vllm.utils.context import set_context, get_context, reset_context
from nano_qwen3tts_vllm.config import Config
from multiprocessing.synchronize import Event

from logging import getLogger

logger = getLogger(__name__)


class TalkerModeModelRunner(ModelRunner):
    # Repetition penalty matching original qwen-tts generate() default (1.05).
    # Penalizes previously generated tokens during decode to prevent codec
    # looping that suppresses EOS probability.
    # Set TTS_REPETITION_PENALTY=1.0 to disable entirely.
    _repetition_penalty: float = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))
    # Window size: only penalize tokens from the last N steps.
    # Prevents over-penalizing naturally repeated codec tokens in long sequences.
    # 0 = no window (penalize all past tokens, original behavior).
    _repetition_penalty_window: int = int(os.environ.get("TTS_REPETITION_PENALTY_WINDOW", "100"))

    # Minimum decode steps before EOS is allowed.
    # Suppresses EOS with -inf for the first N steps to prevent premature termination.
    # Disabled by default: Qwen3-TTS does not prematurely EOS, and suppressing it
    # causes short sentences (e.g. "Hello.") to overshoot and generate garbage.
    _eos_min_steps: int = int(os.environ.get("TTS_EOS_MIN_STEPS", "0"))

    # Progressive EOS logit boost: linearly ramp an additive boost on EOS logits
    # from 0 to _eos_boost_max between _eos_boost_start_step and max_generation_steps.
    # At 12Hz codec rate, step 50 ≈ 4.2s of audio — generous for a single sentence.
    _eos_boost_start_step: int = int(os.environ.get("TTS_EOS_BOOST_START_STEP", "50"))
    _eos_boost_max: float = float(os.environ.get("TTS_EOS_BOOST_MAX", "3.0"))
    _eos_boost_max_step: int = int(os.environ.get("TTS_EOS_BOOST_MAX_STEP", "300"))

    # Codec loop detection: if unique tokens in the last N steps fall below threshold,
    # apply a strong immediate EOS boost to escape the loop.
    _eos_loop_window: int = int(os.environ.get("TTS_EOS_LOOP_DETECT_WINDOW", "20"))
    _eos_loop_threshold: float = float(os.environ.get("TTS_EOS_LOOP_THRESHOLD", "0.3"))
    _eos_loop_boost: float = float(os.environ.get("TTS_EOS_LOOP_BOOST", "5.0"))
    # Force EOS after loop persists for this many consecutive steps.
    # If the model is stuck in a loop for this long, logit boosting isn't enough.
    _eos_loop_force_after: int = int(os.environ.get("TTS_EOS_LOOP_FORCE_AFTER", "30"))

    # Codec whitelist suppress mask: restrict logits to valid codec tokens + EOS.
    # Set TTS_SUPPRESS_MASK=0 to disable (allow all tokens through).
    _suppress_mask_enabled: bool = os.environ.get("TTS_SUPPRESS_MASK", "1").lower() not in ("0", "false", "no")

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        super().__init__(config, rank, event)
        self.model = self.load_model(config)
        self._build_suppress_mask()
        self._loop_streak: dict[int, int] = {}  # seq_id -> consecutive loop steps
        self.post_init(rank)
        # KV cache + CUDA graphs deferred to complete_init() — called by worker
        # after coordinator assigns VRAM budget.

    def _build_suppress_mask(self):
        """Build codec whitelist mask and EOS token ID list.

        Always builds _eos_token_ids (needed by EOS boost/loop detection).
        Conditionally builds _suppress_mask tensor (controlled by TTS_SUPPRESS_MASK env var).
        """
        tc = self.model_config  # Qwen3TTSTalkerConfig
        vocab_size = tc.vocab_size
        codec_eos = tc.codec_eos_token_id
        codebook_vocab_size = tc.code_predictor_config.vocab_size  # 2048

        # Always populate _eos_token_ids (needed by EOS min steps, boost, loop detection)
        codec_eos_secondary = 2157  # QwenLM/Qwen3-TTS#118
        self._eos_token_ids = [codec_eos]
        if codec_eos_secondary != codec_eos:
            self._eos_token_ids.append(codec_eos_secondary)

        if not self._suppress_mask_enabled:
            self._suppress_mask = None
            logger.info(
                f"[TalkerModeModelRunner] suppress mask DISABLED (TTS_SUPPRESS_MASK=0), "
                f"EOS tokens={self._eos_token_ids}"
            )
        else:
            # Whitelist: True = allowed, False = suppressed
            allowed = torch.zeros(vocab_size, dtype=torch.bool, device="cuda")
            lo, hi = 1, min(codebook_vocab_size, vocab_size)
            if hi > lo:
                allowed[lo:hi] = True
            if 0 <= codec_eos < vocab_size:
                allowed[codec_eos] = True
            if 0 <= codec_eos_secondary < vocab_size:
                allowed[codec_eos_secondary] = True
            self._suppress_mask = ~allowed
            num_allowed = allowed.sum().item()
            num_suppressed = self._suppress_mask.sum().item()
            logger.info(
                f"[TalkerModeModelRunner] codec whitelist: allow [{lo}, {hi}) + EOS={codec_eos} "
                f"({num_allowed} allowed, {num_suppressed} suppressed out of {vocab_size})"
            )

        logger.info(
            f"[TalkerModeModelRunner] decode features: "
            f"suppress_mask={'ON' if self._suppress_mask is not None else 'OFF'} "
            f"repetition_penalty={self._repetition_penalty} "
            f"eos_min_steps={self._eos_min_steps} "
            f"eos_boost_max={self._eos_boost_max} "
            f"loop_detect_window={self._eos_loop_window} "
            f"loop_force_after={self._eos_loop_force_after}"
        )

    def load_model(self, config: Config):
        with open(os.path.join(config.model, "config.json"), "r") as f:
            model_config = json.load(f)
            model_config = Qwen3TTSConfig(**model_config)
        
        self.full_config = model_config
            
        model = Qwen3TTSTalkerForCausalLM(model_config.talker_config)
        
        self.model_config = model_config.talker_config
        
        state_dict = load_file(
            os.path.join(config.model, "model.safetensors")
        )
        model.load_state_dict(state_dict)   
        return model

    
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, input_embeds: Optional[torch.Tensor] = None):
        start = time.time()
        use_graph = False
        model_input = input_embeds if input_embeds is not None else input_ids
        if is_prefill:
            num_tokens = model_input.size(0)
            context = get_context()
            num_seqs = context.cu_seqlens_q.size(0) - 1
            # Find the next captured graph size >= num_tokens (sparse bucket)
            graph_size = None
            if (
                not self.enforce_eager
                and getattr(self, "graph_bs_prefill", None) is not None
                and num_seqs == 1
                and context.block_tables is None
            ):
                graph_size = next((s for s in self.graph_bs_prefill if s >= num_tokens), None)

            use_graph = graph_size is not None
            if use_graph:
                graph = self.graphs_prefill[graph_size]
                graph_vars = self.graph_vars_prefill
                # Zero the full buffer slice, then copy actual data
                graph_vars["input_embeds"][:graph_size].zero_()
                graph_vars["input_embeds"][:num_tokens].copy_(model_input)
                graph_vars["positions"][:graph_size].zero_()
                graph_vars["positions"][:num_tokens].copy_(positions)
                graph_vars["cu_seqlens_q"][0] = 0
                graph_vars["cu_seqlens_q"][1] = num_tokens
                graph_vars["cu_seqlens_k"][0] = 0
                graph_vars["cu_seqlens_k"][1] = num_tokens
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:num_tokens].copy_(context.slot_mapping)
                graph.replay()
                hidden_states = graph_vars["outputs"][:num_tokens].clone()
                graph_latency_ms = (time.time() - start) * 1000
                logger.info(f"[talker mode model runner] Prefill graph num_tokens={num_tokens} (bucket={graph_size}) latency={graph_latency_ms:.4f}ms")
            else:
                hidden_states = self.model(model_input, positions)
        elif self.enforce_eager or model_input.size(0) > 512:
            hidden_states = self.model(model_input, positions)
        else:
            use_graph = True
            bs = input_embeds.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_embeds"][:bs] = model_input
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            max_num_blocks = graph_vars["block_tables"].size(1)
            bt_cols = min(context.block_tables.size(1), max_num_blocks)
            graph_vars["block_tables"][:bs, :bt_cols] = context.block_tables[:, :bt_cols]
            graph.replay()
            hidden_states = graph_vars["outputs"][:bs]

        logits = self.model.compute_logits(hidden_states)
        
        if is_prefill:
            context = get_context()
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        
        logger.debug(f"[talker mode model runner] Model run prefill={is_prefill} {input_embeds.shape} latency: {time.time() - start} use_graph={use_graph}")

        return logits, hidden_states

    def prepare_decode_talker(self, seqs: list[Sequence]):
        positions = []
        slot_mapping = []
        context_lens = []
        input_embeds_list = []
        for seq in seqs:
            emb = seq.decode_input_embeds
            if emb is None:
                raise ValueError(f"Sequence {seq.seq_id} has no decode_input_embeds set")
            input_embeds_list.append(emb)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_embeds = torch.cat([e.reshape(-1, e.shape[-1]) if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds_list], dim=0).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)
        else:
            input_embeds = input_embeds.cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        input_ids = torch.zeros(len(seqs), dtype=torch.int64, device="cuda")
        return input_ids, input_embeds, positions

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, input_embeds, positions = self.prepare_decode_talker(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits, hidden_states = self.run_model(input_ids, positions, is_prefill, input_embeds)
        # Suppress invalid tokens — only allow valid codec tokens + EOS.
        # Uses masked_fill (not in-place) to work under @torch.inference_mode().
        if self.rank == 0 and self._suppress_mask is not None:
            logits = logits.masked_fill(self._suppress_mask, float('-inf'))
        # Suppress EOS before minimum steps to prevent premature termination.
        if self.rank == 0 and not is_prefill and self._eos_min_steps > 0:
            for i, seq in enumerate(seqs):
                if len(seq.token_ids) < self._eos_min_steps:
                    for eos_id in self._eos_token_ids:
                        logits[i, eos_id] = float('-inf')
        # Repetition penalty during decode: penalize previously generated tokens
        # so the model doesn't loop on the same codec patterns forever.
        # Uses a sliding window to avoid over-penalizing naturally repeated codec tokens.
        if self.rank == 0 and self._repetition_penalty != 1.0 and not is_prefill:
            logits = logits.clone()
            penalty = self._repetition_penalty
            window = self._repetition_penalty_window
            for i, seq in enumerate(seqs):
                past = seq.token_ids
                if past:
                    if window > 0:
                        past = past[-window:]
                    unique_past = torch.tensor(list(set(past)), device=logits.device, dtype=torch.long)
                    scores = logits[i, unique_past]
                    logits[i, unique_past] = torch.where(
                        scores > 0, scores / penalty, scores * penalty
                    )
        # Progressive EOS logit boost + codec loop detection during decode.
        # Nudges the model toward emitting EOS as generation gets longer,
        # and applies a strong boost if a codec loop is detected.
        force_eos_indices = []
        if self.rank == 0 and not is_prefill and (self._eos_boost_max > 0 or self._eos_loop_boost > 0):
            start = self._eos_boost_start_step
            max_step = self._eos_boost_max_step
            max_boost = self._eos_boost_max
            loop_win = self._eos_loop_window
            loop_thresh = self._eos_loop_threshold
            loop_boost_val = self._eos_loop_boost
            force_after = self._eos_loop_force_after
            eos_ids = self._eos_token_ids
            for i, seq in enumerate(seqs):
                step = len(seq.token_ids)
                boost = 0.0
                # Progressive ramp
                if step >= start and max_step > start:
                    boost = max_boost * min((step - start) / (max_step - start), 1.0)
                # Loop detection: if last N tokens have very few unique values
                in_loop = False
                if loop_win > 0 and len(seq.token_ids) >= loop_win:
                    recent = seq.token_ids[-loop_win:]
                    unique_ratio = len(set(recent)) / loop_win
                    if unique_ratio < loop_thresh:
                        in_loop = True
                        boost += loop_boost_val
                        streak = self._loop_streak.get(seq.seq_id, 0) + 1
                        self._loop_streak[seq.seq_id] = streak
                        if streak % 50 == 1:
                            logger.warning(
                                f"[TalkerModeModelRunner] loop detected for seq {seq.seq_id}: "
                                f"unique_ratio={unique_ratio:.2f} at step {step}, "
                                f"streak={streak}, boost +{loop_boost_val}"
                            )
                        # Force EOS if loop persists too long
                        if force_after > 0 and streak >= force_after:
                            force_eos_indices.append(i)
                            if streak == force_after:
                                logger.warning(
                                    f"[TalkerModeModelRunner] forcing EOS for seq {seq.seq_id}: "
                                    f"loop persisted {streak} steps at step {step}"
                                )
                if not in_loop:
                    self._loop_streak.pop(seq.seq_id, None)
                if boost > 0:
                    for eos_id in eos_ids:
                        logits[i, eos_id] += boost
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # Override sampled tokens with EOS for sequences stuck in persistent loops
        if token_ids and force_eos_indices:
            eos_token = self._eos_token_ids[0]  # primary EOS (2150)
            for i in force_eos_indices:
                token_ids[i] = eos_token
        reset_context()
        return token_ids, hidden_states
    
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = self.model_config
        max_bs = min(self.config.max_num_seqs, 2048)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        input_embeds = torch.zeros(max_bs, hf_config.hidden_size)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_embeds=input_embeds,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        
    @torch.inference_mode()
    def capture_cudagraph_prefill(self):
        """Capture CUDA graphs for prefill: 1 seq, sparse token counts.

        Uses a sparse set of sizes instead of every integer 1-256.
        At runtime, inputs are padded to the next captured size.
        This saves hundreds of MB of VRAM vs capturing all 256 sizes.
        """
        config = self.config
        hf_config = self.model_config
        # Sparse set: fine-grained for small (1-8), then coarser buckets
        self.graph_bs_prefill = sorted(set(
            list(range(1, 9))  # 1-8: exact match for small prefills
            + [12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        ))
        max_num_tokens = self.graph_bs_prefill[-1] + 1  # buffer size
        input_embeds = torch.zeros(max_num_tokens, hf_config.hidden_size, device="cuda")
        positions = torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.zeros(2, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.zeros(2, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size, device="cuda")
        self.graphs_prefill = {}
        graph_pool = self.graph_pool

        for bs in tqdm(reversed(self.graph_bs_prefill), desc="talker prefill graphs"):
            graph = torch.cuda.CUDAGraph()
            cu_seqlens_q[0] = 0
            cu_seqlens_q[1] = bs
            cu_seqlens_k[0] = 0
            cu_seqlens_k[1] = bs
            set_context(True, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=bs, max_seqlen_k=bs, slot_mapping=slot_mapping[:bs])
            outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], positions[:bs])    # capture
            if graph_pool is None:
                graph_pool = graph.pool()
            self.graphs_prefill[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars_prefill = dict(
            input_embeds=input_embeds,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            outputs=outputs,
        )
