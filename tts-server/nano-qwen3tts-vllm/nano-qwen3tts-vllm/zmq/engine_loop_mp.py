"""
Async engine loops when talker and predictor run in separate processes (USE_MULTIPROCESS_ENGINES=1).
Orchestrator: wait until "ready" set matches active requests (or prefill), send run_step to worker,
await result, dispatch to per-request asyncio queues.
"""

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


PREDICTOR_COLLECT_MS = _float_env("PREDICTOR_COLLECT_MS", 3.0)
PREFILL_COLLECT_MS = _float_env("PREFILL_COLLECT_MS", 5.0)
DECODE_COLLECT_MS = _float_env("DECODE_COLLECT_MS", 40.0)


async def run_talker_loop_mp(
    talker_client: Any,
    request_queues: dict,
    queues_lock: asyncio.Lock,
    talker_ready: set,
    kv_ready: asyncio.Event | None = None,
) -> None:
    """
    Replacement for run_talker_loop when using multiprocess talker.
    Waits until talker_ready matches active requests (or timeout), sends run_step, awaits result,
    dispatches (engine_type, msg_type, payload) to request_queues[request_id].
    """
    # Wait for KV cache allocation before sending any steps to the worker.
    if kv_ready is not None:
        logger.info("[talker_loop_mp] waiting for KV cache allocation...")
        await kv_ready.wait()
        logger.info("[talker_loop_mp] KV ready, starting step loop")

    step_count = 0
    last_batch_size = 0  # Track previous batch size for adaptive collection
    while True:
        await asyncio.sleep(0.0005)
        async with queues_lock:
            active = set(request_queues.keys())
        if not talker_ready:
            continue
        # Orphan detection: discard talker_ready entries whose request_queues are gone
        # (safety net for race between clear_request and the ready set)
        orphans = talker_ready - active
        if orphans:
            logger.warning(f"[talker_loop_mp] discarding {len(orphans)} orphan(s) from talker_ready: {list(orphans)[:3]!r}")
            talker_ready -= orphans
        if not talker_ready:
            continue
        # Adaptive collection: use longer window for decode steps (previous batch > 1)
        # to allow all concurrent requests to complete their predictor feedback cycle
        # (~33ms) before running the next talker step.
        if len(talker_ready) < len(active):
            collect_ms = DECODE_COLLECT_MS if last_batch_size > 1 else PREFILL_COLLECT_MS
            t_start = time.perf_counter()
            while (time.perf_counter() - t_start) * 1000 < collect_ms:
                await asyncio.sleep(0.001)
                async with queues_lock:
                    active = set(request_queues.keys())
                if talker_ready >= active:
                    break
        if not talker_ready:
            continue
        # Run step with whoever is ready; do not wait for all active, so chunk-2
        # / time-to-first-audio stays low when many requests are concurrent.
        try:
            logger.debug(
                f"[talker_loop_mp] run_step active={len(active)} talker_ready={len(talker_ready)} "
                f"request_ids={list(talker_ready)[:3]!r}"
            )
            future = talker_client.run_step_async()
            _, outputs_all = await future
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"[talker_loop_mp] step failed: {e}")
            continue

        if not outputs_all:
            continue

        step_count += 1
        completed_this_step = set()
        for item in outputs_all:
            request_id, seq_id, token_ids, hidden_states, is_finished = item
            completed_this_step.add(request_id)
            payload = {"token_ids": token_ids, "hidden_states": hidden_states}
            async with queues_lock:
                q = request_queues.get(request_id)
            if q is not None:
                last_id = token_ids[-1] if token_ids else -1
                try:
                    q.put_nowait(("talker", "token", payload))
                    if step_count <= 1:
                        logger.debug(f"[talker_loop_mp] dispatched token to request_id={request_id[:8]} token_ids={token_ids[:5]!r}")
                except asyncio.QueueFull:
                    if last_id == 2150:
                        await q.put(("talker", "token", payload))
                        logger.warning(f"[talker_loop_mp] queue full but forced EOS delivery for {request_id[:8]}")
                    else:
                        logger.error(
                            f"[talker_loop_mp] DROPPED talker token for {request_id[:8]}! "
                            f"Queue full (size={q.qsize()}, last_id={last_id}). "
                            f"This may cause a hang in generate_async."
                        )
            if is_finished:
                async with queues_lock:
                    q = request_queues.get(request_id)
                if q is not None:
                    try:
                        q.put_nowait(("talker", "done", {}))
                    except asyncio.QueueFull:
                        await q.put(("talker", "done", {}))
                        logger.warning(f"[talker_loop_mp] queue full but forced done delivery for {request_id[:8]}")

        # Clear only the request_ids that were in this step (they've been served)
        talker_ready -= completed_this_step
        last_batch_size = len(completed_this_step)
        if step_count % 200 == 1:
            logger.info(f"[talker_loop_mp] step#{step_count} batch={len(outputs_all)}")


async def run_predictor_loop_mp(
    predictor_client: Any,
    request_queues: dict,
    queues_lock: asyncio.Lock,
    predictor_ready: set,
    kv_ready: asyncio.Event | None = None,
) -> None:
    """
    Replacement for run_predictor_loop when using multiprocess predictor.
    Waits until predictor_ready is non-empty (and optionally all active have sent),
    sends run_step, awaits burst result, dispatches to request_queues.
    """
    if kv_ready is not None:
        logger.info("[predictor_loop_mp] waiting for KV cache allocation...")
        await kv_ready.wait()
        logger.info("[predictor_loop_mp] KV ready, starting step loop")

    burst_count = 0
    while True:
        await asyncio.sleep(0.0005)
        if not predictor_ready:
            continue
        # Orphan detection: discard predictor_ready entries whose request_queues are gone
        async with queues_lock:
            active = set(request_queues.keys())
        orphans = predictor_ready - active
        if orphans:
            logger.warning(f"[predictor_loop_mp] discarding {len(orphans)} orphan(s) from predictor_ready: {list(orphans)[:3]!r}")
            predictor_ready -= orphans
        if not predictor_ready:
            continue
        # Brief yield to let more add_requests arrive (batching)
        async with queues_lock:
            active = set(request_queues.keys())
        if len(predictor_ready) < len(active) and len(active) > 1:
            await asyncio.sleep(PREDICTOR_COLLECT_MS / 1000.0)
            async with queues_lock:
                active = set(request_queues.keys())
                predictor_ready_copy = set(predictor_ready)
        else:
            async with queues_lock:
                active = set(request_queues.keys())
                predictor_ready_copy = set(predictor_ready)

        if not predictor_ready_copy:
            continue

        try:
            future = predictor_client.run_step_async()
            _, outputs_all = await future
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"[predictor_loop_mp] burst failed: {e}")
            continue

        burst_count += 1
        for request_id, seq_id, token_ids in outputs_all:
            payload = {"token_ids": token_ids}
            async with queues_lock:
                q = request_queues.get(request_id)
            if q is not None:
                try:
                    q.put_nowait(("predictor", "token", payload))
                except asyncio.QueueFull:
                    logger.error(
                        f"[predictor_loop_mp] DROPPED predictor tokens for {request_id[:8]}! "
                        f"Queue full (size={q.qsize()}). This will cause a hang in generate_async."
                    )
            predictor_ready.discard(request_id)

        if burst_count % 200 == 1:
            logger.info(f"[predictor_loop_mp] burst#{burst_count} finished={len(outputs_all)}")
