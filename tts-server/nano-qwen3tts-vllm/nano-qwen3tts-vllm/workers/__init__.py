"""Worker processes for talker, predictor (and decoder) to keep engine work off the event loop."""

from nano_qwen3tts_vllm.workers.protocol import (
    deserialize_command,
    serialize_talker_result,
    deserialize_talker_result,
    serialize_predictor_result,
    deserialize_predictor_result,
    serialize_allocate_kv_cache,
    serialize_allocate_kv_cache_ack,
    CMD_ADD_REQUEST,
    CMD_RUN_STEP,
    CMD_CLEAR_REQUEST,
    CMD_SHUTDOWN,
    CMD_ALLOCATE_KV_CACHE,
)

__all__ = [
    "deserialize_command",
    "serialize_talker_result",
    "deserialize_talker_result",
    "serialize_predictor_result",
    "deserialize_predictor_result",
    "serialize_allocate_kv_cache",
    "serialize_allocate_kv_cache_ack",
    "CMD_ADD_REQUEST",
    "CMD_RUN_STEP",
    "CMD_CLEAR_REQUEST",
    "CMD_SHUTDOWN",
    "CMD_ALLOCATE_KV_CACHE",
]
