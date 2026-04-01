"""Tests for processor implementations (CPU-only)."""

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

    _mock_settings = MagicMock()
    _mock_settings.llm_api_url = "http://localhost:8000/v1"
    _mock_settings.llm_system_prompt_path = ""
    _mock_settings.llm_context_window_segments = 3

    src_config = types.ModuleType("src.config")
    src_config.settings = _mock_settings
    sys.modules["src.config"] = src_config

    from src.processors.conversation import ConversationProcessor
    from src.processors.passthrough import PassthroughProcessor

    HAS_PROCESSORS = True
except Exception as e:
    HAS_PROCESSORS = False
    _skip_reason = str(e)

needs_processors = pytest.mark.skipif(
    not HAS_PROCESSORS,
    reason=f"Processors import failed: {_skip_reason if not HAS_PROCESSORS else ''}",
)


@needs_processors
class TestPassthroughProcessor:
    async def test_yields_input_unchanged(self):
        proc = PassthroughProcessor()
        await proc.initialize()
        chunks = [chunk async for chunk in proc.process("hello world", {})]
        assert chunks == ["hello world"]

    async def test_yields_korean_text(self):
        proc = PassthroughProcessor()
        await proc.initialize()
        chunks = [chunk async for chunk in proc.process("안녕하세요", {})]
        assert chunks == ["안녕하세요"]

    async def test_shutdown_is_noop(self):
        proc = PassthroughProcessor()
        await proc.shutdown()  # should not raise


@needs_processors
class TestConversationProcessor:
    async def test_raises_not_implemented(self):
        proc = ConversationProcessor()
        await proc.initialize()
        with pytest.raises(NotImplementedError):
            # process() raises before yielding, so it's a coroutine at runtime
            await proc.process("test", {})
