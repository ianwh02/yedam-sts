"""Shared fixtures for yedam-sts tests."""
import sys
from pathlib import Path

# Add nano-qwen3tts-vllm to path so imports work without pip install
nano_root = Path(__file__).parent.parent / "tts-server" / "nano-qwen3tts-vllm"
if str(nano_root) not in sys.path:
    sys.path.insert(0, str(nano_root))

# Add stt-server to path for korean_endings and other STT module imports
stt_root = Path(__file__).parent.parent / "stt-server"
if str(stt_root) not in sys.path:
    sys.path.insert(0, str(stt_root))
