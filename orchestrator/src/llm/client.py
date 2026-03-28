from __future__ import annotations

import logging
import re
from typing import AsyncGenerator

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

from ..config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for vLLM's OpenAI-compatible API.

    Streams chat completion tokens for low-latency translation output.
    Uses HTTP/2 connection pooling for multiplexed requests across
    concurrent sessions.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.llm_api_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def initialize(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def shutdown(self):
        if self._client:
            await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: float = 0.5,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from vLLM.

        Yields individual text tokens as they're generated.
        """
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "frequency_penalty": 0.3,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        in_think = False
        think_buf = ""
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                import json

                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        # Strip Qwen3 <think>...</think> blocks from stream
                        if in_think:
                            think_buf += content
                            if "</think>" in think_buf:
                                after = think_buf.split("</think>", 1)[1].lstrip()
                                in_think = False
                                think_buf = ""
                                if after:
                                    yield after
                        elif "<think>" in content:
                            in_think = True
                            think_buf = content
                        else:
                            yield content
