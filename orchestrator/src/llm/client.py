from __future__ import annotations

import logging
from typing import AsyncGenerator

import httpx

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
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from vLLM.

        Yields individual text tokens as they're generated.
        """
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

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
                        yield content
