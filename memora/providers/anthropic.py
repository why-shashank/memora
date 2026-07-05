"""Anthropic API LLM provider (BYO key via MEMORA_ANTHROPIC_API_KEY)."""

from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock

from memora.providers.base import LLMProvider


class AnthropicLLM(LLMProvider):
    def __init__(
        self, model: str, api_key: str | None = None, client: AsyncAnthropic | None = None
    ) -> None:
        self._model = model
        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)

    async def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 4096
    ) -> str:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            request["system"] = system
        response = await self._client.messages.create(**request)
        return "".join(block.text for block in response.content if isinstance(block, TextBlock))
