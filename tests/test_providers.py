"""M0.5 — provider abstractions: local embeddings, Anthropic LLM, config-selected factories."""

from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import TextBlock

from memora.config import Settings
from memora.models.orm import EMBEDDING_DIM
from memora.providers import get_embedding_provider, get_llm_provider
from memora.providers.anthropic import AnthropicLLM
from memora.providers.base import EmbeddingProvider


@pytest.fixture(scope="module")
def embedder() -> EmbeddingProvider:
    """The default (config-selected) local embedding provider; model loads once per module."""
    return get_embedding_provider(Settings(_env_file=None))


async def test_embed_returns_one_vector_per_text_at_declared_dimension(
    embedder: EmbeddingProvider,
) -> None:
    vectors = await embedder.embed(["the cat sat on the mat", "quarterly invoice #4821"])
    assert len(vectors) == 2
    assert all(len(v) == embedder.dimension for v in vectors)
    assert embedder.dimension == EMBEDDING_DIM


async def test_embed_is_deterministic_and_order_preserving(embedder: EmbeddingProvider) -> None:
    cat, invoice = await embedder.embed(["the cat sat on the mat", "quarterly invoice #4821"])
    (cat_again,) = await embedder.embed(["the cat sat on the mat"])
    assert cat == pytest.approx(cat_again, abs=1e-5)
    assert cat != pytest.approx(invoice, abs=1e-5)


def test_embedding_factory_rejects_schema_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memora.providers.EMBEDDING_DIM", 9999)
    with pytest.raises(ValueError, match="9999"):
        get_embedding_provider(Settings(_env_file=None))


class _RecordingStubClient:
    """Duck-typed stand-in for AsyncAnthropic that captures the request it was sent."""

    def __init__(self, reply: str) -> None:
        self.last_request: dict[str, Any] | None = None

        async def create(**kwargs: Any) -> Any:
            self.last_request = kwargs
            return SimpleNamespace(content=[TextBlock(type="text", text=reply)])

        self.messages = SimpleNamespace(create=create)


async def test_generate_sends_prompt_and_returns_model_text() -> None:
    client = _RecordingStubClient(reply="PARIS")
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001", client=client)  # type: ignore[arg-type]
    answer = await llm.generate("capital of France?", system="answer in caps")
    assert answer == "PARIS"
    assert client.last_request is not None
    assert client.last_request["model"] == "claude-haiku-4-5-20251001"
    assert client.last_request["messages"] == [{"role": "user", "content": "capital of France?"}]
    assert client.last_request["system"] == "answer in caps"


def test_llm_factory_requires_byo_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORA_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MEMORA_ANTHROPIC_API_KEY"):
        get_llm_provider(Settings(_env_file=None))


def test_llm_factory_builds_provider_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORA_ANTHROPIC_API_KEY", "test-key")
    provider = get_llm_provider(Settings(_env_file=None))
    assert isinstance(provider, AnthropicLLM)
