"""memora.providers — provider factories building the config-selected impl (dev-plan §6)."""

from memora.config import Settings
from memora.models.orm import EMBEDDING_DIM
from memora.providers.anthropic import AnthropicLLM
from memora.providers.base import EmbeddingProvider, LLMProvider
from memora.providers.local import SentenceTransformersEmbedding


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = SentenceTransformersEmbedding(settings.embedding_model)
    if provider.dimension != EMBEDDING_DIM:
        raise ValueError(
            f"embedding model {settings.embedding_model!r} produces {provider.dimension}-d "
            f"vectors but the schema stores {EMBEDDING_DIM}-d — a different model needs a "
            "column migration plus re-embedding of existing memories"
        )
    return provider


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.anthropic_api_key is None:
        raise ValueError(
            "MEMORA_ANTHROPIC_API_KEY is not set — memora brings no LLM key of its own; "
            "put yours in .env (see .env.example)"
        )
    return AnthropicLLM(model=settings.llm_model, api_key=settings.anthropic_api_key)
