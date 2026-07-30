"""Application settings, loaded from MEMORA_-prefixed env vars (and a local .env file)."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORA_", env_file=".env")

    database_url: str = "postgresql+asyncpg://memora:memora@localhost:5432/memora"

    # observability — unset means spans are created but never exported: no egress by
    # default, which is the self-host/air-gap path. Also honours the standard unprefixed
    # OTEL_ name, since that is what an operator with a collector already sets.
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MEMORA_OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"
        ),
    )

    # providers — model choice is per-deployment; changing the embedding model
    # requires a column migration + re-embed (vectors are model-specific)
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str | None = None
