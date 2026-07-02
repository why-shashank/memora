"""Application settings, loaded from MEMORA_-prefixed env vars (and a local .env file)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORA_", env_file=".env")

    database_url: str = "postgresql+asyncpg://memora:memora@localhost:5432/memora"
