"""M0.3 — Settings load from environment with the MEMORA_ prefix."""

import pytest

from memora.config import Settings


def test_database_url_has_local_dev_default() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://memora:memora@localhost:5432/memora"


def test_env_var_with_memora_prefix_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORA_DATABASE_URL", "postgresql+asyncpg://user:pw@db:5432/prod")
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://user:pw@db:5432/prod"
