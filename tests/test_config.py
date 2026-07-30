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


def test_trace_export_is_off_unless_an_endpoint_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # no endpoint = spans go nowhere: memora makes no outbound connection it wasn't told to
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert Settings(_env_file=None).otel_exporter_otlp_endpoint is None

    # an operator running a collector sets the standard unprefixed name out of habit;
    # honouring only MEMORA_-prefixed here would silently export nothing
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/v1/traces")
    assert Settings(_env_file=None).otel_exporter_otlp_endpoint == "http://collector:4318/v1/traces"
