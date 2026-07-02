"""Shared fixtures: a real Postgres+pgvector container with migrations applied."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def migrated_db_url() -> Iterator[str]:
    """Async URL of a throwaway pgvector Postgres migrated to head."""
    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        yield url
