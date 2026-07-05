"""M0.7 — GET /healthz reports real readiness: 200 iff the database answers."""

from fastapi.testclient import TestClient

from memora.api.app import create_app
from memora.config import Settings


def test_healthz_ok_when_db_reachable(migrated_db_url: str) -> None:
    app = create_app(Settings(_env_file=None, database_url=migrated_db_url))
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_503_when_db_unreachable() -> None:
    app = create_app(
        Settings(_env_file=None, database_url="postgresql+asyncpg://x:x@localhost:9/nodb")
    )
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
