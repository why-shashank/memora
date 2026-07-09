"""M0.7 — GET /healthz reports real readiness: 200 iff the database answers.
M1.3 — POST /v1/memories enqueues extraction and returns 202 immediately."""

import asyncio
from uuid import UUID

from fastapi.testclient import TestClient

from memora.api.app import create_app
from memora.config import Settings
from memora.store.postgres import PostgresStorage


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


def test_post_memories_enqueues_and_returns_202(migrated_db_url: str) -> None:
    app = create_app(Settings(_env_file=None, database_url=migrated_db_url))
    with TestClient(app) as client:
        response = client.post(
            "/v1/memories", json={"conversation": "Customer: I moved to Berlin."}
        )

    assert response.status_code == 202
    job_id = UUID(response.json()["job_id"])  # a real, trackable job handle

    # the interaction is on the queue for the worker — nothing was processed inline
    async def fetch_claimed_payload() -> dict[str, object] | None:
        store = PostgresStorage(migrated_db_url)
        try:
            while (claimed := await store.claim_extraction_job()) is not None:
                if claimed.id == job_id:
                    return dict(claimed.payload)
            return None
        finally:
            await store.dispose()

    payload = asyncio.run(fetch_claimed_payload())
    assert payload == {"conversation": "Customer: I moved to Berlin."}


def test_post_memories_rejects_blank_conversation(migrated_db_url: str) -> None:
    app = create_app(Settings(_env_file=None, database_url=migrated_db_url))
    with TestClient(app) as client:
        assert client.post("/v1/memories", json={"conversation": "   "}).status_code == 422
        assert client.post("/v1/memories", json={}).status_code == 422
