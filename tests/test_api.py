"""M0.7 — GET /healthz reports real readiness: 200 iff the database answers.
M1.3 — POST /v1/memories enqueues extraction and returns 202 immediately.
M2.4 — POST /v1/memories/search returns scope-filtered, ranked memories."""

import asyncio
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from memora.api.app import create_app
from memora.config import Settings
from memora.models import MemoryCreate, Scope
from memora.models.orm import EMBEDDING_DIM
from memora.providers.base import EmbeddingProvider
from memora.store.postgres import PostgresStorage


def _vec(x: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0] = x
    return v


_QUERY_VECTOR = _vec(1.0)  # what the stub embedder returns for any query
_AWAY = _vec(-1.0)  # opposite direction → the far vector neighbour


class StubEmbedder(EmbeddingProvider):
    """One fixed vector for any text: these tests own the vector space, so relevance is
    deterministic and the real model (a ~6s load) never enters the API tests."""

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_QUERY_VECTOR for _ in texts]


def _app(db_url: str) -> FastAPI:
    return create_app(Settings(_env_file=None, database_url=db_url), embedder=StubEmbedder())


async def _seed(db_url: str, rows: list[tuple[str, list[float], Scope]]) -> None:
    """Replace the memory table with exactly these memories, so search results are the
    test's own and not whatever an earlier test left behind."""
    store = PostgresStorage(db_url)
    try:
        async with store.session_factory() as session:
            await session.execute(text("TRUNCATE memories, audit_log"))
            await session.commit()
        await store.add_memories(
            [
                MemoryCreate(content=content, type="entity_fact", scope=scope)
                for content, _, scope in rows
            ],
            embeddings=[embedding for _, embedding, _ in rows],
        )
    finally:
        await store.dispose()


def test_healthz_ok_when_db_reachable(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_503_when_db_unreachable() -> None:
    app = _app("postgresql+asyncpg://x:x@localhost:9/nodb")
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_post_memories_enqueues_and_returns_202(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
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
    assert payload is not None
    assert payload["conversation"] == "Customer: I moved to Berlin."
    # attribution defaults when the caller sends none: unscoped, agent actor (M1.4)
    assert payload["scope"] == {"user_id": None, "agent_id": None, "run_id": None, "app_id": None}
    assert payload["actor_type"] == "agent"


def test_post_memories_rejects_blank_conversation(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        assert client.post("/v1/memories", json={"conversation": "   "}).status_code == 422
        assert client.post("/v1/memories", json={}).status_code == 422


def test_post_memories_threads_scope_and_actor_into_the_job(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        response = client.post(
            "/v1/memories",
            json={
                "conversation": "Customer: only email me. (scoped-thread-test)",
                "scope": {"user_id": "u-42", "app_id": "support"},
                "actor_type": "user_stated",
            },
        )
    assert response.status_code == 202
    job_id = UUID(response.json()["job_id"])

    async def fetch_payload() -> dict[str, object] | None:
        store = PostgresStorage(migrated_db_url)
        try:
            while (claimed := await store.claim_extraction_job()) is not None:
                if claimed.id == job_id:
                    return dict(claimed.payload)
            return None
        finally:
            await store.dispose()

    payload = asyncio.run(fetch_payload())
    assert payload is not None
    assert payload["scope"] == {
        "user_id": "u-42",
        "agent_id": None,
        "run_id": None,
        "app_id": "support",
    }
    assert payload["actor_type"] == "user_stated"


def test_post_memories_rejects_bad_scope_or_reserved_actor(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        # typo'd scope key: must not silently file the memory unscoped
        assert (
            client.post(
                "/v1/memories", json={"conversation": "hi", "scope": {"userid": "u1"}}
            ).status_code
            == 422
        )
        # human_* actors are reserved for the correction/review flows (M3.1/M4.2) —
        # extraction output must not be able to masquerade as human-verified
        for reserved in ("human_correction", "human_review"):
            assert (
                client.post(
                    "/v1/memories", json={"conversation": "hi", "actor_type": reserved}
                ).status_code
                == 422
            )


# --- M2.4: POST /v1/memories/search ---


def test_search_returns_ranked_memories_with_scores(migrated_db_url: str) -> None:
    asyncio.run(
        _seed(
            migrated_db_url,
            [
                ("Refund window is thirty days", _QUERY_VECTOR, Scope()),
                ("Prefers aisle seats", _AWAY, Scope()),
            ],
        )
    )
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        response = client.post("/v1/memories/search", json={"query": "refund window"})

    assert response.status_code == 200
    memories = response.json()["memories"]
    assert [m["content"] for m in memories] == [
        "Refund window is thirty days",  # matches on both legs
        "Prefers aisle seats",  # vector leg only, and the far neighbour
    ]
    top = memories[0]
    assert top["type"] == "entity_fact"
    assert top["actor_type"] == "agent"
    assert top["score"] > memories[1]["score"]  # the caller can see *why* it ranked
    UUID(top["id"])


def test_search_limit_caps_the_returned_memories(migrated_db_url: str) -> None:
    asyncio.run(
        _seed(
            migrated_db_url,
            [
                ("Refund window is thirty days", _QUERY_VECTOR, Scope()),
                ("Refund window changed last year", _AWAY, Scope()),
            ],
        )
    )
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        response = client.post("/v1/memories/search", json={"query": "refund window", "limit": 1})

    assert [m["content"] for m in response.json()["memories"]] == ["Refund window is thirty days"]


def test_search_is_isolated_by_scope(migrated_db_url: str) -> None:
    # identical content under two users: the tenant filter is the only thing separating them
    asyncio.run(
        _seed(
            migrated_db_url,
            [
                ("Refund window is thirty days", _QUERY_VECTOR, Scope(user_id="u-1")),
                ("Refund window is thirty days", _QUERY_VECTOR, Scope(user_id="u-2")),
            ],
        )
    )
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        response = client.post(
            "/v1/memories/search",
            json={"query": "refund window", "scope": {"user_id": "u-1"}},
        )

    assert len(response.json()["memories"]) == 1  # u-2's copy is invisible


def test_search_rejects_a_blank_query_bad_scope_or_bad_limit(migrated_db_url: str) -> None:
    app = _app(migrated_db_url)
    with TestClient(app) as client:
        assert client.post("/v1/memories/search", json={"query": "   "}).status_code == 422
        assert client.post("/v1/memories/search", json={}).status_code == 422
        # a typo'd scope key must not silently widen the search across tenants
        assert (
            client.post(
                "/v1/memories/search", json={"query": "refund", "scope": {"userid": "u-1"}}
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/memories/search", json={"query": "refund", "limit": 0}).status_code
            == 422
        )
