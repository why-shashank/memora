"""M2.1 — hybrid retrieval: vector + FTS legs fused with RRF, scope-filtered.

The vector space is stubbed with deterministic literals so these tests exercise
the fusion SQL and ranking behaviour, not embedding quality — S2 already
validated relevance against a real model.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from memora.models import MemoryCreate, Scope
from memora.models.orm import EMBEDDING_DIM
from memora.providers.base import EmbeddingProvider
from memora.retrieval import retrieve
from memora.store.postgres import PostgresStorage


@pytest.fixture
async def store(migrated_db_url: str) -> AsyncIterator[PostgresStorage]:
    storage = PostgresStorage(migrated_db_url)
    async with storage.session_factory() as session:
        await session.execute(text("TRUNCATE memories, audit_log"))
        await session.commit()
    yield storage
    await storage.dispose()


def _vec(x: float, y: float = 0.0) -> list[float]:
    """A vector pointing in a chosen direction, zeros elsewhere. Cosine distance
    depends only on direction, so two axes suffice to place memories near or far."""
    v = [0.0] * EMBEDDING_DIM
    v[0], v[1] = x, y
    return v


_NEAR = _vec(1.0)  # aligned with the query vector → smallest cosine distance
_FAR = _vec(-1.0)  # opposite the query vector → largest cosine distance


class StubEmbedder(EmbeddingProvider):
    """Returns one fixed query vector so the test owns the vector space."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dimension(self) -> int:
        return len(self._vector)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


async def _add(
    store: PostgresStorage, content: str, embedding: list[float], *, scope: Scope | None = None
) -> object:
    (memory_id,) = await store.add_memories(
        [MemoryCreate(content=content, type="entity_fact", scope=scope or Scope())],
        embeddings=[embedding],
    )
    return memory_id


async def test_vector_leg_retrieves_semantic_match_without_lexical_overlap(
    store: PostgresStorage,
) -> None:
    # the query shares no words with either memory → FTS empty → pure vector leg
    near = await _add(store, "The northern lights danced overhead", _NEAR)
    await _add(store, "Refund window is thirty days", _FAR)

    results = await retrieve(store, StubEmbedder(_NEAR), "aurora colours glowing")

    assert results[0].id == near  # nearest embedding wins with zero keyword help


async def test_fts_leg_lifts_a_lexical_match_above_a_closer_vector_neighbor(
    store: PostgresStorage,
) -> None:
    # the lexical match sits FAR in vector space; a non-matching memory sits NEAR.
    # RRF must let the dual-leg (vector + FTS) hit outrank the vector-only neighbour.
    lexical = await _add(store, "Customer requested a refund", _FAR)
    await _add(store, "The sky was clear tonight", _NEAR)

    results = await retrieve(store, StubEmbedder(_NEAR), "refund")

    assert results[0].id == lexical


async def test_scope_filter_restricts_results_to_the_query_scope(
    store: PostgresStorage,
) -> None:
    mine = await _add(store, "refund note", _NEAR, scope=Scope(user_id="u-1"))
    await _add(store, "refund note", _NEAR, scope=Scope(user_id="u-2"))

    results = await retrieve(store, StubEmbedder(_NEAR), "refund", scope=Scope(user_id="u-1"))

    assert [r.id for r in results] == [mine]  # the other user's memory is invisible
