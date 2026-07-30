"""M2.1 — hybrid retrieval: vector + FTS legs fused with RRF, scope-filtered.
M2.2 — effective-confidence weighting: trust + type float above raw relevance.
M2.3 — rerank hook: a supplied reranker has the final say on order.

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
from memora.retrieval.rerank import Reranker
from memora.retrieval.scoring import effective_confidence
from memora.store.base import RetrievedMemory
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
_NEARISH = _vec(1.0, 0.2)  # slightly off-axis → a shade farther, so ranks are deterministic
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
    store: PostgresStorage,
    content: str,
    embedding: list[float],
    *,
    memory_type: str = "entity_fact",
    actor_type: str = "agent",
    confidence: float | None = None,
    scope: Scope | None = None,
) -> object:
    (memory_id,) = await store.add_memories(
        [
            MemoryCreate(
                content=content,
                type=memory_type,
                actor_type=actor_type,
                confidence=confidence,
                scope=scope or Scope(),
            )
        ],
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


# --- M2.2: effective-confidence weighting ---


def test_effective_confidence_combines_confidence_and_actor_trust() -> None:
    # a human-vouched memory outweighs an agent-inferred one at equal confidence
    assert effective_confidence(0.9, "human_correction") > effective_confidence(0.9, "agent")
    # no confidence = an assertion (correction/review), taken at full confidence, not penalised
    assert effective_confidence(None, "human_correction") == effective_confidence(
        1.0, "human_correction"
    )
    # confidence scales it: a hedged inference ranks below a confident one, same actor
    assert effective_confidence(0.3, "agent") < effective_confidence(0.9, "agent")


async def test_correction_floats_above_an_ordinary_fact_of_similar_relevance(
    store: PostgresStorage,
) -> None:
    # the fact is the nearer vector neighbour, so pure relevance ranks it first; the
    # correction's type priority + human-actor trust must float it to the top instead
    await _add(store, "Refund policy is thirty days", _NEAR, memory_type="entity_fact")
    correction = await _add(
        store,
        "Refund policy is thirty days",
        _NEARISH,
        memory_type="correction",
        actor_type="human_correction",
    )

    results = await retrieve(store, StubEmbedder(_NEAR), "refund policy")

    assert results[0].id == correction


async def test_higher_confidence_outranks_lower_confidence_at_equal_trust(
    store: PostgresStorage,
) -> None:
    # same type + actor; the more-confident memory wins even from the farther rank
    await _add(store, "Ships on the Growth plan", _NEAR, confidence=0.3)
    confident = await _add(store, "Ships on the Growth plan", _NEARISH, confidence=0.9)

    results = await retrieve(store, StubEmbedder(_NEAR), "Growth plan")

    assert results[0].id == confident


# --- M2.3: rerank hook ---


class ReversingReranker(Reranker):
    """Stands in for a second-stage model by inverting the order it is handed —
    an ordering no relevance signal in the pipeline would produce on its own."""

    async def rerank(self, query: str, candidates: list[RetrievedMemory]) -> list[RetrievedMemory]:
        return list(reversed(candidates))


async def test_reranker_has_the_final_say_on_order(store: PostgresStorage) -> None:
    await _add(store, "Customer requested a refund", _FAR)
    await _add(store, "The sky was clear tonight", _NEAR)

    weighted = await retrieve(store, StubEmbedder(_NEAR), "refund")
    reranked = await retrieve(store, StubEmbedder(_NEAR), "refund", reranker=ReversingReranker())

    assert len(weighted) == 2  # else reversing proves nothing
    assert [r.id for r in reranked] == [r.id for r in reversed(weighted)]


async def test_reranker_sees_the_whole_pool_before_the_limit_cut(
    store: PostgresStorage,
) -> None:
    # the lexical+vector hit outranks the vector-only one (fusion), so the latter is last;
    # a reranker can only promote it into a limit=1 answer if it saw the untruncated pool
    await _add(store, "Customer requested a refund", _FAR)
    ranked_last = await _add(store, "The sky was clear tonight", _NEAR)

    results = await retrieve(
        store, StubEmbedder(_NEAR), "refund", limit=1, reranker=ReversingReranker()
    )

    assert [r.id for r in results] == [ranked_last]
