"""M2.1 — hybrid retrieval: vector + FTS legs fused with RRF, scope-filtered.
M2.3 — rerank hook: a supplied reranker has the final say on order.
M2.4 — per-phase spans so a slow retrieval says which phase was slow.
M2.5a — ordering is relevance alone; trust must not override it.

M2.2's trust/type weighting stage was removed in M2.5a after the eval harness measured it as
worse than no stage at any strength, so its three tests went with it — they encoded the
behaviour that turned out to be wrong. See `retrieval/pipeline.py` for the finding.

The vector space is stubbed with deterministic literals so these tests exercise
the fusion SQL and ranking behaviour, not embedding quality — S2 already
validated relevance against a real model.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import text

from memora.models import MemoryCreate, Scope
from memora.models.orm import EMBEDDING_DIM
from memora.providers.base import EmbeddingProvider
from memora.retrieval import retrieve
from memora.retrieval.rerank import Reranker
from memora.store.base import RetrievedMemory
from memora.store.postgres import PostgresStorage


@pytest.fixture
async def store(migrated_db_url: str) -> AsyncIterator[PostgresStorage]:
    storage = PostgresStorage(migrated_db_url)
    async with storage.session_factory() as session:
        await session.execute(text("TRUNCATE memories, audit_log, memory_entities"))
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


# --- M2.5c: the keyword leg requires every query term ---


async def test_keyword_leg_ignores_a_memory_matching_only_one_query_term(
    store: PostgresStorage,
) -> None:
    # "policy" is incidental to both memories; only one is about refunds at all. Under the
    # OR-semantics this replaced, sharing that one word earned the aisle-seat memory a
    # full-strength keyword leg — the M2.4 smoke test caught exactly this, and at 20K rows
    # it filled 80% of the fusion pool with matches like it.
    #
    # Asserting the score rather than the order is deliberate: under OR the two memories fuse
    # to *identical* scores (each wins one leg and places second in the other), so an ordering
    # assertion would be decided by the UUID tiebreak — a coin flip, red or green by luck.
    await _add(store, "Aisle seat policy for frequent flyers", _NEAR)
    refunds = await _add(store, "The refund policy is thirty days", _FAR)

    results = await retrieve(store, StubEmbedder(_NEAR), "refund policy")

    scores = {r.content: r.score for r in results}
    # nearest vector, and nothing else: rank 1 of one leg is 1/(60+1). A keyword leg would
    # add another ~1/61 on top, which is precisely the bug.
    assert scores["Aisle seat policy for frequent flyers"] == pytest.approx(1 / 61)
    # and the consequence: the memory that actually answers the query comes first
    assert results[0].id == refunds


# --- M2.5a: trust does not override relevance ---


async def test_a_trusted_correction_cannot_displace_a_far_more_relevant_memory(
    store: PostgresStorage,
) -> None:
    # the golden-set failure in miniature, kept as the guard against reintroducing it. The
    # answer wins *both* legs — roughly twice the RRF of a single-leg hit, the widest gap the
    # pool produces — while carrying the weakest possible trust (agent, hedged); the
    # correction wins one leg and carries the strongest. This failed against M2.2's weighting,
    # and fails again the moment anything reorders the pool on a memory's own attributes.
    answer = await _add(store, "Refund window is thirty days", _NEAR, confidence=0.5)
    await _add(
        store,
        "The sky was clear tonight",
        _FAR,
        memory_type="correction",
        actor_type="human_correction",
    )

    results = await retrieve(store, StubEmbedder(_NEAR), "refund")

    assert results[0].id == answer


# --- M2.4: per-phase latency attribution ---


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Capture emitted spans. Attaches to the SDK provider if an app already installed one
    (the global provider can only be set once per process), else installs one."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    yield exporter
    processor.shutdown()


async def test_retrieve_attributes_latency_to_each_phase(
    store: PostgresStorage, spans: InMemorySpanExporter
) -> None:
    # S3 found the embedding call, not Postgres, is the bottleneck under load — which is
    # only actionable if a slow retrieval says *which* phase was slow.
    await _add(store, "Refund window is thirty days", _NEAR)
    await _add(store, "Prefers aisle seats", _FAR)

    await retrieve(store, StubEmbedder(_NEAR), "refund window")

    emitted = {span.name: span for span in spans.get_finished_spans()}
    assert {"memora.retrieve.embed", "memora.retrieve.search", "memora.retrieve.rank"} <= set(
        emitted
    )
    parent = emitted["memora.retrieve"]
    assert parent.attributes is not None
    assert parent.attributes["memora.results"] == 2
    assert parent.attributes["memora.query_chars"] == len("refund window")
    # the raw query is deliberately absent: traces leave the deployment, queries hold user data
    assert not any("refund" in str(value) for value in parent.attributes.values())
