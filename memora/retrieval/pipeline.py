"""Hybrid retrieval (M2.1): embed the query, then fuse vector + FTS legs in the store.

S3 measured the embedding step — not the SQL — as the latency bottleneck, so the
one model call here is the thing to watch; the fusion itself is a single query.

Which is why each phase gets its own span (M2.4): a p95 regression is only actionable
if the trace says *which* phase moved. The attributes deliberately exclude the query
text — traces leave the deployment for someone's collector, and a query carries the
end user's own words. Length, counts and the top score answer "was this retrieval
healthy?" without shipping the content out.

⚠️ **There is no trust/type weighting stage, and that is a measured decision (M2.5a).**
M2.2 shipped one — ``score × type_weight × effective_confidence``, floating corrections and
policies above ordinary facts per PRD FR-1.1 — and M2.5's golden set found it made retrieval
*worse than not having it*: hit@1 0.65 → 0.40, MRR 0.78 → 0.57. Retuning it into a bounded
bonus (``score × (1 + γ·boost)``, the shape the diagnosis called for) recovered most of that
but still lost, and a sweep of four boost shapes × nine strengths found **no γ > 0 that beat
plain fusion** — the damage fell monotonically to zero as the stage was turned off.

The reason is structural, and worth remembering before adding any similar stage. RRF is
*flat*: inside a pool of 20 the whole score range is 2.62×, and neighbouring ranks differ by
about 1.6%. A boost is a property of a memory *alone* — a correction is boosted in every pool
it lands in, including the many where it is off-topic noise — so any boost big enough to flip
a genuine near-tie is also big enough to flip an arbitrary adjacent pair, and there is nothing
in the scores to tell those two situations apart. On the golden set it promoted the right
memory on 1 query of 20 and demoted it on 6.

FR-1.1's real requirement — a correction beating the stale fact it corrects — is a
*supersession* problem, not a ranking one: returning the stale memory second is still
returning it. It lands in M3.2 as a status filter, where the correction/fact relationship is
recorded rather than guessed at from a score.
"""

from memora.models import Scope
from memora.observability.tracing import tracer
from memora.providers.base import EmbeddingProvider
from memora.retrieval.rerank import NO_OP_RERANKER, Reranker
from memora.store.base import RetrievedMemory, StorageBackend


async def retrieve(
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    query: str,
    *,
    scope: Scope | None = None,
    limit: int = 10,
    reranker: Reranker = NO_OP_RERANKER,
) -> list[RetrievedMemory]:
    """Memories most relevant to ``query``: RRF-fused and scope-filtered, best first.

    The store returns the whole fused pool and the reranker sees all of it, before the
    top-``limit`` cut — a second stage exists precisely to lift a candidate that first-stage
    relevance ranked below the cut. It has the last word on order, and does nothing unless
    one is supplied.

    Ordering is relevance and nothing else. A trust/type weighting stage sat between the two
    until M2.5a, when the eval harness measured it as worse than no stage at any strength —
    see the module docstring for why, and M3.2 for where corrections get handled properly.
    """
    with tracer.start_as_current_span("memora.retrieve") as span:
        with tracer.start_as_current_span("memora.retrieve.embed"):
            [query_embedding] = await embedder.embed([query])
        with tracer.start_as_current_span("memora.retrieve.search"):
            candidates = await storage.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                scope=scope or Scope(),
            )
        with tracer.start_as_current_span("memora.retrieve.rank"):
            ranked = await reranker.rerank(query, candidates)

        results = ranked[:limit]
        span.set_attribute("memora.query_chars", len(query))
        span.set_attribute("memora.candidates", len(candidates))
        span.set_attribute("memora.results", len(results))
        if results:
            # how relevant the best answer was: a healthy retrieval that returns junk
            # looks identical to a good one until you can see this
            span.set_attribute("memora.top_score", results[0].score)
        return results
