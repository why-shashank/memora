"""Hybrid retrieval (M2.1): embed the query, then fuse vector + FTS legs in the store.

S3 measured the embedding step — not the SQL — as the latency bottleneck, so the
one model call here is the thing to watch; the fusion itself is a single query.

Which is why each phase gets its own span (M2.4): a p95 regression is only actionable
if the trace says *which* phase moved. The attributes deliberately exclude the query
text — traces leave the deployment for someone's collector, and a query carries the
end user's own words. Length, counts and the top score answer "was this retrieval
healthy?" without shipping the content out.
"""

from memora.models import Scope
from memora.observability.tracing import tracer
from memora.providers.base import EmbeddingProvider
from memora.retrieval.rerank import NO_OP_RERANKER, Reranker
from memora.retrieval.scoring import rank
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
    """Memories most relevant to ``query``: RRF-fused, scope-filtered, trust-weighted.

    The store returns the whole fused pool; weighting reorders it before the top-``limit``
    cut, so a trusted memory ranked below ``limit`` on pure relevance can still surface.
    The reranker gets that same full pool, for the same reason, and has the last word on
    order — it does nothing unless one is supplied.
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
            ranked = await reranker.rerank(query, rank(candidates))

        results = ranked[:limit]
        span.set_attribute("memora.query_chars", len(query))
        span.set_attribute("memora.candidates", len(candidates))
        span.set_attribute("memora.results", len(results))
        if results:
            # how relevant the best answer was: a healthy retrieval that returns junk
            # looks identical to a good one until you can see this
            span.set_attribute("memora.top_score", results[0].score)
        return results
