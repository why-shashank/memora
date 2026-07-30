"""Hybrid retrieval (M2.1): embed the query, then fuse vector + FTS legs in the store.

S3 measured the embedding step — not the SQL — as the latency bottleneck, so the
one model call here is the thing to watch; the fusion itself is a single query.
"""

from memora.models import Scope
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
    [query_embedding] = await embedder.embed([query])
    candidates = await storage.hybrid_search(
        query_embedding=query_embedding,
        query_text=query,
        scope=scope or Scope(),
    )
    reranked = await reranker.rerank(query, rank(candidates))
    return reranked[:limit]
