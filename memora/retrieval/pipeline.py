"""Hybrid retrieval (M2.1): embed the query, then fuse vector + FTS legs in the store.

S3 measured the embedding step — not the SQL — as the latency bottleneck, so the
one model call here is the thing to watch; the fusion itself is a single query.
"""

from memora.models import Scope
from memora.providers.base import EmbeddingProvider
from memora.store.base import RetrievedMemory, StorageBackend


async def retrieve(
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    query: str,
    *,
    scope: Scope | None = None,
    limit: int = 10,
) -> list[RetrievedMemory]:
    """Return memories most relevant to ``query``, RRF-fused and scope-filtered."""
    [query_embedding] = await embedder.embed([query])
    return await storage.hybrid_search(
        query_embedding=query_embedding,
        query_text=query,
        scope=scope or Scope(),
        limit=limit,
    )
