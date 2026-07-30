"""The rerank seam (M2.3) — where a second-stage relevance model plugs in. No-op by default.

Weighted score fusion (M2.1 + M2.2) is the MVP's entire ranking story; dev-plan §5 keeps a
cross-encoder rerank explicitly optional-and-later, because taking on a model dependency
before measuring the need is the wrong trade. So this ships the seam and nothing more:
``retrieve`` always calls a reranker, and the default one hands the candidates straight back.

A reranker runs **last** — after trust weighting, before the ``limit`` cut. Seeing the whole
pool is the point: a reranker exists to lift a candidate that first-stage relevance ranked
below the cut. The consequence is that its order overrides M2.2's trust weighting, so an
implementation that must not bury corrections should scale each candidate's existing ``score``
rather than reorder from scratch. Whether that matters is a question for M2.5's eval harness.
"""

from abc import ABC, abstractmethod

from memora.store.base import RetrievedMemory


class Reranker(ABC):
    """Reorders one query's retrieval candidates, best first."""

    @abstractmethod
    async def rerank(self, query: str, candidates: list[RetrievedMemory]) -> list[RetrievedMemory]:
        """Return the candidates reordered, best first.

        Takes the query *text*: a cross-encoder scores (query, memory) pairs, where the
        first stage only ever used the query's vector. Async because every plausible
        implementation is a model call — local weights or a remote rerank API.
        """


class NoOpReranker(Reranker):
    """The default — leaves the weighted order exactly as M2.2 produced it."""

    async def rerank(self, query: str, candidates: list[RetrievedMemory]) -> list[RetrievedMemory]:
        return candidates


# A default argument can't be a call (ruff B008), and the no-op holds no state, so one
# shared instance serves every retrieval.
NO_OP_RERANKER: Reranker = NoOpReranker()
