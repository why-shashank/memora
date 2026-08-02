"""StorageBackend — the persistence boundary (swappable per PRD FR; one impl for MVP)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from memora.models import MemoryCreate, Scope


@dataclass(frozen=True)
class ClaimedJob:
    """A queue job handed to exactly one worker (attempts includes this claim)."""

    id: UUID
    payload: dict[str, Any]
    attempts: int


@dataclass(frozen=True)
class RetrievedMemory:
    """A memory returned by retrieval.

    ``score`` is the RRF relevance out of ``hybrid_search``; a reranker may rewrite it.
    ``actor_type`` and ``confidence`` travel with the memory so a caller can see who
    vouched for it and how sure it was (M2.4's search response, provenance-on-retrieval
    in M4.5) — they no longer influence ranking itself, see `retrieval/pipeline.py`.
    """

    id: UUID
    content: str
    type: str
    actor_type: str | None
    confidence: float | None
    score: float


class StorageBackend(ABC):
    """Owns connections to the backing store. Memory operations land here in M1."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the store is reachable and answering."""

    @abstractmethod
    async def dispose(self) -> None:
        """Release all connections."""

    @abstractmethod
    async def enqueue_extraction(self, payload: dict[str, Any]) -> UUID:
        """Queue an interaction for async extraction; returns the job id."""

    @abstractmethod
    async def claim_extraction_job(self) -> ClaimedJob | None:
        """Atomically claim the oldest pending job (None when the queue is empty).

        Concurrent claimers must never receive the same job.
        """

    @abstractmethod
    async def complete_extraction_job(self, job_id: UUID) -> None:
        """Mark a claimed job done."""

    @abstractmethod
    async def fail_extraction_job(self, job_id: UUID, *, error: str, retry: bool) -> None:
        """Record a failed attempt: back to the queue (retry) or dead-lettered."""

    @abstractmethod
    async def add_memories(
        self, items: list[MemoryCreate], embeddings: list[list[float]] | None = None
    ) -> list[UUID]:
        """Persist validated memories (born as candidates); returns their ids.

        Contract: each write lands atomically with a 'created' entry in the
        append-only audit log — no memory may exist without its audit trail.

        ``embeddings`` (when given) must align 1:1 with ``items`` — the vector
        for retrieval's semantic leg. Omitted, memories store no embedding.
        """

    @abstractmethod
    async def resolve_entity(
        self, *, name: str, type: str, aliases: list[str] | None = None
    ) -> UUID:
        """Find the entity these surface forms name, or create it; returns its id.

        Alias matching only — exact, after canonicalization. Lookups are partitioned
        by ``type``: a name shared by entities of different types must resolve to
        different entities, never merge them (S5).
        """

    @abstractmethod
    async def link_memory(self, *, memory_id: UUID, entity_ids: list[UUID]) -> None:
        """Record which entities a memory is about. Idempotent."""

    @abstractmethod
    async def merge_entities(self, *, source: UUID, target: UUID, actor_type: str) -> None:
        """Fold ``source`` into ``target``, moving its aliases and memory links.

        Contract: lands with an ``entity_merged`` audit entry naming what was
        absorbed — a resolution decision is a trust decision.
        """

    @abstractmethod
    async def split_entity(
        self, *, source: UUID, alias_keys: list[str], canonical_name: str, actor_type: str
    ) -> UUID:
        """Pull ``alias_keys`` out of ``source`` into a new entity; returns its id.

        The inverse of ``merge_entities`` when given the absorbed entity's aliases,
        which is what makes a merge reversible. Audited as ``entity_split``.
        """

    @abstractmethod
    async def hybrid_search(
        self,
        *,
        query_embedding: list[float],
        query_text: str,
        scope: Scope,
    ) -> list[RetrievedMemory]:
        """Vector (cosine) + FTS legs fused with RRF; scope-filtered, RRF-ordered.

        Returns the whole fused candidate pool, not a final top-N: trust weighting
        (M2.2) reorders these before the caller truncates, so a trusted memory below
        the output limit must still reach the scorer. Built as a fusion of independent
        legs so a third (entities, M2.8) can be added without disturbing these two.
        """
