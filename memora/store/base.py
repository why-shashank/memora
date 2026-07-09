"""StorageBackend — the persistence boundary (swappable per PRD FR; one impl for MVP)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from memora.models import MemoryCreate


@dataclass(frozen=True)
class ClaimedJob:
    """A queue job handed to exactly one worker (attempts includes this claim)."""

    id: UUID
    payload: dict[str, Any]
    attempts: int


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
    async def add_memories(self, items: list[MemoryCreate]) -> list[UUID]:
        """Persist validated memories (born as candidates); returns their ids.

        Contract: each write lands atomically with a 'created' entry in the
        append-only audit log — no memory may exist without its audit trail.
        """
