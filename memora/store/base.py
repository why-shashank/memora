"""StorageBackend — the persistence boundary (swappable per PRD FR; one impl for MVP)."""

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Owns connections to the backing store. Memory operations land here in M1."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the store is reachable and answering."""

    @abstractmethod
    async def dispose(self) -> None:
        """Release all connections."""
