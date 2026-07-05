"""Provider interfaces — the swappable seams for LLMs and embedding models (dev-plan §6)."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Maps text to vectors.

    One model per deployment: vectors from different models live in unrelated spaces,
    so the model is a setup-time choice (config) — switching later means a column
    migration plus re-embedding every stored memory.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Size of the vectors this model produces — must match the schema's vector column."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, returning one vector per input, in input order."""


class LLMProvider(ABC):
    """Single-turn text generation behind a common interface (BYO key or local model)."""

    @abstractmethod
    async def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 4096
    ) -> str:
        """Generate a completion for the prompt."""
