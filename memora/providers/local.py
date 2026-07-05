"""Local embeddings via sentence-transformers — the zero-key, zero-egress default."""

import asyncio

from memora.providers.base import EmbeddingProvider


class SentenceTransformersEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # torch is heavy, so it ships as an extra, not a core dep
            raise RuntimeError(
                "sentence-transformers is not installed; install the local extra:"
                " uv add 'memora[local]' (or pip install 'memora[local]')"
            ) from exc
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # encode() is blocking CPU inference — keep it off the event loop. S3 flagged this
        # in-process step (not Postgres) as the bottleneck to revisit under concurrency.
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return [vector.tolist() for vector in vectors]
