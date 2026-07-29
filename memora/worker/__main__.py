"""Entrypoint for the compose `worker` service: python -m memora.worker."""

import asyncio

from memora.config import Settings
from memora.observability.logging import configure_logging
from memora.providers import get_embedding_provider, get_llm_provider
from memora.store.postgres import PostgresStorage
from memora.worker.runner import run_forever


def main() -> None:
    configure_logging()
    settings = Settings()
    # fail fast: extraction needs an LLM, so a missing key should stop the boot,
    # not surface later as a queue that never drains
    llm = get_llm_provider(settings)
    # embed-on-write needs the embedding model loaded up front (M2.1)
    embedder = get_embedding_provider(settings)
    storage = PostgresStorage(settings.database_url)
    asyncio.run(run_forever(storage, llm, embedder))


if __name__ == "__main__":
    main()
