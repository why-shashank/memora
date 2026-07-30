"""FastAPI application factory — routes land with their milestones."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memora.config import Settings
from memora.models import ActorType, Scope
from memora.observability.logging import configure_logging
from memora.observability.tracing import configure_tracing
from memora.providers import get_embedding_provider
from memora.providers.base import EmbeddingProvider
from memora.retrieval import retrieve
from memora.store.base import StorageBackend
from memora.store.postgres import PostgresStorage

log = structlog.get_logger()


def get_storage(request: Request) -> StorageBackend:
    storage: StorageBackend = request.app.state.storage
    return storage


def get_embedder(request: Request) -> EmbeddingProvider:
    embedder: EmbeddingProvider = request.app.state.embedder
    return embedder


class IngestRequest(BaseModel):
    """POST /v1/memories body — an interaction to remember (extraction runs async).

    ``actor_type`` is limited to the actors an ingest pipeline can legitimately
    claim; ``human_correction``/``human_review`` are reserved for the correction
    (M3.1) and review (M4.2) flows — they carry trust weight extraction must not
    be able to claim for itself.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    conversation: str = Field(min_length=1)
    scope: Scope = Field(default_factory=Scope)
    actor_type: Literal[ActorType.agent, ActorType.user_stated, ActorType.system] = ActorType.agent


class SearchRequest(BaseModel):
    """POST /v1/memories/search body — the injection call an agent makes per turn."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    scope: Scope = Field(default_factory=Scope)
    # the fused candidate pool bounds what is actually available, so a very large limit
    # buys nothing; the cap keeps one request from asking for an unbounded injection
    limit: int = Field(default=10, ge=1, le=50)


class MemoryHit(BaseModel):
    """One retrieved memory. ``score`` and the actor/confidence pair travel with it so the
    caller can see *why* it ranked — a memory you can't account for isn't trustworthy."""

    id: UUID
    content: str
    type: str
    actor_type: str | None
    confidence: float | None
    score: float


class SearchResponse(BaseModel):
    memories: list[MemoryHit]


def create_app(
    settings: Settings | None = None, embedder: EmbeddingProvider | None = None
) -> FastAPI:
    app_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.storage = PostgresStorage(app_settings.database_url)
        # loading the embedding model costs seconds: pay it at boot, so /healthz only goes
        # green once search can actually serve, and no cold start lands inside the first
        # (p95-measured) request. Injectable so tests need no real model.
        app.state.embedder = (
            embedder if embedder is not None else get_embedding_provider(app_settings)
        )
        log.info("api_started")
        yield
        await app.state.storage.dispose()

    app = FastAPI(title="memora", lifespan=lifespan)
    configure_tracing(app, otlp_endpoint=app_settings.otel_exporter_otlp_endpoint)

    @app.get("/healthz")
    async def healthz(storage: Annotated[StorageBackend, Depends(get_storage)]) -> JSONResponse:
        if await storage.ping():
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.post("/v1/memories", status_code=202)
    async def ingest_memories(
        body: IngestRequest, storage: Annotated[StorageBackend, Depends(get_storage)]
    ) -> dict[str, str]:
        # 202: accepted for async extraction — the API never blocks on the LLM
        job_id = await storage.enqueue_extraction(
            {
                "conversation": body.conversation,
                "scope": body.scope.model_dump(),
                "actor_type": body.actor_type.value,
            }
        )
        return {"job_id": str(job_id)}

    @app.post("/v1/memories/search")
    async def search_memories(
        body: SearchRequest,
        storage: Annotated[StorageBackend, Depends(get_storage)],
        embedder: Annotated[EmbeddingProvider, Depends(get_embedder)],
    ) -> SearchResponse:
        memories = await retrieve(storage, embedder, body.query, scope=body.scope, limit=body.limit)
        return SearchResponse(memories=[MemoryHit(**vars(m)) for m in memories])

    return app


app = create_app()
