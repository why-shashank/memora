"""FastAPI application factory — routes land with their milestones."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memora.config import Settings
from memora.observability.logging import configure_logging
from memora.observability.tracing import configure_tracing
from memora.store.base import StorageBackend
from memora.store.postgres import PostgresStorage

log = structlog.get_logger()


def get_storage(request: Request) -> StorageBackend:
    storage: StorageBackend = request.app.state.storage
    return storage


class IngestRequest(BaseModel):
    """POST /v1/memories body — an interaction to remember (extraction runs async)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    conversation: str = Field(min_length=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.storage = PostgresStorage(app_settings.database_url)
        log.info("api_started")
        yield
        await app.state.storage.dispose()

    app = FastAPI(title="memora", lifespan=lifespan)
    configure_tracing(app)

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
        job_id = await storage.enqueue_extraction({"conversation": body.conversation})
        return {"job_id": str(job_id)}

    return app


app = create_app()
