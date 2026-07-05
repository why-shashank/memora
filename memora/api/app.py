"""FastAPI application factory — routes land with their milestones."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from memora.config import Settings
from memora.observability.logging import configure_logging
from memora.observability.tracing import configure_tracing
from memora.store.base import StorageBackend
from memora.store.postgres import PostgresStorage

log = structlog.get_logger()


def get_storage(request: Request) -> StorageBackend:
    storage: StorageBackend = request.app.state.storage
    return storage


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

    return app


app = create_app()
