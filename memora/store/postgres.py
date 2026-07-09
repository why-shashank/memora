"""Postgres + pgvector implementation of StorageBackend."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from memora.models import MemoryCreate
from memora.models.orm import AuditLog, ExtractionJob, Memory
from memora.store.base import ClaimedJob, StorageBackend


class PostgresStorage(StorageBackend):
    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def ping(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def enqueue_extraction(self, payload: dict[str, Any]) -> UUID:
        async with self.session_factory() as session:
            job = ExtractionJob(payload=payload)
            session.add(job)
            await session.commit()
            return job.id

    async def claim_extraction_job(self) -> ClaimedJob | None:
        # The classic single-statement Postgres queue claim: SKIP LOCKED makes
        # concurrent claimers pass over rows another transaction is taking, so a
        # job is handed out exactly once — no Redis, no advisory locks.
        oldest_pending = (
            select(ExtractionJob.id)
            .where(ExtractionJob.status == "pending")
            .order_by(ExtractionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        stmt = (
            update(ExtractionJob)
            .where(ExtractionJob.id == oldest_pending)
            .values(status="processing", attempts=ExtractionJob.attempts + 1)
            .returning(ExtractionJob.id, ExtractionJob.payload, ExtractionJob.attempts)
        )
        async with self.session_factory() as session:
            row = (await session.execute(stmt)).one_or_none()
            await session.commit()
        if row is None:
            return None
        return ClaimedJob(id=row.id, payload=row.payload, attempts=row.attempts)

    async def complete_extraction_job(self, job_id: UUID) -> None:
        async with self.session_factory() as session:
            await session.execute(
                update(ExtractionJob).where(ExtractionJob.id == job_id).values(status="done")
            )
            await session.commit()

    async def fail_extraction_job(self, job_id: UUID, *, error: str, retry: bool) -> None:
        async with self.session_factory() as session:
            await session.execute(
                update(ExtractionJob)
                .where(ExtractionJob.id == job_id)
                .values(status="pending" if retry else "failed", error=error)
            )
            await session.commit()

    async def add_memories(self, items: list[MemoryCreate]) -> list[UUID]:
        rows = [
            Memory(
                content=item.content,
                type=item.type.value,
                user_id=item.scope.user_id,
                agent_id=item.scope.agent_id,
                run_id=item.scope.run_id,
                app_id=item.scope.app_id,
                actor_type=item.actor_type.value,
                confidence=item.confidence,
                source=item.source,
            )
            for item in items
        ]
        async with self.session_factory() as session:
            session.add_all(rows)
            await session.flush()  # assigns ids so the audit rows can name them
            session.add_all(
                AuditLog(memory_id=row.id, action="created", actor_type=item.actor_type.value)
                for item, row in zip(items, rows, strict=True)
            )
            # one commit: a memory and its audit row land together or not at all
            await session.commit()
        return [row.id for row in rows]
