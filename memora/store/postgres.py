"""Postgres + pgvector implementation of StorageBackend."""

from typing import Any
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from memora.models import MemoryCreate, Scope
from memora.models.orm import AuditLog, ExtractionJob, Memory
from memora.store.base import ClaimedJob, RetrievedMemory, StorageBackend

# Reciprocal-rank fusion: a memory's score is the sum over legs of 1/(k + rank).
# k=60 is the canonical RRF constant — it damps the top ranks so no single leg
# dominates. _POOL caps how many candidates each leg contributes before fusion.
_RRF_K = 60
_POOL = 20

# The hybrid query: two independent ranked legs (vector, FTS) over the same
# scope-filtered set, fused by RRF. {scope_where} is composed from fixed column
# names (never user input); all values bind as parameters. A third leg (entity
# traversal, M2.8) slots in as another CTE + COALESCE term without touching these.
_HYBRID_SQL = """
WITH q AS (
    -- S2 found FTS needs OR-semantics (any term may match), so turn plainto's
    -- AND-joined lexemes into an OR query, keeping stemming + stopword removal.
    SELECT replace(plainto_tsquery('english', :query_text)::text, ' & ', ' | ')::tsquery AS query
),
scoped AS (
    SELECT id, content, type, actor_type, confidence, embedding, content_tsv
    FROM memories
    WHERE TRUE {scope_where}
),
vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS rank
    FROM scoped
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :pool
),
fts_hits AS (
    SELECT scoped.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(scoped.content_tsv, q.query) DESC) AS rank
    FROM scoped, q
    WHERE scoped.content_tsv @@ q.query
    ORDER BY ts_rank_cd(scoped.content_tsv, q.query) DESC
    LIMIT :pool
),
fused AS (
    SELECT COALESCE(v.id, f.id) AS id,
           COALESCE(1.0 / (:k + v.rank), 0.0) + COALESCE(1.0 / (:k + f.rank), 0.0) AS score
    FROM vector_hits v
    FULL OUTER JOIN fts_hits f ON v.id = f.id
)
SELECT scoped.id, scoped.content, scoped.type, scoped.actor_type, scoped.confidence, fused.score
FROM fused
JOIN scoped ON scoped.id = fused.id
ORDER BY fused.score DESC, scoped.id
"""


def _vector_literal(vector: list[float]) -> str:
    """pgvector text form '[1.0,0.0,...]' — cast to vector in-query, no codec needed."""
    return "[" + ",".join(str(x) for x in vector) + "]"


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

    async def add_memories(
        self, items: list[MemoryCreate], embeddings: list[list[float]] | None = None
    ) -> list[UUID]:
        if embeddings is not None and len(embeddings) != len(items):
            raise ValueError("embeddings must align 1:1 with items")
        vectors = embeddings if embeddings is not None else [None] * len(items)
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
                embedding=vector,
            )
            for item, vector in zip(items, vectors, strict=True)
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

    async def hybrid_search(
        self,
        *,
        query_embedding: list[float],
        query_text: str,
        scope: Scope,
    ) -> list[RetrievedMemory]:
        params: dict[str, Any] = {
            "query_text": query_text,
            "query_embedding": _vector_literal(query_embedding),
            "pool": _POOL,
            "k": _RRF_K,
        }
        # only non-None scope fields filter; column names are fixed literals, so
        # composing them into the SQL is safe — the values still bind as params
        clauses = ""
        for field in ("user_id", "agent_id", "run_id", "app_id"):
            value = getattr(scope, field)
            if value is not None:
                clauses += f" AND {field} = :{field}"
                params[field] = value

        stmt = text(_HYBRID_SQL.format(scope_where=clauses))
        async with self.session_factory() as session:
            rows = (await session.execute(stmt, params)).all()
        return [
            RetrievedMemory(
                id=row.id,
                content=row.content,
                type=row.type,
                actor_type=row.actor_type,
                confidence=row.confidence,
                score=float(row.score),
            )
            for row in rows
        ]
