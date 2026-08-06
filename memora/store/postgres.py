"""Postgres + pgvector implementation of StorageBackend."""

from typing import Any
from uuid import UUID

from sqlalchemy import bindparam, delete, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from memora.entities import candidate_keys, canonical_key
from memora.models import MemoryCreate, Scope
from memora.models.orm import (
    AuditLog,
    Entity,
    EntityAlias,
    ExtractionJob,
    Memory,
    MemoryEntity,
)
from memora.store.base import ClaimedJob, RetrievedMemory, StorageBackend

# Reciprocal-rank fusion: a memory's score is the sum over legs of 1/(k + rank).
# k=60 is the canonical RRF constant — it damps the top ranks so no single leg
# dominates. _POOL caps how many candidates each leg contributes before fusion.
_RRF_K = 60
_POOL = 20

# The hybrid query: two independent ranked legs (vector, FTS) fused by RRF.
# {scope_where} is composed from fixed column names (never user input); all values
# bind as parameters.
#
# Each leg's ORDER BY ... LIMIT must sit DIRECTLY ON `memories`. Factoring the
# scope-filtered set into a shared CTE reads better, but Postgres materialises any
# CTE referenced more than once: the legs then scan that copy, where the HNSW and
# GIN indexes are unreachable, and the vector leg silently degrades to sorting every
# row in the table (measured at 20K rows: 25.4ms -> 8.6ms once the legs reach the
# index, vector leg 7.7ms -> 0.9ms, and the bad shape grows linearly with corpus
# size). The entity leg (M2.8) must follow the same shape: its own subquery on the
# base table plus one more COALESCE term in `fused`.
_HYBRID_SQL = """
WITH q AS (
    -- AND-semantics: every query term must appear. `plainto_tsquery` gives this natively.
    --
    -- S2 relaxed it to OR (any term may match) to protect recall, and M2.5b measured the
    -- price at volume: the OR leg matched a fixed *fraction* of the corpus -- a median of
    -- 1,097 rows at 20K -- filling 80% of the fusion pool with memories that shared one
    -- incidental word, each earning a full-strength RRF leg against what the vector leg
    -- actually found. Head to head on the same corpus, AND took hit@1 0.55 -> 0.70, MRR
    -- 0.72 -> 0.78, search p95 ~20ms -> ~6ms, and growth-per-4x-rows 2.89 -> 1.73. It fixed
    -- paraphrase queries outright (MRR 0.67 -> 1.00): the vector leg had already ranked the
    -- right memory first and keyword noise was pushing it down.
    --
    -- The cost, accepted knowingly: hit@5 0.90 -> 0.80. A query whose exact words aren't all
    -- present now leans entirely on the vector leg, which is the leg whose job that is.
    SELECT plainto_tsquery('english', :query_text) AS query
),
vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> CAST(:query_embedding AS vector) AS distance
        FROM memories
        WHERE embedding IS NOT NULL {scope_where}
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :pool
    ) nearest
),
fts_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY relevance DESC) AS rank
    FROM (
        SELECT memories.id, ts_rank_cd(memories.content_tsv, q.query) AS relevance
        FROM memories, q
        WHERE memories.content_tsv @@ q.query {scope_where}
        ORDER BY ts_rank_cd(memories.content_tsv, q.query) DESC
        LIMIT :pool
    ) matched
),
entity_hits AS (
    -- M2.8's third leg: the question named something, and these memories are about it.
    --
    -- One hop, not a graph walk. M2.6 built memory<->entity links and no entity<->entity
    -- edges, so 'the customer on account BR-88214' cannot be followed to that customer --
    -- the question has to name the entity itself. That is the hypothesis being measured.
    --
    -- Ranked by how many of the named entities a memory is about, then by vector distance.
    -- The tiebreak is not decoration: M2.5b measured what happens when a leg contributes a
    -- fixed *fraction* of the corpus, and a customer with 500 memories is exactly that
    -- shape. Ordering the entity's own memories by relevance means the pool fills with the
    -- most relevant things about the right customer instead of an arbitrary twenty.
    --
    -- Unlike a per-memory boost (M2.2, removed in M2.5a) this is evidence about the
    -- query-memory *pair*: it fires only for memories about an entity this question named.
    SELECT id, ROW_NUMBER() OVER (ORDER BY named DESC, distance) AS rank
    FROM (
        SELECT memories.id,
               COUNT(DISTINCT link.entity_id) AS named,
               memories.embedding <=> CAST(:query_embedding AS vector) AS distance
        FROM memories
        JOIN memory_entities link ON link.memory_id = memories.id
        JOIN entity_aliases alias ON alias.entity_id = link.entity_id
        WHERE alias.alias_key IN :entity_keys {scope_where}
        GROUP BY memories.id, memories.embedding
        ORDER BY named DESC, distance
        LIMIT :pool
    ) about
),
fused AS (
    SELECT COALESCE(v.id, f.id, e.id) AS id,
           COALESCE(1.0 / (:k + v.rank), 0.0)
         + COALESCE(1.0 / (:k + f.rank), 0.0)
         + COALESCE(1.0 / (:k + e.rank), 0.0) AS score
    FROM vector_hits v
    FULL OUTER JOIN fts_hits f ON v.id = f.id
    FULL OUTER JOIN entity_hits e ON e.id = COALESCE(v.id, f.id)
)
SELECT memories.id, memories.content, memories.type,
       memories.actor_type, memories.confidence, fused.score
FROM fused
JOIN memories ON memories.id = fused.id
ORDER BY fused.score DESC, memories.id
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

    async def resolve_entity(
        self, *, name: str, type: str, aliases: list[str] | None = None
    ) -> UUID:
        keys = {canonical_key(form) for form in [name, *(aliases or [])]}
        keys.discard("")

        async with self.session_factory() as session:
            existing = (
                await session.execute(
                    select(EntityAlias.entity_id)
                    .where(EntityAlias.entity_type == type)
                    .where(EntityAlias.alias_key.in_(keys))
                    .limit(1)
                )
            ).scalar_one_or_none()

            if existing is not None:
                # Same thing, new way of naming it — learn the forms we hadn't seen.
                await session.execute(
                    insert(EntityAlias)
                    .values(
                        [
                            {"entity_type": type, "alias_key": key, "entity_id": existing}
                            for key in sorted(keys)
                        ]
                    )
                    .on_conflict_do_nothing()
                )
                await session.commit()
                return existing

            entity = Entity(type=type, canonical_name=name)
            session.add(entity)
            await session.flush()
            session.add_all(
                EntityAlias(entity_type=type, alias_key=key, entity_id=entity.id)
                for key in sorted(keys)
            )

            # One of these names is already taken by an entity of another type. That is
            # usually correct and must not merge them (S5) — but it is also the signal a
            # human needs to spot a genuine duplicate, so it is recorded rather than
            # dropped. Once per new entity, not once per lookup: the next resolution of
            # this name finds the entity above and never reaches here.
            collision = (
                await session.execute(
                    select(EntityAlias.alias_key, EntityAlias.entity_type)
                    .where(EntityAlias.alias_key.in_(keys))
                    .where(EntityAlias.entity_type != type)
                    .limit(1)
                )
            ).one_or_none()
            if collision is not None:
                session.add(
                    AuditLog(
                        entity_id=entity.id,
                        action="entity_near_miss",
                        actor_type="system",
                        reason=f"alias {collision.alias_key!r} also names a"
                        f" {collision.entity_type}; kept separate",
                    )
                )
            await session.commit()
            return entity.id

    async def link_memory(self, *, memory_id: UUID, entity_ids: list[UUID]) -> None:
        if not entity_ids:
            return
        async with self.session_factory() as session:
            await session.execute(
                insert(MemoryEntity)
                .values([{"memory_id": memory_id, "entity_id": e} for e in entity_ids])
                .on_conflict_do_nothing()
            )
            await session.commit()

    async def merge_entities(self, *, source: UUID, target: UUID, actor_type: str) -> None:
        async with self.session_factory() as session:
            absorbed = (
                await session.execute(select(Entity).where(Entity.id == source))
            ).scalar_one()
            # Repoint before deleting the source, or its rows cascade away with it.
            # A memory linked to both entities would collide on the composite PK, so
            # drop those links rather than moving them.
            await session.execute(
                delete(MemoryEntity).where(
                    MemoryEntity.entity_id == source,
                    MemoryEntity.memory_id.in_(
                        select(MemoryEntity.memory_id).where(MemoryEntity.entity_id == target)
                    ),
                )
            )
            await session.execute(
                update(MemoryEntity)
                .where(MemoryEntity.entity_id == source)
                .values(entity_id=target)
            )
            await session.execute(
                update(EntityAlias).where(EntityAlias.entity_id == source).values(entity_id=target)
            )
            await session.execute(delete(Entity).where(Entity.id == source))
            session.add(
                AuditLog(
                    entity_id=target,
                    action="entity_merged",
                    actor_type=actor_type,
                    reason=f"absorbed {absorbed.canonical_name!r} ({source})",
                )
            )
            await session.commit()

    async def split_entity(
        self, *, source: UUID, alias_keys: list[str], canonical_name: str, actor_type: str
    ) -> UUID:
        async with self.session_factory() as session:
            parent = (await session.execute(select(Entity).where(Entity.id == source))).scalar_one()
            entity = Entity(type=parent.type, canonical_name=canonical_name)
            session.add(entity)
            await session.flush()

            await session.execute(
                update(EntityAlias)
                .where(EntityAlias.entity_type == parent.type)
                .where(EntityAlias.alias_key.in_(alias_keys))
                .values(entity_id=entity.id)
            )
            await session.execute(
                insert(EntityAlias)
                .values(
                    entity_type=parent.type,
                    alias_key=canonical_key(canonical_name),
                    entity_id=entity.id,
                )
                .on_conflict_do_nothing()
            )
            session.add(
                AuditLog(
                    entity_id=entity.id,
                    action="entity_split",
                    actor_type=actor_type,
                    reason=f"split from {parent.canonical_name!r} ({source}): {alias_keys}",
                )
            )
            await session.commit()
            return entity.id

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
            # every short run of words in the question, for the entity leg to look up
            "entity_keys": sorted(candidate_keys(query_text)),
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

        # expanding: the key list is per-query, and an empty one renders as a false
        # predicate — a question naming nobody simply contributes no entity leg
        stmt = text(_HYBRID_SQL.format(scope_where=clauses)).bindparams(
            bindparam("entity_keys", expanding=True)
        )
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
