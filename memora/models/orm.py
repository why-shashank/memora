"""SQLAlchemy ORM — the memories table and the extraction job queue (dev-plan §5)."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Float, ForeignKey, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Locked to the local sentence-transformers path (all-MiniLM-L6-v2) for MVP.
# A different embedding provider/dimension requires a migration of this column.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    content: Mapped[str] = mapped_column(Text)
    # correction | policy | entity_fact | preference | commitment | procedure
    # (validated at the DTO layer, M1.1 — kept as text in the DB to avoid enum migrations)
    type: Mapped[str] = mapped_column(Text)
    # candidate | verified | promoted | superseded | deleted. Default is candidate, but
    # human corrections write straight to promoted (FR-1.1/FR-2.3 — trusted on write)
    status: Mapped[str] = mapped_column(Text, server_default=text("'candidate'"))

    # scope
    user_id: Mapped[str | None] = mapped_column(Text)
    agent_id: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    app_id: Mapped[str | None] = mapped_column(Text)

    # provenance
    source: Mapped[str | None] = mapped_column(Text)
    # human_correction | human_review | agent | user_stated | system (PRD §13 — drives weighting)
    actor_type: Mapped[str | None] = mapped_column(Text)

    # temporal validity
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    confidence: Mapped[float | None] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float, server_default=text("1.0"))
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id")
    )

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=text("now()")
    )


class AuditLog(Base):
    """Append-only accountability trail — what happened to each memory record.

    A DB trigger (migration 0003) rejects UPDATE/DELETE: history can't be
    rewritten, even by our own code. Deliberately **no FK** to memories:
    right-to-be-forgotten (M4.6) deletes the memory, but the fact that it
    existed and was deleted must survive it.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # exactly one of memory_id / entity_id names the subject (CHECK, migration 0004)
    memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # action vocabulary grows with the flows: created (M1.5); entity_near_miss/
    # entity_merged/entity_split (M2.6); superseded/promoted/deleted/... with M3/M4
    action: Mapped[str] = mapped_column(Text)
    actor_type: Mapped[str] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class Entity(Base):
    """A real-world thing memories can be about (dev-plan §5).

    ``canonical_name`` is display only; every lookup goes through EntityAlias.
    """

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # person | organization | product (validated at the boundary, free text here —
    # same reasoning as Memory.type)
    type: Mapped[str] = mapped_column(Text)
    canonical_name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class EntityAlias(Base):
    """Every surface form that names an entity, one row each.

    A table rather than an ``aliases`` array column on `entities` so that
    ``PRIMARY KEY (entity_type, alias_key)`` can enforce *one entity per name* in
    the database. Two workers resolving the same customer concurrently then collide
    on the constraint instead of quietly creating duplicate entities — which is the
    exact failure entity resolution exists to prevent.

    ``entity_type`` is denormalized from `entities` because it is half of that key.
    S5 is why it is in the key at all: the extraction model correctly lists a
    customer's email as a mention of both the person and their employer, so a
    type-blind index fuses `Leo Tran` into `Stackpine` (measured: 3 collapses in 3
    reps; partitioning by type took it to 0).
    """

    __tablename__ = "entity_aliases"

    entity_type: Mapped[str] = mapped_column(Text, primary_key=True)
    alias_key: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE")
    )


class MemoryEntity(Base):
    """Which memories are about which entities — the join M2.8's CTE traverses."""

    __tablename__ = "memory_entities"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )


class ExtractionJob(Base):
    """Postgres-backed work queue (dev-plan §3 — async writes without Redis).

    Claiming flips pending→processing via FOR UPDATE SKIP LOCKED, so concurrent
    workers never grab the same job. JSONB payload so M1.4 can add scope/actor
    keys without a migration.
    """

    __tablename__ = "extraction_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # pending | processing | done | failed (failed = dead-lettered after MAX_ATTEMPTS)
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=text("now()")
    )
