"""SQLAlchemy ORM — the memories table (dev-plan §5)."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Float, ForeignKey, Text, text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
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
    # candidate | verified | promoted | superseded | deleted — nothing is born promoted
    status: Mapped[str] = mapped_column(Text, server_default=text("'candidate'"))

    # scope
    user_id: Mapped[str | None] = mapped_column(Text)
    agent_id: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    app_id: Mapped[str | None] = mapped_column(Text)

    # provenance
    source: Mapped[str | None] = mapped_column(Text)
    actor_type: Mapped[str | None] = mapped_column(Text)  # human | agent | system

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
