"""M0.4 — migrated schema stores and returns memories; StorageBackend pings.
M1.5 — every memory write lands with its audit row, and the audit log is append-only."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memora.models import MemoryCreate
from memora.models.orm import EMBEDDING_DIM, Memory
from memora.store.postgres import PostgresStorage


async def test_ping_reports_reachable_database(migrated_db_url: str) -> None:
    store = PostgresStorage(migrated_db_url)
    assert await store.ping() is True
    await store.dispose()


async def test_ping_reports_unreachable_database() -> None:
    store = PostgresStorage("postgresql+asyncpg://nobody:nope@localhost:1/void")
    assert await store.ping() is False
    await store.dispose()


async def test_memory_roundtrip_with_vector_and_generated_tsv(migrated_db_url: str) -> None:
    engine = create_async_engine(migrated_db_url)
    try:
        async with async_sessionmaker(engine)() as session:
            memory = Memory(
                content="Customer prefers email over phone for account matters",
                type="preference",
                user_id="u-roundtrip",
                embedding=[0.1] * EMBEDDING_DIM,
            )
            session.add(memory)
            await session.commit()

            row = (
                await session.execute(select(Memory).where(Memory.user_id == "u-roundtrip"))
            ).scalar_one()
            assert row.id is not None
            assert row.status == "candidate"  # trust gate: nothing is born promoted
            assert row.created_at is not None
            assert row.valid_from is not None
            assert list(row.embedding)[:2] == [0.1, 0.1]
            # tsvector is DB-generated from content — proves FTS leg has data
            await session.refresh(row, ["content_tsv"])
            assert "email" in str(row.content_tsv)
    finally:
        await engine.dispose()


async def test_add_memories_records_provenance_and_audit(migrated_db_url: str) -> None:
    store = PostgresStorage(migrated_db_url)
    try:
        ids = await store.add_memories(
            [
                MemoryCreate(
                    content="Prefers invoices in PDF (audit-test)",
                    type="preference",
                    actor_type="user_stated",
                    source="extraction:job-audit-test",
                ),
                MemoryCreate(
                    content="On the Growth plan (audit-test)",
                    type="entity_fact",
                    source="extraction:job-audit-test",
                ),
            ]
        )

        async with store.session_factory() as session:
            sources = (
                await session.execute(
                    text("SELECT source FROM memories WHERE id = ANY(:ids)"), {"ids": ids}
                )
            ).scalars()
            assert list(sources) == ["extraction:job-audit-test"] * 2

            # accountability: exactly one 'created' audit row per memory, naming its actor
            audit = (
                await session.execute(
                    text(
                        "SELECT memory_id, action, actor_type, created_at FROM audit_log"
                        " WHERE memory_id = ANY(:ids)"
                    ),
                    {"ids": ids},
                )
            ).all()
        assert {(row.memory_id, row.action) for row in audit} == {(m, "created") for m in ids}
        assert {row.actor_type for row in audit} == {"user_stated", "agent"}
        assert all(row.created_at is not None for row in audit)
    finally:
        await store.dispose()


async def test_audit_log_is_append_only(migrated_db_url: str) -> None:
    # the trust promise is structural: even our own code cannot rewrite history
    store = PostgresStorage(migrated_db_url)
    try:
        (memory_id,) = await store.add_memories(
            [MemoryCreate(content="immutable-audit-test", type="policy")]
        )
        for tampering in (
            "UPDATE audit_log SET action = 'promoted' WHERE memory_id = :id",
            "DELETE FROM audit_log WHERE memory_id = :id",
        ):
            async with store.session_factory() as session:
                with pytest.raises(DBAPIError, match="append-only"):
                    await session.execute(text(tampering), {"id": memory_id})
    finally:
        await store.dispose()
