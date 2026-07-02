"""M0.4 — migrated schema stores and returns memories; StorageBackend pings."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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
