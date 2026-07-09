"""M1.3 — Postgres-backed job queue + async extraction worker.

Queue semantics run against real Postgres (SKIP LOCKED is the thing under test);
the LLM is stubbed — the model's judgment was S1/M1.2's problem, the worker's
job is claim → extract → store → complete/retry.
"""

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy import text

from memora.providers.base import LLMProvider
from memora.store.postgres import PostgresStorage
from memora.worker import MAX_ATTEMPTS, process_one, run_forever


@pytest.fixture
async def store(migrated_db_url: str) -> AsyncIterator[PostgresStorage]:
    """A storage backend on the migrated test DB, with clean job/memory tables."""
    storage = PostgresStorage(migrated_db_url)
    async with storage.session_factory() as session:
        # TRUNCATE (not DELETE) on audit_log: deliberate admin/test cleanup stays
        # possible — the append-only trigger blocks row-level UPDATE/DELETE only
        await session.execute(text("TRUNCATE extraction_jobs, memories, audit_log"))
        await session.commit()
    yield storage
    await storage.dispose()


class StubLLM(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 4096
    ) -> str:
        return self.reply


async def _job_status(store: PostgresStorage, job_id: UUID) -> str:
    async with store.session_factory() as session:
        row = await session.execute(
            text("SELECT status FROM extraction_jobs WHERE id = :id"), {"id": job_id}
        )
        return str(row.scalar_one())


# --- queue semantics ---


async def test_enqueue_then_claim_roundtrip(store: PostgresStorage) -> None:
    job_id = await store.enqueue_extraction({"conversation": "Customer: hi"})

    claimed = await store.claim_extraction_job()
    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.payload == {"conversation": "Customer: hi"}
    assert claimed.attempts == 1
    # the job is now processing — nothing left to claim
    assert await store.claim_extraction_job() is None


async def test_claim_is_oldest_first(store: PostgresStorage) -> None:
    first = await store.enqueue_extraction({"conversation": "first"})
    second = await store.enqueue_extraction({"conversation": "second"})

    one, two = await store.claim_extraction_job(), await store.claim_extraction_job()
    assert one is not None and two is not None
    assert [one.id, two.id] == [first, second]


async def test_concurrent_claims_never_hand_out_the_same_job(store: PostgresStorage) -> None:
    # THE queue property (FOR UPDATE SKIP LOCKED): n jobs, 2n concurrent claimers,
    # every job claimed exactly once, the rest get None — no double-processing.
    job_ids = {await store.enqueue_extraction({"conversation": f"c{i}"}) for i in range(3)}

    results = await asyncio.gather(*(store.claim_extraction_job() for _ in range(6)))

    claimed = [r for r in results if r is not None]
    assert {c.id for c in claimed} == job_ids
    assert len(claimed) == 3  # no duplicates, no phantom claims


# --- worker behavior ---

_EXTRACTION_REPLY = """{"memories": [
  {"type": "preference", "content": "Prefers email over phone.", "confidence": 0.9},
  {"type": "policy", "content": "Refunds allowed within 30 days.", "confidence": 0.8}
]}"""


async def test_process_one_extracts_and_stores_candidates(store: PostgresStorage) -> None:
    job_id = await store.enqueue_extraction({"conversation": "Customer chat..."})

    assert await process_one(store, StubLLM(_EXTRACTION_REPLY)) is True

    async with store.session_factory() as session:
        rows = (
            await session.execute(text("SELECT content, type, status, confidence FROM memories"))
        ).all()
    assert {(r.content, r.type, r.status) for r in rows} == {
        ("Prefers email over phone.", "preference", "candidate"),
        ("Refunds allowed within 30 days.", "policy", "candidate"),
    }
    assert await _job_status(store, job_id) == "done"


async def test_process_one_returns_false_on_empty_queue(store: PostgresStorage) -> None:
    assert await process_one(store, StubLLM("unused")) is False


async def test_failed_extraction_retries_then_dead_letters(store: PostgresStorage) -> None:
    job_id = await store.enqueue_extraction({"conversation": "chat"})
    broken_llm = StubLLM("I'm sorry, I can't produce JSON today.")

    for _ in range(MAX_ATTEMPTS - 1):  # every failed attempt but the last requeues
        assert await process_one(store, broken_llm) is True
        assert await _job_status(store, job_id) == "pending"

    assert await process_one(store, broken_llm) is True  # final attempt
    assert await _job_status(store, job_id) == "failed"
    assert await process_one(store, broken_llm) is False  # dead-lettered, not claimable

    async with store.session_factory() as session:
        error = (
            await session.execute(
                text("SELECT error FROM extraction_jobs WHERE id = :id"), {"id": job_id}
            )
        ).scalar_one()
        n_memories = (await session.execute(text("SELECT count(*) FROM memories"))).scalar_one()
    assert error  # the failure reason is recorded for inspection
    assert n_memories == 0  # nothing half-written


async def test_process_one_threads_scope_and_actor_onto_memories(store: PostgresStorage) -> None:
    await store.enqueue_extraction(
        {
            "conversation": "chat",
            "scope": {"user_id": "u-42", "agent_id": "helper", "run_id": None, "app_id": "support"},
            "actor_type": "user_stated",
        }
    )

    assert await process_one(store, StubLLM(_EXTRACTION_REPLY)) is True

    async with store.session_factory() as session:
        rows = (
            await session.execute(
                text("SELECT user_id, agent_id, run_id, app_id, actor_type FROM memories")
            )
        ).all()
    assert len(rows) == 2  # every extracted memory carries the interaction's attribution
    for row in rows:
        assert (row.user_id, row.agent_id, row.run_id, row.app_id) == (
            "u-42",
            "helper",
            None,
            "support",
        )
        assert row.actor_type == "user_stated"


async def test_process_one_stamps_provenance_and_audit(store: PostgresStorage) -> None:
    job_id = await store.enqueue_extraction({"conversation": "chat", "actor_type": "user_stated"})

    assert await process_one(store, StubLLM(_EXTRACTION_REPLY)) is True

    async with store.session_factory() as session:
        rows = (await session.execute(text("SELECT id, source FROM memories"))).all()
        audit = (
            await session.execute(text("SELECT memory_id, action, actor_type FROM audit_log"))
        ).all()
    # every memory points back at the exact interaction (job) that produced it...
    assert len(rows) == 2
    assert {row.source for row in rows} == {f"extraction:{job_id}"}
    # ...and carries a 'created' audit entry naming the interaction's actor
    assert {(a.memory_id, a.action, a.actor_type) for a in audit} == {
        (row.id, "created", "user_stated") for row in rows
    }


async def test_process_one_defaults_scope_and_actor_when_absent(store: PostgresStorage) -> None:
    # a payload without attribution (M1.3-era shape) still processes: unscoped, agent actor
    await store.enqueue_extraction({"conversation": "chat"})

    assert await process_one(store, StubLLM(_EXTRACTION_REPLY)) is True

    async with store.session_factory() as session:
        rows = (
            await session.execute(text("SELECT user_id, app_id, actor_type FROM memories"))
        ).all()
    assert len(rows) == 2
    for row in rows:
        assert (row.user_id, row.app_id, row.actor_type) == (None, None, "agent")


async def test_worker_loop_survives_unexpected_errors(store: PostgresStorage) -> None:
    # an LLM/API blip (auth, rate limit, network) is not ExtractionError — the loop
    # must absorb it and keep polling, not die and take the compose service with it
    class ExplodingLLM(LLMProvider):
        async def generate(
            self, prompt: str, *, system: str | None = None, max_tokens: int = 4096
        ) -> str:
            raise RuntimeError("provider exploded")

    await store.enqueue_extraction({"conversation": "boom"})

    loop_task = asyncio.create_task(run_forever(store, ExplodingLLM()))
    await asyncio.sleep(0.3)  # long enough for at least one claim → explode cycle
    try:
        assert not loop_task.done(), loop_task.exception()
    finally:
        loop_task.cancel()


async def test_invalid_candidate_is_dropped_not_fatal(store: PostgresStorage) -> None:
    # one candidate violates the MemoryCreate contract (blank content) — the other lands
    reply = """{"memories": [
      {"type": "policy", "content": "   "},
      {"type": "preference", "content": "Wants monthly summaries."}
    ]}"""
    job_id = await store.enqueue_extraction({"conversation": "chat"})

    assert await process_one(store, StubLLM(reply)) is True

    async with store.session_factory() as session:
        contents = (await session.execute(text("SELECT content FROM memories"))).scalars().all()
    assert contents == ["Wants monthly summaries."]
    assert await _job_status(store, job_id) == "done"
