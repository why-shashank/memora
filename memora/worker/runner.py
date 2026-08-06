"""Async extraction worker — drains the Postgres job queue off the agent's hot path.

Claim → extract (M1.2) → store candidates → link their entities (M2.6) → complete.
A broken model response
(`ExtractionError`) requeues the job up to MAX_ATTEMPTS, then dead-letters it as
status='failed' with the reason recorded. A broken individual candidate is
dropped, never the whole job. Stuck-'processing' recovery (worker killed
mid-job) is Phase-2 hardening.
"""

import asyncio

import structlog
from pydantic import ValidationError

from memora.extraction import ExtractedMemory, ExtractionError, extract_memories
from memora.models import ActorType, MemoryCreate, Scope
from memora.providers.base import EmbeddingProvider, LLMProvider
from memora.store.base import StorageBackend

log = structlog.get_logger()

MAX_ATTEMPTS = 3
POLL_SECONDS = 1.0


async def process_one(
    storage: StorageBackend, llm: LLMProvider, embedder: EmbeddingProvider
) -> bool:
    """Claim and handle one queued job; False when the queue is empty."""
    job = await storage.claim_extraction_job()
    if job is None:
        return False

    try:
        extracted = await extract_memories(job.payload["conversation"], llm)
    except ExtractionError as exc:
        retry = job.attempts < MAX_ATTEMPTS
        await storage.fail_extraction_job(job.id, error=str(exc), retry=retry)
        log.warning(
            "extraction_job_failed",
            job_id=str(job.id),
            attempts=job.attempts,
            retry=retry,
            error=str(exc),
        )
        return True

    # every memory from this interaction carries its attribution (API-validated;
    # .get defaults keep pre-M1.4 payloads processable)
    scope = Scope(**job.payload.get("scope", {}))
    actor_type = ActorType(job.payload.get("actor_type", ActorType.agent))

    items: list[MemoryCreate] = []
    kept: list[ExtractedMemory] = []  # the candidate behind each item, for entity linking
    for candidate in extracted:
        try:
            # supersedes resolution lands in M3.2
            item = MemoryCreate(
                content=candidate.content,
                type=candidate.type,
                confidence=candidate.confidence,
                scope=scope,
                actor_type=actor_type,
                # the job row persists with its payload — this points every
                # memory back at the exact interaction that produced it
                source=f"extraction:{job.id}",
            )
        except ValidationError:
            log.warning(
                "extraction_dropped_invalid_candidate",
                job_id=str(job.id),
                candidate=candidate.model_dump(),
            )
            continue
        items.append(item)
        kept.append(candidate)

    # embed-on-write: retrieval's vector leg (M2.1) reads this column, so each
    # candidate is embedded as it lands rather than back-filled later
    embeddings = await embedder.embed([item.content for item in items])
    memory_ids = await storage.add_memories(items, embeddings=embeddings)

    # Entity linking: every mention resolves to one entity (M2.6), and the memory
    # records what it is about. Resolved per memory, not per transcript — the same
    # customer named three ways is one entity, while a general policy is about no
    # one and links to nothing. Each call also teaches the alias index the surface
    # forms it hadn't seen, so repeats across candidates are wanted, not waste.
    for memory_id, candidate in zip(memory_ids, kept, strict=True):
        entity_ids = [
            await storage.resolve_entity(
                name=entity.canonical_name, type=entity.type.value, aliases=entity.mentions
            )
            for entity in candidate.entities
        ]
        await storage.link_memory(memory_id=memory_id, entity_ids=entity_ids)

    await storage.complete_extraction_job(job.id)
    log.info("extraction_job_done", job_id=str(job.id), memories=len(items))
    return True


async def run_forever(
    storage: StorageBackend, llm: LLMProvider, embedder: EmbeddingProvider
) -> None:
    """The worker loop: drain the queue, sleep briefly when it's empty.

    Unexpected errors (LLM auth/rate-limit/network, DB blips) are logged and
    absorbed — one bad iteration must not take the service down. The claimed job
    stays 'processing' in that case (recovery is the Phase-2 hardening noted above).
    """
    log.info("worker_started")
    while True:
        try:
            worked = await process_one(storage, llm, embedder)
        except Exception:
            log.exception("worker_iteration_failed")
            worked = False
        if not worked:
            await asyncio.sleep(POLL_SECONDS)
