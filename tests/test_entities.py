"""M2.6 — alias resolution: one entity per real-world thing, never two, never fused.

The cross-type tests encode S5's finding: the extraction model correctly reports a
customer's email as a mention of *both* the person and their employer, so an alias
index that ignores type merges a person into an organization.
"""

from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy import text

from memora.entities import canonical_key
from memora.models import MemoryCreate
from memora.store.postgres import PostgresStorage


@pytest.fixture
async def store(migrated_db_url: str) -> AsyncIterator[PostgresStorage]:
    backend = PostgresStorage(migrated_db_url)
    yield backend
    await backend.dispose()


async def _aliases(store: PostgresStorage, entity_id: UUID) -> set[str]:
    async with store.session_factory() as session:
        rows = await session.execute(
            text("SELECT alias_key FROM entity_aliases WHERE entity_id = :id"), {"id": entity_id}
        )
        return set(rows.scalars())


async def _audit(store: PostgresStorage, entity_id: UUID) -> list[tuple[str, str | None]]:
    async with store.session_factory() as session:
        rows = await session.execute(
            text("SELECT action, reason FROM audit_log WHERE entity_id = :id ORDER BY created_at"),
            {"id": entity_id},
        )
        return [(row.action, row.reason) for row in rows]


# ------------------------------------------------------------------ canonicalization


def test_canonical_key_folds_case_whitespace_and_punctuation() -> None:
    assert canonical_key("ACME  Corp.") == canonical_key("acme corp")
    assert canonical_key("  Jane Doe ") == canonical_key("jane   doe")


def test_canonical_key_strips_a_legal_suffix() -> None:
    # S5: 'Lumen Health' and 'Lumen Health Inc.' are one customer, and punctuation
    # folding alone leaves them as two entities (measured: 11/12 merges -> 12/12)
    assert canonical_key("Lumen Health Inc.") == canonical_key("Lumen Health")
    assert canonical_key("Fieldworks LLC") == canonical_key("Fieldworks")


def test_canonical_key_keeps_a_name_that_is_only_a_suffix() -> None:
    # stripping must never empty a key — an empty key would collide with every other
    assert canonical_key("Ltd") != ""


# ------------------------------------------------------------------ resolution


async def test_resolving_the_same_name_twice_returns_one_entity(store: PostgresStorage) -> None:
    first = await store.resolve_entity(name="Stackpine", type="organization")
    again = await store.resolve_entity(name="stackpine.", type="organization")
    assert first == again


async def test_resolving_a_known_alias_returns_the_entity_and_learns_new_ones(
    store: PostgresStorage,
) -> None:
    entity_id = await store.resolve_entity(
        name="Priya Nair", type="person", aliases=["priya@lumenhealth.org"]
    )
    # a later conversation calls her 'P. Nair' and shares only the email
    again = await store.resolve_entity(
        name="P. Nair", type="person", aliases=["priya@lumenhealth.org"]
    )
    assert again == entity_id
    assert {"priya nair", "p nair", "priya lumenhealth org"} <= await _aliases(store, entity_id)


async def test_an_alias_shared_across_types_stays_two_entities(store: PostgresStorage) -> None:
    # S5's failure, made structural: 'leo@stackpine.dev' is a true mention of both
    person = await store.resolve_entity(
        name="Leo Tran", type="person", aliases=["leo@stackpine.dev"]
    )
    org = await store.resolve_entity(
        name="Stackpine", type="organization", aliases=["leo@stackpine.dev", "stackpine.dev"]
    )
    assert person != org


async def test_a_cross_type_alias_collision_is_recorded_as_a_near_miss(
    store: PostgresStorage,
) -> None:
    await store.resolve_entity(name="Marcus Webb", type="person", aliases=["m@fieldworks.com"])
    org = await store.resolve_entity(
        name="Fieldworks", type="organization", aliases=["m@fieldworks.com"]
    )
    # the collision is a real signal for a human reviewer — it must not be silent
    assert [action for action, _ in await _audit(store, org)] == ["entity_near_miss"]


# ------------------------------------------------------------------ admin merge / split


async def test_merge_moves_aliases_and_links_then_records_the_decision(
    store: PostgresStorage,
) -> None:
    keep = await store.resolve_entity(name="Lumen Health", type="organization")
    dupe = await store.resolve_entity(name="Lumen Heath", type="organization")  # typo'd
    (memory_id,) = await store.add_memories(
        [MemoryCreate(content="Renewal moved to April 15", type="entity_fact")]
    )
    await store.link_memory(memory_id=memory_id, entity_ids=[dupe])

    await store.merge_entities(source=dupe, target=keep, actor_type="human_review")

    assert {"lumen health", "lumen heath"} <= await _aliases(store, keep)
    async with store.session_factory() as session:
        linked = (
            await session.execute(
                text("SELECT entity_id FROM memory_entities WHERE memory_id = :m"),
                {"m": memory_id},
            )
        ).scalars()
        assert list(linked) == [keep]
    # a resolution decision is a trust decision: auditable, and naming what it absorbed
    (action, reason) = (await _audit(store, keep))[-1]
    assert action == "entity_merged"
    assert str(dupe) in (reason or "")


async def test_split_extracts_aliases_into_a_new_entity_and_records_it(
    store: PostgresStorage,
) -> None:
    fused = await store.resolve_entity(
        name="Acme", type="organization", aliases=["acme.co", "acme.com"]
    )
    new_id = await store.split_entity(
        source=fused,
        alias_keys=["acme com"],
        canonical_name="Acme Legacy",
        actor_type="human_review",
    )

    assert new_id != fused
    assert await _aliases(store, new_id) == {"acme com", "acme legacy"}
    assert "acme com" not in await _aliases(store, fused)
    assert [action for action, _ in await _audit(store, new_id)] == ["entity_split"]
