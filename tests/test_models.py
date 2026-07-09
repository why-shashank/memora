"""M1.1 — the DTO layer enforces the PRD §13 taxonomy at memora's boundary."""

import pytest
from pydantic import ValidationError

from memora.models import ActorType, MemoryCreate, MemoryType, Scope


def test_valid_create_exposes_typed_enums_and_defaults() -> None:
    mem = MemoryCreate(content="customer prefers email over phone", type="preference")

    assert mem.content == "customer prefers email over phone"
    # a plain string coerces to the enum member
    assert mem.type is MemoryType.preference
    # defaults: agent actor, empty (fully unscoped) scope
    assert mem.actor_type is ActorType.agent
    assert mem.scope == Scope()
    assert mem.scope.user_id is None


def test_scope_accepts_nested_fields() -> None:
    mem = MemoryCreate(
        content="ticket 42 escalated",
        type="entity_fact",
        scope={"user_id": "u1", "app_id": "support"},
        actor_type="human_review",
    )

    assert mem.scope.user_id == "u1"
    assert mem.scope.app_id == "support"
    assert mem.scope.agent_id is None
    assert mem.actor_type is ActorType.human_review


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryCreate(content="x", type="nonsense")


def test_scope_rejects_unknown_fields() -> None:
    # a typo'd scope key must fail loudly, not silently store an unscoped memory
    with pytest.raises(ValidationError):
        Scope(userid="u1")  # type: ignore[call-arg]


def test_unknown_actor_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryCreate(content="x", type="policy", actor_type="intruder")


def test_empty_or_whitespace_content_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryCreate(content="", type="policy")
    with pytest.raises(ValidationError):
        MemoryCreate(content="   ", type="policy")


def test_server_controlled_fields_cannot_be_set_by_client() -> None:
    # status/id/timestamps are assigned by the write path, never accepted on input
    with pytest.raises(ValidationError):
        MemoryCreate(content="x", type="policy", status="promoted")
    with pytest.raises(ValidationError):
        MemoryCreate(content="x", type="policy", id="00000000-0000-0000-0000-000000000000")
