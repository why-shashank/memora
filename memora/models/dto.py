"""Pydantic DTOs — the validated taxonomy at memora's boundary (PRD §13).

The ORM (orm.py) stores type/status/actor_type as free text to avoid enum
migrations; these DTOs are where the closed vocabularies get enforced, before
anything reaches the store.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(StrEnum):
    correction = "correction"
    policy = "policy"
    entity_fact = "entity_fact"
    preference = "preference"
    commitment = "commitment"
    procedure = "procedure"


class MemoryStatus(StrEnum):
    candidate = "candidate"
    verified = "verified"
    promoted = "promoted"
    superseded = "superseded"
    deleted = "deleted"


class ActorType(StrEnum):
    """Who produced a memory — drives trust weighting (PRD §13)."""

    human_correction = "human_correction"
    human_review = "human_review"
    agent = "agent"
    user_stated = "user_stated"
    system = "system"


class Scope(BaseModel):
    """Who/what a memory belongs to (PRD §13). All optional — absence = unscoped.

    ``extra="forbid"``: a typo'd key must fail loudly, not silently misfile a
    memory outside the scope filters it belongs in.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    app_id: str | None = None


class MemoryCreate(BaseModel):
    """The write-path input.

    ``status``/``id``/timestamps are server-assigned, so ``extra="forbid"`` keeps
    a client from injecting them; ``str_strip_whitespace`` rejects blank content.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1)
    type: MemoryType
    scope: Scope = Field(default_factory=Scope)
    actor_type: ActorType = ActorType.agent
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # provenance: which flow produced this memory, pointing at its originating
    # record — "extraction:<job_id>" (M1.5); "correction:<...>" arrives with M3.1
    source: str | None = None
