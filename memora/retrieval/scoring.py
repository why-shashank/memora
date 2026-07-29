"""Effective-confidence weighting (M2.2) — float trusted memory above merely-relevant memory.

Retrieval ranks on two axes. RRF (M2.1) answers "how relevant?"; this stage answers "how
much do we trust it, and does its kind matter?" The final score multiplies relevance by a
structural type priority (PRD: corrections/policies/procedures float above ordinary facts)
and an *effective confidence* — the memory's own confidence scaled by the trust of whoever
vouched for it (truthgraph's per-source ``trust_weight``, harvested).

The constants below are a defensible starting point documented in one place; M2.5's eval
harness is where they get tuned against a golden set. (The stored per-memory ``weight``
column stays out of this until decay/manual-boost writes it — FR-1.3, Phase 2.)
"""

from dataclasses import replace

from memora.store.base import RetrievedMemory

# Structural priority by type: corrections are the crown jewel (PRD §goals), policies and
# procedures rank above ordinary facts; entity_fact/preference/commitment sit at baseline.
_TYPE_WEIGHT: dict[str, float] = {
    "correction": 3.0,
    "policy": 2.0,
    "procedure": 2.0,
}
_DEFAULT_TYPE_WEIGHT = 1.0

# Trust by who vouched for the memory. Human-vouched is fully trusted; a user asserting
# something about themselves is fairly trusted but unverified; agent/system inferences must
# still earn trust through the promotion gate, so they weigh least (and are the default).
_TRUST_WEIGHT: dict[str, float] = {
    "human_correction": 1.0,
    "human_review": 1.0,
    "user_stated": 0.7,
    "agent": 0.5,
    "system": 0.5,
}
_DEFAULT_TRUST = 0.5


def effective_confidence(confidence: float | None, actor_type: str | None) -> float:
    """How much to believe a memory: its own confidence × the trust of its actor.

    A memory with no confidence is an *assertion* (a human correction or review, not a
    hedged inference), so it is taken at full confidence rather than penalised.
    """
    base = confidence if confidence is not None else 1.0
    return base * _TRUST_WEIGHT.get(actor_type or "", _DEFAULT_TRUST)


def rank(candidates: list[RetrievedMemory]) -> list[RetrievedMemory]:
    """Re-score RRF candidates by type priority + effective confidence, best first.

    Input order is the RRF ranking, so Python's stable sort breaks score ties by
    relevance. Each candidate's ``score`` is replaced with its effective score.
    """
    weighted = [
        replace(
            m,
            score=m.score
            * _TYPE_WEIGHT.get(m.type, _DEFAULT_TYPE_WEIGHT)
            * effective_confidence(m.confidence, m.actor_type),
        )
        for m in candidates
    ]
    weighted.sort(key=lambda m: m.score, reverse=True)
    return weighted
