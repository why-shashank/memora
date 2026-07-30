"""Retrieval metrics for the eval harness (M2.5).

Deliberately dumb arithmetic over already-collected results, so it can be unit-tested
without a database, a model, or a corpus. The runner produces `QueryResult`s; everything
here turns them into the numbers in dev-plan §15.

Only *retrieval* metrics live here. The rest of §15 — correction propagation, staleness,
promotion rate, procedure reuse — needs features that don't exist yet (M3/M4/M5); each
lands with the milestone that gives it something to measure.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    """One golden query's outcome. ``retrieved_ids`` is best-first, in golden-set ids."""

    category: str
    expected_id: str
    retrieved_ids: list[str]
    latency_s: float


@dataclass(frozen=True)
class Summary:
    queries: int
    hit_at_k: dict[int, float]
    mrr: float
    p95_latency_s: float


def hit_at_k(results: Sequence[QueryResult], k: int) -> float:
    """Fraction of queries whose expected memory appears in the top ``k``.

    S2's headline lesson: report several k. A memory ranked 3rd is useless to an agent
    injecting one memory and perfectly fine to one injecting five, and a single number
    can't say which situation you're in.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.expected_id in r.retrieved_ids[:k]) / len(results)


def mrr(results: Sequence[QueryResult]) -> float:
    """Mean reciprocal rank: 1/rank of the expected memory, 0 when it never appears.

    Sharper than hit@k for tuning, because it moves when a fix promotes the right memory
    from 4th to 2nd — a change hit@5 would report as no change at all.
    """
    if not results:
        return 0.0
    ranked = (
        r.retrieved_ids.index(r.expected_id) + 1
        for r in results
        if r.expected_id in r.retrieved_ids
    )
    return sum(1.0 / rank for rank in ranked) / len(results)


def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile: the smallest sample with at least ``p``% of samples at or
    below it. No interpolation — with the handful of samples a golden set produces,
    interpolating invents precision that isn't there."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = math.ceil(len(ordered) * p / 100)
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def summarize(
    results: Sequence[QueryResult], *, k_values: Sequence[int] = (1, 5)
) -> dict[str, Summary]:
    """Per-category summaries plus an ``OVERALL`` row.

    Per-category is where the signal is: S2's overall 0.94 hit@5 hid a 0.70 on scoped
    queries, which was the one finding worth acting on.
    """
    by_category: dict[str, list[QueryResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    summaries = {
        category: _summarize(group, k_values) for category, group in sorted(by_category.items())
    }
    summaries["OVERALL"] = _summarize(results, k_values)
    return summaries


def _summarize(group: Sequence[QueryResult], k_values: Sequence[int]) -> Summary:
    return Summary(
        queries=len(group),
        hit_at_k={k: hit_at_k(group, k) for k in k_values},
        mrr=mrr(group),
        p95_latency_s=percentile([r.latency_s for r in group], 95),
    )
