"""M2.5 — the eval harness's metric math.

The runner itself is verified by running it (it needs a corpus, a real model and a
container); what belongs in pytest is the arithmetic that turns retrieved ids into a
number someone will make decisions from.
"""

from evals.metrics import QueryResult, hit_at_k, mrr, percentile, summarize


def _result(
    category: str, expected: str, retrieved: list[str], latency_s: float = 0.01
) -> QueryResult:
    return QueryResult(
        category=category, expected_id=expected, retrieved_ids=retrieved, latency_s=latency_s
    )


def test_hit_at_k_only_counts_matches_inside_the_cut() -> None:
    # S2's lesson: gold labels must be top-K-aware. The same result is a miss at k=1 and
    # a hit at k=5 — reporting only one of those hides how retrieval actually behaves.
    results = [_result("semantic", "m3", ["m1", "m2", "m3"])]
    assert hit_at_k(results, 1) == 0.0
    assert hit_at_k(results, 5) == 1.0


def test_hit_at_k_is_the_fraction_of_queries_that_hit() -> None:
    results = [
        _result("semantic", "m1", ["m1"]),
        _result("semantic", "m2", ["m9"]),
        _result("semantic", "m3", ["m3"]),
        _result("semantic", "m4", []),  # retrieval returned nothing at all
    ]
    assert hit_at_k(results, 5) == 0.5


def test_mrr_rewards_rank_and_scores_a_miss_as_zero() -> None:
    assert mrr([_result("semantic", "m1", ["m1", "m2"])]) == 1.0
    assert mrr([_result("semantic", "m2", ["m1", "m2"])]) == 0.5
    # absent entirely contributes 0, so MRR can't be gamed by returning a huge pool
    assert mrr([_result("semantic", "m9", ["m1", "m2"])]) == 0.0


def test_percentile_reports_a_value_at_or_above_the_requested_share() -> None:
    assert percentile([0.5], 95) == 0.5  # a single sample is its own p95
    assert percentile([float(n) for n in range(1, 101)], 95) == 95.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0  # unsorted input is fine


def test_summarize_reports_each_category_and_an_overall_row() -> None:
    results = [
        _result("semantic", "m1", ["m1"], latency_s=0.10),
        _result("semantic", "m2", ["m9"], latency_s=0.20),
        _result("policy", "m3", ["m3"], latency_s=0.30),
    ]

    summary = summarize(results, k_values=(1, 5))

    assert summary["semantic"].hit_at_k[1] == 0.5
    assert summary["policy"].hit_at_k[1] == 1.0
    overall = summary["OVERALL"]
    assert overall.queries == 3
    assert overall.hit_at_k[1] == 2 / 3
    assert overall.p95_latency_s == 0.30  # p95 across every query, not just one category
