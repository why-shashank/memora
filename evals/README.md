# Eval harness

Measures retrieval against a golden set instead of arguing about it. Run it on every change
that could move relevance (dev-plan §10).

```sh
uv run python -m evals.runner                  # the production retrieval path
uv run python -m evals.runner --variant rrf    # baseline: raw fusion, pipeline bypassed
```

It starts its own throwaway `pgvector` container, migrates it, seeds `golden/memories.json`
with embeddings from the configured model, then runs `golden/queries.json` through the real
retrieval path. Nothing touches a live deployment, and two runs of one commit agree.

## The golden set

42 memories over 6 customers plus 5 global policies/procedures, and 20 labelled queries in
five categories, each aimed at one capability:

| category | what it exercises | scoped? |
|---|---|---|
| `exact_term` | FTS leg on a unique token (account numbers, "Okta") against near-identical siblings | no |
| `semantic` | vector leg on a paraphrase that shares no content words with its target | yes |
| `scoped` | tenant isolation — the same question returns a different memory per customer | yes |
| `policy` | global memories, reached without a scope | no |
| `correction` | a human correction outranking the ordinary fact it contradicts | yes |

Gold labels are single-best-answer and top-K-aware: hit@5 is the headline, hit@1 is a
diagnostic, and MRR is the tuning signal because it moves when a fix promotes the right
memory from 4th to 2nd. S2's `temporal` category is deliberately absent — there is no
validity filter to test until M3.2.

## Baseline — 2026-07-30, `all-MiniLM-L6-v2`, 20 queries

| category | queries | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| `correction` | 3 | 0.67 | 1.00 | 1.00 | 0.83 |
| `exact_term` | 4 | 0.50 | 1.00 | 1.00 | 0.71 |
| `policy` | 4 | 0.75 | 0.75 | 0.75 | 0.75 |
| `scoped` | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| `semantic` | 5 | 0.40 | 1.00 | 1.00 | 0.67 |
| **OVERALL** | 20 | **0.65** | **0.95** | **0.95** | **0.78** |

`pipeline` and `rrf` currently produce identical numbers, because retrieval ranks on
relevance alone. Any future ranking stage has to beat this table to earn its place.

### What this harness killed on its first run

M2.2 shipped a trust/type weighting stage — `score × type_weight × effective_confidence`,
floating corrections and policies above ordinary facts per PRD FR-1.1. Measured against the
golden set, it made retrieval **worse than not having it**: hit@1 0.65 → 0.40, MRR 0.78 →
0.57. Retuning it into a bounded bonus (`score × (1 + γ·boost)`) recovered most of the loss —
0.55 / MRR 0.69 — but still lost, and a sweep of four boost shapes × nine strengths found
**no γ > 0 that beat plain fusion**; the damage fell monotonically to zero as the stage was
turned off. So it was removed (M2.5a) rather than tuned.

The reason is worth keeping, because it applies to any similar stage. RRF is *flat*: inside a
pool of 20 the entire score range is **2.62×** and neighbouring ranks differ by ~1.6%. A boost
is a property of a memory *by itself* — a correction is boosted in every pool it lands in,
including the many where it is off-topic noise — so any boost large enough to flip a genuine
near-tie is also large enough to flip an arbitrary adjacent pair, and nothing in the scores
distinguishes the two cases. Concretely, it promoted the expected memory on **1 query of 20**
and demoted it on **6**; for `"Which customer has billing account BR-51330?"` the correct
memory is rank 1 on fusion and fell out of the top 5 entirely, displaced by a refund
correction about a different customer.

FR-1.1's actual requirement is a *supersession* problem, not a ranking one — returning a
stale memory second is still returning it — and lands in M3.2 as a status filter.
