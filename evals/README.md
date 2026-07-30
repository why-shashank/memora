# Eval harness

Measures retrieval against a golden set instead of arguing about it. Run it on every change
that could move relevance (dev-plan §10).

```sh
uv run python -m evals.runner                  # production path: RRF fusion + weighting
uv run python -m evals.runner --variant rrf    # baseline: fusion only, weighting skipped
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

| variant | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| `weighted` (production path) | 0.40 | 0.70 | 0.90 | 0.57 |
| `rrf` (fusion only) | **0.65** | **0.95** | **0.95** | **0.78** |

**The weighting stage currently makes retrieval worse on every metric**, which is the first
thing this harness was built to find out. Cause: the type/trust multiplier spans **12×**
across this corpus (0.25 for an agent-extracted fact at confidence 0.5, up to 3.0 for a
human correction) while RRF scores span only **2.62×** inside a pool of 20. A multiplier
range wider than the relevance range stops being a tiebreak and becomes a total order —
relevance can no longer influence the result at all.

It shows up as absurd answers rather than subtle ones: for `"Which customer has billing
account BR-51330?"` the correct memory is **rank 1** on pure fusion and **absent from the
top 5** once weighted, displaced by a refund correction about a different customer.

Tracked as the retune decision in `TASKS.md`; the numbers above are the before-picture.
