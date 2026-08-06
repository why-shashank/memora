# Eval harness

Measures retrieval against a golden set instead of arguing about it. Run it on every change
that could move relevance (dev-plan §10).

```sh
uv run python -m evals.runner                  # the production retrieval path
uv run python -m evals.runner --variant rrf    # baseline: raw fusion, pipeline bypassed
uv run python -m evals.volume                  # the same queries against 20K distractors
uv run python -m evals.runner --no-entities    # either harness, without the M2.8 entity leg
```

Both start their own throwaway `pgvector` container, migrate it, seed with embeddings from
the configured model, then run `golden/queries.json` through the real retrieval path. Nothing
touches a live deployment, and two runs of one commit agree.

`runner` answers *"is retrieval correct?"* against a 41-memory corpus. `volume` answers
*"does it stay correct and fast as the corpus grows?"* by adding 20,000 generated distractors
— and exits non-zero when search starts tracking the row count. Run `volume` when you touch
the SQL; run `runner` when you touch ranking.

## The golden set

42 memories over 6 customers plus 5 global policies/procedures, and 27 labelled queries in
seven categories, each aimed at one capability:

| category | what it exercises | scoped? |
|---|---|---|
| `exact_term` | FTS leg on a unique token (account numbers, "Okta") against near-identical siblings | no |
| `semantic` | vector leg on a paraphrase that shares no content words with its target | yes |
| `scoped` | tenant isolation — the same question returns a different memory per customer | yes |
| `policy` | global memories, reached without a scope | no |
| `correction` | a human correction outranking the ordinary fact it contradicts | yes |
| `relational` | the question **names** a customer and asks something generic about them — only the entity picks the right one out of six near-identical answers | no |
| `multi_hop` | the question names something *related* to the answer's subject ("Dana's company", "the customer on account BR-88214") | no |

Gold labels are single-best-answer and top-K-aware: hit@5 is the headline, hit@1 is a
diagnostic, and MRR is the tuning signal because it moves when a fix promotes the right
memory from 4th to 2nd. S2's `temporal` category is deliberately absent — there is no
validity filter to test until M3.2.

Memories carry hand-labelled `entities` in the shape M2.7's extraction emits, which the runner
resolves and links (`--no-entities` skips it, leaving the entity leg with nothing to return —
bit-for-bit the pre-M2.8 baseline). Hand-labelling makes every entity number here an **upper
bound: the leg measured under perfect extraction.** It is the same kind of label `type` and
`confidence` already are.

## Baseline — 2026-07-31, `all-MiniLM-L6-v2`, 20 queries

| category | queries | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| `correction` | 3 | 0.67 | 1.00 | 1.00 | 0.83 |
| `exact_term` | 4 | 0.75 | 1.00 | 1.00 | 0.83 |
| `policy` | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| `scoped` | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| `semantic` | 5 | 1.00 | 1.00 | 1.00 | 1.00 |
| **OVERALL** | 20 | **0.90** | **1.00** | **1.00** | **0.94** |

Up from 0.65 / 0.95 / 0.95 / 0.78 when the keyword leg used OR-semantics (M2.5c). Every
category except `correction` is now perfect at this corpus size; the one `correction` miss is
a genuine near-tie between a correction and the fact it corrects, which M3.2 resolves by
filtering the superseded memory out rather than out-ranking it.

`pipeline` and `rrf` produce identical numbers, because retrieval ranks on relevance alone.
Any future ranking stage has to beat this table to earn its place.

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

## Volume — 2026-07-31, 20,041 memories

`evals/volume.py` seeds the golden set alongside 20,000 generated distractors over ~2,500
tenants, measures at 5K and again at 20K, and compares the growth.

### The guard: scaling, not a stopwatch

An absolute latency budget can't be set honestly — measured p95 sat at 21.5ms against the
only defensible budget of 25ms, so it would fire on a busy laptop. But the defect has a
*shape*: an unreachable index degrades a query to a scan, and a scan tracks the row count.

| shape | growth for 4× rows |
|---|---|
| current query | **1.7 – 2.1×** |
| ceiling (`corpus_growth ** 0.75`) | 2.83× |
| M2.1 defect, replayed from git | **3.8×** |
| a pure sequential scan would be | 4.0× |

The defect row is the important one: the shared-CTE shape from `cacc174` was recovered from
git and run against the same corpus, so the guard is known to fire on the exact bug it was
built for rather than assumed to.

Three things had to be right, and each was wrong first:

- **Verdict on unscoped queries only.** A tenant-scoped query filters to a handful of rows,
  where scanning *is* the right plan — measured against the defect, scoped queries grew 1.07×
  while unscoped grew 3.70×. The golden set is 60% scoped, so pooling them hid a completely
  broken index.
- **Best-of-N per query, not pooled samples.** Interference only ever makes a query slower,
  so the minimum is the cleanest estimate of work done. Pooling made the verdict swing between
  1.48× and 2.47× on identical code.
- **`ANALYZE` after each bulk load.** Otherwise the planner works from pre-load statistics —
  a state no running deployment is ever in — and the two stages get planned differently.

### Relevance decays with corpus size

| corpus | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| 41 memories | **0.90** | **1.00** | **1.00** | **0.94** |
| 20,041 memories | 0.75 | 0.80 | 0.80 | 0.78 |

Worth re-reading whenever the golden-set numbers look reassuring — they are measured against
41 distractors, and a deployment is not. The gap is concentrated in `exact_term` and `policy`
(both 0.50 at volume, both 1.00 on the golden set): unique tokens like account numbers survive
scale fine, but a *paraphrased* policy question has to find one memory among 20,000 on meaning
alone. That was the entity leg's case to make in M2.8, and it made it — see below, where the
new `relational` category goes from **1.00 on the golden set to 0.00 at volume** without the
leg. This decay is the single most reliable thing this harness measures.

*(hit@1 wobbles ±0.05 between runs: `ORDER BY fused.score DESC, memories.id` breaks score
ties on randomly-generated UUIDs. Real but small; the ranks it flips are genuine ties.)*

### What OR-semantics cost — the M2.4 smoke test, quantified (resolved in M2.5c)

S2 required OR-semantics so any query term may match, and justified it on recall. Nobody had
measured the price. At 20K rows the keyword leg matches a **median of 1,097 rows and a p95 of
6,063** — up to 30% of the corpus — and fills **80%** of the 20-slot RRF pool. Every one of
those gets a full-strength fusion leg, competing with what the vector leg actually found.

Measured against `plainto_tsquery`'s native AND-semantics on the identical corpus:

| | hit@1 | hit@5 | MRR | `semantic` MRR | search p95 | growth for 4× rows |
|---|---|---|---|---|---|---|
| OR (current) | 0.55 | **0.90** | 0.72 | 0.67 | 19–21ms | 2.89× |
| AND | **0.70–0.75** | 0.80 | **0.75–0.78** | **1.00** | **5.4–6.1ms** | **1.73×** |

AND wins hit@1, MRR, latency (3.3×) and scaling, and fixes the `semantic` category outright —
under OR the vector leg *found* the right memory at rank 1 and twenty keyword matches pushed
it down. AND's cost is real though: hit@5 falls 0.90 → 0.80, losing a policy query and
sometimes an exact-term one, because AND requires *every* term present. That is exactly the
recall loss S2 warned about.

A fallback (AND, then OR only when AND finds nothing) was built and rejected: for the median
natural-language question AND matches **zero** rows, so the fallback fires almost always and
degenerates into OR plus an extra subquery — p95 78–148ms.

**Outcome (M2.5c): switched to AND.** Measured after the change, at 20K:

| | keyword rows p50 | pool filled | search p95 | growth for 4× rows | hit@1 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| OR | 1,097 | 80% | ~21ms | 1.83–2.10× | 0.55 | **0.95** | 0.72 |
| AND | **0** | **26%** | **11.2ms** | **1.62×** | **0.75** | 0.80 | **0.78** |

The keyword leg now contributes *nothing* to the median natural-language question and fires
only on the exact-token queries it exists for, which is the behaviour a hybrid system wants:
keyword handles tokens, vector handles meaning. On the golden set the switch was a clean
sweep with no downside at all (0.65 → 0.90 hit@1, 0.78 → 0.94 MRR, hit@5 to a perfect 1.00);
the hit@5 cost only appears at volume, and it lands on `exact_term` and `policy`.

## The entity leg: prove-or-kill — 2026-08-06 (M2.8)

M2.8 shipped a third RRF leg on the condition that it beat vector+FTS on relational queries or
be deleted. **It was measured twice, and only the second measurement could answer the question.**

### On the golden set: no effect, and no ability to detect one

41 memories, 27 queries, `--no-entities` vs linked: **every number identical**, category by
category. Not a wash — a saturated metric. `relational` scored hit@5 **1.00 in both arms**,
so the criterion had nothing to compare. With six customers to choose between, the vector leg
lands the right one near the top by default.

The leg was firing the whole time. Probing the fused scores directly, `"Remind me of the
account number we bill Orchid against"` returns four *different* customers' billing memories
without links and **five Orchid memories** with them — a real precision gain that hit@k cannot
see, because the expected answer was already rank 1.

### At 20,041 memories: the leg is the difference between working and not

| category | queries | hit@1 off → on | hit@3 off → on | hit@5 off → on | MRR off → on |
|---|---|---|---|---|---|
| `relational` | 5 | 0.00 → **0.20** | 0.00 → **0.80** | **0.00 → 1.00** | 0.00 → **0.54** |
| `exact_term` | 4 | 0.00 → 0.25 | 0.50 → 0.75 | 0.50 → 0.75 | 0.25 → 0.50 |
| `policy` | 4 | 0.25 → 0.50 | 0.25 → 0.50 | 0.25 → 0.50 | 0.25 → 0.50 |
| `multi_hop` | 2 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| `correction` | 3 | 0.67 → 0.67 | 1.00 → 1.00 | 1.00 → 1.00 | 0.83 → 0.83 |
| `scoped` | 4 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| `semantic` | 5 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| **OVERALL** | 27 | 0.44 → **0.56** | 0.56 → **0.78** | 0.56 → **0.81** | 0.50 → **0.67** |

**`relational` hit@5 goes 0.00 → 1.00.** Vector+FTS finds *none* of the five at any k against
20,000 distractors; with the leg it finds all five. Nothing regressed in any category.

Read the smaller movements with care. `exact_term` has a plausible mechanism (`"who signs in
through Okta"` — `Okta` is a labelled product entity) but it is one query of four, and
`policy` moving 0.25 → 0.50 has **no mechanism at all**: policies carry no entities and the
policy questions name none, so both arms run identical SQL. That pair is the honest size of
run-to-run noise on a 4-query category at volume — worth remembering before believing any
future ±1-query result here. Only `relational` is far enough outside it to claim.

### What it costs

| | search p50 | search p95 | growth for 4× rows (ceiling 2.82×) |
|---|---|---|---|
| off | 6.9ms | 11.6ms | 1.51× |
| on | 8.7ms | 14.9ms | **2.07×** |

~26% slower, and the scaling margin narrows from comfortable to noticeable — still sub-linear,
so the guard passes, but this is the number to watch. Likely cause: `entity_aliases` is keyed
`(entity_type, alias_key)` and the leg filters on `alias_key` alone, which cannot use that
index's leading column. Unverified, and filed as **M2.8a** rather than fixed on a hunch.

### What it does not do

`multi_hop` is **0.00 in both arms**, and that is a property of the design, not a tuning
failure. M2.6 built memory↔entity links and no entity↔entity edges, so the leg is strictly
one hop: the question must name the entity itself. `"Where is the head office of the customer
on account BR-88214?"` produces byte-identical scores in both arms — `br 88214` is nobody's
alias, so the leg contributes nothing. `"What plan is Dana's company on?"` is worse than
nothing: the leg boosts the *Dana* memory and entrenches the wrong answer at rank 1.

Two hops needs an edge table that does not exist. Kept as a labelled, failing category so the
gap stays measured instead of remembered.
