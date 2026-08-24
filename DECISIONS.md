# Design decisions

Why memora is built the way it is. Each entry states the decision, the reasoning, and — where one exists — the measurement that settled it.

The rule the project runs on: **a claim without a number is a guess.** Several decisions below reversed the intuition that preceded them, and one deleted a finished feature.

---

## Storage

### One database, not three

**Decision.** PostgreSQL 16 + pgvector in a single container. No separate vector store, no graph database.

**Why.** Postgres does vector search (pgvector/HNSW), keyword search (native full-text), and relational storage well enough, in one process. A dedicated vector database would be better at one of those and would cost two more containers to operate. Self-hosting is the product — the thing a user runs has to stay a single `docker compose up`.

The interface (`StorageBackend`) exists so this is a swap rather than a rewrite, but only one implementation ships.

### Postgres is also the job queue

**Decision.** Extraction jobs live in a Postgres table, claimed with `SELECT … FOR UPDATE SKIP LOCKED`.

**Why.** Async writes without adding Redis. `SKIP LOCKED` makes concurrent workers pass over rows another transaction is taking, so a job is handed out exactly once — one statement, no external broker, no second thing to operate.

### Embeddings run locally by default

**Decision.** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) is the default; API providers are optional.

**Why.** The default has to work with no API key and no outbound connection. Embedding spaces are model-specific, so this is a setup-time choice — switching later means a column migration plus re-embedding every stored memory. The provider reports its dimension and startup fails loudly on a mismatch rather than storing vectors that can never be compared.

### Writes never block on the model

**Decision.** `POST /v1/memories` enqueues and returns `202` with a job id. A worker does the LLM call.

**Why.** Extraction takes seconds. An agent's turn cannot wait on it. The cost is that memories are eventually consistent — a fact isn't searchable the instant it's mentioned — which is the right trade for a system read on every turn and written to occasionally.

---

## Retrieval

### Fuse by rank, not by score

**Decision.** Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` over the legs, `k = 60`.

**Why.** The legs produce incomparable numbers — cosine distance (0–2, lower is better), `ts_rank_cd` (unbounded, higher is better), and an entity count. There is no principled way to add them. RRF discards the scores and keeps only positions, which *are* comparable across any legs.

`k = 60` compresses the spread between rank 1 and rank 20 from 20× to 1.31×, so **two legs weakly agreeing outrank one leg shouting.** That's the behaviour a hybrid system should have, and it meant adding a third leg later required no re-tuning of the first two.

### Each leg queries the base table directly

**Decision.** Every leg repeats `FROM memories WHERE … ORDER BY … LIMIT`. The scope filter is *not* factored into a shared CTE.

**Why.** The factored version reads better and was measurably slower. Postgres materialises any CTE referenced more than once, so both legs ended up scanning a temporary copy — where the HNSW and GIN indexes do not exist. The vector leg silently degraded to sorting the entire table, and it grew linearly with the corpus, so it looked fine in tests and got steadily worse in production.

**Measured (20K rows).** Total 25.4 ms → **8.6 ms**. Vector leg 7.7 ms → **0.9 ms**, once the planner could use `ix_memories_embedding_hnsw`.

Found with `EXPLAIN`, not with a profiler and not with tests — the results were correct the whole time.

### Keyword search requires every query term

**Decision.** `plainto_tsquery` (AND) rather than rewriting to OR.

**Why.** OR was chosen first to protect recall. At volume it matched a fixed *fraction* of the corpus — a median of 1,097 rows out of 20,000 — filling 80% of the fusion pool with memories that shared one incidental word, each earning a full-strength vote against what the vector leg actually found.

**Measured.** hit@1 0.65 → **0.90**, MRR 0.78 → **0.94** on the golden set; search p95 ~21 ms → **11.2 ms** at 20K; keyword leg p50 1,097 → **0** rows on natural-language questions.

**The cost, accepted knowingly.** hit@5 0.95 → 0.80 at volume. A question whose exact words aren't all present now leans entirely on the vector leg — which is the leg whose job that is. Keyword for tokens, vector for meaning.

### Trust weighting was built, measured, and deleted

**Decision.** Retrieval ranks on relevance alone. There is no boost for corrections, policies, or high-confidence memories.

**Why.** The intuition — a human-verified correction should outrank an ordinary fact — is reasonable and wrong in this position. A bounded bonus was built and did improve on the no-boost baseline, and still lost to plain fusion. A sweep of **4 boost shapes × 9 strengths**, all scored against identical candidate pools, found **no setting that beat plain RRF**: degradation was monotone, and the only value that tied the baseline did so by changing no ranking at all.

**Why no constant could have worked.** RRF is deliberately flat — about 1.6% between neighbouring ranks. A boost is a property of a memory *alone*, applied in every pool it lands in, including the many where it's off-topic. Any boost large enough to flip a genuine near-tie also flips an arbitrary adjacent pair, and the scores contain nothing that distinguishes the two cases.

**Measured.** Across 20 queries the boost promoted the expected memory on **1** and demoted it on **6**.

The scoring module was deleted. The lesson generalised: relevance-versus-trust is not decidable from a relevance score, so it belongs where the correction→fact relationship is *recorded*, not where it is ranked.

---

## Entities

### Alias matching only — no embedding similarity

**Decision.** Two names refer to the same thing when they fold to an identical key. Nothing fuzzy.

**Why.** A pre-build spike expected the opposite: that models report names inconsistently and fuzzy matching would be needed to clean up after them. The measurement said surface-form consistency was the model's *strongest* result (`Priya Nair` → `Priya` → `P. Nair`, 3/3 reps). The problem embeddings were going to solve did not exist.

Embeddings would also have made the known failure worse: a person and their employer co-occur in every sentence, so they embed close together — pushing toward exactly the merge that must not happen.

### The alias index is partitioned by entity type

**Decision.** The key is `(entity_type, alias_key)`, not `alias_key` alone.

**Why.** An email like `leo@stackpine.dev` is a legitimate mention of *both* a person and their employer. With a type-blind index, that one shared key fuses the person into the company.

**Measured.** 3 collapses in 3 reps without partitioning; **0** with it, at no cost in correct merges.

Stripping legal suffixes (`Inc.`, `Ltd`, `LLC`) took cross-conversation resolution from 11/12 to **12/12** — matched as whole words only, so `Cisco` and `BrightCo` keep their endings.

### Constraints do the enforcing, not our code

**Decision.** Invariants live in the schema wherever they can.

**Why.** Careful code is a promise every future code path has to keep. A constraint cannot be forgotten.

- `PRIMARY KEY (entity_type, alias_key)` means "one entity per name" is enforced by Postgres. Two workers resolving the same new customer concurrently **collide on the constraint** instead of quietly creating duplicates — the exact failure entity resolution exists to prevent. A loud error you can handle beats silent corruption you never notice.
- A trigger rejects `UPDATE` and `DELETE` on the audit log, so history cannot be rewritten by memora's own code.
- The audit log has **no foreign key** to memories, deliberately: a deletion request must remove the memory while the record that it existed and was deleted survives it.

---

## Method

### The eval guard measures growth, not wall-clock

**Decision.** The volume harness fails on how latency *scales* with corpus size, not on an absolute millisecond budget.

**Why.** No honest budget could be set — measured p95 was 21.5 ms against a defensible ceiling of 25 ms, which would cry wolf constantly. But the defect being guarded against has a *shape*: an unreachable index degrades to a scan, and a scan tracks the row count.

**Validated against the real bug**, not assumed: the CTE defect replayed out of git history grows **3.8×** for 4× the rows, where the current query grows 1.7–2.1×. The ceiling sits between them.

Three design errors in the harness itself, each caught by measuring rather than reasoning: the verdict must use **unscoped** queries only (against the real defect, scoped queries grew 1.07× while unscoped grew 3.70×, so pooling hid a completely broken index); samples must be **best-of-N per query**, since pooling swung the verdict between 1.48× and 2.47× on identical code; and `ANALYZE` must run after each bulk load, or the planner uses pre-load statistics — a state no deployment is ever in.

### Failing categories stay in the eval output

**Decision.** `multi_hop` sits at 0.00 in the published results and is not removed.

**Why.** Memories link to entities; entities don't link to each other. So *"what plan is Dana's company on?"* cannot be answered, and one such query is made actively *worse* by the entity leg. That gap is real, and a metric that only reports what already works cannot tell you when it closes.

---

*Numbers here come from `evals/` — a hand-labelled golden set plus a generated 20,000-memory corpus. Both are reproducible: see `evals/README.md`.*
