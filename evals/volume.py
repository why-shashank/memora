"""Volume + latency guard (M2.5b) — the defect class a golden set cannot see.

    uv run python -m evals.volume                     # 5K -> 20K, exits non-zero on regression
    uv run python -m evals.volume --memories 50000

The M2.1 HNSW regression was invisible at test-row counts: at forty rows every query plan is
identical and every latency is a rounding error, so a green suite said nothing. It took
EXPLAIN against 20,000 rows to show the vector leg sorting the whole table. This harness runs
the *labelled* golden queries against a generated corpus big enough for the planner's choices
to matter, and measures three things the golden-set runner structurally cannot.

**1. Whether search cost scales with the corpus.** This is the guard, and it is a ratio rather
than a threshold on purpose. An absolute budget has to be re-tuned per machine, and picking
one honestly turned out to be impossible here: measured p95 sat at 21.5ms against the only
defensible budget of 25ms, so the guard would have cried wolf on a busy laptop. But the defect
has a *shape* — an unusable index degrades a query to a scan, and a scan grows linearly with
row count while HNSW and GIN grow far slower. So the corpus is measured at two sizes and the
growth compared: sub-linear passes, linear fails, on any hardware.

Verified against the real thing rather than assumed: replaying the pre-6ac8c65 shared-CTE
shape out of git against this corpus measures **3.6×** growth for 4× the rows, while the
current shape measures 1.7–1.9×. A guard nobody has seen fire is a guard nobody should trust.

**2. Latency per phase, not end to end.** S3 found the embedding call dominates, which makes
an end-to-end number nearly blind to the SQL: the M2.1 defect moved the vector leg 0.9ms →
7.7ms, a change that disappears inside a ~25ms embed. Timing the search alone turns a masked
25% wobble into an unmissable 8×.

**3. How wide the keyword leg casts, and what volume does to relevance.** hit@k against 41
distractors and hit@k against 20,000 are different questions, and only the second resembles a
deployment. This is the measurement that condemned OR-semantics (M2.5b → M2.5c): the cost was
*volume-dependent* and so invisible everywhere else — "how long do I have to get my money
back" matched a handful of rows in the golden set and 1,097 here, filling 80% of the fusion
pool. Under AND the same probe reads 0 rows and 26%. Watch it for drift upward.

Deliberately not a pytest test: a run takes minutes, and asserting on wall-clock timings in
the suite buys flakiness rather than confidence. It is a tool you run when you touch
retrieval — verified by running green, like `docker compose up`.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

from evals.metrics import QueryResult, percentile, summarize
from evals.runner import GOLDEN, K_VALUES
from memora.config import Settings
from memora.models import ActorType, MemoryCreate, MemoryType, Scope
from memora.providers import get_embedding_provider
from memora.providers.base import EmbeddingProvider
from memora.store.postgres import PostgresStorage

# Insert and embed in chunks: 20K memories in one transaction means 20K ORM objects plus
# 20K audit rows resident at once, and a single embed call that big just waits.
_BATCH = 1000

# Each query is timed several times. Twenty samples make a meaningless p95, and a served
# endpoint runs warm anyway — a full untimed pass precedes the timed ones for that reason.
_REPEATS = 10

# The corpus is measured at 1/4 size and full size. Search cost may grow, but it must grow
# *sub-linearly*: an index that stopped being used tracks the row count.
_FIRST_STAGE = 0.25

# Where to draw the fail line, as an exponent of the corpus growth: search may grow at most
# as fast as the ¾ power of the row count (4× more rows → at most 2.83×).
#
# Calibrated from measurement, not theory. The real M2.1 defect — the shared-CTE shape
# replayed out of git against this same corpus — grows **3.8×**, near-perfectly linear.
# The current query grew 1.7–2.1× when this was calibrated under OR-semantics and 1.62×
# since M2.5c switched the keyword leg to AND. 2.83× keeps margin on both sides.
#
# The ceiling is deliberately looser than theory (which would argue nearer 0.5, since HNSW
# and GIN are both far better than linear). Under OR the keyword leg was genuinely
# near-linear — it matched a fixed *fraction* of the corpus, so more rows meant
# proportionally more ranking work even with GIN doing its job perfectly. AND removed most
# of that, and the extra room is now headroom rather than slack. Left as-is on purpose: a
# guard retuned tight after every improvement is a guard that fires on the next one.
_GROWTH_EXPONENT = 0.75

_FIRST = [
    "Dana",
    "Marcus",
    "Priya",
    "Leo",
    "Omar",
    "Yuki",
    "Carlos",
    "Ingrid",
    "Aisha",
    "Viktor",
    "Mei",
    "Sam",
    "Nadia",
    "Piotr",
    "Elena",
    "Ravi",
    "Greta",
    "Kofi",
]
_LAST = [
    "Reyes",
    "Webb",
    "Nair",
    "Tran",
    "Haddad",
    "Sato",
    "Vega",
    "Larsen",
    "Silva",
    "Khan",
    "Novak",
    "Lin",
    "Ortiz",
    "Petrov",
    "Rossi",
    "Iyer",
    "Berg",
    "Mensah",
]
_COMPANIES = [
    "BrightCo",
    "FieldWorks",
    "LumenHealth",
    "Stackpine",
    "NorthBay",
    "Quantex",
    "HelioSoft",
    "Vantage",
    "BlueRidge",
    "Corelink",
    "Zenwave",
    "Optima",
    "GreenGrid",
    "Silverline",
    "Foundry",
    "Peakstone",
    "Clearwater",
    "Arclight",
]
_PLANS = ["Starter", "Growth", "Pro", "Scale", "Enterprise"]
_ENVS = [
    "routes all traffic through a Zscaler corporate proxy",
    "runs the sync client on Windows 11 behind a strict firewall",
    "integrates with us through the Salesforce connector",
    "signs in through Okta for every login",
    "deploys our agent inside an air-gapped network segment",
]
_CHANNELS = [
    "prefers email and never answers phone calls",
    "wants a phone call for anything urgent",
    "wants updates posted in the shared Slack channel",
]
_FIXES = [
    "reconfiguring the proxy",
    "reissuing the API key",
    "clearing the client cache",
    "rotating the SSO certificate",
    "rebuilding the search index",
]


def generate(count: int) -> list[dict[str, Any]]:
    """Synthetic support memories spread over many tenants.

    Seeded, so two runs of one commit measure the same corpus. The text matters more than it
    looks: the vector index only behaves realistically if the embeddings have genuine cluster
    structure (random vectors are near-equidistant — the pathological case for HNSW), and the
    keyword leg only behaves realistically against varied natural language.
    """
    rng = random.Random(7)
    out: list[dict[str, Any]] = []
    customer = 0
    while len(out) < count:
        uid = f"gen-{customer:05d}"
        company = f"{rng.choice(_COMPANIES)} {customer}"
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        rows = [
            (
                "entity_fact",
                f"{company} is on the {rng.choice(_PLANS)} plan with "
                f"{rng.choice(['monthly', 'annual'])} billing.",
            ),
            (
                "entity_fact",
                f"Billing account number for {company} is GN-{100000 + customer * 37}.",
            ),
            (
                "entity_fact",
                f"{name}'s contact email is "
                f"{name.split()[0].lower()}@{company.split()[0].lower()}.example.com.",
            ),
            ("entity_fact", f"{company} {rng.choice(_ENVS)}."),
            ("preference", f"{name} {rng.choice(_CHANNELS)}."),
            (
                "commitment",
                f"Support owes {name} a follow-up about the "
                f"{rng.choice(['pricing sheet', 'migration plan', 'security review'])}.",
            ),
        ]
        for _ in range(rng.randint(2, 5)):  # resolved-issue noise, the bulk of a real corpus
            code = f"{rng.choice(['SYNC', 'AUTH', 'NET', 'BILL'])}-{rng.randint(100, 999)}"
            rows.append(
                (
                    "entity_fact",
                    f"Previously resolved a {code} error for {company} by {rng.choice(_FIXES)}.",
                )
            )
        out += [{"user_id": uid, "type": t, "content": c} for t, c in rows]
        customer += 1
    return out[:count]


async def _seed(
    store: PostgresStorage, embedder: EmbeddingProvider, rows: list[dict[str, Any]]
) -> list[Any]:
    ids: list[Any] = []
    for start in range(0, len(rows), _BATCH):
        chunk = rows[start : start + _BATCH]
        items = [
            MemoryCreate(
                content=m["content"],
                type=MemoryType(m["type"]),
                actor_type=ActorType(m.get("actor_type", "agent")),
                confidence=m.get("confidence"),
                scope=Scope(user_id=m.get("user_id")),
            )
            for m in chunk
        ]
        vectors = await embedder.embed([m["content"] for m in chunk])
        ids += await store.add_memories(items, embeddings=vectors)
        print(f"\r  seeded {len(ids):,}/{len(rows):,}", end="", flush=True)
    print()
    # A bulk insert leaves the planner reading statistics from before the load, which is a
    # state no running deployment is ever in — autovacuum keeps them current. Without this
    # the two measurement stages are planned against different-quality stats and the
    # comparison measures ANALYZE, not the index.
    async with store.session_factory() as session:
        await session.execute(text("ANALYZE memories"))
        await session.commit()
    return ids


async def _fts_breadth(store: PostgresStorage, query: str) -> int:
    """How many rows the keyword leg matches before the pool truncates it.

    Mirrors `_HYBRID_SQL`'s tsquery exactly — keep the two in step. This number is what
    condemned OR-semantics in M2.5b (median 1,097 rows at 20K, 80% of the pool); under the
    AND-semantics of M2.5c it should be small, and a drift upward means the keyword leg has
    started casting wide again.
    """
    sql = text("SELECT count(*) FROM memories WHERE content_tsv @@ plainto_tsquery('english', :q)")
    async with store.session_factory() as session:
        return int((await session.execute(sql, {"q": query})).scalar_one())


async def _time_queries(
    store: PostgresStorage,
    embedder: EmbeddingProvider,
    queries: list[dict[str, Any]],
    label: dict[Any, str],
) -> tuple[list[float], list[float], dict[str, list[float]], list[QueryResult]]:
    """Returns raw embed times, raw search times, per-query *best* search times bucketed by
    scoped/unscoped, and the retrieval results.

    Two deliberate choices, both learned by getting them wrong first.

    **Bucketing by scope**, because only one bucket can see the defect this harness exists
    for. A tenant-scoped query filters to a handful of rows, where scanning *is* the right
    plan and the planner picks btree over HNSW regardless — so it looks identical whether the
    vector index is reachable or not. Measured against the M2.1 defect: scoped queries grew
    1.07× while the corpus grew 4×, unscoped grew 3.70×. Pool the two and a broken index
    vanishes completely, and the golden set is 60% scoped.

    **Best-of-N per query for the guard**, raw samples for the reported latency. Interference
    is one-sided — a busy machine or a cold cache only ever makes a query slower — so the
    minimum is the least-contaminated estimate of the work actually being done, and it still
    grows linearly when the work does. Pooling raw samples instead made the verdict swing
    between 1.48× and 2.47× across two runs of identical code, which is a guard that cannot
    be trusted either way. The latency table keeps the raw distribution, because there the
    noise is the point.
    """
    embed_times: list[float] = []
    search_raw: list[float] = []
    search_best: dict[str, list[float]] = {"scoped": [], "unscoped": []}
    results: list[QueryResult] = []

    # One untimed pass over every query first. Timing the very first touch of a freshly
    # grown corpus measures cold index pages, not the query — and that cost lands entirely
    # on the larger stage, which is exactly where it would fake a scaling regression.
    vectors = {q["query"]: (await embedder.embed([q["query"]]))[0] for q in queries}
    for q in queries:
        await store.hybrid_search(
            query_embedding=vectors[q["query"]],
            query_text=q["query"],
            scope=Scope(user_id=q.get("user_id")),
        )

    for q in queries:
        scope = Scope(user_id=q.get("user_id"))
        bucket = "scoped" if q.get("user_id") else "unscoped"
        hits = []
        this_query: list[float] = []
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            [vector] = await embedder.embed([q["query"]])
            t1 = time.perf_counter()
            hits = await store.hybrid_search(
                query_embedding=vector, query_text=q["query"], scope=scope
            )
            t2 = time.perf_counter()
            embed_times.append(t1 - t0)
            this_query.append(t2 - t1)
        search_raw += this_query
        search_best[bucket].append(min(this_query))
        results.append(
            QueryResult(
                category=q["category"],
                expected_id=q["expected_id"],
                # generated distractors are unlabelled: anything outside the golden set is by
                # construction a wrong answer, so a placeholder id is the honest record
                retrieved_ids=[label.get(h.id, "-") for h in hits[: max(K_VALUES)]],
                latency_s=0.0,
            )
        )
    return embed_times, search_raw, search_best, results


def run(memories: int) -> int:
    corpus = generate(memories)
    golden = json.loads((GOLDEN / "memories.json").read_text())
    queries = json.loads((GOLDEN / "queries.json").read_text())

    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", url)
        command.upgrade(config, "head")
        return asyncio.run(_measure(url, corpus, golden, queries))


async def _measure(
    url: str,
    corpus: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> int:
    embedder = get_embedding_provider(Settings())
    store = PostgresStorage(url)
    try:
        split = int(len(corpus) * _FIRST_STAGE)
        started = time.perf_counter()
        print(f"seeding {len(golden)} golden + {split:,} generated memories")
        golden_ids = await _seed(store, embedder, golden)
        label = {stored: m["id"] for stored, m in zip(golden_ids, golden, strict=True)}
        await _seed(store, embedder, corpus[:split])
        small_total = split + len(golden)
        *_, small_search, _ = await _time_queries(store, embedder, queries, label)

        print(f"growing to {len(corpus):,} generated memories")
        await _seed(store, embedder, corpus[split:])
        full_total = len(corpus) + len(golden)
        embed_times, search_times, full_search, results = await _time_queries(
            store, embedder, queries, label
        )
        print(f"  seeded and measured in {time.perf_counter() - started:.0f}s")

        breadth = [await _fts_breadth(store, q["query"]) for q in queries]
        pool_share = [min(b, 20) / 20 for b in breadth]

        print(
            f"\ncorpus  {full_total:,} memories  ·  {len(queries)} queries × "
            f"{_REPEATS} timed passes"
        )

        print(f"\n{'phase':<10}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}")
        print("-" * 37)
        for name, samples in (("embed", embed_times), ("search", search_times)):
            print(
                f"{name:<10}{percentile(samples, 50) * 1000:>9.1f}"
                f"{percentile(samples, 95) * 1000:>9.1f}{max(samples) * 1000:>9.1f}"
            )

        print(f"\n{'keyword leg':<14}{'p50 rows':>10}{'p95 rows':>10}{'pool filled':>13}")
        print("-" * 47)
        print(
            f"{'and-semantics':<14}{percentile(breadth, 50):>10,.0f}"
            f"{percentile(breadth, 95):>10,.0f}{sum(pool_share) / len(pool_share):>12.0%}"
        )

        print(f"\nrelevance against {len(corpus):,} distractors")
        header = f"{'category':<12}{'queries':>8}" + "".join(f"{f'hit@{k}':>8}" for k in K_VALUES)
        print(f"{header}{'MRR':>8}")
        print("-" * len(f"{header}{'MRR':>8}"))
        for name, s in summarize(results, k_values=K_VALUES).items():
            scores = "".join(f"{s.hit_at_k[k]:>8.2f}" for k in K_VALUES)
            print(f"{name:<12}{s.queries:>8}{scores}{s.mrr:>8.2f}")

        # The guard: an index that stopped being used makes search track the row count.
        # Only the unscoped verdict counts — see `_time_queries` for why scoped queries are
        # blind to this. Scoped is printed anyway, because a change there is worth a look.
        corpus_growth = full_total / small_total
        ceiling = corpus_growth**_GROWTH_EXPONENT
        print(
            f"\nscaling  {small_total:,} → {full_total:,} rows ({corpus_growth:.1f}×)  ·  "
            f"ceiling {ceiling:.2f}×, linear would be {corpus_growth:.1f}×"
        )
        print(f"\n{'queries':<10}{'p50 small':>11}{'p50 full':>11}{'growth':>9}   verdict")
        print("-" * 52)
        over = False
        for bucket in ("scoped", "unscoped"):
            small_p50 = percentile(small_search[bucket], 50)
            full_p50 = percentile(full_search[bucket], 50)
            growth = full_p50 / small_p50 if small_p50 else 0.0
            failed = growth > ceiling
            verdict = "REGRESSION" if failed else "ok, sub-linear"
            if bucket == "unscoped":
                over = failed
            else:
                verdict += " (advisory)" if not failed else " (advisory — investigate)"
            print(
                f"{bucket:<10}{small_p50 * 1000:>10.1f}ms{full_p50 * 1000:>10.1f}ms"
                f"{growth:>8.2f}×   {verdict}"
            )
        if over:
            print("\nsearch is tracking the row count: an index is no longer being reached.")
        return 1 if over else 0
    finally:
        await store.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memories", type=int, default=20_000)
    args = parser.parse_args()
    sys.exit(run(args.memories))


if __name__ == "__main__":
    main()
