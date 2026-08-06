"""Golden-set eval runner (M2.5) — measure retrieval instead of arguing about it.

    uv run python -m evals.runner                    # the production retrieval path
    uv run python -m evals.runner --variant rrf      # baseline: raw fusion, pipeline bypassed
    uv run python -m evals.runner --no-entities      # baseline: no entity leg (pre-M2.8)

Seeds a *throwaway* pgvector container, so a run can never touch a real deployment's data
and two runs of the same commit give the same answer. Embeddings come from the configured
model, because relevance is a property of the model and the SQL together — stubbing the
model here would measure nothing worth knowing.

dev-plan §10 wants this run on every meaningful change from M2 onward: it is the thing
that stops M2.6–M2.8's entity layer from being added on faith.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from testcontainers.postgres import PostgresContainer

from evals.metrics import QueryResult, Summary, summarize
from memora.config import Settings
from memora.models import ActorType, MemoryCreate, MemoryType, Scope
from memora.providers import get_embedding_provider
from memora.providers.base import EmbeddingProvider
from memora.retrieval import retrieve
from memora.store.base import RetrievedMemory
from memora.store.postgres import PostgresStorage

GOLDEN = Path(__file__).parent / "golden"
K_VALUES = (1, 3, 5)


async def _search(
    store: PostgresStorage,
    embedder: EmbeddingProvider,
    query: str,
    scope: Scope,
    variant: str,
    limit: int,
) -> list[RetrievedMemory]:
    if variant == "rrf":
        # raw fusion straight out of the store, pipeline bypassed — the baseline any ranking
        # stage has to beat. It is what retired M2.2's weighting in M2.5a.
        [vector] = await embedder.embed([query])
        candidates = await store.hybrid_search(
            query_embedding=vector, query_text=query, scope=scope
        )
        return candidates[:limit]
    return await retrieve(store, embedder, query, scope=scope, limit=limit)


async def _link_entities(
    store: PostgresStorage, stored_ids: list[Any], corpus: list[dict[str, Any]]
) -> None:
    """Resolve and link the golden set's hand-labelled entities (M2.8).

    Hand-labelled in the exact shape M2.7's extraction emits, and an upper bound on
    purpose: this measures the entity leg against *perfect* extraction, so a win here is
    the ceiling rather than the deployed number. It is the same kind of label `type` and
    `confidence` already are.

    `--no-entities` skips this entirely, which leaves `memory_entities` empty and makes the
    entity leg contribute zero rows to fusion — bit-for-bit the M2.5 vector+FTS baseline,
    which is the only honest thing to compare the leg against.
    """
    for memory_id, memory in zip(stored_ids, corpus, strict=True):
        entity_ids = [
            await store.resolve_entity(
                name=entity["canonical_name"], type=entity["type"], aliases=entity["mentions"]
            )
            for entity in memory.get("entities", [])
        ]
        await store.link_memory(memory_id=memory_id, entity_ids=entity_ids)


def run(
    corpus_path: Path, queries_path: Path, variant: str, limit: int, link_entities: bool
) -> list[QueryResult]:
    """Spin up a throwaway migrated Postgres, then measure inside it.

    Sync on purpose: alembic's env.py calls `asyncio.run` itself, so migrations have to
    happen before an event loop exists.
    """
    corpus = json.loads(corpus_path.read_text())
    queries = json.loads(queries_path.read_text())

    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", url)
        command.upgrade(config, "head")
        return asyncio.run(_measure(url, corpus, queries, variant, limit, link_entities))


async def _measure(
    url: str,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    variant: str,
    limit: int,
    link_entities: bool,
) -> list[QueryResult]:
    embedder = get_embedding_provider(Settings())
    store = PostgresStorage(url)
    try:
        # the enums are constructed explicitly so a typo in the golden set fails loudly
        # here rather than quietly changing what the run measures
        items = [
            MemoryCreate(
                content=m["content"],
                type=MemoryType(m["type"]),
                actor_type=ActorType(m.get("actor_type", "agent")),
                confidence=m.get("confidence"),
                scope=Scope(user_id=m.get("user_id")),
            )
            for m in corpus
        ]
        vectors = await embedder.embed([m["content"] for m in corpus])
        stored_ids = await store.add_memories(items, embeddings=vectors)
        if link_entities:
            await _link_entities(store, stored_ids, corpus)
        # retrieval speaks in UUIDs; the golden set speaks in stable labels like "a01"
        label = {stored: m["id"] for stored, m in zip(stored_ids, corpus, strict=True)}

        results = []
        for q in queries:
            scope = Scope(user_id=q.get("user_id"))
            started = time.perf_counter()
            hits = await _search(store, embedder, q["query"], scope, variant, limit)
            results.append(
                QueryResult(
                    category=q["category"],
                    expected_id=q["expected_id"],
                    retrieved_ids=[label[h.id] for h in hits],
                    latency_s=time.perf_counter() - started,
                )
            )
        return results
    finally:
        await store.dispose()


def print_report(summaries: dict[str, Summary], results: list[QueryResult]) -> None:
    header = f"{'category':<12}{'queries':>8}" + "".join(f"{f'hit@{k}':>8}" for k in K_VALUES)
    print(f"\n{header}{'MRR':>8}{'p95 s':>8}")
    print("-" * len(f"{header}{'MRR':>8}{'p95 s':>8}"))
    for name, s in summaries.items():
        scores = "".join(f"{s.hit_at_k[k]:>8.2f}" for k in K_VALUES)
        print(f"{name:<12}{s.queries:>8}{scores}{s.mrr:>8.2f}{s.p95_latency_s:>8.3f}")

    misses = [r for r in results if r.expected_id not in r.retrieved_ids[: max(K_VALUES)]]
    if misses:
        print(f"\n{len(misses)} miss(es) — the only rows worth reading closely:")
        for r in misses:
            print(f"  [{r.category}] expected {r.expected_id}, got {r.retrieved_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=GOLDEN / "memories.json")
    parser.add_argument("--queries", type=Path, default=GOLDEN / "queries.json")
    parser.add_argument(
        "--variant",
        choices=("pipeline", "rrf"),
        default="pipeline",
        help="pipeline = the production retrieval path; rrf = raw fusion, for comparison",
    )
    parser.add_argument("--limit", type=int, default=max(K_VALUES))
    parser.add_argument(
        "--no-entities",
        action="store_true",
        help="skip entity linking — the pre-M2.8 vector+FTS baseline",
    )
    args = parser.parse_args()

    results = run(args.corpus, args.queries, args.variant, args.limit, not args.no_entities)
    entities = "off" if args.no_entities else "on"
    print(f"\nvariant={args.variant} entities={entities} queries={len(results)}")
    print_report(summarize(results, k_values=K_VALUES), results)


if __name__ == "__main__":
    main()
