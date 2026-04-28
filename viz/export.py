"""
Export pipeline run outputs to a single snapshot.parquet for the viz app.

Usage
-----
    python viz/export.py --run-dir /path/to/run --out viz/data/snapshot.parquet

Options
-------
    --run-dir   Path to the pipeline run directory (the base_path in config).
    --out       Output path for the snapshot Parquet file.
                Default: viz/data/snapshot.parquet
    --sample    Maximum number of prompts to include in the browser snapshot.
                Default: 50000. Full-data aggregate stats are always computed
                over the entire dataset regardless of this value.
    --seed      Random seed for sampling. Default: 42
    --umap      Compute UMAP 2D coords for sampled prompts (slow; requires
                umap-learn and embeddings shards). Off by default.

The script auto-detects which stages are complete by probing for output
directories, and joins available data in a single pass. Full-data aggregate
statistics (domain, difficulty, language, source, topics, cluster distributions)
are computed via Polars over all records and stored in meta.json under "stats".
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Reservoir sampling — O(sample_size) memory, works on any iterable
# ---------------------------------------------------------------------------

def _reservoir_sample(iterable, n: int, seed: int) -> tuple[list[dict], int]:
    """
    Return (sample, total_seen) using reservoir sampling.
    Never loads more than n records into memory at once.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0
    for rec in iterable:
        total += 1
        if len(reservoir) < n:
            reservoir.append(rec)
        else:
            j = rng.randint(0, total - 1)
            if j < n:
                reservoir[j] = rec
    return reservoir, total


# ---------------------------------------------------------------------------
# Full-data aggregate stats via Polars (shard-by-shard, low memory)
# ---------------------------------------------------------------------------

def _compute_full_stats(stage3_dir: Path) -> dict:
    """
    Scan all part-*.jsonl shards shard-by-shard and compute aggregate stats.
    Peak memory ≈ one shard at a time (~50–100 MB per shard parsed).
    Returns a dict ready to embed in meta.json under "stats".
    """
    try:
        import polars as pl
    except ImportError:
        logger.warning("Polars not available — skipping full stats computation.")
        return {}

    shards = sorted(stage3_dir.glob("part-*.jsonl"))
    if not shards:
        return {}

    logger.info("Computing full-data stats over %d shards …", len(shards))

    domain_counts:     Counter = Counter()
    difficulty_counts: Counter = Counter()
    language_counts:   Counter = Counter()
    source_counts:     Counter = Counter()
    topics_counts:     Counter = Counter()

    # Cross-tab accumulators
    domain_difficulty: Counter = Counter()   # (domain, difficulty) → count
    domain_language:   Counter = Counter()   # (domain, language)   → count

    # Topics per domain
    topics_by_domain: dict[str, Counter] = defaultdict(Counter)

    # Cluster stats
    cluster_size:         Counter = Counter()           # cluster_id → count
    cluster_domain:       dict[int, Counter] = defaultdict(Counter)  # cluster_id → {domain: count}

    total = 0

    WANTED = ["domain", "difficulty", "language", "source", "topics", "cluster_id"]

    for idx, shard in enumerate(shards):
        try:
            df = pl.read_ndjson(str(shard))
        except Exception as exc:
            logger.warning("  Could not read %s: %s — skipping.", shard.name, exc)
            continue

        cols = [c for c in WANTED if c in df.columns]
        df = df.select(cols)
        total += len(df)

        def _str_col(col: str) -> list:
            return df[col].to_list() if col in df.columns else []

        domains     = _str_col("domain")
        difficulties= _str_col("difficulty")
        languages   = _str_col("language")
        sources     = _str_col("source")
        cluster_ids = _str_col("cluster_id")
        topics_col  = _str_col("topics") if "topics" in df.columns else []

        for v in domains:
            if v: domain_counts[v] += 1
        for v in difficulties:
            if v: difficulty_counts[v] += 1
        for v in languages:
            if v: language_counts[v] += 1
        for v in sources:
            if v: source_counts[v] += 1
        for v in cluster_ids:
            if v is not None: cluster_size[int(v)] += 1

        # Topics
        for t_list in topics_col:
            if not t_list:
                continue
            for t in t_list:
                if t and t.strip():
                    topics_counts[t.strip()] += 1

        # Cross-tabs
        for d, diff in zip(domains, difficulties):
            if d and diff:
                domain_difficulty[(d, diff)] += 1
        for d, lang in zip(domains, languages):
            if d and lang:
                domain_language[(d, lang)] += 1

        # Topics by domain
        for d, t_list in zip(domains, topics_col):
            if not d or not t_list:
                continue
            for t in t_list:
                if t and t.strip():
                    topics_by_domain[d][t.strip()] += 1

        # Cluster dominant domain
        for cid, d in zip(cluster_ids, domains):
            if cid is not None and d and int(cid) >= 0:
                cluster_domain[int(cid)][d] += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(shards):
            logger.info("  %d / %d shards processed (%d records) …", idx + 1, len(shards), total)

    logger.info("Full stats done: %d records across %d domains.", total, len(domain_counts))

    # ── Build structured dicts ─────────────────────────────────────────────

    all_domains   = [d for d, _ in domain_counts.most_common()]
    difficulties  = ["easy", "medium", "hard"]
    top_languages = [lang for lang, _ in language_counts.most_common(10)]

    dd_matrix = {
        "domains":      all_domains,
        "difficulties": difficulties,
        "counts": [
            [domain_difficulty.get((d, diff), 0) for diff in difficulties]
            for d in all_domains
        ],
    }

    dl_matrix = {
        "domains":   all_domains,
        "languages": top_languages,
        "counts": [
            [domain_language.get((d, lang), 0) for lang in top_languages]
            for d in all_domains
        ],
    }

    topics_by_domain_top = {
        d: [[t, c] for t, c in counter.most_common(15)]
        for d, counter in topics_by_domain.items()
        if d in domain_counts
    }

    # Cluster stats
    valid_clusters = {cid: sz for cid, sz in cluster_size.items() if cid >= 0}
    n_clusters = len(valid_clusters)
    sizes = sorted(valid_clusters.values())

    # Size histogram with log-friendly bins
    import numpy as np
    if sizes:
        max_sz = max(sizes)
        bins = [0, 5, 20, 50, 100, 250, 500, 1000, 5000, max(max_sz + 1, 5001)]
        hist_counts, hist_bins = np.histogram(sizes, bins=bins)
        size_histogram = {
            "bin_edges": [int(b) for b in hist_bins.tolist()],
            "counts":    [int(c) for c in hist_counts.tolist()],
        }
    else:
        size_histogram = {"bin_edges": [], "counts": []}

    # Dominant domain per cluster → clusters-per-domain count
    clusters_per_domain: Counter = Counter()
    cluster_dominant: dict[int, str] = {}
    for cid, d_counter in cluster_domain.items():
        if d_counter:
            dom = max(d_counter, key=d_counter.get)
            cluster_dominant[cid] = dom
            clusters_per_domain[dom] += 1

    # Top 50 clusters by size
    top_clusters = [
        {
            "cluster_id": int(cid),
            "size":       int(sz),
            "domain":     cluster_dominant.get(int(cid), "unknown"),
        }
        for cid, sz in sorted(valid_clusters.items(), key=lambda x: -x[1])[:50]
    ]

    return {
        "total":            total,
        "domain_counts":    dict(domain_counts.most_common()),
        "difficulty_counts":dict(difficulty_counts),
        "language_counts":  dict(language_counts.most_common(30)),
        "source_counts":    dict(source_counts.most_common(30)),
        "topics_counts":    dict(topics_counts.most_common(100)),
        "domain_difficulty_matrix": dd_matrix,
        "domain_language_matrix":   dl_matrix,
        "topics_by_domain": topics_by_domain_top,
        "cluster_stats": {
            "n_clusters":          n_clusters,
            "clusters_per_domain": dict(clusters_per_domain.most_common()),
            "size_histogram":      size_histogram,
            "top_clusters":        top_clusters,
        },
    }


# ---------------------------------------------------------------------------
# Optional UMAP (only if --umap flag is passed)
# ---------------------------------------------------------------------------

def _compute_umap(
    sampled_ids: list[str],
    embeddings_dir: Path,
    seed: int,
) -> dict[str, tuple[float, float]] | None:
    shards = list(embeddings_dir.glob("embeddings_*.parquet"))
    if not shards:
        logger.info("  No embedding shards found — skipping UMAP.")
        return None
    try:
        import numpy as np
        import polars as pl
        import umap as umap_lib

        id_set = set(sampled_ids)
        emb_df = (
            pl.scan_parquet(str(embeddings_dir / "embeddings_*.parquet"))
            .filter(pl.col("prompt_id").is_in(list(id_set)))
            .collect()
        )
        if emb_df.is_empty():
            logger.warning("  No embeddings matched sampled IDs.")
            return None
        ids_found = emb_df["prompt_id"].to_list()
        vectors   = np.array(emb_df["embedding"].to_list(), dtype=np.float32)
        logger.info("  Running UMAP on %d vectors …", len(vectors))
        coords = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            random_state=seed, verbose=False,
        ).fit_transform(vectors)
        return {pid: (float(coords[i, 0]), float(coords[i, 1])) for i, pid in enumerate(ids_found)}
    except Exception as exc:
        logger.warning("  UMAP failed (%s).", exc)
        return None


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(
    run_dir: Path,
    out_path: Path,
    sample: int,
    seed: int,
    compute_umap: bool = False,
) -> None:
    run_dir  = run_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_dir":     str(run_dir),
        "stages":      {},
    }

    # ------------------------------------------------------------------
    # 1. Locate prompt source — Stage 3 preferred over Stage 1
    # ------------------------------------------------------------------
    stage3_dir = run_dir / "stage3"
    stage1_dir = run_dir / "stage1"

    if list(stage3_dir.glob("part-*.jsonl")):
        source_dir = stage3_dir
        meta["stages"]["stage3"] = True
        meta["stages"]["stage1"] = True
        logger.info("Reading Stage 3 prompts from %s …", stage3_dir)
    elif list(stage1_dir.glob("part-*.jsonl")):
        source_dir = stage1_dir
        meta["stages"]["stage1"] = True
        logger.info("Stage 3 not found — reading Stage 1 from %s …", stage1_dir)
    else:
        logger.error("No Stage 1 or Stage 3 output found in %s", run_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Full-data aggregate stats (Polars, shard-by-shard)
    # ------------------------------------------------------------------
    if meta["stages"].get("stage3"):
        stats = _compute_full_stats(stage3_dir)
        if stats:
            meta["stats"] = stats
            meta["total_prompts"] = stats["total"]
            logger.info("Full stats stored in meta.json['stats'].")

    # ------------------------------------------------------------------
    # 3. Reservoir-sampled snapshot for the prompt browser
    # ------------------------------------------------------------------
    from sft_pipeline.storage import iter_jsonl_dir

    logger.info("Reservoir-sampling %d prompts (seed=%d) …", sample, seed)
    rows, total_seen = _reservoir_sample(iter_jsonl_dir(source_dir), sample, seed)
    meta.setdefault("total_prompts", total_seen)
    meta["sample_size"] = len(rows)
    logger.info("  Sampled %d / %d prompts.", len(rows), total_seen)

    sampled_ids = [r["prompt_id"] for r in rows]

    # ------------------------------------------------------------------
    # 4. Optional UMAP
    # ------------------------------------------------------------------
    umap_coords: dict | None = None
    if compute_umap:
        embeddings_dir = stage3_dir / "embeddings"
        if embeddings_dir.exists():
            umap_coords = _compute_umap(sampled_ids, embeddings_dir, seed)
            if umap_coords:
                meta["stages"]["umap"] = True

    # ------------------------------------------------------------------
    # 5. Stage 5 responses
    # ------------------------------------------------------------------
    stage5_dir = run_dir / "stage5"
    resp_lookup: dict[str, dict] = {}
    if list(stage5_dir.glob("part-*.jsonl")):
        logger.info("Reading Stage 5 responses …")
        for rec in iter_jsonl_dir(stage5_dir):
            pid = rec.get("prompt_id")
            if pid:
                resp_lookup[pid] = rec
        logger.info("  %d responses loaded.", len(resp_lookup))
        meta["stages"]["stage5"] = True

    # ------------------------------------------------------------------
    # 6. Stage 6 filter results
    # ------------------------------------------------------------------
    stage6_dir = run_dir / "stage6"
    filter_lookup: dict[str, bool] = {}
    if list(stage6_dir.glob("part-*.jsonl")):
        logger.info("Reading Stage 6 filter results …")
        for rec in iter_jsonl_dir(stage6_dir):
            pid = rec.get("prompt_id")
            if pid and "passed_filters" in rec:
                filter_lookup[pid] = bool(rec["passed_filters"])
        logger.info("  %d filter results loaded.", len(filter_lookup))
        meta["stages"]["stage6"] = True

    # ------------------------------------------------------------------
    # 7. Assemble snapshot rows
    # ------------------------------------------------------------------
    logger.info("Assembling snapshot …")
    final_rows = []
    for rec in rows:
        pid = rec["prompt_id"]
        row: dict = {
            "prompt_id":  pid,
            "prompt":     rec.get("prompt", ""),
            "source":     rec.get("source", ""),
            "domain":     rec.get("domain", "other"),
            "difficulty": rec.get("difficulty"),
            "cluster_id": rec.get("cluster_id"),
            "topics":     ", ".join(rec.get("topics") or []),
            "language":   rec.get("language", "en"),
            "summary":    rec.get("summary", ""),
        }
        if umap_coords and pid in umap_coords:
            row["umap_x"], row["umap_y"] = umap_coords[pid]
        else:
            row["umap_x"] = row["umap_y"] = None
        if pid in resp_lookup:
            row["reasoning"] = resp_lookup[pid].get("reasoning")
            row["answer"]    = resp_lookup[pid].get("answer")
        else:
            row["reasoning"] = row["answer"] = None
        row["passed_filters"] = filter_lookup.get(pid)
        final_rows.append(row)

    # ------------------------------------------------------------------
    # 8. Write snapshot.parquet + meta.json
    # ------------------------------------------------------------------
    import polars as pl

    df = pl.DataFrame(final_rows)
    df.write_parquet(str(out_path), compression="zstd")
    logger.info("Snapshot → %s  (%d rows, %d cols)", out_path, len(df), len(df.columns))

    meta_path = out_path.parent / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata → %s", meta_path)
    logger.info("Stages present: %s", [k for k, v in meta["stages"].items() if v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pipeline outputs to viz snapshot.")
    parser.add_argument("--run-dir",  required=True, help="Pipeline run directory (base_path)")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data" / "snapshot.parquet"),
        help="Output snapshot.parquet path",
    )
    parser.add_argument("--sample", type=int, default=50_000, help="Max prompts to sample")
    parser.add_argument("--seed",   type=int, default=42,     help="Random seed")
    parser.add_argument(
        "--umap", action="store_true",
        help="Compute UMAP 2D coords for sampled prompts (slow; requires umap-learn)",
    )
    args = parser.parse_args()

    export(
        run_dir=Path(args.run_dir),
        out_path=Path(args.out),
        sample=args.sample,
        seed=args.seed,
        compute_umap=args.umap,
    )


if __name__ == "__main__":
    main()
