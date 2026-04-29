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
# PDF export — static report from meta.json stats
# ---------------------------------------------------------------------------

def _export_pdf(stats: dict, meta: dict, out_pdf: Path) -> None:
    """
    Render all distribution charts to a multi-page PDF.
    Reads pre-computed stats from meta.json — no raw data re-scan needed.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.backends.backend_pdf import PdfPages

    ACCENT   = "#4C78A8"
    GREY     = "#888888"
    DIFFICULTIES = ["easy", "medium", "hard"]
    DIFF_COLORS  = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}

    def _annotation_page(pdf, title: str, body: str) -> None:
        """Insert a plain-text annotation page (title + body paragraph)."""
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.axis("off")
        ax.text(0.0, 0.95, title, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.0, 0.72, body, transform=ax.transAxes,
                fontsize=9, va="top", color="#333333",
                wrap=True, linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#cccccc"))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    total = stats.get("total", meta.get("total_prompts", 0))

    def _hbar(ax, labels, values, color=ACCENT, pct_total=None):
        """Horizontal bar chart, sorted descending."""
        order = sorted(range(len(labels)), key=lambda i: values[i])
        labs  = [labels[i] for i in order]
        vals  = [values[i] for i in order]
        bars  = ax.barh(labs, vals, color=color)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar, v in zip(bars, vals):
            pct = f"  {v/pct_total*100:.1f}%" if pct_total else ""
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {v:,}{pct}", va="center", fontsize=8, color=GREY)
        ax.spines[["top", "right"]].set_visible(False)

    def _heatmap(ax, row_labels, col_labels, counts_2d, row_totals=None, fmt_abs=True):
        """
        Normalised heatmap (row-normalised).
        Hover replacement: shows absolute count in each cell annotation.
        """
        arr = np.array(counts_2d, dtype=float)
        row_sums = arr.sum(axis=1, keepdims=True)
        normed   = np.where(row_sums > 0, arr / row_sums, 0.0)

        im = ax.imshow(normed, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                abs_val = int(arr[i, j])
                pct_val = normed[i, j]
                if abs_val == 0:
                    continue
                text_color = "white" if pct_val > 0.55 else "black"
                cell_text  = f"{pct_val*100:.0f}%\n{abs_val:,}" if fmt_abs else f"{pct_val*100:.0f}%"
                ax.text(j, i, cell_text, ha="center", va="center",
                        fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, fraction=0.03, label="Row fraction")

    with PdfPages(out_pdf) as pdf:

        # ── Title page ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        lines = [
            ("SFT Pipeline — Data Distribution Report", 28, "bold"),
            ("", 12, "normal"),
            (f"Run directory:  {meta.get('run_dir', '—')}", 11, "normal"),
            (f"Exported at:    {meta.get('exported_at', '—')}", 11, "normal"),
            (f"Total prompts:  {total:,}", 11, "normal"),
            (f"Stages present: {', '.join(k for k, v in meta.get('stages', {}).items() if v)}", 11, "normal"),
            ("", 12, "normal"),
            (f"Domains:     {len(stats.get('domain_counts', {}))}", 10, "normal"),
            (f"Languages:   {len(stats.get('language_counts', {}))}", 10, "normal"),
            (f"Sources:     {len(stats.get('source_counts', {}))}", 10, "normal"),
            (f"Topics:      {len(stats.get('topics_counts', {}))}", 10, "normal"),
        ]
        y = 0.85
        for text, size, weight in lines:
            ax.text(0.5, y, text, transform=ax.transAxes,
                    ha="center", fontsize=size, fontweight=weight)
            y -= 0.06
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Domain distribution ───────────────────────────────────────────
        _annotation_page(pdf, "Domain Distribution",
            "Each prompt is annotated with a high-level domain (e.g. math, code, science) "
            "by the LLM annotator. This chart shows how many prompts fall into each domain "
            "across the full dataset. A healthy distribution avoids extreme skew — Stage 4 "
            "quota sampling will rebalance this, so large imbalances here are expected and fine. "
            "Domains with very few prompts may be merged into 'other' at sampling time.")
        domain_counts = stats.get("domain_counts", {})
        if domain_counts:
            n = len(domain_counts)
            fig, ax = plt.subplots(figsize=(11, max(4, n * 0.35 + 1.5)))
            labels = list(domain_counts.keys())
            values = list(domain_counts.values())
            _hbar(ax, labels, values, pct_total=total)
            ax.set_title(f"Domain Distribution  (N={total:,})", fontsize=13, pad=10)
            ax.set_xlabel("Prompts")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Difficulty distribution ───────────────────────────────────────
        _annotation_page(pdf, "Difficulty Distribution",
            "Difficulty (easy / medium / hard) is assigned by the LLM annotator based on "
            "the complexity of reasoning required. Ideally the dataset skews toward medium "
            "and hard to maximise learning signal. A large 'easy' fraction means many prompts "
            "require only shallow reasoning; Stage 4 quota sampling can correct this by "
            "down-weighting easy and up-weighting hard prompts.")
        diff_counts = stats.get("difficulty_counts", {})
        if diff_counts:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ordered = [d for d in DIFFICULTIES if d in diff_counts]
            ordered += [d for d in diff_counts if d not in DIFFICULTIES]
            vals   = [diff_counts[d] for d in ordered]
            colors = [DIFF_COLORS.get(d, ACCENT) for d in ordered]
            bars   = ax.bar(ordered, vals, color=colors)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:,}\n({v/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
            ax.set_title("Difficulty Distribution", fontsize=13, pad=10)
            ax.set_ylabel("Prompts")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Language distribution (top 30) ───────────────────────────────
        _annotation_page(pdf, "Language Distribution",
            "Language is detected by the LLM annotator from the prompt text. "
            "Most source datasets are English-dominant, so 'en' will usually be the "
            "overwhelming majority. Non-English prompts come primarily from multilingual "
            "sources (e.g. OPUS, CC-100 derivatives). This chart shows the top 30 "
            "languages by raw count. If Stage 7 translation is enabled in v2, these "
            "non-English prompts will be translated to English before fine-tuning.")
        lang_counts = stats.get("language_counts", {})
        if lang_counts:
            items = list(lang_counts.items())[:30]
            labels, values = zip(*items)
            fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.32 + 1.5)))
            _hbar(ax, list(labels), list(values), pct_total=total)
            ax.set_title("Language Distribution (top 30)", fontsize=13, pad=10)
            ax.set_xlabel("Prompts")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Source dataset distribution (top 30) ─────────────────────────
        _annotation_page(pdf, "Source Dataset Distribution",
            "Shows which HuggingFace datasets (or local sources) contributed how many "
            "prompts after Stage 1 collection and exact deduplication. Large sources "
            "dominate here; near-duplicate removal in Stage 4 will further reduce the "
            "effective contribution of sources with repetitive content. Use this chart to "
            "spot sources with unexpectedly low yield (possible download or parsing issues) "
            "or sources that are disproportionately large.")
        src_counts = stats.get("source_counts", {})
        if src_counts:
            items = list(src_counts.items())[:30]
            labels, values = zip(*items)
            fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.35 + 1.5)))
            _hbar(ax, list(labels), list(values), pct_total=total)
            ax.set_title("Source Dataset Distribution (top 30)", fontsize=13, pad=10)
            ax.set_xlabel("Prompts")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Domain × Difficulty heatmap ───────────────────────────────────
        _annotation_page(pdf, "Domain × Difficulty Heatmap",
            "Cross-tabulation of domain and difficulty, row-normalised (each row sums to 100%). "
            "Each cell shows the percentage AND absolute count for that (domain, difficulty) pair. "
            "Darker blue = higher fraction. Use this to spot domains where the LLM annotator "
            "systematically assigned one difficulty — e.g. if 'math' is 90% hard, that may "
            "reflect genuine data characteristics or annotator bias. These distributions directly "
            "inform the quota targets you should set in stage4_sample.yaml.")
        dd = stats.get("domain_difficulty_matrix")
        if dd and dd.get("domains") and dd.get("difficulties"):
            n_dom = len(dd["domains"])
            fig, ax = plt.subplots(figsize=(8, max(4, n_dom * 0.4 + 2)))
            _heatmap(ax, dd["domains"], dd["difficulties"], dd["counts"])
            ax.set_title("Domain × Difficulty  (row-normalised; cell = % | count)",
                         fontsize=12, pad=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Domain × Language heatmap ─────────────────────────────────────
        _annotation_page(pdf, "Domain × Language Heatmap",
            "Cross-tabulation of domain and language (top 10 languages), row-normalised. "
            "Shows whether non-English prompts are evenly spread across domains or "
            "concentrated in specific ones. For example, coding prompts are often English-only "
            "while math word problems may appear in many languages. Cells with high non-English "
            "fraction in a domain suggest that domain's quota may need adjustment if the "
            "fine-tuning target is English-only.")
        dl = stats.get("domain_language_matrix")
        if dl and dl.get("domains") and dl.get("languages"):
            n_dom  = len(dl["domains"])
            n_lang = len(dl["languages"])
            fig, ax = plt.subplots(figsize=(max(8, n_lang * 0.8 + 2), max(4, n_dom * 0.4 + 2)))
            _heatmap(ax, dl["domains"], dl["languages"], dl["counts"])
            ax.set_title("Domain × Language  (row-normalised; cell = % | count)",
                         fontsize=12, pad=10)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Top topics overall ────────────────────────────────────────────
        _annotation_page(pdf, "Topics",
            "Each prompt is tagged with 1–5 fine-grained topic strings by the LLM annotator "
            "(e.g. 'linear algebra', 'sorting algorithms', 'organic chemistry'). "
            "Topics are stored as a list and exploded for counting — a single prompt contributes "
            "one count per topic it is tagged with. The overall chart shows the 50 most frequent "
            "topics across all domains. The per-domain pages that follow show the top 15 topics "
            "within each domain, which is more actionable for identifying coverage gaps.")
        topic_counts = stats.get("topics_counts", {})
        if topic_counts:
            top_n   = min(50, len(topic_counts))
            items   = list(topic_counts.items())[:top_n]
            labels, values = zip(*items)
            fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.28 + 1.5)))
            _hbar(ax, list(labels), list(values), pct_total=total)
            ax.set_title(f"Top {top_n} Topics (overall)", fontsize=13, pad=10)
            ax.set_xlabel("Prompts")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Top topics per domain ─────────────────────────────────────────
        tbd = stats.get("topics_by_domain", {})
        if tbd:
            domain_order = list(stats.get("domain_counts", tbd).keys())
            per_page     = 4  # domains per PDF page

            sorted_domains = [d for d in domain_order if d in tbd] + \
                             [d for d in tbd if d not in domain_order]

            for page_start in range(0, len(sorted_domains), per_page):
                page_domains = sorted_domains[page_start:page_start + per_page]
                cols = 2
                rows = (len(page_domains) + 1) // 2
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(14, max(4, rows * 3.5)),
                                         squeeze=False)
                fig.suptitle("Top Topics per Domain", fontsize=13, y=1.01)

                for idx, domain in enumerate(page_domains):
                    ax   = axes[idx // cols][idx % cols]
                    data = tbd[domain]          # [[topic, count], ...]
                    if not data:
                        ax.axis("off")
                        continue
                    top_n  = min(15, len(data))
                    labs   = [d[0] for d in data[:top_n]]
                    vals   = [d[1] for d in data[:top_n]]
                    _hbar(ax, labs, vals)
                    dom_total = domain_counts.get(domain, 0)
                    ax.set_title(f"{domain}  ({dom_total:,} prompts)", fontsize=10)
                    ax.tick_params(axis="y", labelsize=8)

                # Hide unused axes
                for idx in range(len(page_domains), rows * cols):
                    axes[idx // cols][idx % cols].axis("off")

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # ── Cluster size histogram ────────────────────────────────────────
        _annotation_page(pdf, "Cluster Analysis",
            "Stage 3 grouped all prompts into 100,000 clusters using GPU-accelerated K-Means "
            "on 1024-dimensional sentence embeddings. Clusters represent semantic neighbourhoods "
            "— prompts within a cluster ask about similar concepts. "
            "The size histogram shows how evenly sized the clusters are: many singleton or "
            "tiny clusters indicate sparse/noisy semantic regions; very large clusters suggest "
            "a topic is over-represented. Stage 4 near-dedup uses cosine similarity within "
            "clusters to remove near-duplicates, so cluster quality directly affects sample diversity. "
            "The 'clusters per domain' chart shows how many clusters are dominated by each domain.")
        cs = stats.get("cluster_stats", {})
        hist = cs.get("size_histogram", {})
        if hist.get("bin_edges") and hist.get("counts"):
            edges  = hist["bin_edges"]
            counts = hist["counts"]
            labels = [f"{edges[i]}–{edges[i+1]-1}" for i in range(len(counts))]
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(labels, counts, color=ACCENT)
            for bar, v in zip(bars, counts):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:,}", ha="center", va="bottom", fontsize=8)
            ax.set_title(
                f"Cluster Size Distribution  ({cs.get('n_clusters', 0):,} clusters)",
                fontsize=13, pad=10,
            )
            ax.set_xlabel("Prompts per cluster")
            ax.set_ylabel("Number of clusters")
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Clusters per domain ───────────────────────────────────────────
        cpd = cs.get("clusters_per_domain", {})
        if cpd:
            n = len(cpd)
            fig, ax = plt.subplots(figsize=(11, max(4, n * 0.35 + 1.5)))
            labels = list(cpd.keys())
            values = list(cpd.values())
            _hbar(ax, labels, values, pct_total=sum(values))
            ax.set_title("Clusters Dominated per Domain", fontsize=13, pad=10)
            ax.set_xlabel("Clusters")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── PDF metadata ──────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"]   = "SFT Pipeline Data Distribution Report"
        d["Subject"] = f"Run: {meta.get('run_dir', '')}"

    logger.info("PDF report → %s  (%d pages)", out_pdf, pdf.get_pagecount() if hasattr(pdf, 'get_pagecount') else "?")


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
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-export even if meta.json already exists (re-scans all shards).",
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        const=str(Path(__file__).parent / "data" / "report.pdf"),
        metavar="PATH",
        help=(
            "Export all distribution charts to a PDF report. "
            "Optionally specify output path (default: viz/data/report.pdf). "
            "Reads stats from meta.json — run export first if meta.json is missing."
        ),
    )
    args = parser.parse_args()

    meta_path = Path(args.out).parent / "meta.json"

    # Skip full re-export when --pdf is requested and meta.json already exists.
    # Pass --refresh to force a re-scan even when meta.json is present.
    if args.pdf and meta_path.exists() and not args.refresh:
        logger.info("meta.json found — skipping snapshot re-export (pass --refresh to force).")
    else:
        export(
            run_dir=Path(args.run_dir),
            out_path=Path(args.out),
            sample=args.sample,
            seed=args.seed,
            compute_umap=args.umap,
        )

    if args.pdf:
        if not meta_path.exists():
            logger.error("meta.json not found at %s — run export first.", meta_path)
            sys.exit(1)
        meta = json.loads(meta_path.read_text())
        stats = meta.get("stats")
        if not stats:
            logger.error("No 'stats' key in meta.json — was Stage 3 output present?")
            sys.exit(1)
        _export_pdf(stats, meta, Path(args.pdf))


if __name__ == "__main__":
    main()
