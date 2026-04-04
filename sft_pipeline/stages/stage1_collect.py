"""
Stage 1 — Prompt Collection from Existing Datasets.

Ingests prompts from HuggingFace datasets or local JSONL files,
normalizes them, deduplicates by SHA256 prompt_id, and writes a JSONL
pool of candidate prompts.

Prompt field extraction
-----------------------
The configured prompt_field may contain:
  - A plain string  → used directly.
  - A JSON-encoded string containing a conversation → parsed first.
  - A Python list (HF datasets can auto-decode JSON arrays) → used directly.

Supported conversation formats:
  OpenAI  — list of {"role": "system"|"user"|"assistant", "content": "..."}
  ShareGPT — list of {"from": "system"|"human"|"gpt",    "value":   "..."}

Extraction rule:
  1. Take the first user utterance (role=="user" / from=="human").
  2. If a system message precedes it, prepend: "{system}\n\n{user}".
  3. If no user turn is found, the record is skipped.

Output schema per record:
  {
    "prompt_id": "sha256:...",
    "prompt":    "...",
    "source":    "gsm8k",
    "domain":    "math",       # from domain_hint or heuristic
    "stage":     "stage1"
  }
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import DatasetSource, PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "stage1_collect"

# Tuning constants
_QUEUE_MAXSIZE = 20_000          # max items buffered between producers and consumer
_MAX_WORKER_THREADS = 8          # cap concurrent source-loader threads

# Sentinel object used to signal a worker thread has finished
_SENTINEL = object()

# Heuristic domain keywords for labelling when no domain_hint is given
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "math": [
        "calculate", "compute", "solve", "equation", "integral", "derivative",
        "probability", "algebra", "geometry", "arithmetic", "theorem", "proof",
        "matrix", "vector", "polynomial",
    ],
    "code": [
        "write a function", "implement", "code", "python", "javascript", "algorithm",
        "data structure", "debug", "program", "class", "method", "api", "sql",
    ],
    "science": [
        "physics", "chemistry", "biology", "molecule", "reaction", "force",
        "energy", "quantum", "cell", "organism", "evolution", "atom",
    ],
    "language": [
        "translate", "grammar", "essay", "summarize", "paraphrase", "syntax",
        "vocabulary", "word", "sentence", "paragraph", "write", "explain",
    ],
}

# Pre-compiled regex patterns — compiled once at import time, reused for every prompt
_DOMAIN_PATTERNS: dict[str, re.Pattern] = {
    domain: re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)
    for domain, keywords in _DOMAIN_KEYWORDS.items()
}


def _infer_domain(text: str) -> str:
    best_domain = "general"
    best_count = 0
    for domain, pattern in _DOMAIN_PATTERNS.items():
        count = len(pattern.findall(text))
        if count > best_count:
            best_count = count
            best_domain = domain
    return best_domain


def _normalize(text: str) -> str:
    """Unicode normalize, collapse whitespace, strip."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Conversation field extraction
# ---------------------------------------------------------------------------

def _extract_from_openai_messages(messages: list[dict]) -> str | None:
    """
    Extract prompt from OpenAI chat format.
    Messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    """
    system_text: str | None = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            # content can be a list of parts (vision API); join text parts
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and "text" in p
            )
        content = content.strip()
        if role == "system" and content:
            system_text = content
        elif role == "user" and content:
            return f"{system_text}\n\n{content}" if system_text else content
    return None


def _extract_from_sharegpt_messages(messages: list[dict]) -> str | None:
    """
    Extract prompt from ShareGPT format.
    Messages: [{"from": "system"|"human"|"gpt", "value": "..."}, ...]
    """
    system_text: str | None = None
    for msg in messages:
        from_ = msg.get("from", "")
        value = (msg.get("value") or "").strip()
        if from_ == "system" and value:
            system_text = value
        elif from_ in ("human", "user") and value:
            return f"{system_text}\n\n{value}" if system_text else value
    return None


def _extract_prompt(val: Any) -> str | None:
    """
    Extract a prompt string from a dataset field value.

    Handles:
      - Plain string               → returned as-is (after strip).
      - JSON-encoded string        → parsed, then dispatched by format.
      - Python list (already parsed) → dispatched by format.

    Conversation formats detected:
      OpenAI   — messages have a "role" key.
      ShareGPT — messages have a "from" key.

    Returns None if no usable prompt can be extracted (skip this record).
    """
    # If it's a string, try to detect whether it's a JSON conversation.
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                val = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped if stripped else None
        else:
            return stripped if stripped else None

    # At this point val is a Python object (list or dict).
    if isinstance(val, list) and val:
        first = val[0]
        if not isinstance(first, dict):
            return None
        if "role" in first:
            return _extract_from_openai_messages(val)
        if "from" in first:
            return _extract_from_sharegpt_messages(val)
        return None

    if isinstance(val, dict):
        # Single-message dict — treat as one turn based on available keys.
        content = val.get("content") or val.get("value") or val.get("text") or ""
        return content.strip() if content.strip() else None

    return None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _get_field(row: dict, field: str) -> Any:
    """
    Extract a (potentially nested) field from a row using dot notation.

    Examples:
      "question"                     → row["question"]
      "responses_create_params.input" → row["responses_create_params"]["input"]
    """
    val: Any = row
    for key in field.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(key)
        if val is None:
            return None
    return val


def _load_hf_dataset(src: DatasetSource):
    """Yield raw prompt strings from a HuggingFace dataset."""
    from datasets import load_dataset

    kwargs: dict = {"split": src.hf_split}
    if src.hf_config:
        kwargs["name"] = src.hf_config

    logger.info("Loading HF dataset %s (split=%s, config=%s)", src.hf_repo_id, src.hf_split, src.hf_config)
    try:
        ds = load_dataset(src.hf_repo_id, **kwargs, streaming=True, trust_remote_code=False)
    except Exception as exc:
        logger.error(
            "Failed to load HF dataset %s — skipping source. Error: %s",
            src.hf_repo_id, exc,
        )
        return

    # kwargs passed through so _iter_hf_rows can reload with features=None on schema errors
    yield from _iter_hf_rows(ds, src.hf_repo_id, src.prompt_field, kwargs)


def _iter_hf_rows(ds, repo_id: str, field: str, load_kwargs: dict):
    """
    Iterate over a streaming HF dataset, yielding prompt strings.

    If Arrow schema-casting fails mid-iteration (TypeError / ValueError from
    the datasets Arrow layer), we reload the dataset with ``features=None``
    to bypass schema enforcement and retry from the beginning.  This handles
    datasets whose registered feature schema disagrees with the actual data
    (e.g. ``string`` vs ``null`` type mismatches).
    """
    from datasets import load_dataset

    def _yield_rows(dataset):
        row_errors = 0
        for row in dataset:
            try:
                val = _get_field(row, field)
                if val is None:
                    continue
                text = _extract_prompt(val)
                if text:
                    yield text
            except Exception as exc:
                row_errors += 1
                if row_errors <= 5:
                    logger.warning(
                        "Skipping malformed row from %s (field=%s): %s",
                        repo_id, field, exc,
                    )
                elif row_errors == 6:
                    logger.warning("Further row errors from %s will be suppressed.", repo_id)
        if row_errors:
            logger.info("Skipped %d malformed rows from %s.", row_errors, repo_id)

    try:
        yield from _yield_rows(ds)
    except (TypeError, ValueError) as exc:
        err = str(exc)
        if "cast" in err.lower() or "couldn't cast" in err.lower() or "null" in err.lower():
            logger.warning(
                "Schema cast error from %s — retrying with features=None: %s", repo_id, exc
            )
            try:
                ds2 = load_dataset(
                    repo_id, **{k: v for k, v in load_kwargs.items() if k != "features"},
                    streaming=True, trust_remote_code=False, features=None,
                )
                yield from _yield_rows(ds2)
            except Exception as exc2:
                logger.warning("Retry with features=None also failed for %s: %s", repo_id, exc2)
        else:
            raise


def _load_local_jsonl(src: DatasetSource):
    """Yield raw prompt strings from a local JSONL file."""
    path = Path(src.path)
    logger.info("Loading local JSONL %s", path)
    field = src.prompt_field
    for record in iter_jsonl(path):
        val = _get_field(record, field)
        if val is None:
            continue
        text = _extract_prompt(val)
        if text:
            yield text



def _source_worker(
    src: DatasetSource,
    seen_ids: set[str],
    out_q: queue.Queue,
) -> None:
    """
    Producer thread: load one dataset source, normalize rows, and enqueue
    (pid, text, source_name, domain_hint) tuples.

    Pre-filters against seen_ids (the set of prompt_ids already written to
    output shards) so the consumer doesn't have to process duplicates from
    a previous run.  seen_ids is read-only here — the consumer is the sole
    writer.

    Puts _SENTINEL on the queue when done (even on error) so the consumer
    always knows this worker has finished.
    """
    source_name = src.hf_repo_id or Path(src.path).name
    seen = 0
    enqueued = 0
    try:
        if src.source == "hf_dataset":
            raw_iter = _load_hf_dataset(src)
        else:
            raw_iter = _load_local_jsonl(src)

        if src.max_examples is not None:
            raw_iter = itertools.islice(raw_iter, src.max_examples)
            logger.info("Stage1: capping %s at %d examples", source_name, src.max_examples)

        for raw_text in raw_iter:
            seen += 1
            normalized = _normalize(raw_text)
            if len(normalized) < 10:
                continue
            pid = prompt_id(normalized)
            # Skip prompts already written in a previous run
            if pid in seen_ids:
                continue
            out_q.put((pid, normalized, source_name, src.domain_hint))
            enqueued += 1
    except Exception as exc:
        logger.error("Source %s failed after %d rows: %s", source_name, seen, exc)
    finally:
        logger.info(
            "Source %s worker done: %d rows seen, %d items enqueued.",
            source_name, seen, enqueued,
        )
        out_q.put(_SENTINEL)


def run_stage1(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    if cfg.stage1_collect.distributed:
        run_stage1_distributed(cfg, cm)
        return

    s1 = cfg.stage1_collect
    if not s1.datasets:
        logger.warning("stage1_collect: no datasets configured — skipping")
        return

    if cfg.global_.hf_home:
        os.environ["HF_HOME"] = cfg.global_.hf_home
        logger.info("Stage1: HF_HOME set to %s", cfg.global_.hf_home)

    output_dir = Path(s1.output_path).parent
    ensure_dir(output_dir)

    cm.mark_stage_started(STAGE)

    # ── Resume: load prompt_ids already written in previous runs ─────────────
    # Scanning existing output shards (large sequential reads) is far faster
    # than per-record DuckDB WAL writes on Lustre.  No mark_processed_batch
    # calls are made during the hot loop — only mark_stage_complete at the end.
    seen_ids: set[str] = set()
    existing_shards = sorted(output_dir.glob("part-*.jsonl"))
    if existing_shards:
        logger.info(
            "Stage1: loading %d existing output shard(s) for resume …",
            len(existing_shards),
        )
        for shard_path in existing_shards:
            for rec in iter_jsonl(shard_path):
                seen_ids.add(rec["prompt_id"])
        logger.info("Stage1: %d prompt_ids pre-loaded from existing shards.", len(seen_ids))

    n_sources = len(s1.datasets)
    n_workers = min(n_sources, _MAX_WORKER_THREADS)
    out_q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)

    total_processed = 0   # items received from workers
    total_written = len(seen_ids)  # count existing as already written

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for src in s1.datasets:
            executor.submit(_source_worker, src, seen_ids, out_q)

        active = n_sources
        with ShardedJSONLWriter(output_dir, shard_size_mb=200) as writer:
            while active > 0:
                item = out_q.get()

                if item is _SENTINEL:
                    active -= 1
                    continue

                pid, normalized, source_name, domain_hint = item
                total_processed += 1

                if pid in seen_ids:
                    continue

                seen_ids.add(pid)
                domain = domain_hint or _infer_domain(normalized)
                writer.write({
                    "prompt_id": pid,
                    "prompt": normalized,
                    "source": source_name,
                    "domain": domain,
                    "stage": "stage1",
                })
                total_written += 1

                if total_processed % 10_000 == 0:
                    dedup_rate = 100.0 * (1 - total_written / total_processed) if total_processed else 0.0
                    logger.info(
                        "Stage1: processed %d  |  written %d  |  dedup rate %.1f%%",
                        total_processed, total_written, dedup_rate,
                    )

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage1 complete: written=%d prompts", total_written)


# ---------------------------------------------------------------------------
# Distributed Stage 1 — Ray-based multi-node collection
# ---------------------------------------------------------------------------

def _source_slug(src: DatasetSource) -> str:
    """Return a short filesystem-safe identifier for a dataset source."""
    name = (src.hf_repo_id or src.path or "unknown").replace("/", "_").replace(":", "_")
    return f"{name}__{src.hf_split}"[:120]


def _collect_source(
    src_dict: dict,
    hf_home: str | None,
    out_path: str,
) -> dict:
    """
    Collect one dataset source and write SHA256-deduplicated records to out_path.

    This is the body of the Ray remote task — it runs on a worker node.
    All imports are local so Ray can serialise and ship this function cleanly.

    Returns {"source": str, "written": int, "output": str}
    """
    import itertools
    import logging
    import os
    import orjson
    from pathlib import Path

    from sft_pipeline.checkpoint import prompt_id
    from sft_pipeline.config import DatasetSource as _DS
    from sft_pipeline.stages.stage1_collect import (
        _infer_domain,
        _load_hf_dataset,
        _load_local_jsonl,
        _normalize,
    )

    _log = logging.getLogger(__name__)
    src = _DS(**src_dict)
    source_name = src.hf_repo_id or Path(src.path).name

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    if src.source == "hf_dataset":
        raw_iter = _load_hf_dataset(src)
    else:
        raw_iter = _load_local_jsonl(src)

    if src.max_examples is not None:
        raw_iter = itertools.islice(raw_iter, src.max_examples)

    seen_ids: set[str] = set()
    written = 0
    tmp_path = out_path + ".tmp"

    try:
        with open(tmp_path, "wb") as f:
            for raw_text in raw_iter:
                try:
                    normalized = _normalize(raw_text)
                    if len(normalized) < 10:
                        continue
                    pid = prompt_id(normalized)
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    record = {
                        "prompt_id": pid,
                        "prompt": normalized,
                        "source": source_name,
                        "domain": src.domain_hint or _infer_domain(normalized),
                        "stage": "stage1",
                    }
                    f.write(orjson.dumps(record) + b"\n")
                    written += 1
                except Exception as exc:
                    _log.warning("Skipping row from %s: %s", source_name, exc)
        # Atomic rename — only exists as a complete file after success
        Path(tmp_path).rename(out_path)
    except Exception as exc:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise RuntimeError(f"Source {source_name} failed: {exc}") from exc

    _log.info("Source %s: wrote %d prompts → %s", source_name, written, Path(out_path).name)
    return {"source": source_name, "written": written, "output": out_path}


def _merge_and_dedup(
    phase1_dir: Path,
    output_dir: Path,
    s1,             # Stage1Config  (kept for signature compatibility)
    cm: CheckpointManager,
) -> int:
    """
    Read all phase1 JSONL files, cross-source exact-dedup by prompt_id
    (SHA256), and write the final sharded output.

    Near-duplicate dedup (Jaccard / cosine) is deferred to Stage 4, which
    has embeddings and FAISS and can do it far more accurately. A set lookup
    is O(1) and doesn't degrade with corpus size.

    **No per-record DuckDB writes in the hot loop.** Flushing the DuckDB
    checkpoint every N records on a Lustre shared filesystem causes WAL syncs
    that take ~10 s each, capping throughput at ~100 rec/s. Instead:
      - Resume is handled by scanning existing output shards at startup and
        loading their prompt_ids into seen_ids (large sequential reads — fast
        on Lustre).
      - DuckDB is updated once at the end via mark_stage_complete only.

    Returns total prompts written.
    """
    from sft_pipeline.storage import iter_jsonl, iter_jsonl_dir

    # ── Resume: load prompt_ids already written in previous runs ─────────────
    seen_ids: set[str] = set()
    existing_shards = sorted(output_dir.glob("part-*.jsonl"))
    if existing_shards:
        logger.info(
            "Merge: loading %d existing output shard(s) for resume …",
            len(existing_shards),
        )
        for shard_path in existing_shards:
            for rec in iter_jsonl(shard_path):
                seen_ids.add(rec["prompt_id"])
        logger.info("Merge: %d prompt_ids pre-loaded from existing shards.", len(seen_ids))

    total_written = len(seen_ids)  # count existing as already written
    total_seen = 0

    phase1_shards = sorted(phase1_dir.glob("*.jsonl"))
    logger.info("Merge: reading %d phase1 shard(s) from %s …", len(phase1_shards), phase1_dir)

    with ShardedJSONLWriter(output_dir, shard_size_mb=200) as writer:
        for rec in iter_jsonl_dir(phase1_dir):
            total_seen += 1
            pid = rec["prompt_id"]

            if pid in seen_ids:
                continue

            seen_ids.add(pid)
            writer.write(rec)
            total_written += 1

            if total_written % 50_000 == 0:
                logger.info(
                    "Merge: written %d / %d seen  (dedup rate %.1f%%)",
                    total_written, total_seen,
                    100.0 * (1 - total_written / max(total_seen, 1)),
                )

    logger.info(
        "Merge complete: written=%d / seen=%d  (dedup rate %.1f%%)",
        total_written, total_seen,
        100.0 * (1 - total_written / max(total_seen, 1)),
    )
    return total_written


def run_stage1_distributed(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    """
    Multi-node Stage 1 using Ray.

    Phase 1 (parallel, all nodes):
        One Ray task per dataset source. Each task streams its source,
        normalises, SHA256-deduplicates within the source, and writes to
        {output_dir}/_phase1/{source_slug}.jsonl on the shared filesystem.
        Tasks are idempotent — if the output file already exists, the source
        is skipped, making the whole phase crash-resumable.

    Phase 2 (single node, head):
        Reads all phase1 files, deduplicates by prompt_id (SHA256) across
        sources, writes the final sharded output, and updates the DuckDB
        checkpoint.

    Config:
        stage1_collect.distributed: true
        global.ray_address: "auto"   # or "ray://<head-ip>:10001"
    """
    import ray

    s1 = cfg.stage1_collect
    if not s1.datasets:
        logger.warning("stage1_collect (distributed): no datasets configured — skipping")
        return

    if cfg.global_.hf_home:
        os.environ["HF_HOME"] = cfg.global_.hf_home

    output_dir = Path(s1.output_path).parent
    phase1_dir = output_dir / "_phase1"
    ensure_dir(output_dir)
    ensure_dir(phase1_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    # ── Phase 1: parallel source collection ──────────────────────────────────
    ray.init(address=cfg.global_.ray_address, ignore_reinit_error=True)

    # Wrap the plain function as a Ray remote task (num_cpus=2 to avoid
    # over-subscribing nodes — each task is I/O-bound, not CPU-bound).
    collect_remote = ray.remote(num_cpus=2)(_collect_source)

    futures = {}
    skipped = 0
    for src in s1.datasets:
        slug = _source_slug(src)
        out_path = phase1_dir / f"{slug}.jsonl"
        if out_path.exists():
            logger.info("Phase1: %s already complete, skipping.", slug)
            skipped += 1
            continue
        future = collect_remote.remote(
            src.model_dump(),
            cfg.global_.hf_home,
            str(out_path),
        )
        futures[future] = src.hf_repo_id or src.path

    total = len(futures) + skipped
    logger.info(
        "Phase1: %d sources total — %d submitted to Ray, %d already done.",
        total, len(futures), skipped,
    )

    done = skipped
    failed = 0
    # Process results as they complete rather than waiting for all at once
    remaining = list(futures.keys())
    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        future = ready[0]
        source_name = futures[future]
        try:
            result = ray.get(future)
            done += 1
            logger.info(
                "Phase1 [%d/%d] ✓  %s — %d prompts",
                done, total, result["source"], result["written"],
            )
        except Exception as exc:
            failed += 1
            done += 1
            logger.error("Phase1 [%d/%d] ✗  %s — %s", done, total, source_name, exc)

    if failed:
        logger.warning(
            "Phase1: %d/%d sources failed. Proceeding with successful outputs.",
            failed, len(futures),
        )

    # ── Phase 2: merge + cross-source dedup on head node ─────────────────────
    logger.info("Phase2: merging phase1 outputs …")
    total_written = _merge_and_dedup(phase1_dir, output_dir, s1, cm)

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage1 distributed complete: %d prompts written.", total_written)
