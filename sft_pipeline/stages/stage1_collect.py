"""
Stage 1 — Prompt Collection from Existing Datasets.

Ingests prompts from HuggingFace datasets or local JSONL files,
normalizes them, deduplicates with MinHash LSH, and writes a JSONL
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
import re
import unicodedata
from pathlib import Path
from typing import Any

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus, prompt_id
from sft_pipeline.config import DatasetSource, PipelineConfig
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "stage1_collect"

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


def _infer_domain(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


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

    field = src.prompt_field
    row_errors = 0
    for row in ds:
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
                    src.hf_repo_id, field, exc,
                )
            elif row_errors == 6:
                logger.warning(
                    "Further row errors from %s will be suppressed.", src.hf_repo_id
                )
    if row_errors:
        logger.info("Skipped %d malformed rows from %s.", row_errors, src.hf_repo_id)


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


def _make_lsh(num_perm: int, threshold: float):
    from datasketch import MinHashLSH
    return MinHashLSH(threshold=threshold, num_perm=num_perm)


def _make_minhash(text: str, num_perm: int):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for token in text.lower().split():
        m.update(token.encode())
    return m


def run_stage1(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s1 = cfg.stage1_collect
    if not s1.datasets:
        logger.warning("stage1_collect: no datasets configured — skipping")
        return

    # Set HuggingFace cache directory before any HF calls if configured.
    if cfg.global_.hf_home:
        os.environ["HF_HOME"] = cfg.global_.hf_home
        logger.info("Stage1: HF_HOME set to %s", cfg.global_.hf_home)

    output_dir = Path(s1.output_path).parent
    ensure_dir(output_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    lsh = _make_lsh(s1.minhash_num_perm, s1.dedup_threshold)
    seen_in_lsh: set[str] = set()

    total_seen = 0
    total_written = 0

    with ShardedJSONLWriter(output_dir, shard_size_mb=200) as writer:
        for src in s1.datasets:
            source_name = src.hf_repo_id or Path(src.path).name
            domain_hint = src.domain_hint

            if src.source == "hf_dataset":
                raw_iter = _load_hf_dataset(src)
            else:
                raw_iter = _load_local_jsonl(src)

            if src.max_examples is not None:
                raw_iter = itertools.islice(raw_iter, src.max_examples)
                logger.info("Stage1: capping %s at %d examples", source_name, src.max_examples)

            source_written = 0
            try:
                for raw_text in raw_iter:
                    total_seen += 1
                    normalized = _normalize(raw_text)
                    if len(normalized) < 10:
                        continue

                    pid = prompt_id(normalized)

                    # Skip if already written in a prior run
                    if cm.is_processed(pid, STAGE):
                        continue

                    # Near-duplicate check via MinHash LSH
                    mh = _make_minhash(normalized, s1.minhash_num_perm)
                    if pid not in seen_in_lsh:
                        duplicates = lsh.query(mh)
                        if duplicates:
                            # Near-duplicate found — skip
                            cm.mark_processed(pid, STAGE, status=ItemStatus.SKIPPED)
                            continue
                        lsh.insert(pid, mh)
                        seen_in_lsh.add(pid)

                    domain = domain_hint or _infer_domain(normalized)

                    record = {
                        "prompt_id": pid,
                        "prompt": normalized,
                        "source": source_name,
                        "domain": domain,
                        "stage": "stage1",
                    }
                    writer.write(record)
                    cm.mark_processed(pid, STAGE, shard=writer.written_shards[-1] if writer.written_shards else None)
                    total_written += 1
                    source_written += 1

                    if total_written % 10_000 == 0:
                        logger.info(
                            "Stage1: written %d total (%d this source, %d seen)",
                            total_written, source_written, total_seen,
                        )
            except Exception as exc:
                logger.error(
                    "Source %s failed during iteration after %d rows — skipping remainder. Error: %s",
                    source_name, source_written, exc,
                )
            else:
                logger.info("Source %s complete: wrote %d prompts.", source_name, source_written)

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info(
        "Stage1 complete: seen=%d, written=%d, dedup_rate=%.1f%%",
        total_seen, total_written,
        100.0 * (1 - total_written / max(total_seen, 1)),
    )
