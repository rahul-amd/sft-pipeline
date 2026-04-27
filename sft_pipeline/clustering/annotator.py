"""
LLM annotation of prompts via an OpenAI-compatible API server.

Sends each prompt to the configured model and parses a JSON annotation:
  {
    "domain":     "math",
    "topics":     ["linear algebra", "eigenvalues"],
    "difficulty": "medium",
    "language":   "en",
    "summary":    "Find the eigenvalues of a 3×3 matrix."
  }

Designed for Qwen3-30B-A3B-Thinking-2507 (and similar thinking models): the model
wraps its reasoning in <think>...</think> before the JSON; this block is stripped
before parsing.

Key properties:
  - Async with configurable concurrency (semaphore-bounded asyncio tasks)
  - Checkpoint to Parquet every N records — safe to restart mid-run
  - Falls back to validated defaults on any API/parse error (never crashes the pipeline)
  - Progress logged every 1 000 completions
  - Prompt truncated to last 512 whitespace tokens to stay within context window
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THINK_STRIP_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_VALID_DOMAINS = {
    "math", "code", "science", "reasoning",
    "writing", "language", "knowledge", "instruction", "other",
}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}

# Number of whitespace-split tokens to keep from the END of the prompt.
# Keeping the tail preserves the actual question in prompts that start with
# lengthy context / background (common in coding and science prompts).
_PROMPT_MAX_TOKENS = 512

_SYSTEM_PROMPT = """\
You are a prompt classifier. Classify the given user prompt and return a JSON object with these fields:

"domain": classify into exactly one of:
  "math"        — calculations, proofs, algebra, geometry, statistics, discrete math
  "code"        — programming, algorithms, debugging, software design, data structures
  "science"     — physics, chemistry, biology, medicine, earth science, astronomy
  "reasoning"   — logic, deduction, argumentation, puzzles, critical thinking
  "writing"     — essays, creative writing, summarization, editing, storytelling
  "language"    — translation, grammar, linguistics, vocabulary, rhetoric
  "knowledge"   — factual Q&A, history, geography, culture, definitions
  "instruction" — how-to guides, tutorials, step-by-step procedures, recipes
  "other"       — anything that does not fit the categories above

"topics": list of 1–3 specific topic strings (e.g. ["matrix multiplication", "eigenvalues"])

"difficulty":
  "easy"   — single-step, factual recall, straightforward
  "medium" — multi-step or requires moderate background knowledge
  "hard"   — complex, requires deep expertise or creative reasoning

"language": ISO 639-1 code of the prompt language (e.g. "en", "zh", "fr", "de", "es")

"summary": one sentence, at most 15 words, describing what the prompt asks for

Output ONLY the JSON object. No explanation, no markdown, no extra text."""

# Parquet schema for the checkpoint file
_CHECKPOINT_SCHEMA = pa.schema([
    pa.field("prompt_id", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("topics", pa.string()),       # JSON-encoded list
    pa.field("difficulty", pa.string()),
    pa.field("language", pa.string()),
    pa.field("summary", pa.string()),
])


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_annotation(text: str) -> dict:
    """
    Strip thinking block and extract JSON from model output.

    Qwen3 thinking models always emit <think>...</think> before the answer.
    After stripping, we try a direct JSON parse then fall back to regex
    extraction of the outermost {...} block.
    """
    text = _THINK_STRIP_RE.sub("", text).strip()
    # Strip markdown code fences if present (```json ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _truncate_prompt(prompt: str, max_tokens: int = _PROMPT_MAX_TOKENS) -> str:
    """
    Return the last `max_tokens` whitespace-split tokens of the prompt.

    Keeping the tail (rather than the head) preserves the actual question in
    prompts that open with lengthy context, code blocks, or background text.
    No tokenizer dependency — whitespace splitting is a fast, reasonable proxy.
    """
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt
    return " ".join(words[-max_tokens:])


def _validate_annotation(ann: dict) -> dict:
    """Coerce annotation to valid values; fill missing keys with defaults."""
    domain = ann.get("domain", "other")
    difficulty = ann.get("difficulty", "medium")
    topics = ann.get("topics", [])
    language = ann.get("language", "en")
    summary = ann.get("summary", "")
    return {
        "domain": domain if domain in _VALID_DOMAINS else "other",
        "topics": topics if isinstance(topics, list) else [],
        "difficulty": difficulty if difficulty in _VALID_DIFFICULTIES else "medium",
        "language": language if isinstance(language, str) and len(language) <= 10 else "en",
        "summary": summary if isinstance(summary, str) else "",
    }


# ---------------------------------------------------------------------------
# Async annotation core
# ---------------------------------------------------------------------------

async def _annotate_one(
    client,
    model: str,
    record: dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
    counter: list,        # mutable [done, failed] for progress tracking
    total: int,
) -> tuple[str, dict]:
    """Annotate a single prompt. Returns (prompt_id, annotation_dict)."""
    pid = record["prompt_id"]
    prompt = record["prompt"]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _truncate_prompt(prompt)},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            ann = _validate_annotation(_parse_annotation(text))
        except Exception as exc:
            logger.debug("Annotation failed for %s: %s", pid, exc)
            ann = _validate_annotation({})
            counter[1] += 1   # failed

        counter[0] += 1   # done
        done = counter[0]
        if done % 1000 == 0:
            logger.info(
                "Annotation progress: %d / %d  (%.1f%%)  failures so far: %d",
                done, total, 100 * done / total, counter[1],
            )
        return pid, ann


async def _run_async(
    records: list[dict],
    model: str,
    api_base: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, dict]:
    """Drive all annotation tasks in a single event loop."""
    import httpx
    from openai import AsyncOpenAI

    # httpx defaults to max_connections=100 (some versions) or 1000.  Either
    # way, high-concurrency runs (e.g. 4096 against 64 vLLM workers) need the
    # pool sized to match — otherwise requests queue inside httpx and workers
    # are underutilised.  We also raise the timeout: large thinking models can
    # take 30–60 s per request; the httpx default (5 s) is too aggressive.
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrency + 128,
            max_keepalive_connections=concurrency,
        ),
        timeout=httpx.Timeout(timeout=120.0),
    )

    # Use async-with so the client (and its underlying httpx connection pool)
    # is closed before asyncio.run() tears down the event loop.  Without this,
    # httpx schedules aclose() tasks that run after the loop is closed, causing
    # "RuntimeError: Event loop is closed" noise in the logs.
    async with AsyncOpenAI(base_url=api_base, api_key=api_key, http_client=http_client) as client:
        semaphore = asyncio.Semaphore(concurrency)
        counter = [0, 0]   # [done, failed]
        total = len(records)

        tasks = [
            asyncio.create_task(
                _annotate_one(client, model, rec, semaphore, max_tokens, temperature, counter, total)
            )
            for rec in records
        ]

        results: dict[str, dict] = {}
        for coro in asyncio.as_completed(tasks):
            pid, ann = await coro
            results[pid] = ann

    return results


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Load existing annotation checkpoint. Returns {} if file absent."""
    if not path.exists():
        return {}
    table = pq.read_table(path, schema=_CHECKPOINT_SCHEMA)
    annotation_map: dict[str, dict] = {}
    for row in table.to_pylist():
        annotation_map[row["prompt_id"]] = {
            "domain": row["domain"],
            "topics": json.loads(row["topics"]) if row["topics"] else [],
            "difficulty": row["difficulty"],
            "language": row["language"],
            "summary": row.get("summary", ""),
        }
    return annotation_map


def _save_checkpoint(annotation_map: dict[str, dict], path: Path) -> None:
    """Persist annotation map to Parquet (overwrites previous checkpoint)."""
    rows = [
        {
            "prompt_id": pid,
            "domain": ann.get("domain", "other"),
            "topics": json.dumps(ann.get("topics", [])),
            "difficulty": ann.get("difficulty", "medium"),
            "language": ann.get("language", "en"),
            "summary": ann.get("summary", ""),
        }
        for pid, ann in annotation_map.items()
    ]
    table = pa.Table.from_pylist(rows, schema=_CHECKPOINT_SCHEMA)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Public helpers for offline annotation workflow
# ---------------------------------------------------------------------------

def build_annotation_request(record: dict) -> dict:
    """Build an OpenAI-compatible request dict for a single prompt record.

    Returns a dict suitable for writing to a JSONL file that can be sent to any
    OpenAI-compatible inference server::

        {"prompt_id": "sha256:...", "messages": [{"role": "system", ...}, {"role": "user", ...}]}
    """
    return {
        "prompt_id": record["prompt_id"],
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _truncate_prompt(record["prompt"])},
        ],
    }


def parse_and_validate_annotation(response_text: str) -> dict:
    """Parse a raw LLM response text into a validated annotation dict.

    Handles thinking-model output (<think>...</think> blocks), markdown fences,
    and partial JSON — identical to the parsing done during online annotation.
    Returns validated defaults for any missing or invalid fields.
    """
    return _validate_annotation(_parse_annotation(response_text))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def annotate_prompts(
    prompt_records: list[dict],         # [{"prompt_id": ..., "prompt": ...}, ...]
    model: str,
    api_base: str,
    api_key: str = "EMPTY",
    concurrency: int = 64,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50_000,
) -> dict[str, dict]:
    """
    Annotate all prompts using an OpenAI-compatible API server.

    Args:
        prompt_records:   list of {"prompt_id": str, "prompt": str} dicts.
        model:            model name as recognised by the API server.
        api_base:         base URL of the OpenAI-compatible server (e.g. http://host:8001/v1).
        api_key:          API key; use "EMPTY" for vLLM servers that don't require one.
        concurrency:      number of parallel requests in flight at any time.
        max_tokens:       max tokens per response (thinking + JSON).
        temperature:      sampling temperature (lower = more deterministic).
        checkpoint_path:  path to Parquet checkpoint file; set to resume mid-run.
        checkpoint_every: save a checkpoint after every this many new annotations.

    Returns:
        annotation_map: {prompt_id: {"domain", "topics", "difficulty", "language", "summary"}}
    """
    # Resume from checkpoint
    annotation_map = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    if annotation_map:
        logger.info("Annotation: resumed %d records from checkpoint %s", len(annotation_map), checkpoint_path)

    remaining = [r for r in prompt_records if r["prompt_id"] not in annotation_map]
    if not remaining:
        logger.info("Annotation: all %d records already annotated (checkpoint)", len(annotation_map))
        return annotation_map

    logger.info(
        "Annotation: %d to annotate, %d already done | model=%s  api=%s  concurrency=%d",
        len(remaining), len(annotation_map), model, api_base, concurrency,
    )

    t0 = time.monotonic()
    newly_done = 0
    already_done = len(prompt_records) - len(remaining)  # records loaded from checkpoint

    # Process in chunks so we checkpoint periodically without losing progress
    for chunk_start in range(0, len(remaining), checkpoint_every):
        chunk = remaining[chunk_start: chunk_start + checkpoint_every]
        chunk_end = chunk_start + len(chunk)
        logger.info(
            "Annotation: chunk %d–%d / %d ...",
            already_done + chunk_start,
            already_done + chunk_end,
            len(prompt_records),
        )

        chunk_results = asyncio.run(_run_async(
            records=chunk,
            model=model,
            api_base=api_base,
            api_key=api_key,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
        ))
        annotation_map.update(chunk_results)
        newly_done += len(chunk_results)

        if checkpoint_path:
            _save_checkpoint(annotation_map, checkpoint_path)
            elapsed = time.monotonic() - t0
            rate = newly_done / elapsed if elapsed > 0 else 0
            remaining_n = len(remaining) - chunk_end
            eta_s = remaining_n / rate if rate > 0 else 0
            logger.info(
                "Annotation: checkpoint saved — %d total | %.0f rec/s | ETA %.0f min",
                len(annotation_map), rate, eta_s / 60,
            )

    elapsed = time.monotonic() - t0
    logger.info(
        "Annotation: complete. %d records annotated in %.1f min (%.0f rec/s).",
        len(annotation_map), elapsed / 60, newly_done / elapsed if elapsed > 0 else 0,
    )
    return annotation_map
