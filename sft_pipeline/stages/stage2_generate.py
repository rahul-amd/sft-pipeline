"""
Stage 2 — Prompt Generation from Knowledge-Rich Corpora.

Reads document corpora (local dir or HF dataset), splits into token-aware
chunks, calls a lightweight LLM to generate N prompts per chunk, and
appends them to the prompt pool.

Output schema per record (same as Stage 1):
  {
    "prompt_id": "sha256:...",
    "prompt":    "...",
    "source":    "arxiv/1234.5678",
    "domain":    "science",
    "stage":     "stage2",
    "chunk_id":  "arxiv/...:chunk_003"
  }
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator

import httpx
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_exponential

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus, prompt_id
from sft_pipeline.config import CorpusSource, PipelineConfig
from sft_pipeline.stages.stage1_collect import _infer_domain, _normalize
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl, iter_jsonl_dir

logger = logging.getLogger(__name__)

STAGE = "stage2_generate"

_PROMPT_GENERATION_SYSTEM = """\
You are an expert at creating educational and challenging questions from text passages.
Given a text passage, generate diverse, self-contained questions that:
- Can be fully answered from the passage content or general knowledge
- Test different skills: comprehension, explanation, derivation, multi-step reasoning
- Are clear and unambiguous
- Do NOT reference "the passage" or "the text" — questions must stand alone

Output ONLY a JSON array of question strings. No other text."""

_PROMPT_GENERATION_USER = """\
Text passage:
{passage}

Generate {n} diverse questions. Output as JSON array of strings."""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-aware chunks using tiktoken."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def _len_fn(t: str) -> int:
            return len(enc.encode(t))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=_len_fn,
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback: rough character-based splitting
        char_size = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + char_size, len(text))
            chunks.append(text[start:end])
            start += char_size - char_overlap
        return chunks


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_generator(
    passage: str,
    n: int,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str = "EMPTY",
) -> list[str]:
    """Call the generator LLM and return a list of prompt strings."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _PROMPT_GENERATION_SYSTEM},
            {"role": "user", "content": _PROMPT_GENERATION_USER.format(
                passage=passage, n=n
            )},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = httpx.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json=payload,
        timeout=60.0,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Generator response was truncated; increase generator_max_tokens")
    content = choice["message"].get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Try to parse JSON; repair if malformed
    try:
        prompts = json.loads(content)
    except json.JSONDecodeError:
        repaired = repair_json(content)
        try:
            prompts = json.loads(repaired)
        except Exception:
            # Last resort: extract quoted strings
            prompts = re.findall(r'"([^"]{20,})"', content)

    if not isinstance(prompts, list):
        raise ValueError("Generator did not return a JSON array of questions")
    valid = [p for p in prompts if isinstance(p, str) and len(p.strip()) >= 20]
    if not valid:
        raise ValueError("Generator returned no usable questions")
    return valid[:n]


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def _iter_local_corpus(src: CorpusSource) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, text) pairs from a local directory."""
    base = Path(src.path)
    if not base.exists():
        raise FileNotFoundError(f"Corpus not found: {base}")
    extensions = {".txt", ".jsonl", ".json", ".md"}
    paths = [base] if base.is_file() else sorted(base.rglob("*"))
    for fp in paths:
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in extensions:
            continue
        if fp.suffix.lower() in {".jsonl", ".json"}:
            if fp.suffix.lower() == ".json":
                data = json.loads(fp.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else [data]
            else:
                rows = iter_jsonl(fp, skip_errors=False)
            for rec in rows:
                text = rec.get(src.text_field, "")
                if isinstance(text, str) and len(text) > 100:
                    doc_id = f"{fp.resolve()}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
                    yield doc_id, text
        else:
            text = fp.read_text(errors="replace")
            if len(text) > 100:
                yield str(fp.resolve()), text


def _iter_hf_corpus(src: CorpusSource) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, text) pairs from a HuggingFace dataset."""
    from datasets import load_dataset

    logger.info("Loading HF corpus %s (split=%s)", src.hf_repo_id, src.hf_split)
    ds = load_dataset(src.hf_repo_id, split=src.hf_split, streaming=True)
    field = src.text_field
    for i, row in enumerate(ds):
        text = row.get(field, "")
        if text and len(text) > 100:
            yield f"{src.hf_repo_id}:{i}", text


# ---------------------------------------------------------------------------
# Main stage runner
# ---------------------------------------------------------------------------

def run_stage2(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s2 = cfg.stage2_generate
    if not s2.corpora:
        raise ValueError("stage2_generate is enabled but no corpora are configured")

    output_dir = Path(s2.output_dir)
    ensure_dir(output_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    total_chunks = 0
    seen_ids = {r["prompt_id"] for r in iter_jsonl_dir(output_dir)}
    total_written = len(seen_ids)
    failures = 0

    def _process_chunk(
        doc_id: str, chunk_idx: int, chunk_text: str, domain_hint: str | None
    ) -> tuple[str, list[dict], str | None]:
        digest = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
        chunk_id = f"{doc_id}:chunk_{chunk_idx:04d}:{digest}"
        chunk_hash = "sha256:" + hashlib.sha256(chunk_id.encode()).hexdigest()[:32]

        if cm.is_processed(chunk_hash, STAGE):
            return chunk_hash, [], None

        try:
            raw_prompts = _call_generator(
                chunk_text,
                n=s2.prompts_per_chunk,
                endpoint=s2.generator_endpoint,
                model=s2.generator_model,
                temperature=s2.generator_temperature,
                max_tokens=s2.generator_max_tokens,
                api_key=s2.generator_api_key,
            )
        except Exception as exc:
            logger.warning("Generator failed for chunk %s: %s", chunk_id, exc)
            return chunk_hash, [], str(exc)

        records = []
        for raw in raw_prompts:
            normalized = _normalize(raw)
            if len(normalized.strip()) < 20:
                continue
            pid = prompt_id(normalized)
            domain = domain_hint or _infer_domain(normalized)
            records.append({
                "prompt_id": pid,
                "prompt": normalized,
                "source": doc_id,
                "domain": domain,
                "stage": "stage2",
                "chunk_id": chunk_id,
            })

        return chunk_hash, records, None

    with ShardedJSONLWriter(output_dir, shard_size_mb=200) as writer:
        for src in s2.corpora:
            domain_hint = src.domain_hint
            if src.source == "local":
                doc_iter = _iter_local_corpus(src)
            else:
                doc_iter = _iter_hf_corpus(src)

            def chunk_args():
                for doc_id, doc_text in doc_iter:
                    chunks = _chunk_text(doc_text, s2.chunk_size_tokens, s2.chunk_overlap_tokens)
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:
                            yield doc_id, i, chunk, domain_hint

            args_iter = iter(chunk_args())
            with ThreadPoolExecutor(max_workers=s2.max_workers) as pool:
                pending = set()

                def submit_next():
                    args = next(args_iter, None)
                    if args is not None:
                        pending.add(pool.submit(_process_chunk, *args))

                for _ in range(s2.max_workers * 2):
                    submit_next()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_hash, records, error = future.result()
                        total_chunks += 1
                        if error is not None:
                            failures += 1
                            cm.mark_processed(chunk_hash, STAGE, ItemStatus.FAILED, error_msg=error)
                        elif records:
                            for rec in records:
                                if rec["prompt_id"] not in seen_ids:
                                    writer.write(rec)
                                    seen_ids.add(rec["prompt_id"])
                                    total_written += 1
                            writer.flush()
                            cm.mark_processed(chunk_hash, STAGE, ItemStatus.SUCCESS)
                        submit_next()

    if failures or not total_written:
        raise RuntimeError(f"Stage 2 incomplete: {failures} failed chunks, {total_written} prompts")

    cm.mark_stage_complete(STAGE, output_count=total_written)
    logger.info("Stage2 complete: chunks=%d, prompts_written=%d", total_chunks, total_written)
