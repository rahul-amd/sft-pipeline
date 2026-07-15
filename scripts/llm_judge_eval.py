#!/usr/bin/env python3
"""
LLM-as-a-judge evaluator for response quality.

Reads a JSONL file where each record has a 'messages' array:
  messages[0] — user prompt
  messages[1] — assistant response (may contain <think>...</think> block)

Calls a configurable judge model to score each response on:
  - thinking_coherence (1–5): reasoning steps are logically connected
  - thinking_clarity   (1–5): thinking is clearly written and easy to follow
  - answer_correctness (1–5): final answer is correct and complete

Writes a JSONL file with all original fields plus judge scores.
Prints summary statistics at the end.

Provider support
----------------
The --provider flag controls which API client is used.  It defaults to "auto",
which picks the provider from the model name (claude-* → anthropic, else openai).

  vLLM (local):
    python scripts/llm_judge_eval.py responses.jsonl scores.jsonl \\
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
        --api-base http://localhost:9000/v1 \\
        --concurrency 64

  OpenAI (chat completions — gpt-4o and older):
    python scripts/llm_judge_eval.py responses.jsonl scores.jsonl \\
        --model gpt-4o \\
        --api-key $OPENAI_API_KEY \\
        --concurrency 32

  OpenAI (responses API — gpt-4.5 and newer):
    python scripts/llm_judge_eval.py responses.jsonl scores.jsonl \\
        --model gpt-4.5 \\
        --provider openai-responses \\
        --api-key $OPENAI_API_KEY \\
        --concurrency 32

  Anthropic:
    python scripts/llm_judge_eval.py responses.jsonl scores.jsonl \\
        --model claude-opus-4-7 \\
        --api-key $ANTHROPIC_API_KEY \\
        --concurrency 16

  Custom thinking delimiters (e.g. a model that uses |think|/|/think|):
    python scripts/llm_judge_eval.py responses.jsonl scores.jsonl \\
        --model my-model \\
        --think-open '|think|' --think-close '|/think|'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import orjson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(
    raw: str,
    think_open: str,
    think_close: str,
) -> tuple[str | None, str | None, bool]:
    """
    Split an assistant response into (thinking, answer, thinking_complete).

    Returns:
        thinking:           text inside think_open…think_close, or None if absent
        answer:             text after think_close, or None if model didn't finish
        thinking_complete:  False when think_open seen but think_close never closed
    """
    close_pos = raw.find(think_close)
    open_pos = raw.find(think_open)

    if close_pos != -1:
        # Close tag present. The open tag is optional — many reasoning models
        # emit the opening delimiter implicitly and only produce </think>.
        start = open_pos + len(think_open) if (open_pos != -1 and open_pos < close_pos) else 0
        thinking = raw[start:close_pos].strip()
        answer = raw[close_pos + len(think_close):].strip() or None
        return thinking, answer, True

    if open_pos != -1:
        # Opened but never closed — model stopped mid-thought
        thinking = raw[open_pos + len(think_open):].strip()
        return thinking, None, False

    # No think tags at all — treat entire response as the answer
    return None, raw.strip() or None, True


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + f"\n… [truncated — {len(words) - max_words} more words]"


def extract_json(text: str, think_open: str, think_close: str) -> dict | None:
    """Extract a JSON object from judge output, handling think blocks and code fences."""
    # Strip any think block the judge itself produced
    pattern = re.compile(re.escape(think_open) + r".*?" + re.escape(think_close), re.DOTALL)
    text = pattern.sub("", text).strip()

    # Try markdown code fence first
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an expert AI evaluator. You will be given a user question and an "
    "AI assistant's response, which may include a reasoning trace. Evaluate the "
    "response quality and return a JSON object with your assessment."
)


def build_judge_prompt(
    question: str,
    thinking: str | None,
    answer: str | None,
    thinking_complete: bool,
    max_thinking_words: int,
) -> str:
    lines = [f"**User Question:**\n{question}\n"]

    if thinking is not None:
        truncated = truncate_words(thinking, max_thinking_words)
        tag = "" if thinking_complete else " *(INCOMPLETE — model stopped mid-thought)*"
        lines.append(f"**Thinking Trace{tag}:**\n{truncated}\n")
    else:
        lines.append("**Thinking Trace:** [none — model produced no thinking trace]\n")

    if answer is not None:
        lines.append(f"**Final Answer:**\n{answer}\n")
    else:
        lines.append("**Final Answer:** [none — model did not finish thinking]\n")

    lines.append(
        "Score the response on the following axes (1 = very poor, 5 = excellent):\n"
        "- **thinking_coherence**: Are the reasoning steps logically connected and "
        "do they lead toward the correct conclusion?\n"
        "- **thinking_clarity**: Is the thinking clearly written and easy to follow?\n"
        "- **answer_correctness**: Is the final answer correct and complete?\n\n"
        "Rules:\n"
        "- If there is no thinking trace, score thinking_coherence and thinking_clarity as 1.\n"
        "- If the thinking is incomplete (model stopped mid-thought), factor that into thinking_coherence.\n"
        "- If there is no final answer, score answer_correctness as 1.\n\n"
        "Respond ONLY with a JSON object, no other text:\n"
        '{"reasoning": "<2-3 sentence evaluation>", '
        '"thinking_coherence": <1-5>, '
        '"thinking_clarity": <1-5>, '
        '"answer_correctness": <1-5>}'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async judge calls
# ---------------------------------------------------------------------------

async def _call_judge(client, provider: str, model: str, prompt: str, max_tokens: int) -> str:
    """Call the judge API and return the raw text response."""
    if provider == "anthropic":
        resp = await client.messages.create(
            model=model,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.content[0].text or ""
    elif provider == "openai-responses":
        resp = await client.responses.create(
            model=model,
            instructions=_JUDGE_SYSTEM,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return resp.output_text or ""
    else:  # openai / vllm
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


async def judge_one(
    client,
    provider: str,
    model: str,
    record: dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    max_thinking_words: int,
    think_open: str,
    think_close: str,
    counter: list,  # [done, start_time, errors]
    total: int,
) -> dict:
    messages_raw = record.get("messages", [])
    if len(messages_raw) < 2:
        counter[2] += 1
        return {**record, "_judge_error": "insufficient_messages"}

    question = messages_raw[0].get("content", "")
    raw_response = messages_raw[1].get("content", "")

    thinking, answer, thinking_complete = parse_response(raw_response, think_open, think_close)
    prompt = build_judge_prompt(question, thinking, answer, thinking_complete, max_thinking_words)

    scores = None
    error = None

    async with semaphore:
        try:
            raw_judge = await _call_judge(client, provider, model, prompt, max_tokens)
            scores = extract_json(raw_judge, think_open, think_close)
            if scores is None:
                error = f"json_parse_failed: {raw_judge[:300]}"
                counter[2] += 1
        except Exception as exc:
            error = str(exc)
            counter[2] += 1

    counter[0] += 1
    done = counter[0]
    if done % 200 == 0 or done == total:
        elapsed = time.time() - counter[1]
        rate = done / elapsed if elapsed > 0 else 0
        logger.info(
            "Judge: %d / %d  (%.1f%%)  errors=%d  rate=%.1f/s",
            done, total, 100.0 * done / total, counter[2], rate,
        )

    result = dict(record)
    result["_thinking_complete"] = thinking_complete
    if scores:
        result["judge_reasoning"] = scores.get("reasoning", "")
        result["judge_thinking_coherence"] = scores.get("thinking_coherence")
        result["judge_thinking_clarity"] = scores.get("thinking_clarity")
        result["judge_answer_correctness"] = scores.get("answer_correctness")
    else:
        result["_judge_error"] = error

    return result


async def run_async(
    records: list[dict],
    model: str,
    provider: str,
    api_base: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    max_thinking_words: int,
    think_open: str,
    think_close: str,
    output_file: Path | None = None,
) -> list[dict]:
    http_client = None

    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)
    else:
        import httpx
        from openai import AsyncOpenAI
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency + 16,
                max_keepalive_connections=concurrency,
            ),
            timeout=httpx.Timeout(120.0),
        )
        client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key or "none",
            http_client=http_client,
        )

    semaphore = asyncio.Semaphore(concurrency)
    counter = [0, time.time(), 0]  # [done, start_time, errors]
    total = len(records)

    logger.info(
        "Starting evaluation: %d records  model=%s  provider=%s  concurrency=%d",
        total, model, provider, concurrency,
    )

    tasks = [
        asyncio.create_task(
            judge_one(
                client, provider, model, rec, semaphore,
                max_tokens, max_thinking_words,
                think_open, think_close,
                counter, total,
            )
        )
        for rec in records
    ]

    # Stream each result to disk as it completes: a mid-run crash (or an API
    # account running out of balance) must not lose the calls already paid for.
    results = []
    out_fh = output_file.open("wb") if output_file is not None else None
    try:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if out_fh is not None:
                out_fh.write(orjson.dumps(result) + b"\n")
                out_fh.flush()
    finally:
        if out_fh is not None:
            out_fh.close()

    if http_client is not None:
        await http_client.aclose()

    elapsed = time.time() - counter[1]
    logger.info(
        "Done: %d evaluated  %d errors  %.0fs total  (%.1f/s avg)",
        total, counter[2], elapsed, total / elapsed if elapsed else 0,
    )
    return results


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(results: list[dict]) -> None:
    total = len(results)
    errors = [r for r in results if "_judge_error" in r]
    scored = [r for r in results if "judge_thinking_coherence" in r]
    incomplete = [r for r in scored if not r.get("_thinking_complete", True)]

    print(f"\n{'='*62}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*62}")
    print(f"  Total records      : {total}")
    print(f"  Successfully scored: {len(scored)}")
    print(f"  Errors / skipped   : {len(errors)}")
    print(f"  Incomplete thinking: {len(incomplete)}  "
          f"({100*len(incomplete)/max(len(scored),1):.1f}% of scored)")

    axes = [
        ("thinking_coherence", "Thinking Coherence"),
        ("thinking_clarity",   "Thinking Clarity"),
        ("answer_correctness", "Answer Correctness"),
    ]

    print(f"\n  {'Axis':<22} {'Mean':>5} {'Std':>5}   "
          f"{'1':>5} {'2':>5} {'3':>5} {'4':>5} {'5':>5}")
    print(f"  {'─'*58}")

    for key, label in axes:
        values = [
            r[f"judge_{key}"]
            for r in scored
            if isinstance(r.get(f"judge_{key}"), (int, float))
        ]
        if not values:
            print(f"  {label:<22}  (no data)")
            continue
        dist = {i: values.count(i) for i in range(1, 6)}
        avg = mean(values)
        sd = stdev(values) if len(values) > 1 else 0.0
        print(
            f"  {label:<22} {avg:>5.2f} {sd:>5.2f}   "
            + "  ".join(f"{dist.get(i, 0):>5}" for i in range(1, 6))
        )

    print(f"  {'─'*58}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge evaluator for response quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Windows consoles default to cp1252, which cannot encode the box-drawing
    # characters used in print_stats(). Force UTF-8 so stats never crash the run.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser.add_argument("input_file", type=Path, help="Input JSONL file")
    parser.add_argument("output_file", type=Path, help="Output JSONL file with judge scores")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="Judge model served by the vLLM API",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:9000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument("--api-key", default="none", help="API key")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "openai", "openai-responses", "anthropic"],
        help=(
            "API provider. 'auto' infers from model name (claude-* → anthropic, else openai). "
            "Use 'openai-responses' for models that require the Responses API (gpt-4.5+)."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrent in-flight requests")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for judge response")
    parser.add_argument(
        "--max-thinking-words",
        type=int,
        default=1500,
        help="Truncate thinking traces to this many words before sending to the judge",
    )
    parser.add_argument(
        "--think-open",
        default="<think>",
        help="Opening tag for the model's thinking block",
    )
    parser.add_argument(
        "--think-close",
        default="</think>",
        help="Closing tag for the model's thinking block",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Load input
    logger.info("Loading %s ...", args.input_file)
    records: list[dict] = []
    with args.input_file.open("rb") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(orjson.loads(line))
            except Exception as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)

    logger.info("Loaded %d records", len(records))
    if not records:
        logger.error("No records to evaluate")
        sys.exit(1)

    provider = args.provider
    if provider == "auto":
        provider = "anthropic" if args.model.startswith("claude") else "openai"
        logger.info("Auto-detected provider: %s", provider)

    # Evaluate — results are streamed to the output file as they complete.
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run_async(
        records,
        model=args.model,
        provider=provider,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_thinking_words=args.max_thinking_words,
        think_open=args.think_open,
        think_close=args.think_close,
        output_file=args.output_file,
    ))
    logger.info("Scores written to %s", args.output_file)

    # Print stats
    print_stats(results)


if __name__ == "__main__":
    main()
