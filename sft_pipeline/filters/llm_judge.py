"""
LLM-based quality judge (samples a configurable fraction of responses).

Uses a lightweight model (Qwen2.5-7B via vLLM HTTP) to score responses
on a 1-10 scale. Responses below the threshold are filtered out.
Only `sample_rate` fraction of examples are judged (cost control).
"""
from __future__ import annotations

import json
import logging
import random

import httpx
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_exponential

from sft_pipeline.config import LLMJudgeConfig
from sft_pipeline.filters.structural import FilterResult

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You are an expert evaluator of AI-generated educational content.
Rate the following prompt-response pair on these criteria:
- accuracy (1-10): Is the final answer correct and well-supported?
- reasoning (1-10): Is the reasoning trace logical, complete, and clear?
- format (1-10): Is the response well-formatted and easy to follow?

Respond ONLY with valid JSON in this exact format:
{"accuracy": <int>, "reasoning": <int>, "format": <int>, "overall": <int>, "reason": "<one sentence>"}"""

_JUDGE_USER = """\
Prompt: {prompt}

Reasoning:
{reasoning}

Answer: {answer}

Rate this response."""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def _call_judge(
    prompt: str,
    reasoning: str,
    answer: str,
    endpoint: str,
    model: str,
) -> dict | None:
    """Call the LLM judge and return the parsed score dict."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_USER.format(
                    prompt=prompt[:500],
                    reasoning=reasoning[:1500],
                    answer=answer[:500],
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    resp = httpx.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        repaired = repair_json(content)
        try:
            return json.loads(repaired)
        except Exception:
            return None


def check_llm_judge(
    record: dict,
    cfg: LLMJudgeConfig,
    rng: random.Random | None = None,
) -> FilterResult:
    """
    Optionally call the LLM judge on this record.
    Skips non-sampled records (returns passed=True immediately).
    """
    if not cfg.enabled:
        return FilterResult(True)

    _rng = rng or random.Random()
    if _rng.random() > cfg.sample_rate:
        return FilterResult(True)  # Not selected for judging

    try:
        scores = _call_judge(
            prompt=record.get("prompt", ""),
            reasoning=record.get("reasoning", ""),
            answer=record.get("answer", ""),
            endpoint=cfg.model_endpoint,
            model=cfg.model,
        )
    except Exception as exc:
        logger.warning("LLM judge call failed: %s — passing record through", exc)
        return FilterResult(True, reason="judge_error")

    if scores is None:
        return FilterResult(True, reason="judge_parse_error")

    overall = scores.get("overall", 10)
    try:
        overall = float(overall)
    except (TypeError, ValueError):
        return FilterResult(True, reason="judge_score_invalid")

    if overall < cfg.score_threshold:
        return FilterResult(
            False,
            reason=f"judge_score:{overall:.1f}<{cfg.score_threshold}",
        )

    return FilterResult(True)
