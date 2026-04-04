"""
Shared pytest fixtures.

All fixtures are CPU-only and lightweight — suitable for running on
the development laptop without GPU or heavy dependencies.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Tiny synthetic datasets
# ---------------------------------------------------------------------------

SAMPLE_MATH_PROMPTS = [
    "What is the derivative of x^2 + 3x + 5?",
    "Solve the equation 2x + 4 = 10.",
    "Calculate the integral of sin(x) from 0 to pi.",
    "What is the probability of rolling a 6 on a fair die?",
    "Find the roots of the quadratic equation x^2 - 5x + 6 = 0.",
]

SAMPLE_CODE_PROMPTS = [
    "Write a Python function to reverse a string.",
    "Implement a binary search algorithm in Python.",
    "Write a function that checks if a number is prime.",
    "Implement a stack data structure in Python.",
    "Write a Python function to find the factorial of n.",
]

SAMPLE_SCIENCE_PROMPTS = [
    "Explain Newton's second law of motion.",
    "What is the difference between an atom and a molecule?",
    "Describe the process of photosynthesis.",
    "What is the speed of light in a vacuum?",
    "Explain the concept of entropy in thermodynamics.",
]

SAMPLE_GENERAL_PROMPTS = [
    "What are the main causes of World War I?",
    "Explain the water cycle.",
    "What is the capital of France?",
    "Describe the process of natural selection.",
    "What is the difference between a virus and a bacterium?",
]

ALL_SAMPLE_PROMPTS = (
    SAMPLE_MATH_PROMPTS
    + SAMPLE_CODE_PROMPTS
    + SAMPLE_SCIENCE_PROMPTS
    + SAMPLE_GENERAL_PROMPTS
)


def make_prompt_record(
    prompt: str,
    domain: str = "general",
    difficulty: str = "medium",
    source: str = "test",
) -> dict:
    from sft_pipeline.checkpoint import prompt_id
    return {
        "prompt_id": prompt_id(prompt),
        "prompt": prompt,
        "source": source,
        "domain": domain,
        "difficulty": difficulty,
        "stage": "test",
    }


def make_response_record(prompt: str, domain: str = "general") -> dict:
    base = make_prompt_record(prompt, domain=domain)
    base.update({
        "reasoning": (
            f"Let me think through this carefully. The question asks: '{prompt[:40]}'. "
            "To solve this I need to break the problem into clear steps and analyze each part. "
            "First I will identify the key concepts involved and their relationships. "
            "Next I will apply the relevant principles to work toward a solution. "
            "The core insight is that we must consider all relevant constraints and conditions. "
            "Working through the logic step by step leads us to the correct conclusion. "
            "Therefore the answer follows directly from this systematic analysis."
        ),
        "answer": "Based on the analysis above, this is the final answer to the question.",
        "teacher_model": "test-model",
        "used_fallback_parse": False,
    })
    return base


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path):
    """A minimal pipeline config pointing to tmp_path."""
    config_text = f"""
global:
  seed: 42
  dry_run: false
  run_id: test_run
  base_path: "{tmp_path}"
  checkpoint_db: "{tmp_path}/checkpoints.duckdb"
  device: cpu

stage1_collect:
  enabled: true
  datasets: []
  output_dir: "{tmp_path}/stage1"

stage2_generate:
  enabled: false
  corpora: []
  output_dir: "{tmp_path}/stage2"

stage3_cluster:
  enabled: false
  embeddings_dir: "{tmp_path}/stage3/embeddings"
  faiss_index_path: "{tmp_path}/stage3/faiss.index"
  output_dir: "{tmp_path}/stage3"

stage4_sample:
  enabled: false
  total_prompts: 20
  output_dir: "{tmp_path}/stage4"

stage5_inference:
  enabled: false
  model: "test-model"
  n_replicas: 1
  output_dir: "{tmp_path}/stage5"

stage6_filter:
  enabled: false
  output_dir: "{tmp_path}/stage6"
  report_path: "{tmp_path}/stage6/filter_report.json"
  llm_judge:
    enabled: false

export:
  final_jsonl_path: "{tmp_path}/final/dataset.jsonl"
  push_to_hub: false
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(config_text)

    from sft_pipeline.config import load_config
    return load_config(cfg_file)


@pytest.fixture
def checkpoint_manager(tmp_path):
    """A fresh CheckpointManager backed by a tmp_path DuckDB file."""
    from sft_pipeline.checkpoint import CheckpointManager
    cm = CheckpointManager(tmp_path / "checkpoints.duckdb")
    cm.open()
    yield cm
    cm.close()


@pytest.fixture
def sample_jsonl_dir(tmp_path) -> Path:
    """Write 20 sample prompt records to a JSONL dir and return the path."""
    d = tmp_path / "prompts"
    d.mkdir()
    records = [
        make_prompt_record(p, domain="math" if i < 5 else "general")
        for i, p in enumerate(ALL_SAMPLE_PROMPTS[:20])
    ]
    shard = d / "part-000000.jsonl"
    with shard.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return d


@pytest.fixture
def sample_response_jsonl_dir(tmp_path) -> Path:
    """Write 20 sample response records (with reasoning+answer) to a JSONL dir."""
    d = tmp_path / "responses"
    d.mkdir()
    records = [
        make_response_record(p, domain="math" if i < 5 else
                               "code" if i < 10 else "general")
        for i, p in enumerate(ALL_SAMPLE_PROMPTS[:20])
    ]
    shard = d / "part-000000.jsonl"
    with shard.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return d
