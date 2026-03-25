"""
Dry-run cost and time estimator.

Counts input prompts, estimates GPU-hours for Stage 5 inference,
and prints a human-readable table. No actual processing occurs.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sft_pipeline.config import PipelineConfig
from sft_pipeline.storage import count_jsonl_lines, iter_jsonl_dir

logger = logging.getLogger(__name__)
console = Console()

# Rough throughput constants
# Qwen3.5-122B-A10B MoE (10B active): ~800 output tok/s per replica on MI250X
# With 64 replicas: 64 * 800 = 51,200 tok/s total
_DEFAULT_OUTPUT_TOKENS_PER_PROMPT = 2048  # avg reasoning trace + answer
_DEFAULT_TOK_PER_SEC_PER_REPLICA = 800
_EMBEDDING_THROUGHPUT_PER_GPU = 500  # prompts/sec on one GPU (bge-m3)


def estimate_and_print(cfg: PipelineConfig) -> None:
    console.rule("[bold blue]SFT Pipeline — Run Estimate")
    console.print(f"  Run ID: [cyan]{cfg.run_id}[/cyan]")
    console.print(f"  Base path: [cyan]{cfg.base_path}[/cyan]\n")

    rows = []

    # Stage 1 + 2: count existing output if available
    s1_dir = Path(cfg.stage1_collect.output_path).parent
    s2_dir = Path(cfg.stage2_generate.output_path).parent
    s1_count = _count_dir(s1_dir)
    s2_count = _count_dir(s2_dir)
    rows.append(("Stage 1 — Collect", _fmt(s1_count), "~fast", "—"))
    rows.append(("Stage 2 — Generate", _fmt(s2_count), "~fast (depends on LLM)", "—"))

    # Stage 3: embedding estimate
    total_pool = max(s1_count + s2_count, cfg.stage4_sample.total_prompts)
    emb_gpus = 1 if cfg.global_.device == "cpu" else 1
    emb_secs = total_pool / _EMBEDDING_THROUGHPUT_PER_GPU / emb_gpus / 3600
    rows.append(("Stage 3 — Cluster+Embed", _fmt(total_pool), f"~{emb_secs:.1f} GPU-hrs", "—"))

    # Stage 4: fast CPU/Polars op
    rows.append(("Stage 4 — Sample", _fmt(cfg.stage4_sample.total_prompts), "< 1 hr (CPU)", "—"))

    # Stage 5: inference estimate
    n_prompts = cfg.stage4_sample.total_prompts
    n_replicas = cfg.stage5_inference.n_replicas
    avg_out_tokens = _DEFAULT_OUTPUT_TOKENS_PER_SEC = (
        cfg.stage5_inference.generation.max_tokens // 2
    )
    total_tokens = n_prompts * avg_out_tokens
    total_tok_per_sec = _DEFAULT_TOK_PER_SEC_PER_REPLICA * n_replicas
    wall_hrs = total_tokens / total_tok_per_sec / 3600
    gpu_hrs = wall_hrs * n_replicas * cfg.stage5_inference.vllm_engine.tensor_parallel_size
    rows.append((
        "Stage 5 — Inference",
        _fmt(n_prompts),
        f"~{wall_hrs:.1f} wall-hrs ({n_replicas} replicas)",
        f"~{gpu_hrs:,.0f} GPU-hrs",
    ))

    # Stage 6: filtering estimate (fast, mostly CPU)
    rows.append(("Stage 6 — Filter", _fmt(n_prompts), "~2–4 hrs (CPU + judge)", "—"))

    # Estimated final output
    expected_output = int(n_prompts * 0.70)  # ~30% filtered
    rows.append(("Expected final dataset", _fmt(expected_output), "—", "—"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Prompts")
    table.add_column("Est. Wall Time")
    table.add_column("Est. GPU-hrs")

    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]Model:[/bold] {cfg.stage5_inference.model}  "
        f"[bold]Replicas:[/bold] {n_replicas}  "
        f"[bold]TP:[/bold] {cfg.stage5_inference.vllm_engine.tensor_parallel_size}"
    )
    console.print(
        "[yellow]Note: wall-time estimates assume ~800 tok/s/replica (MI250X).[/yellow]"
    )


def _count_dir(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        count_jsonl_lines(f)
        for f in sorted(directory.glob("*.jsonl"))
    )


def _fmt(n: int) -> str:
    if n == 0:
        return "[dim]not yet run[/dim]"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
