"""
Typer CLI for the SFT dataset construction pipeline.

Commands:
  run           Run all enabled stages end-to-end
  run-stage     Run a single stage by name
  status        Show checkpoint status for a run
  estimate      Dry-run cost/time estimation (no actual processing)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sft-pipeline",
    help="Scalable SFT dataset construction pipeline.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load(config_path: str):
    from sft_pipeline.config import load_config
    return load_config(config_path)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost/time without processing"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip already-processed items"),
) -> None:
    """Run all enabled pipeline stages end-to-end."""
    cfg = _load(config)
    _setup_logging(cfg.global_.log_level)

    if dry_run or cfg.global_.dry_run:
        from sft_pipeline.cost_estimator import estimate_and_print
        estimate_and_print(cfg)
        return

    from sft_pipeline.checkpoint import CheckpointManager
    from sft_pipeline.stages.stage1_collect import run_stage1
    from sft_pipeline.stages.stage2_generate import run_stage2
    from sft_pipeline.stages.stage3_cluster import run_stage3
    from sft_pipeline.stages.stage4_sample import run_stage4
    from sft_pipeline.stages.stage5_inference import run_stage5
    from sft_pipeline.stages.stage6_filter import run_stage6

    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        stages = [
            ("stage1_collect", cfg.stage1_collect.enabled, lambda: run_stage1(cfg, cm)),
            ("stage2_generate", cfg.stage2_generate.enabled, lambda: run_stage2(cfg, cm)),
            ("stage3_cluster", cfg.stage3_cluster.enabled, lambda: run_stage3(cfg, cm)),
            ("stage4_sample", cfg.stage4_sample.enabled, lambda: run_stage4(cfg, cm)),
            ("stage5_inference", cfg.stage5_inference.enabled, lambda: run_stage5(cfg, cm)),
            ("stage6_filter", cfg.stage6_filter.enabled, lambda: run_stage6(cfg, cm)),
        ]
        for stage_name, enabled, fn in stages:
            if not enabled:
                console.print(f"[dim]Skipping {stage_name} (disabled)[/dim]")
                continue
            if resume and cm.is_stage_complete(stage_name):
                console.print(f"[green]✓[/green] {stage_name} already complete — skipping")
                continue
            console.rule(f"[bold blue]{stage_name}")
            try:
                fn()
            except Exception as exc:
                console.print(f"[red]✗ {stage_name} FAILED: {exc}[/red]")
                cm.mark_stage_failed(stage_name, str(exc))
                raise typer.Exit(1) from exc

    console.print("\n[bold green]Pipeline complete.[/bold green]")


# ---------------------------------------------------------------------------
# run-stage
# ---------------------------------------------------------------------------

STAGE_NAMES = [
    "stage1_collect",
    "stage2_generate",
    "stage3_cluster",
    "stage4_sample",
    "stage5_inference",
    "stage6_filter",
]


@app.command(name="run-stage")
def run_stage(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    stage: str = typer.Argument(help=f"Stage name: {', '.join(STAGE_NAMES)}"),
) -> None:
    """Run a single pipeline stage by name."""
    if stage not in STAGE_NAMES:
        console.print(f"[red]Unknown stage '{stage}'. Valid: {', '.join(STAGE_NAMES)}[/red]")
        raise typer.Exit(1)

    cfg = _load(config)
    _setup_logging(cfg.global_.log_level)

    from sft_pipeline.checkpoint import CheckpointManager

    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        console.rule(f"[bold blue]{stage}")
        _dispatch_stage(stage, cfg, cm)

    console.print(f"\n[bold green]{stage} complete.[/bold green]")


def _dispatch_stage(stage: str, cfg, cm) -> None:
    if stage == "stage1_collect":
        from sft_pipeline.stages.stage1_collect import run_stage1
        run_stage1(cfg, cm)
    elif stage == "stage2_generate":
        from sft_pipeline.stages.stage2_generate import run_stage2
        run_stage2(cfg, cm)
    elif stage == "stage3_cluster":
        from sft_pipeline.stages.stage3_cluster import run_stage3
        run_stage3(cfg, cm)
    elif stage == "stage4_sample":
        from sft_pipeline.stages.stage4_sample import run_stage4
        run_stage4(cfg, cm)
    elif stage == "stage5_inference":
        from sft_pipeline.stages.stage5_inference import run_stage5
        run_stage5(cfg, cm)
    elif stage == "stage6_filter":
        from sft_pipeline.stages.stage6_filter import run_stage6
        run_stage6(cfg, cm)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """Show checkpoint status for a pipeline run."""
    cfg = _load(config)

    from sft_pipeline.checkpoint import CheckpointManager

    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        rows = cm.all_stage_statuses()

    if not rows:
        console.print("[yellow]No pipeline activity recorded yet.[/yellow]")
        return

    table = Table(title=f"Run: {cfg.run_id}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Input")
    table.add_column("Output")
    table.add_column("Started")
    table.add_column("Completed")

    status_colors = {
        "completed": "green", "running": "yellow",
        "failed": "red", "pending": "dim",
    }
    for r in rows:
        color = status_colors.get(r["status"], "white")
        table.add_row(
            r["stage"],
            f"[{color}]{r['status']}[/{color}]",
            str(r["input_count"] or "—"),
            str(r["output_count"] or "—"),
            str(r["started_at"] or "—"),
            str(r["completed_at"] or "—"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------

@app.command()
def estimate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """Print cost/time estimates for a pipeline run without processing any data."""
    cfg = _load(config)
    _setup_logging(cfg.global_.log_level)
    from sft_pipeline.cost_estimator import estimate_and_print
    estimate_and_print(cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
