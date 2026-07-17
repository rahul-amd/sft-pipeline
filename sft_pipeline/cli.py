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
    # Suppress noisy HTTP-level logs from libraries used by HF datasets streaming
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _apply_global_env(cfg) -> None:
    """Apply global config settings to the process environment.

    Called once in the main process before any stage runs.
    Ray remote workers are separate processes and apply their own env via
    the arguments passed to each remote function (e.g. _collect_source).
    """
    import os
    if cfg.global_.hf_home:
        os.environ["HF_HOME"] = cfg.global_.hf_home
        logging.getLogger(__name__).info("HF_HOME set to %s", cfg.global_.hf_home)


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
    _apply_global_env(cfg)

    if dry_run or cfg.global_.dry_run:
        from sft_pipeline.cost_estimator import estimate_and_print
        estimate_and_print(cfg)
        return

    from sft_pipeline.checkpoint import CheckpointManager
    from sft_pipeline.stages.decontaminate import run_decontaminate
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
            ("decontaminate",
             cfg.decontaminate.enabled and bool(cfg.decontaminate.evals),
             lambda: run_decontaminate(cfg, cm)),
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
    "decontaminate",
    "stage3_cluster",
    "stage4_sample",
    "stage5_inference",
    "stage6_filter",
]


@app.command(name="run-stage")
def run_stage(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    stage: str = typer.Argument(help=f"Stage name: {', '.join(STAGE_NAMES)}"),
    dump_annotations: Optional[Path] = typer.Option(
        None, "--dump-annotations",
        help=(
            "[stage3_cluster only] Write annotation requests to PATH as JSONL "
            "(OpenAI-compatible messages format) and exit. "
            "Run inference elsewhere, then re-run with --import-annotations."
        ),
    ),
    import_annotations: Optional[Path] = typer.Option(
        None, "--import-annotations",
        help=(
            "[stage3_cluster only] Read pre-computed annotation responses from PATH "
            "(JSONL with prompt_id + response fields). Falls back to online inference "
            "for any prompt_ids missing from the file (requires annotation_enabled: true)."
        ),
    ),
) -> None:
    """Run a single pipeline stage by name."""
    if stage not in STAGE_NAMES:
        console.print(f"[red]Unknown stage '{stage}'. Valid: {', '.join(STAGE_NAMES)}[/red]")
        raise typer.Exit(1)

    if (dump_annotations or import_annotations) and stage != "stage3_cluster":
        console.print(
            "[red]--dump-annotations and --import-annotations are only valid for stage3_cluster[/red]"
        )
        raise typer.Exit(1)

    cfg = _load(config)
    _setup_logging(cfg.global_.log_level)
    _apply_global_env(cfg)

    from sft_pipeline.checkpoint import CheckpointManager

    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        console.rule(f"[bold blue]{stage}")
        _dispatch_stage(
            stage, cfg, cm,
            dump_annotations_path=dump_annotations,
            import_annotations_path=import_annotations,
        )

    console.print(f"\n[bold green]{stage} complete.[/bold green]")


def _dispatch_stage(
    stage: str,
    cfg,
    cm,
    dump_annotations_path: Optional[Path] = None,
    import_annotations_path: Optional[Path] = None,
) -> None:
    if stage == "stage1_collect":
        from sft_pipeline.stages.stage1_collect import run_stage1
        run_stage1(cfg, cm)
    elif stage == "stage2_generate":
        from sft_pipeline.stages.stage2_generate import run_stage2
        run_stage2(cfg, cm)
    elif stage == "decontaminate":
        from sft_pipeline.stages.decontaminate import run_decontaminate
        run_decontaminate(cfg, cm)
    elif stage == "stage3_cluster":
        from sft_pipeline.stages.stage3_cluster import run_stage3
        run_stage3(cfg, cm,
                   dump_annotations_path=dump_annotations_path,
                   import_annotations_path=import_annotations_path)
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
# annotate
# ---------------------------------------------------------------------------

@app.command()
def annotate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """
    Annotate prompts via an external OpenAI-compatible API (no GPU needed).

    Reads all prompts from stage1_collect.output_dir (and stage2_generate.output_dir
    if it exists), calls the configured vLLM annotation endpoint, and writes a
    checkpoint to {stage3_cluster.output_dir}/annotations.parquet.

    Safe to interrupt and resume — the checkpoint is updated every
    stage3_cluster.annotation_checkpoint_every records, and already-annotated
    prompts are skipped on restart.

    The checkpoint file is the same one that run_stage3() reads automatically,
    so when you later run the full Stage 3 (with clustering) it will pick up
    the pre-computed annotations and skip re-annotating.

    Typical usage:
      sft-pipeline annotate --config config/stage3_annotate.yaml
    """
    cfg = _load(config)
    _setup_logging(cfg.global_.log_level)
    _apply_global_env(cfg)

    s3 = cfg.stage3_cluster
    if not s3.annotation_enabled:
        console.print(
            "[yellow]annotation_enabled is false in config — nothing to do.\n"
            "Set stage3_cluster.annotation_enabled: true to enable.[/yellow]"
        )
        raise typer.Exit(0)

    from sft_pipeline.clustering.annotator import annotate_prompts
    from sft_pipeline.storage import ensure_dir, iter_jsonl_dir

    stage1_dir = Path(cfg.stage1_collect.output_dir)
    stage2_dir = Path(cfg.stage2_generate.output_dir)
    out_dir = Path(s3.output_dir)
    ann_checkpoint = out_dir / "annotations.parquet"
    ensure_dir(out_dir)

    # Collect prompt_id + prompt text from stage1 (and stage2 if present)
    prompt_records: list[dict] = []
    seen_ids: set[str] = set()
    for source_dir in (stage1_dir, stage2_dir):
        if not source_dir.exists():
            console.print(f"[dim]  {source_dir} not found — skipping[/dim]")
            continue
        n_before = len(prompt_records)
        for rec in iter_jsonl_dir(source_dir):
            pid = rec.get("prompt_id")
            if pid and pid not in seen_ids:
                prompt_records.append({"prompt_id": pid, "prompt": rec.get("prompt", "")})
                seen_ids.add(pid)
        console.print(f"  {source_dir}: {len(prompt_records) - n_before:,} prompts loaded")

    if not prompt_records:
        console.print(f"[red]No prompts found in {stage1_dir} or {stage2_dir}.[/red]")
        raise typer.Exit(1)

    console.rule("[bold cyan]Annotation")
    console.print(
        f"  [bold]{len(prompt_records):,}[/bold] prompts total\n"
        f"  model         = {s3.annotation_model}\n"
        f"  api_base      = {s3.annotation_api_base}\n"
        f"  concurrency   = {s3.annotation_concurrency}\n"
        f"  checkpoint    = {ann_checkpoint}\n"
        f"  save every    = {s3.annotation_checkpoint_every:,} records\n"
    )

    annotate_prompts(
        prompt_records=prompt_records,
        model=s3.annotation_model,
        api_base=s3.annotation_api_base,
        api_key=s3.annotation_api_key,
        concurrency=s3.annotation_concurrency,
        max_tokens=s3.annotation_max_tokens,
        temperature=s3.annotation_temperature,
        checkpoint_path=ann_checkpoint,
        checkpoint_every=s3.annotation_checkpoint_every,
    )

    console.print(f"\n[bold green]✓ Annotation complete.[/bold green]  Checkpoint: {ann_checkpoint}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
