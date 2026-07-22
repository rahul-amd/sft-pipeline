"""
Small shared helpers for the Ray-distributed stages.

Stages 1, 3, 6 and decontaminate all fan a set of independent tasks out to a Ray
cluster and drain them as they finish, logging ``[done/total]`` progress and
tolerating individual task failures. This module holds that mechanism in one
place so each stage keeps only its own task body and result handling.

Design contract (why local tests translate to the cluster): workers write their
own uniquely-named output files and return small results; the **driver** is the
sole writer of any shared bookkeeping (ledger, report, checkpoint marks). That
keeps multi-node runs free of write contention, so the only things a single-node
local test can't reproduce are performance and cluster ops — not correctness.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def ensure_ray(cfg) -> None:
    """Attach to the configured Ray cluster, unless one is already running.

    Idempotent: if ``ray.init`` has already been called (e.g. a test that spun up
    a local Ray, or an earlier stage in the same process), this is a no-op. That
    lets tests pre-initialize a local cluster with ``ray.init(num_cpus=...)`` and
    have the stage attach to it rather than dialing ``ray_address``.
    """
    import ray

    if not ray.is_initialized():
        ray.init(address=cfg.global_.ray_address, ignore_reinit_error=True)


def as_completed(
    future_to_label: dict[Any, Any],
    *,
    desc: str,
) -> Iterator[tuple[int, int, Any, Any, BaseException | None]]:
    """Drain Ray futures as they finish, yielding one tuple per completed task.

    Yields ``(done, total, label, result, error)``:
      - ``result`` is the task's return value and ``error`` is ``None`` on
        success;
      - ``result`` is ``None`` and ``error`` is the raised exception on failure.

    This never raises for a failed task — the caller decides whether to skip it,
    retry it, or abort. ``label`` is whatever value the caller mapped each future
    to (e.g. a shard key), used for logging and result handling.
    """
    import ray

    remaining = list(future_to_label)
    total = len(remaining)
    done = 0
    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        future = ready[0]
        label = future_to_label[future]
        done += 1
        try:
            result = ray.get(future)
            yield done, total, label, result, None
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the drain loop
            logger.error("%s [%d/%d] failed: %s — %s", desc, done, total, label, exc)
            yield done, total, label, None, exc
