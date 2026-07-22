"""Tests for the shared Ray helpers."""
from __future__ import annotations

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")

from sft_pipeline import ray_utils  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _local_ray():
    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False,
             configure_logging=False, log_to_driver=False)
    yield
    ray.shutdown()


def _double(x):
    return x * 2


def _boom(x):
    raise ValueError(f"boom {x}")


def test_as_completed_collects_all_results():
    remote = ray.remote(_double)
    future_to_label = {remote.remote(i): i for i in range(5)}
    out = {}
    for done, total, label, result, err in ray_utils.as_completed(future_to_label, desc="t"):
        assert total == 5
        assert err is None
        out[label] = result
    assert out == {i: i * 2 for i in range(5)}


def test_as_completed_surfaces_errors_without_raising():
    remote_ok = ray.remote(_double)
    remote_bad = ray.remote(_boom)
    future_to_label = {remote_ok.remote(1): "ok", remote_bad.remote(9): "bad"}
    results, errors = {}, {}
    for _done, _total, label, result, err in ray_utils.as_completed(future_to_label, desc="t"):
        if err is None:
            results[label] = result
        else:
            errors[label] = err
    assert results == {"ok": 2}
    assert "bad" in errors and isinstance(errors["bad"], BaseException)


def test_ensure_ray_is_noop_when_initialized():
    # Ray is already up from the fixture; ensure_ray must not raise or reinit.
    class _Cfg:
        class _G:
            ray_address = "auto"
        global_ = _G()
    ray_utils.ensure_ray(_Cfg())
    assert ray.is_initialized()
