"""Tests for config loading and validation."""
from __future__ import annotations

import pytest
import yaml


def test_load_minimal_config(tmp_path):
    from sft_pipeline.config import load_config

    cfg_text = """
global:
  seed: 42
  run_id: test001
  base_path: /tmp/sft/{run_id}
  device: cpu
"""
    f = tmp_path / "cfg.yaml"
    f.write_text(cfg_text)
    cfg = load_config(f)
    assert cfg.global_.seed == 42
    assert cfg.run_id == "test001"
    assert cfg.base_path == "/tmp/sft/test001"


def test_placeholder_resolution(tmp_path):
    from sft_pipeline.config import load_config

    f = tmp_path / "cfg.yaml"
    f.write_text("""
global:
  run_id: run_abc
  base_path: /data/{run_id}
stage5_inference:
  output_path: "{base_path}/stage5/responses.jsonl"
""")
    cfg = load_config(f)
    assert cfg.stage5_inference.output_path == "/data/run_abc/stage5/responses.jsonl"


def test_domain_quotas_validation(tmp_path):
    from sft_pipeline.config import load_config
    from pydantic import ValidationError

    f = tmp_path / "cfg.yaml"
    f.write_text("""
stage4_sample:
  domain_quotas:
    math: 0.50
    code: 0.60
""")
    with pytest.raises((ValidationError, ValueError)):
        load_config(f)


def test_hf_dataset_source_requires_repo_id(tmp_path):
    from sft_pipeline.config import load_config
    from pydantic import ValidationError

    f = tmp_path / "cfg.yaml"
    f.write_text("""
stage1_collect:
  datasets:
    - source: hf_dataset
""")
    with pytest.raises((ValidationError, ValueError)):
        load_config(f)


def test_device_options(tmp_path):
    from sft_pipeline.config import load_config

    for device in ("cuda", "rocm", "cpu"):
        f = tmp_path / f"cfg_{device}.yaml"
        f.write_text(f"global:\n  device: {device}\n")
        cfg = load_config(f)
        assert cfg.global_.device == device
