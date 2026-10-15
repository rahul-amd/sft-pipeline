"""Release-blocking retry, reproducibility, benchmark and judge regressions."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import EvalDatasetSource, LLMJudgeConfig, load_config
from sft_pipeline.decontam.eval_index import EvalLoadError, build_index
from sft_pipeline.storage import iter_jsonl_dir, replacing_jsonl_writer


def test_replacement_failure_preserves_old_shards_and_retry_removes_surplus(tmp_path):
    (tmp_path / "part-000000.jsonl").write_text('{"old":0}\n')
    (tmp_path / "part-000001.jsonl").write_text('{"old":1}\n')
    (tmp_path / "cache.done").write_text("keep")
    with pytest.raises(RuntimeError):
        with replacing_jsonl_writer(tmp_path) as writer:
            writer.write({"new": 0})
            raise RuntimeError("interrupted before publishing")
    assert list(iter_jsonl_dir(tmp_path)) == [{"old": 0}, {"old": 1}]
    for _ in range(2):
        with replacing_jsonl_writer(tmp_path) as writer:
            writer.write({"new": 0})
        assert list(iter_jsonl_dir(tmp_path)) == [{"new": 0}]
    assert (tmp_path / "cache.done").read_text() == "keep"


def test_retry_recovers_interrupted_publication(tmp_path, monkeypatch):
    for i in range(3):
        (tmp_path / f"part-{i:06d}.jsonl").write_text(json.dumps({"old": i}) + "\n")
    real_replace = Path.replace
    calls = 0
    def interrupt(path, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication interrupted")
        return real_replace(path, target)
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", interrupt)
        with pytest.raises(OSError):
            with replacing_jsonl_writer(tmp_path, shard_size_mb=0) as writer:
                writer.write({"new": 0})
                writer.write({"new": 1})
    with replacing_jsonl_writer(tmp_path, shard_size_mb=0) as writer:
        writer.write({"new": 0})
        writer.write({"new": 1})
    assert list(iter_jsonl_dir(tmp_path)) == [{"new": 0}, {"new": 1}]


def _config(tmp_path):
    import yaml
    raw = {"global": {"base_path": str(tmp_path / "run"), "seed": 91},
           "decontaminate": {"enabled": False},
           "stage3_cluster": {"annotation_enabled": False, "faiss_index_type": "Flat"},
           "stage4_sample": {"total_prompts": 4, "domain_quotas": {"math": 1.0},
                             "difficulty_quotas": {"default": {"easy": 1.0}}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_stage3_and_stage4_retries_do_not_duplicate_outputs(tmp_path, monkeypatch):
    from sft_pipeline.stages import stage3_cluster as stage3
    from sft_pipeline.stages.stage4_sample import run_stage4
    from sft_pipeline.storage import ShardedJSONLWriter
    cfg = _config(tmp_path)
    inputs = Path(cfg.stage1_collect.output_dir)
    inputs.mkdir(parents=True)
    rows = [{"prompt_id": str(i), "prompt": f"Solve equation x + {i} = 7."} for i in range(4)]
    (inputs / "part-000000.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    def embed(**kwargs):
        import pyarrow as pa
        import pyarrow.parquet as pq
        dst = Path(kwargs["output_dir"])
        pq.write_table(pa.table({"prompt_id": [r["prompt_id"] for r in rows],
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(np.eye(4, dtype=np.float16).ravel(), type=pa.float16()), 4)}),
                       dst / "embeddings_0000.parquet")
        return 4
    monkeypatch.setattr(stage3, "embed_prompts", embed)
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        stage3.run_stage3(cfg, cm)
        run_stage4(cfg, cm)
        for fn, name, out_dir in ((stage3.run_stage3, "stage3_cluster", cfg.stage3_cluster.output_dir),
                                  (run_stage4, "stage4_sample", cfg.stage4_sample.output_dir)):
            before = list(iter_jsonl_dir(out_dir))
            original = ShardedJSONLWriter.write
            calls = 0
            def fail(writer, row):
                nonlocal calls
                calls += 1
                original(writer, row)
                if calls == 2:
                    raise RuntimeError("interrupted shard write")
            with patch.object(ShardedJSONLWriter, "write", fail):
                with pytest.raises(RuntimeError, match="interrupted"):
                    fn(cfg, cm)
            assert not cm.is_stage_complete(name)
            assert list(iter_jsonl_dir(out_dir)) == before
            fn(cfg, cm)
            after = list(iter_jsonl_dir(out_dir))
            assert len(after) == len({r["prompt_id"] for r in after}) == 4


def test_sampling_report_accounts_for_empty_domains_cells_and_rounding(tmp_path):
    from sft_pipeline.stages.stage4_sample import run_stage4
    cfg = _config(tmp_path)
    cfg.stage4_sample.total_prompts = 11
    cfg.stage4_sample.domain_quotas = {"math": 0.5, "code": 0.5}
    cfg.stage4_sample.difficulty_quotas = {"default": {"easy": 0.6, "hard": 0.4}}
    directory = Path(cfg.stage3_cluster.output_dir)
    directory.mkdir(parents=True)
    row = {"prompt_id": "one", "prompt": "Solve x + 1 = 2", "domain": "math",
           "difficulty": "easy", "cluster_id": 0, "centroid_sim": 0.5}
    (directory / "part-000000.jsonl").write_text(json.dumps(row)+"\n")
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage4(cfg, cm)
    report = json.loads((Path(cfg.stage4_sample.output_dir) / "sampling_report.json").read_text())
    assert len(report["cells"]) == 4
    assert report["shortfall"] == 10
    assert report["quota_rounding_shortfall"] == 1
    assert sum(c["shortfall"] for c in report["cells"]) + report["quota_rounding_shortfall"] == 10


@pytest.mark.parametrize("failure", ["split", "configs", "empty", "fields"])
def test_partial_eval_loading_fails_with_audit(monkeypatch, failure):
    def load(repo, name, split):
        if name == "bad":
            if failure == "split":
                raise OSError("unavailable")
            if failure == "empty":
                return []
            if failure == "fields":
                return [{"wrong_field": "text"}]
        return [{"question": "What is the sum of these five whole numbers?"}]
    def names(repo):
        if failure == "configs":
            raise OSError("cannot enumerate")
        return ["good", "bad"]
    monkeypatch.setattr("datasets.load_dataset", load)
    monkeypatch.setattr("datasets.get_dataset_config_names", names)
    src = EvalDatasetSource(name="eval", source="hf_dataset", hf_repo_id="fixture",
                            hf_configs="all", splits=["test"], match_fields=["question"])
    with pytest.raises(EvalLoadError) as error:
        build_index([src], 13)
    assert any(row["status"] == "failed" for row in error.value.report)


def test_eval_row_cap_is_reported_for_unvisited_subsets(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: [
        {"question": "What is the sum of these five whole numbers?"}] * 3)
    src = EvalDatasetSource(name="eval", source="hf_dataset", hf_repo_id="fixture",
        hf_configs=["first", "second"], splits=["test"], match_fields=["question"], max_examples=1)
    index, counts = build_index([src], 13)
    assert counts == {"eval": 1}
    assert [r["status"] for r in index.loading_report] == ["capped", "skipped_limit"]


def test_single_eval_without_match_fields_reports_failure(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"wrong_field":"Missing the requested question"}\n')
    src = EvalDatasetSource(name="empty-fields", source="local_jsonl", path=str(path),
                            match_fields=["question"])
    with pytest.raises(EvalLoadError) as error:
        build_index([src], 13)
    assert error.value.report[0]["status"] == "failed"
    assert error.value.report[0]["rows"] == 1


def test_judge_uses_configured_timeout_and_surfaces_cause(monkeypatch):
    import httpx
    from sft_pipeline.filters.llm_judge import _call_judge, check_llm_judge
    seen = []
    def post(*args, **kwargs):
        seen.append(kwargs["timeout"])
        raise httpx.ReadTimeout("fixture deadline exceeded")
    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(_call_judge.retry, "sleep", lambda _: None)
    cfg = LLMJudgeConfig(sample_rate=1.0, request_timeout=211)
    with pytest.raises(RuntimeError, match="ReadTimeout: fixture deadline exceeded"):
        check_llm_judge({"prompt": "test"}, cfg)
    assert seen == [211, 211]


def test_annotation_import_fills_missing_and_empty_online(tmp_path, monkeypatch):
    from sft_pipeline.stages.stage3_cluster import run_stage3
    cfg = _config(tmp_path)
    cfg.stage3_cluster.annotation_enabled = True
    rows = [{"prompt_id": str(i), "prompt": f"Question {i}", "domain": "math",
             "difficulty": "easy", "centroid_sim": 0.75} for i in range(3)]
    for directory in (cfg.stage1_collect.output_dir, cfg.stage3_cluster.output_dir):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "part-000000.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    imported = tmp_path / "responses.jsonl"
    imported.write_text(json.dumps({"prompt_id": "0", "response": {
        "domain": "science", "difficulty": "medium"}})+"\n"+
        json.dumps({"prompt_id": "1", "response": ""})+"\n")
    def annotate(**kwargs):
        assert {r["prompt_id"] for r in kwargs["prompt_records"]} == {"1", "2"}
        return {r["prompt_id"]: {"domain": "code", "difficulty": "hard"}
                for r in kwargs["prompt_records"]}
    monkeypatch.setattr("sft_pipeline.clustering.annotator.annotate_prompts", annotate)
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage3(cfg, cm, import_annotations_path=imported)
    after = {r["prompt_id"]: r for r in iter_jsonl_dir(cfg.stage3_cluster.output_dir)}
    assert len(after) == 3 and after["0"]["domain"] == "science"
    assert after["1"]["domain"] == after["2"]["domain"] == "code"
    assert all(r["centroid_sim"] == 0.75 for r in after.values())


def test_torch_clustering_seed_is_local_and_reproducible():
    import torch
    from sft_pipeline.clustering.clusterer import _cluster_with_torch_kmeans
    vectors = np.random.default_rng(7).normal(size=(256, 32)).astype("float32")
    state = torch.random.get_rng_state().clone()
    first = _cluster_with_torch_kmeans(vectors, 8, 128, "cpu", seed=91)
    second = _cluster_with_torch_kmeans(vectors, 8, 128, "cpu", seed=91)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1], atol=1e-6)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [True, False])
async def test_annotation_validation_retries_with_feedback_and_still_fails_closed(monkeypatch, recovers):
    import asyncio
    from types import SimpleNamespace
    from sft_pipeline.clustering.annotator import _annotate_one
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        domain = "other" if recovers and len(calls) > 1 else "craft"
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"domain": domain, "difficulty": "medium"})))])
    async def no_wait(*args):
        pass
    monkeypatch.setattr("sft_pipeline.clustering.annotator.asyncio.sleep", no_wait)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pending = _annotate_one(client, "test", {"prompt_id": "craft", "prompt": "Embroidery stitches?"},
                            asyncio.Semaphore(1), 1024, 0, [0, 0], 1)
    if recovers:
        pid, annotation = await pending
        assert pid == "craft" and annotation["domain"] == "other"
        assert len(calls) == 2
    else:
        with pytest.raises(RuntimeError, match="invalid after 3 attempts"):
            await pending
        assert len(calls) == 3
    assert "never invent a domain" in calls[1]["messages"][-1]["content"]
