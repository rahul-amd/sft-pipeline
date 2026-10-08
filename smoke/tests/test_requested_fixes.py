"""Behavioral coverage for formatting, global semantic dedup and sandbox failures."""
import json
import sys

import numpy as np
import pytest

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import CodeFilterConfig, PipelineConfig, Stage4Config, Stage6Config
from sft_pipeline.filters.code_verifier import _run_e2b, check_code
from sft_pipeline.filters.docker_sandbox import SandboxError, run_docker
from sft_pipeline.stages.stage1_collect import _extract_prompt, _normalize, run_stage1
from sft_pipeline.stages.stage2_generate import run_stage2
from sft_pipeline.stages.stage4_sample import _SemanticDeduper, run_stage4
from sft_pipeline.storage import iter_jsonl_dir


CODE_PROMPT = '    if ready:\n\tprint("a  b")\n\n    # Keep ℌ and ① literally\n'


@pytest.mark.parametrize("format", ["text", "openai", "sharegpt", "json", "dict"])
def test_multiline_prompt_extraction_preserves_format(format):
    value = {
        "text": CODE_PROMPT,
        "openai": [{"role": "user", "content": CODE_PROMPT}],
        "sharegpt": [{"from": "human", "value": CODE_PROMPT}],
        "json": json.dumps([{"role": "user", "content": CODE_PROMPT}]),
        "dict": {"text": CODE_PROMPT},
    }[format]
    assert _normalize(_extract_prompt(value)) == CODE_PROMPT


def test_prompt_ids_preserve_semantically_significant_whitespace():
    assert prompt_id(_normalize(CODE_PROMPT)) != prompt_id(_normalize(CODE_PROMPT.replace('\t', '')))
    assert _normalize(CODE_PROMPT.replace('\n', '\r\n')) == CODE_PROMPT


def test_both_prompt_stages_preserve_code(tmp_path, monkeypatch):
    source = tmp_path / "prompts.jsonl"
    source.write_text(json.dumps({"prompt": CODE_PROMPT}) + "\n")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A source document discussing the behavior of Python programs. " * 4)
    cfg = PipelineConfig.model_validate({
        "global": {"base_path": str(tmp_path)},
        "stage1_collect": {"output_dir": str(tmp_path / "s1"), "datasets": [
            {"source": "local_jsonl", "path": str(source)}]},
        "stage2_generate": {"output_dir": str(tmp_path / "s2"), "corpora": [
            {"source": "local", "path": str(corpus)}]},
    })
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._call_generator",
                        lambda *args, **kwargs: [CODE_PROMPT])
    with CheckpointManager(tmp_path / "checks.duckdb") as cm:
        run_stage1(cfg, cm)
        run_stage2(cfg, cm)
    for path in (tmp_path / "s1", tmp_path / "s2"):
        records = list(iter_jsonl_dir(path))
        assert len(records) == 1
        assert records[0]["prompt"] == CODE_PROMPT


def test_exact_global_dedup_backfills_across_batches_and_cells():
    vectors = np.array([[1, 0, 0], [5, 0.01, 0], [0, 3, 0], [0, 0, 9]], dtype=np.float16)
    dedup = _SemanticDeduper(vectors, dict(zip('abcd', range(4))), .92, batch_size=1)
    assert dedup.select(['a'], 1) == ['a']
    assert dedup.select(['b', 'c', 'd'], 2) == ['c', 'd']
    assert dedup.rejected == 1


def test_dedup_is_greedy_against_retained_not_rejected_prompts():
    # A~B, B~C, but A !~ C: B's rejection must not suppress C.
    angles = np.deg2rad([0, 20, 40])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    dedup = _SemanticDeduper(vectors, dict(zip('abc', range(3))), .9)
    assert dedup.select(list('abc'), 3) == ['a', 'c']


def test_identical_embeddings_reject_at_threshold_one():
    vectors = np.array([[1., 2., 3.], [1., 2., 3.]])
    dedup = _SemanticDeduper(vectors, {'a': 0, 'b': 1}, 1.0, batch_size=1)
    assert dedup.select(['a', 'b'], 2) == ['a']


@pytest.mark.parametrize('bad', [[0., 0.], [float('nan'), 1], [float('inf'), 1]])
def test_invalid_embeddings_fail_instead_of_bypassing_dedup(bad):
    dedup = _SemanticDeduper(np.array([bad]), {'a': 0}, .92)
    with pytest.raises(ValueError, match='finite, nonzero'):
        dedup.select(['a'], 1)


@pytest.mark.parametrize('enabled', [False, True])
def test_stage4_dedup_crosses_clusters_domains_and_refills(tmp_path, enabled):
    import pyarrow as pa
    import pyarrow.parquet as pq

    s3 = tmp_path / 's3'
    emb = s3 / 'embeddings'
    emb.mkdir(parents=True)
    records = [
        {'prompt_id': p, 'prompt': p, 'domain': domain, 'difficulty': 'easy',
         'cluster_id': i, 'centroid_sim': 1.0}
        for i, (p, domain) in enumerate([('a', 'math'), ('b', 'code'), ('c', 'code')])
    ]
    (s3 / 'part-000000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    vectors = np.array([[1, 0], [1, .01], [0, 1]], dtype=np.float16)
    pq.write_table(pa.table({'prompt_id': list('abc'), 'embedding':
        pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), type=pa.float16()), 2)}),
        emb / 'embeddings_0000.parquet')
    cfg = PipelineConfig.model_validate({
        'global': {'device': 'cpu'},
        'stage3_cluster': {'output_dir': str(s3), 'embeddings_dir': str(emb)},
        'stage4_sample': {'output_dir': str(tmp_path / 's4'), 'total_prompts': 4,
                         'semantic_dedup_enabled': enabled,
                         'domain_quotas': {'math': .5, 'code': .5},
                         'difficulty_quotas': {'default': {'easy': 1.}}},
    })
    with CheckpointManager(tmp_path / 'checks.duckdb') as cm:
        run_stage4(cfg, cm)
    expected = {'a', 'c'} if enabled else {'a', 'b', 'c'}
    assert {r['prompt_id'] for r in iter_jsonl_dir(tmp_path / 's4')} == expected
    report = json.loads((tmp_path / 's4/sampling_report.json').read_text())
    assert report['semantic_dedup_enabled'] is enabled
    assert report['duplicate_candidates'] == int(enabled)
    assert report['cells'][1]['shortfall'] == int(enabled)


@pytest.mark.parametrize('centroid_available', [False, True])
def test_default_sampling_needs_no_embeddings_or_deduper(tmp_path, monkeypatch, centroid_available):
    assert Stage4Config().semantic_dedup_enabled is False
    source = tmp_path / 's3'
    source.mkdir()
    records = [{'prompt_id': p, 'prompt': p, 'domain': 'math', 'difficulty': 'easy',
                'cluster_id': 0, **({'centroid_sim': .5} if centroid_available else {})}
               for p in ('a', 'b')]
    (source / 'part-000000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    if centroid_available:
        # Cached embeddings must not be read just for disabled deduplication.
        (source / 'embeddings').mkdir()
        (source / 'embeddings/embeddings_0000.parquet').write_bytes(b'unreadable')
    def unexpected(*args, **kwargs):
        raise AssertionError('Disabled semantic deduplication performed embedding work')
    monkeypatch.setattr('sft_pipeline.stages.stage4_sample.load_embeddings', unexpected)
    monkeypatch.setattr('sft_pipeline.stages.stage4_sample._SemanticDeduper', unexpected)
    cfg = PipelineConfig.model_validate({
        'stage3_cluster': {'output_dir': str(source), 'embeddings_dir': str(source / 'embeddings')},
        'stage4_sample': {'output_dir': str(tmp_path / 's4'), 'total_prompts': 2,
                         'domain_quotas': {'math': 1.0},
                         'difficulty_quotas': {'default': {'easy': 1.0}}},
    })
    with CheckpointManager(tmp_path / 'checks.duckdb') as cm:
        run_stage4(cfg, cm)
    assert len(list(iter_jsonl_dir(tmp_path / 's4'))) == 2


def test_stage6_has_no_math_filter():
    assert 'math' not in Stage6Config.model_fields


def test_docker_missing_image_fails_closed(monkeypatch):
    import subprocess
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '',
            'No such image' if 'create' in command else 'No such container')
    monkeypatch.setattr(subprocess, 'run', invoke)
    with pytest.raises(SandboxError, match='Cannot create'):
        run_docker('print(42)', CodeFilterConfig())
    assert all(c[0] == 'docker' for c in calls)


def test_e2b_missing_package_never_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, 'e2b_code_interpreter', None)
    with pytest.raises(RuntimeError, match='not installed'):
        _run_e2b('print(42)', 3)


def test_default_code_backend_is_isolated():
    assert CodeFilterConfig().sandbox == 'docker'
    result = check_code({'answer': '```python\n' + 'x=1\n' * 20 + '```'},
                        CodeFilterConfig(max_code_bytes=10))
    assert not result.passed and result.reason == 'code_too_large'
