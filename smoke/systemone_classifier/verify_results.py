"""Independently verify coverage, frozen inputs, full context, and reported metrics."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
LABELS = ['no_defect_found', 'minor_defect', 'major_defect', 'uncertain']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / 'experiment-manifest.json').read_text())
    assert sha(ROOT / 'questions.json') == manifest['questions_sha256']
    for dataset, folder in [('sft', 'source_quality_audit'), ('dolci', 'dolci_quality_audit')]:
        source = ROOT.parent / folder
        assert sha(source / 'samples.jsonl') == manifest['input_hashes'][dataset]['samples_sha256']
        assert sha(source / 'verdicts.jsonl') == manifest['input_hashes'][dataset]['verdicts_sha256']
    predictions = [json.loads(line) for line in (ROOT / 'predictions.jsonl').open()]
    truth = [json.loads(line) for line in (ROOT / 'reference_labels.jsonl').open()]
    pred = {p['id']: p for p in predictions}
    reference = {r['id']: r for r in truth}
    assert len(predictions) == len(pred) == len(truth) == len(reference) == 6600
    assert pred.keys() == reference.keys()
    request_ids = set()
    for line in (ROOT / 'requests.jsonl').open():
        request = json.loads(line)
        request_ids.add(request['id'])
        p = pred[request['id']]
        assert hashlib.sha256(json.dumps(request['payload'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == request['payload_sha256'] == p['payload_sha256']
        assert p['full_input_tokens'] == request['full_input_tokens'] == p['response']['usage']['input_tokens']
        assert p['response']['usage']['output_tokens'] == 0
        answer = p['response']['answers']['quality']
        probabilities = answer['probabilities']
        assert set(probabilities) == set(LABELS)
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-4
        assert probabilities[answer['choice']] >= max(probabilities.values()) - 1e-7
    assert request_ids == pred.keys()
    assert sum(p['full_input_tokens'] for p in predictions) == manifest['total_input_tokens']
    report = json.loads((ROOT / 'report/metrics.json').read_text())
    verified = []
    for dataset in ['sft', 'dolci', 'pooled']:
        rows = [r for r in truth if dataset == 'pooled' or r['dataset'] == dataset]
        y = [r['sol_label'] for r in rows]
        z = [pred[r['id']]['response']['answers']['quality']['choice'] for r in rows]
        modes = [(None, report['cohorts'][dataset], 'equal_record')]
        if dataset != 'pooled':
            folder = 'source_quality_audit' if dataset == 'sft' else 'dolci_quality_audit'
            populations = {s['source']: s['population'] for s in json.loads((ROOT.parent / folder / 'sources.json').read_text())}
            counts = Counter(r['source'] for r in rows)
            assert set(counts.values()) == ({100} if dataset == 'sft' else {200})
            weights = [populations[r['source']] / counts[r['source']] for r in rows]
            modes.append((weights, report['cohorts'][dataset]['population_weighted'], 'population_weighted'))
        for weights, reported, mode in modes:
            checks = {
                'multiclass_MCC': matthews_corrcoef(y, z, sample_weight=weights),
                'Cohen_kappa': cohen_kappa_score(y, z, labels=LABELS, sample_weight=weights),
                'accuracy': accuracy_score(y, z, sample_weight=weights),
                'macro_F1': f1_score(y, z, labels=LABELS, average='macro', sample_weight=weights, zero_division=0),
            }
            for metric, expected in checks.items():
                assert abs(expected - reported[metric]) < 1e-10, (dataset, mode, metric)
            verified.append(f'{dataset}/{mode}')
    output = {
        'complete': True, 'unique_predictions': len(pred), 'full_context_matches': len(pred),
        'frozen_question_and_audit_hashes_match': True, 'metrics_match_sklearn': verified,
        'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
        'predictions_sha256': sha(ROOT / 'predictions.jsonl'),
        'metrics_sha256': sha(ROOT / 'report/metrics.json'),
    }
    (ROOT / 'result-verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
