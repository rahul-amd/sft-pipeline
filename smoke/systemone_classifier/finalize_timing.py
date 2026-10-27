"""Combine resumed execution into an honest end-to-end bulk timing record."""
from datetime import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    completion_path = ROOT / 'completion.json'
    completion = json.loads(completion_path.read_text())
    assert completion['complete'] and completion['completed'] == 6600
    if completion.get('timing_scope') == 'end_to_end_including_interruption':
        return
    segments = json.loads((ROOT / 'run-segments.json').read_text())
    pilot = json.loads((ROOT / 'pilot-completion.json').read_text())
    predictions = [json.loads(line) for line in (ROOT / 'predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == len({p['id'] for p in predictions}) == 6600
    pilot_end = datetime.fromisoformat(pilot['updated_utc'])
    bulk = [p for p in predictions if datetime.fromisoformat(p['completed_utc']) > pilot_end]
    assert len(bulk) == 6594
    wall = (datetime.fromisoformat(completion['updated_utc']) - datetime.fromisoformat(segments['bulk_start_utc'])).total_seconds()
    tokens = sum(p['full_input_tokens'] for p in bulk)
    manifest = json.loads((ROOT / 'experiment-manifest.json').read_text())
    assert tokens + pilot['new_input_tokens'] == manifest['total_input_tokens']
    latencies = sorted(p['latency_seconds'] for p in bulk)
    (ROOT / 'completion-resumed-segment.json').write_text(json.dumps(completion, indent=2) + '\n')
    result = dict(completion)
    result.update(
        cached=6, new_records=len(bulk), new_input_tokens=tokens,
        wall_seconds=wall, records_per_second=len(bulk)/wall,
        input_tokens_per_second=tokens/wall,
        latency_seconds={'median': statistics.median(latencies), 'p95': latencies[int(len(latencies)*.95)]},
        timing_scope='end_to_end_including_interruption',
        bulk_start_utc=segments['bulk_start_utc'],
        operational_interruptions=[segments['interruption']],
        resumed_segment=completion,
        latency_note='Successful request latency only; aggregate wall time includes the interruption, restart, and failed requests.',
    )
    temporary = completion_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(completion_path)
    print(json.dumps({k:result[k] for k in ['completed','cached','wall_seconds','input_tokens_per_second','timing_scope']}))


if __name__ == '__main__':
    main()
