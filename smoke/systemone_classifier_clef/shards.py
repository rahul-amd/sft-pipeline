"""Balance fixed requests across independent replicas and verify the final merge."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(args):
    requests=[json.loads(line) for line in (ROOT/'requests.jsonl').open()]
    assert len(requests)==len({r['id'] for r in requests})==6600
    if args.action=='assign':
        loads=[0.]*args.count;assignments={}
        for row in sorted(requests,key=lambda r:(-r['full_input_tokens'],r['id'])):
            shard=min(range(args.count),key=lambda i:(loads[i],i))
            n=row['full_input_tokens'];loads[shard]+=n+n*n/131072
            assignments[row['id']]=shard
        write(ROOT/'shard-assignment.json',{'shard_count':args.count,'assignments':assignments,'estimated_costs':loads,'policy':'greedy full-input length cost n+n^2/131072; no labels or predictions used'})
        return
    predictions=[];missing=[]
    manifest=json.loads((ROOT/'experiment-manifest.json').read_text())
    config_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for row in requests:
        path=ROOT/'predictions'/(row['id'].replace(':','_')+'.json')
        if not path.exists():missing.append(row['id']);continue
        p=json.loads(path.read_text())
        assert p['id']==row['id'] and p['config_sha256']==config_hash
        assert p['payload_sha256']==row['payload_sha256']
        assert p['full_input_tokens']==row['full_input_tokens']==p['response']['usage']['input_tokens']
        assert p['response']['usage']['output_tokens']==0
        prob=p['response']['answers']['quality']['probabilities']
        assert set(prob)=={'major_defect','minor_defect','uncertain','no_defect_found'}
        assert all(0<=v<=1 for v in prob.values()) and abs(sum(prob.values())-1)<1e-4
        assert prob[p['response']['answers']['quality']['choice']]>=max(prob.values())-1e-7
        predictions.append(p)
    if missing:
        write(ROOT/'incomplete.json',{'completed':len(predictions),'missing':len(missing),'missing_ids':missing,'complete':False})
        raise SystemExit(f'{len(missing)} predictions missing; no complete report will be generated')
    with (ROOT/'predictions.jsonl').open('w') as stream:
        for p in sorted(predictions,key=lambda p:p['id']):stream.write(json.dumps(p,ensure_ascii=False)+'\n')
    assignment=json.loads((ROOT/'shard-assignment.json').read_text())
    completions=[json.loads((ROOT/f'completion-shard-{i}.json').read_text()) for i in range(assignment['shard_count'])] if assignment['shard_count']>1 else [json.loads((ROOT/'completion.json').read_text())]
    assert all(c['complete'] for c in completions)
    attempt=json.loads((ROOT/'runtime-manifest.json').read_text())
    ended=datetime.now(timezone.utc)
    wall=(ended-datetime.fromisoformat(attempt['classification_started_utc'])).total_seconds()
    new=sum(c['new_records'] for c in completions);tokens=sum(c['new_input_tokens'] for c in completions)
    result={'complete':True,'completed':6600,'target':6600,'cached':6600-new,'new_records':new,'new_input_tokens':tokens,
        'wall_seconds':wall,'records_per_second':new/wall,'input_tokens_per_second':tokens/wall,
        'updated_utc':ended.isoformat(),'shard_count':assignment['shard_count'],'shard_completions':completions,
        'timing_note':'parallel bulk interval; excludes initial model loading and six-record pilot; resume cache is explicit'}
    write(ROOT/'completion.json',result)
    for name in ['predictions.jsonl','reference_labels.jsonl']:
        with (ROOT/(name+'.gz')).open('wb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as archive:archive.write((ROOT/name).read_bytes())
    write(ROOT/'inference-verification.json',{'complete':True,'unique_predictions':6600,'full_context_matches':6600,'payload_hashes_match':True,
        'predictions_sha256':hashlib.sha256((ROOT/'predictions.jsonl').read_bytes()).hexdigest(),'total_input_tokens':sum(p['full_input_tokens'] for p in predictions)})
    print(json.dumps({k:result[k] for k in ['complete','completed','wall_seconds','shard_count','input_tokens_per_second']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['assign','merge']);parser.add_argument('--count',type=int,default=8);main(parser.parse_args())
