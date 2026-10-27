"""Query Clef's four-class joint head, checking that every full input was read."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import aiohttp

ROOT = Path(__file__).resolve().parent
LABELS = {'major_defect','minor_defect','uncertain','no_defect_found'}


def atomic(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


async def main(args):
    items=[json.loads(line) for line in (ROOT/'requests.jsonl').read_text().splitlines()]
    assert len(items)==len({r['id'] for r in items})==6600
    manifest=json.loads((ROOT/'experiment-manifest.json').read_text())
    config_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if args.shard_count>1 and not args.pilot:
        assignment=json.loads((ROOT/'shard-assignment.json').read_text())
        assert assignment['shard_count']==args.shard_count
        items=[r for r in items if assignment['assignments'][r['id']]==args.shard_index]
    suffix=f'-shard-{args.shard_index}' if args.shard_count>1 and not args.pilot else ''
    if args.pilot:
        selected=[]
        for dataset in ['sft','dolci']:
            subset=sorted((r for r in items if r['dataset']==dataset),key=lambda r:r['full_input_tokens'])
            selected.extend([subset[0],subset[len(subset)//2],subset[-1]])
        items=selected
    directory=ROOT/'predictions';directory.mkdir(exist_ok=True)
    errors=[];completed=0;cached=0;tokens=0;elapsed=[]
    started=time.perf_counter()
    semaphore=asyncio.Semaphore(args.concurrency)
    timeout=aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout,trust_env=False) as session:
        async def one(item):
            nonlocal completed,cached,tokens
            path=directory/(item['id'].replace(':','_')+'.json')
            if path.exists():
                previous=json.loads(path.read_text())
                assert previous['payload_sha256']==item['payload_sha256'] and previous['config_sha256']==config_hash
                assert previous['full_input_tokens']==previous['response']['usage']['input_tokens']
                completed+=1;cached+=1
                return
            async with semaphore:
                for attempt in range(3):
                    begin=time.perf_counter()
                    try:
                        async with session.post(args.url.rstrip('/')+'/v1/systemone',json=item['payload']) as response:
                            text=await response.text()
                            if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {text[:1500]}')
                        result=json.loads(text)
                        answer=result['answers']['quality'];prob=answer['probabilities']
                        assert set(prob)==LABELS and answer['choice'] in LABELS
                        assert all(isinstance(p,(float,int)) and 0<=p<=1 for p in prob.values())
                        assert abs(sum(prob.values())-1)<0.0001
                        assert prob[answer['choice']]>=max(prob.values())-1e-7
                        assert result['usage']['input_tokens']==item['full_input_tokens'], 'Server token count differs: input was truncated or encoding changed'
                        assert result['usage']['output_tokens']==0, 'Expected a pooling decision, not generated text'
                        duration=time.perf_counter()-begin
                        atomic(path,{'id':item['id'],'dataset':item['dataset'],'audit_id':item['audit_id'],
                            'payload_sha256':item['payload_sha256'],'config_sha256':config_hash,
                            'full_input_tokens':item['full_input_tokens'],'latency_seconds':duration,
                            'attempt':attempt+1,'completed_utc':datetime.now(timezone.utc).isoformat(),
                            'response':result})
                        completed+=1;tokens+=item['full_input_tokens'];elapsed.append(duration)
                        if completed%50==0 or args.pilot:
                            progress={'completed':completed,'target':len(items),'cached':cached,'failed':len(errors),
                                'elapsed_seconds':time.perf_counter()-started,'new_input_tokens':tokens,
                                'updated_utc':datetime.now(timezone.utc).isoformat(),'pilot':args.pilot}
                            atomic(ROOT/('progress'+suffix+'.json'),progress);print(json.dumps(progress),flush=True)
                        return
                    except Exception as exc:
                        if attempt==2:
                            errors.append({'id':item['id'],'error':str(exc)})
                            print(json.dumps(errors[-1]),flush=True)
                            raise RuntimeError('Stopping this shard after repeated request failure') from exc
                        else:await asyncio.sleep(2**attempt)
        tasks=[asyncio.create_task(one(r)) for r in items]
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            if not errors:errors.append({'error':'Unhandled validation failure; see log'})
    duration=time.perf_counter()-started
    result={'complete':not errors,'pilot':args.pilot,'completed':completed,'target':len(items),'cached':cached,
        'errors':errors,'wall_seconds':duration,'new_input_tokens':tokens,'new_records':len(elapsed),
        'records_per_second':len(elapsed)/duration,'input_tokens_per_second':tokens/duration,
        'latency_seconds':{'median':sorted(elapsed)[len(elapsed)//2] if elapsed else None,
                           'p95':sorted(elapsed)[min(len(elapsed)-1,int(len(elapsed)*.95))] if elapsed else None},
        'concurrency':args.concurrency,'shard_index':args.shard_index,'shard_count':args.shard_count,'updated_utc':datetime.now(timezone.utc).isoformat()}
    atomic(ROOT/('pilot-completion.json' if args.pilot else 'completion'+suffix+'.json'),result)
    if errors:raise SystemExit('Classification failures; no missing records are treated as passes')
    if not args.pilot:
        with (ROOT/('predictions'+suffix+'.jsonl')).open('w') as stream:
            for item in sorted(items,key=lambda r:r['id']):
                prediction=json.loads((directory/(item['id'].replace(':','_')+'.json')).read_text())
                stream.write(json.dumps(prediction,ensure_ascii=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    asyncio.run(main(parser.parse_args()))
