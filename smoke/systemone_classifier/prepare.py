"""Freeze identical audit content and four-class questions without Sol label leakage."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

from transformers import AutoTokenizer
from vllm.entrypoints.openai.decisions.strategies import ClefFlashStrategy
from vllm.entrypoints.openai.decisions.question_types import build_question

ROOT = Path('/work')
REPO = Path('/repo')
MODEL = 'Cloudflare/clef-flash'
REVISION = 'f0b7d3cd65296abfe459e6c228b618f5d8b26d2f'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    question_dict = json.loads((ROOT/'questions.json').read_text())
    # Use the same installed vLLM encoder to count exact prefix/schema overhead.
    # The schema parser is loaded below after inspecting the installed API.
    questions = [build_question(k,v['type'],v['instructions'],v['criteria'],26) for k,v in question_dict.items()]
    fake_renderer = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda:tokenizer))
    strategy = ClefFlashStrategy(SimpleNamespace(online_renderer=fake_renderer,engine_client=SimpleNamespace(model_config=SimpleNamespace(max_model_len=262144))))
    strategy.max_length = 262144
    overhead = len(strategy.encode(questions,'')[0])
    spec = importlib.util.spec_from_file_location('dolci_adapter',REPO/'smoke/dolci_quality_audit/review.py')
    adapter = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    items,truth,inputs = [],[],{}
    for dataset,dirname,expected in [('sft','source_quality_audit',3400),('dolci','dolci_quality_audit',3200)]:
        source = REPO/'smoke'/dirname
        path=source/'samples.jsonl'
        raw=path.read_bytes() if path.exists() else gzip.decompress((source/'samples.jsonl.gz').read_bytes())
        samples=[json.loads(line) for line in raw.splitlines()]
        verdict_bytes=(source/'verdicts.jsonl').read_bytes()
        verdicts={v['audit_id']:v for v in map(json.loads,verdict_bytes.splitlines())}
        assert len(samples)==len(verdicts)==expected
        assert sha(raw)==json.loads((source/'sample-manifest.json').read_text())['samples_sha256']
        inputs[dataset]={'samples_sha256':sha(raw),'verdicts_sha256':sha(verdict_bytes)}
        for s in samples:
            if dataset=='sft':
                content={k:s['record'].get(k,'') for k in ['prompt','reasoning','answer','response']}
            else:
                content={'messages':adapter.to_review(s)['messages']}
            state=json.dumps(content,ensure_ascii=False,separators=(',',':'),sort_keys=True)
            state_tokens=len(tokenizer.encode(state,add_special_tokens=False))
            full_tokens=state_tokens+overhead
            assert full_tokens<=262144, (dataset,s['audit_id'],full_tokens)
            uid=dataset+':'+s['audit_id']
            payload={'model':MODEL,'state':state,'questions':question_dict}
            items.append({'id':uid,'dataset':dataset,'audit_id':s['audit_id'],
                'full_input_tokens':full_tokens,'state_tokens':state_tokens,'payload':payload,
                'payload_sha256':sha(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode())})
            v=verdicts[s['audit_id']]
            truth.append({'id':uid,'dataset':dataset,'audit_id':s['audit_id'],'source':s['source'],
                'sol_label':v['overall'],'sol_dimensions':v['dimensions'],'full_input_tokens':full_tokens})
        print(json.dumps({'dataset':dataset,'prepared':expected}),flush=True)
    random.Random(20261027).shuffle(items)
    for name,rows in [('requests.jsonl',items),('reference_labels.jsonl',truth)]:
        (ROOT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    lengths=sorted(x['full_input_tokens'] for x in items)
    manifest={'model':MODEL,'model_revision':REVISION,'samples':len(items),'input_hashes':inputs,
        'questions_sha256':sha((ROOT/'questions.json').read_bytes()),'schema_overhead_tokens':overhead,
        'max_input_tokens':max(lengths),'median_input_tokens':lengths[len(lengths)//2],
        'over_16384':sum(n>16384 for n in lengths),'total_input_tokens':sum(lengths),
        'labels':['major_defect','minor_defect','uncertain','no_defect_found'],
        'primary_metrics':['multiclass_MCC','Cohen_kappa','accuracy','macro_F1','confusion_matrix'],
        'prompt_policy':'Frozen before inference; identical class definitions and original audit evidence; no source names or Sol labels in requests',
        'context_policy':'Full inputs; local adapter lifts reference 16k cap to model position limit; no truncation',
        'threshold_policy':'Four-class argmax; no fitting, calibration, or threshold tuning on these samples',
        'gpu_partition':'hissu','order_seed':20261027}
    (ROOT/'experiment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
