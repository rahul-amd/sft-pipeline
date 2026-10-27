"""Preserve frozen audit evidence and independently encode the 27B checkpoint."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/repo/smoke/systemone_classifier') if Path('/repo').exists() else ROOT.parent/'systemone_classifier'
REVISION = 'a30162b99d68a481d8bd75d38df069a0cf83df40'
os.environ['HF_HOME'] = str(ROOT/'cache/hf')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    questions = json.loads((ROOT/'questions.json').read_text())
    assert hashlib.sha256((ROOT/'questions.json').read_bytes()).hexdigest() == protocol['rubric_sha256']
    tokenizer = AutoTokenizer.from_pretrained('Cloudflare/clef', revision=REVISION, cache_dir=str(ROOT/'cache/hf/hub'), local_files_only=True)
    spec = importlib.util.spec_from_file_location('clef_reference', ROOT/'docs/joint_schema_model.py')
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    empty = reference.encode_record(tokenizer, {'state':'','questions':questions}, max_length=262144)
    overhead = len(empty.input_ids)
    old = json.loads((BASE/'experiment-manifest.json').read_text())
    truth = {r['id']:r for r in map(json.loads,(BASE/'reference_labels.jsonl').open())}
    lengths=[]; seen=set(); parity=[]
    with (ROOT/'requests.jsonl').open('w') as out, (ROOT/'reference_labels.jsonl').open('w') as labels:
        for line in (BASE/'requests.jsonl').open():
            row=json.loads(line);seen.add(row['id'])
            assert row['payload']['questions']==questions
            payload={**row['payload'],'model':'Cloudflare/clef'}
            state_tokens=len(tokenizer.encode(payload['state'],add_special_tokens=False))
            n=state_tokens+overhead
            assert n<=98304, (row['id'],n)
            updated={**row,'payload':payload,'full_input_tokens':n,'state_tokens':state_tokens,
                'payload_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
            out.write(json.dumps(updated,ensure_ascii=False)+'\n')
            labels.write(json.dumps({**truth[row['id']],'full_input_tokens':n},ensure_ascii=False)+'\n')
            lengths.append(n)
            # Keep a deterministic set for full token-ID parity with native vLLM.
            if len(parity)<3:parity.append(updated)
    assert len(lengths)==len(seen)==6600 and seen==truth.keys()
    rows=[json.loads(l) for l in (ROOT/'requests.jsonl').open()]
    for dataset in ['sft','dolci']:
        subset=sorted((r for r in rows if r['dataset']==dataset),key=lambda r:r['full_input_tokens'])
        parity.extend([subset[0],subset[len(subset)//2],subset[-1]])
    # This check runs inside the pinned container before the API is started.
    from types import SimpleNamespace
    from vllm.entrypoints.openai.decisions.strategies import ClefFlashStrategy
    from vllm.entrypoints.openai.decisions.question_types import build_question
    native=ClefFlashStrategy(SimpleNamespace(online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda:tokenizer)),engine_client=SimpleNamespace(model_config=SimpleNamespace(max_model_len=262144))))
    native.max_length=262144
    qs=[build_question(k,v['type'],v['instructions'],v['criteria'],26) for k,v in questions.items()]
    for row in parity:
        expected=reference.encode_record(tokenizer, row['payload'],max_length=262144)
        actual,_,_=native.encode(qs,row['payload']['state'])
        assert list(expected.input_ids)==actual and len(actual)==row['full_input_tokens']
    manifest={**old,'model':'Cloudflare/clef','model_revision':REVISION,
        'schema_overhead_tokens':overhead,'max_input_tokens':max(lengths),'median_input_tokens':sorted(lengths)[len(lengths)//2],
        'over_16384':sum(n>16384 for n in lengths),'total_input_tokens':sum(lengths),
        'gpu_partition':'see runtime-manifest.json and resource-authorization.json',
        'threshold_policy':'raw four-class argmax; separately gated calibration on frozen split per protocol.json',
        'calibration_protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        'native_runtime_class':'ClefFlashForJointSchema (generic Qwen3.5/Qwen3.8 joint-head implementation; checkpoint is Clef-27B)',
        'reference_encoder_parity_records':len(parity)}
    (ROOT/'experiment-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
