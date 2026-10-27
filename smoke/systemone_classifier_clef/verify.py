"""Independently check evidence identity, split coverage, and held-out statistics."""
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,f1_score,log_loss,matthews_corrcoef

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'systemone_classifier'
LABELS=['no_defect_found','minor_defect','major_defect','uncertain']


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert hashlib.sha256((ROOT/'questions.json').read_bytes()).hexdigest()==protocol['rubric_sha256']
    assert hashlib.sha256((ROOT/'calibration-split.json').read_bytes()).hexdigest()==protocol['split_sha256']
    split=json.loads((ROOT/'calibration-split.json').read_text())
    truth={r['id']:r for r in map(json.loads,(BASE/'reference_labels.jsonl').open())}
    new={r['id']:r for r in map(json.loads,(ROOT/'predictions.jsonl').open())}
    flash={r['id']:r for r in map(json.loads,(BASE/'predictions.jsonl').open())}
    assert len(new)==len(truth)==len(flash)==6600 and new.keys()==truth.keys()==flash.keys()==split.keys()
    seen=set()
    for before_line,after_line in zip((BASE/'requests.jsonl').open(),(ROOT/'requests.jsonl').open(),strict=True):
        before=json.loads(before_line);after=json.loads(after_line);seen.add(after['id'])
        assert before['id']==after['id']
        assert before['payload']['state']==after['payload']['state'] and before['payload']['questions']==after['payload']['questions']
        assert after['payload']['model']=='Cloudflare/clef'
        p=new[after['id']]
        expected=hashlib.sha256(json.dumps(after['payload'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        assert expected==after['payload_sha256']==p['payload_sha256']
        assert after['full_input_tokens']==p['response']['usage']['input_tokens'] and p['response']['usage']['output_tokens']==0
    assert len(seen)==6600
    heldout=sorted(i for i,v in split.items() if v=='heldout');calibration=set(split)-set(heldout)
    assert len(heldout)==len(calibration)==3300 and not calibration.intersection(heldout)
    counts=Counter((truth[i]['dataset'],truth[i]['source']) for i in heldout)
    assert all(n==(50 if d=='sft' else 100) for (d,_),n in counts.items())
    metrics=json.loads((ROOT/'report/metrics.json').read_text());checks=[]
    for model,source in [('flash',flash),('clef',new)]:
        fit=json.loads((ROOT/f'calibration-fit-{model}.json').read_text())
        assert not fit['heldout_used_for_fitting_or_gate'] and fit['fit_records']==3300
        assert fit['calibration_adopted']==(fit['out_of_fold_log_loss']<=0.99*fit['raw_calibration_log_loss'])
        raw=np.array([[source[i]['response']['answers']['quality']['probabilities'][label] for label in LABELS] for i in heldout])
        raw_choices=np.array([LABELS.index(source[i]['response']['answers']['quality']['choice']) for i in heldout])
        selected=raw.copy()
        if fit['calibration_adopted']:
            z=np.log(np.maximum(raw,1e-12))@np.array(fit['coefficients']).T+np.array(fit['intercepts'])
            z-=z.max(axis=1,keepdims=True);selected=np.exp(z);selected/=selected.sum(axis=1,keepdims=True)
        for dataset in ['sft','dolci','pooled']:
            mask=np.array([dataset=='pooled' or truth[i]['dataset']==dataset for i in heldout])
            y=np.array([LABELS.index(truth[i]['sol_label']) for i in heldout])[mask]
            for suffix,probs in [('raw',raw[mask]),('selected',selected[mask])]:
                prediction=probs.argmax(axis=1) if suffix=='selected' and fit['calibration_adopted'] else raw_choices[mask]
                reported=metrics['heldout'][dataset]['variants'][model+'_'+suffix]
                independent={'multiclass_MCC':matthews_corrcoef(y,prediction),'macro_F1':f1_score(y,prediction,labels=range(4),average='macro',zero_division=0),'Cohen_kappa':cohen_kappa_score(y,prediction,labels=range(4)),'accuracy':accuracy_score(y,prediction)}
                clipped=np.clip(probs,1e-12,1);sums=clipped.sum(axis=1,keepdims=True)
                # sklearn renormalizes; reported NLL uses the recorded probabilities.
                # Undo the tiny row-sum correction from finite-precision API scores.
                independent['log_loss']=log_loss(y,clipped/sums,labels=range(4))-np.log(sums).mean()
                for key,value in independent.items():assert abs(value-reported[key])<1e-9,(model,dataset,suffix,key)
                checks.append(f'{model}/{dataset}/{suffix}')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.hrefs=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.hrefs.extend(v for k,v in attrs if k=='href')
    parser=Links();parser.feed((ROOT/'report/index.html').read_text())
    for link in parser.hrefs:
        if not link.startswith(('http:','https:','#')):assert (ROOT/'report'/link).exists()
    result={'complete':True,'identical_record_evidence_and_questions':6600,'full_input_checks':6600,
        'calibration_rows':3300,'heldout_rows':3300,'disjoint_split':True,'metrics_independently_verified':checks,'report_links_valid':True,
        'metrics_sha256':hashlib.sha256((ROOT/'report/metrics.json').read_bytes()).hexdigest()}
    (ROOT/'result-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
