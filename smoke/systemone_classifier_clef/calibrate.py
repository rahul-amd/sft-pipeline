"""Frozen calibration protocol with a source-stratified, untouched fitting holdout."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent
BASE=Path('/repo/smoke/systemone_classifier') if Path('/repo').exists() else ROOT.parent/'systemone_classifier'
sys.path.insert(0,str(BASE))
from analyze import matrix_metrics

LABELS=['no_defect_found','minor_defect','major_defect','uncertain']
EPS=1e-12


def read_jsonl(path):
    data=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
    return [json.loads(line) for line in data.splitlines()]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def scores(y,p,prediction=None):
    prediction=p.argmax(axis=1) if prediction is None else prediction
    cm=np.bincount(4*y+prediction,minlength=16).reshape(4,4).astype(float)
    result=matrix_metrics(cm)
    result['n']=len(y)
    result['log_loss']=float(np.mean(-np.log(np.clip(p[np.arange(len(y)),y],EPS,1))))
    result['Brier']=float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1)))
    confidence=p.max(axis=1);correct=prediction==y
    bins=np.minimum((15*confidence).astype(int),14)
    result['ECE_15_bins']=float(sum(np.sum(bins==b)/len(y)*abs(np.mean(correct[bins==b])-np.mean(confidence[bins==b])) for b in range(15) if np.any(bins==b)))
    result['major_defect_recall']=result['per_class']['major_defect']['recall']
    keep=prediction==0
    result['no_defect_decisions']=int(keep.sum())
    result['major_fraction_in_no_defect']=float(np.mean(y[keep]==2)) if keep.any() else None
    result['majority_baseline_accuracy']=float(np.bincount(y,minlength=4).max()/len(y))
    certain=y!=3
    result['major_AUROC_excluding_sol_uncertain']=float(roc_auc_score(y[certain]==2,p[certain,2])) if len(np.unique(y[certain]==2))==2 else None
    return result


def fit_calibrator(name,ids,y,p,calibration):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    fingerprint=hashlib.sha256(json.dumps({'ids':[ids[i] for i in np.flatnonzero(calibration)],'labels':y[calibration].tolist(),'probabilities':p[calibration].tolist(),'protocol':protocol},sort_keys=True).encode()).hexdigest()
    path=ROOT/f'calibration-fit-{name}.json'
    if path.exists():
        fitted=json.loads(path.read_text())
        assert fitted['fit_fingerprint']==fingerprint
        return fitted
    X=np.log(np.clip(p[calibration],EPS,1));target=y[calibration]
    assert set(target)==set(range(4))
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261028)
    out=np.empty_like(X)
    for train,valid in folds.split(X,target):
        model=LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',random_state=20261028)
        model.fit(X[train],target[train]);assert np.array_equal(model.classes_,np.arange(4))
        assert model.n_iter_.max()<2000
        out[valid]=model.predict_proba(X[valid])
    raw_nll=float(np.mean(-np.log(np.clip(p[calibration][np.arange(len(target)),target],EPS,1))))
    oof_nll=float(np.mean(-np.log(np.clip(out[np.arange(len(target)),target],EPS,1))))
    adopted=oof_nll<=0.99*raw_nll
    fitted={'model':name,'fit_fingerprint':fingerprint,'fit_records':int(calibration.sum()),'classes':LABELS,
        'method':'multinomial logistic regression on clipped log probabilities, L2 C=1.0',
        'probability_clip':EPS,'cv_folds':5,'seed':20261028,'raw_calibration_log_loss':raw_nll,
        'out_of_fold_log_loss':oof_nll,'relative_oof_log_loss_improvement':1-oof_nll/raw_nll,
        'calibration_adopted':adopted,'gate':'at least 1% relative out-of-fold log-loss improvement',
        'heldout_used_for_fitting_or_gate':False}
    if adopted:
        model=LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',random_state=20261028)
        model.fit(X,target);assert model.n_iter_.max()<2000
        fitted.update(coefficients=model.coef_.tolist(),intercepts=model.intercept_.tolist())
        portable=softmax(X@model.coef_.T+model.intercept_,axis=1)
        assert np.allclose(portable,model.predict_proba(X),atol=1e-12)
    write(path,fitted)
    return fitted


def apply_calibrator(fitted,p):
    if not fitted['calibration_adopted']:return p.copy()
    return softmax(np.log(np.clip(p,EPS,1))@np.asarray(fitted['coefficients']).T+np.asarray(fitted['intercepts']),axis=1)


def paired_bootstrap(rows,y,variants,repetitions=1000,decisions=None):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['dataset'],row['source'])].append(i)
    groups=[np.asarray(v) for v in groups.values()]
    rng=np.random.default_rng(20261028)
    encoded={name:4*y+(p.argmax(axis=1) if decisions is None else decisions[name]) for name,p in variants.items()}
    nll={name:-np.log(np.clip(p[np.arange(len(y)),y],EPS,1)) for name,p in variants.items()}
    values={name:defaultdict(list) for name in variants}
    comparisons=[('clef_raw','flash_raw'),('clef_selected','flash_selected'),('clef_selected','clef_raw'),('flash_selected','flash_raw')]
    changes={f'{a}_minus_{b}':defaultdict(list) for a,b in comparisons}
    for _ in range(repetitions):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);stats={}
        for name in variants:
            cm=np.bincount(encoded[name][idx],minlength=16).reshape(4,4).astype(float)
            m=matrix_metrics(cm);m['log_loss']=float(nll[name][idx].mean())
            stats[name]={key:m[key] for key in ['multiclass_MCC','macro_F1','accuracy','log_loss']}
            for key,value in stats[name].items():values[name][key].append(value)
        for a,b in comparisons:
            for key in stats[a]:changes[f'{a}_minus_{b}'][key].append(stats[a][key]-stats[b][key])
    intervals=lambda mapping:{name:{key:np.quantile(v,[.025,.975]).tolist() for key,v in metrics.items()} for name,metrics in mapping.items()}
    return {'metrics_95':intervals(values),'paired_differences_95':intervals(changes)}


def main(args):
    rows=sorted(read_jsonl(BASE/'reference_labels.jsonl'),key=lambda r:r['id'])
    assert len(rows)==len({r['id'] for r in rows})==6600
    ids=[r['id'] for r in rows];y=np.array([LABELS.index(r['sol_label']) for r in rows])
    split=json.loads((ROOT/'calibration-split.json').read_text())
    assert set(split)==set(ids)
    calibration=np.array([split[i]=='calibration' for i in ids]);heldout=~calibration
    assert calibration.sum()==heldout.sum()==3300
    raw={};raw_choices={};fitted={};selected={};tie_audit={}
    for name,folder in [('flash',BASE),('clef',ROOT)]:
        if args.baseline_only and name=='clef':break
        predictions=read_jsonl(folder/'predictions.jsonl');mapping={p['id']:p for p in predictions}
        assert len(predictions)==len(mapping)==6600 and set(mapping)==set(ids)
        for record in predictions:
            assert record['full_input_tokens']==record['response']['usage']['input_tokens']
            assert record['response']['usage']['output_tokens']==0
        p=np.array([[mapping[i]['response']['answers']['quality']['probabilities'][label] for label in LABELS] for i in ids])
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1,atol=1e-5)
        raw[name]=p
        raw_choices[name]=np.array([LABELS.index(mapping[i]['response']['answers']['quality']['choice']) for i in ids])
        assert np.allclose(p[np.arange(len(p)),raw_choices[name]],p.max(axis=1),atol=1e-7,rtol=0)
        differing=np.flatnonzero(raw_choices[name]!=p.argmax(axis=1))
        assert all(p[i,raw_choices[name][i]]==p[i].max() for i in differing)
        tie_audit[name]={'api_choice_differs_from_label_order_argmax':len(differing),'ids':[ids[i] for i in differing],
            'policy':'Preserve the API choice for raw decisions, including exact probability ties; calibrated decisions use argmax.'}
        fitted[name]=fit_calibrator(name,ids,y,p,calibration)
        if not args.baseline_only:selected[name]=apply_calibrator(fitted[name],p[heldout])
    if args.baseline_only:
        print(json.dumps({'flash_calibration_diagnostics':fitted['flash'],'heldout_not_evaluated':True},indent=2));return
    old=json.loads((BASE/'report/metrics.json').read_text())
    for dataset in ['sft','dolci']:
        mask=np.array([r['dataset']==dataset for r in rows]);m=scores(y[mask],raw['flash'][mask],raw_choices['flash'][mask])
        for key in ['multiclass_MCC','Cohen_kappa','macro_F1','accuracy']:
            assert abs(m[key]-old['cohorts'][dataset][key])<1e-10
    hold_rows=[r for r,h in zip(rows,heldout) if h];hold_y=y[heldout]
    variants={name+'_raw':p[heldout] for name,p in raw.items()}
    variants.update({name+'_selected':p for name,p in selected.items()})
    decisions={name+'_raw':choice[heldout] for name,choice in raw_choices.items()}
    decisions.update({name+'_selected':p.argmax(axis=1) if fitted[name]['calibration_adopted'] else raw_choices[name][heldout] for name,p in selected.items()})
    result={'complete':True,'labels':LABELS,'protocol':json.loads((ROOT/'protocol.json').read_text()),
        'fitted_calibrators':fitted,'raw_full':{},'heldout':{},'by_source':[],
        'calibration_rows':3300,'heldout_rows':3300,'heldout_labels_used_for_fit_or_gate':False,'raw_decision_ties':tie_audit}
    for dataset in ['sft','dolci','pooled']:
        mask=np.array([dataset=='pooled' or r['dataset']==dataset for r in rows])
        result['raw_full'][dataset]={name:scores(y[mask],p[mask],raw_choices[name][mask]) for name,p in raw.items()}
        hm=np.array([dataset=='pooled' or r['dataset']==dataset for r in hold_rows])
        subset=[r for r,m in zip(hold_rows,hm) if m]
        result['heldout'][dataset]={'variants':{name:scores(hold_y[hm],p[hm],decisions[name][hm]) for name,p in variants.items()},
            'bootstrap':paired_bootstrap(subset,hold_y[hm],{name:p[hm] for name,p in variants.items()},decisions={name:d[hm] for name,d in decisions.items()})}
    for key in sorted({(r['dataset'],r['source']) for r in hold_rows}):
        mask=np.array([(r['dataset'],r['source'])==key for r in hold_rows])
        result['by_source'].append({'dataset':key[0],'source':key[1],'variants':{name:scores(hold_y[mask],p[mask],decisions[name][mask]) for name,p in variants.items()}})
    report=ROOT/'report';report.mkdir(exist_ok=True)
    write(report/'metrics.json',result)
    with (report/'heldout_predictions.csv').open('w',newline='') as stream:
        fields=['id','dataset','source','sol_label']+[f'{name}_decision' for name in variants]+[f'{name}_{label}' for name in variants for label in LABELS]
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for i,r in enumerate(hold_rows):writer.writerow({**{k:r[k] for k in fields if k in r},**{f'{name}_decision':LABELS[d[i]] for name,d in decisions.items()},**{f'{name}_{label}':float(p[i,j]) for name,p in variants.items() for j,label in enumerate(LABELS)}})
    from render_report import render
    render(result)
    print(json.dumps({d:{name:{key:m[key] for key in ['multiclass_MCC','macro_F1','accuracy','major_defect_recall']} for name,m in result['heldout'][d]['variants'].items()} for d in ['sft','dolci']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');main(parser.parse_args())
