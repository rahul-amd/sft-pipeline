"""Four-class agreement with Sol; Sol is a reference reviewer, not ground truth."""
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report'
LABELS=['no_defect_found','minor_defect','major_defect','uncertain']
DISPLAY=['No defect found','Minor defect','Major defect','Uncertain']


def matrix_metrics(cm):
    n=float(cm.sum());diagonal=np.diag(cm);true=cm.sum(axis=1);pred=cm.sum(axis=0)
    observed=float(diagonal.sum()/n)
    expected=float(np.dot(true,pred)/(n*n))
    denominator=math.sqrt(max(0,(n*n-float(np.dot(pred,pred)))*(n*n-float(np.dot(true,true)))))
    mcc=(float(diagonal.sum())*n-float(np.dot(true,pred)))/denominator if denominator else 0.0
    precision=np.divide(diagonal,pred,out=np.zeros(4),where=pred>0)
    recall=np.divide(diagonal,true,out=np.zeros(4),where=true>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=(precision+recall)>0)
    return {'accuracy':observed,'multiclass_MCC':mcc,'Cohen_kappa':(observed-expected)/(1-expected) if expected<1 else 0.,
        'macro_F1':float(f1.mean()),'balanced_accuracy':float(recall[true>0].mean()),
        'confusion_matrix':cm.tolist(),
        'per_class':{label:{'precision':float(precision[i]),'recall':float(recall[i]),'F1':float(f1[i]),'support':float(true[i])} for i,label in enumerate(LABELS)}}


def metrics(rows,weights=None):
    truth=np.array([LABELS.index(r['sol_label']) for r in rows]);pred=np.array([LABELS.index(r['clef_label']) for r in rows])
    cm=np.bincount(4*truth+pred,weights=weights,minlength=16).reshape(4,4).astype(float)
    result=matrix_metrics(cm)
    probs=np.array([[r['probabilities'][label] for label in LABELS] for r in rows])
    weights=np.ones(len(rows))/len(rows) if weights is None else np.asarray(weights)/sum(weights)
    result['multiclass_Brier']=float(np.sum(weights*np.sum((probs-np.eye(4)[truth])**2,axis=1)))
    result['log_loss']=float(-np.sum(weights*np.log(np.maximum(probs[np.arange(len(rows)),truth],1e-15))))
    result['majority_baseline_accuracy']=float(np.max(cm.sum(axis=1))/cm.sum())
    # Supplemental one-vs-rest ranking of material defects; uncertainty remains
    # a separate class in the primary analysis and is excluded only here.
    certain=np.array([r['sol_label']!='uncertain' for r in rows])
    y=truth[certain]==2;p=probs[certain,2]
    if len(set(y))==2:
        result['major_vs_nonmajor_excluding_sol_uncertain']={
            'n':int(certain.sum()),'AUROC':float(roc_auc_score(y,p,sample_weight=weights[certain])),
            'positive_prevalence':float(np.average(y,weights=weights[certain])),
            'average_precision':float(average_precision_score(y,p,sample_weight=weights[certain]))}
    result['n']=len(rows)
    return result


def bootstrap(rows,repetitions=1000):
    rng=np.random.default_rng(20261027)
    strata=defaultdict(list)
    for i,r in enumerate(rows):strata[(r['dataset'],r['source'])].append(i)
    strata=[np.array(v) for v in strata.values()]
    codes=np.array([4*LABELS.index(r['sol_label'])+LABELS.index(r['clef_label']) for r in rows])
    estimates=defaultdict(list)
    for _ in range(repetitions):
        idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        cm=np.bincount(codes[idx],minlength=16).reshape(4,4).astype(float)
        m=matrix_metrics(cm)
        for key in ['accuracy','multiclass_MCC','Cohen_kappa','macro_F1']:estimates[key].append(m[key])
    return {key:np.quantile(values,[.025,.975]).tolist() for key,values in estimates.items()}


def main():
    OUT.mkdir(exist_ok=True)
    truth_bytes=(ROOT/'reference_labels.jsonl').read_bytes()
    truth={r['id']:r for r in map(json.loads,truth_bytes.splitlines())}
    prediction_bytes=(ROOT/'predictions.jsonl').read_bytes()
    predictions=[json.loads(line) for line in prediction_bytes.splitlines()]
    assert len(predictions)==len(truth)==len({p['id'] for p in predictions})==6600
    assert set(truth)=={p['id'] for p in predictions}
    joined=[]
    for p in predictions:
        t=truth[p['id']];answer=p['response']['answers']['quality']
        assert p['full_input_tokens']==p['response']['usage']['input_tokens']==t['full_input_tokens']
        assert p['response']['usage']['output_tokens']==0
        joined.append({**t,'clef_label':answer['choice'],'probabilities':answer['probabilities'],
            'confidence':answer['confidence'],'latency_seconds':p['latency_seconds']})
    result={'labels':LABELS,'n':len(joined),'cohorts':{},'by_source':[],
        'method':{'reference':'GPT-6.1 Sol frozen audit labels; not verified truth','prediction':'one fixed four-class choice; argmax of probabilities',
            'uncertain':'retained as a nominal fourth class, never assigned an ordinal severity',
            'correlation':'generalized multiclass Matthews correlation coefficient (MCC)',
            'intervals':'1000 percentile bootstrap replicates; resample within each dataset/source stratum; seed 20261027; conditional on these labels',
            'long_context':'local vLLM encoder cap override; exact full input-token match checked for all predictions; no truncation',
            'tuning':'no fitting, prompt selection, or threshold calibration on this evaluation set'},
        'input_hashes':{'reference_labels':hashlib.sha256(truth_bytes).hexdigest(),'predictions':hashlib.sha256(prediction_bytes).hexdigest()},
        'completion':json.loads((ROOT/'completion.json').read_text()),
        'runtime':json.loads((ROOT/'runtime-manifest.json').read_text()),
        'experiment':json.loads((ROOT/'experiment-manifest.json').read_text())}
    for dataset in ['sft','dolci','pooled']:
        rows=[r for r in joined if dataset=='pooled' or r['dataset']==dataset]
        m=metrics(rows);m['bootstrap_95']=bootstrap(rows)
        m['sol_counts']=dict(Counter(r['sol_label'] for r in rows));m['clef_counts']=dict(Counter(r['clef_label'] for r in rows))
        kept=[r for r in rows if r['clef_label']=='no_defect_found']
        m['clef_no_defect_subset']={
            'n':len(kept),
            'sol_label_counts':dict(Counter(r['sol_label'] for r in kept)),
            'sol_major_fraction':sum(r['sol_label']=='major_defect' for r in kept)/len(kept) if kept else None,
            'sol_no_defect_fraction':sum(r['sol_label']=='no_defect_found' for r in kept)/len(kept) if kept else None}
        m['major_defect_dimensions']={}
        for dimension in ['prompt_quality','answer_correctness','reasoning_quality','instruction_following','parsing_integrity','training_suitability']:
            affected=[r for r in rows if r['sol_dimensions'].get(dimension)=='major_issue']
            if affected:
                m['major_defect_dimensions'][dimension]={
                    'n':len(affected),
                    'clef_major_recall':sum(r['clef_label']=='major_defect' for r in affected)/len(affected),
                    'clef_no_defect_rate':sum(r['clef_label']=='no_defect_found' for r in affected)/len(affected)}
        m['length_groups']={name:metrics([r for r in rows if (r['full_input_tokens']>16384)==long]) for long,name in [(False,'at_most_16k'),(True,'above_16k')]}
        if dataset!='pooled':
            dirname='source_quality_audit' if dataset=='sft' else 'dolci_quality_audit'
            populations={s['source']:s['population'] for s in json.loads((ROOT.parent/dirname/'sources.json').read_text())}
            counts=Counter(r['source'] for r in rows)
            weights=[populations[r['source']]/sum(populations.values())/counts[r['source']] for r in rows]
            m['population_weighted']=metrics(rows,weights)
            for source in sorted(counts):
                subset=[r for r in rows if r['source']==source];sm=metrics(subset)
                result['by_source'].append({'dataset':dataset,'source':source,**{k:sm[k] for k in ['n','accuracy','multiclass_MCC','Cohen_kappa','macro_F1']}})
        result['cohorts'][dataset]=m
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'source_agreement.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['by_source'][0]),lineterminator='\n');writer.writeheader();writer.writerows(result['by_source'])
    with (OUT/'record_comparison.csv').open('w',newline='') as f:
        fields=['id','dataset','source','sol_label','clef_label','full_input_tokens','confidence','latency_seconds']+[f'p_{l}' for l in LABELS]
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in joined:writer.writerow({**{k:r[k] for k in fields if k in r},**{f'p_{l}':r['probabilities'][l] for l in LABELS}})
    render(result)
    print(json.dumps({d:{k:result['cohorts'][d][k] for k in ['n','accuracy','multiclass_MCC','Cohen_kappa','macro_F1']} for d in result['cohorts']},indent=2))


def render(result):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    figures=[]
    for dataset in ['sft','dolci']:
        cm=np.array(result['cohorts'][dataset]['confusion_matrix']);normalized=cm/cm.sum(axis=1,keepdims=True)
        fig,ax=plt.subplots(figsize=(7.5,5.7));im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{int(cm[i,j]):,}\n{normalized[i,j]*100:.1f}%',ha='center',va='center',color='white' if normalized[i,j]>.5 else '#17353d')
        ax.set_xticks(range(4),DISPLAY,rotation=20,ha='right');ax.set_yticks(range(4),DISPLAY)
        ax.set_xlabel('Clef prediction');ax.set_ylabel('Sol reference label');ax.set_title(('Our SFT audit' if dataset=='sft' else 'Dolci audit')+' · row-normalized agreement')
        fig.colorbar(im,ax=ax,label='Share of each Sol label');fig.tight_layout()
        path=OUT/f'confusion-{dataset}.svg';fig.savefig(path,bbox_inches='tight',metadata={'Date':None});fig.savefig(path.with_suffix('.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        text=path.read_text();figures.append(text[text.index('<svg'):])
    esc=html.escape
    def number(x):return f'{x:.3f}'
    headers=['Audit','N','MCC','95% bootstrap interval','Kappa','Macro-F1','Accuracy','Majority baseline']
    rows=[]
    for dataset,m in result['cohorts'].items():
        rows.append([dataset,f"{m['n']:,}",number(m['multiclass_MCC']),' – '.join(number(x) for x in m['bootstrap_95']['multiclass_MCC']),number(m['Cohen_kappa']),number(m['macro_F1']),f"{100*m['accuracy']:.1f}%",f"{100*m['majority_baseline_accuracy']:.1f}%"])
    def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(str(v))}</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    cohort=result['cohorts']['pooled'];completion=result['completion'];experiment=result['experiment']
    sft=result['cohorts']['sft'];dolci=result['cohorts']['dolci']
    interpretation=f'''<section><h2>The decision: calibrate before filtering</h2><p><strong>The current four-class argmax decisions are too lenient to replace Sol.</strong> Clef labels {100*sft['clef_counts']['no_defect_found']/sft['n']:.1f}% of SFT records and {100*dolci['clef_counts']['no_defect_found']/dolci['n']:.1f}% of Dolci records “no defect found,” compared with Sol’s {100*sft['sol_counts']['no_defect_found']/sft['n']:.1f}% and {100*dolci['sol_counts']['no_defect_found']/dolci['n']:.1f}%. It identifies only {100*sft['per_class']['major_defect']['recall']:.1f}% and {100*dolci['per_class']['major_defect']['recall']:.1f}% of Sol’s major-defect cases as major defects.</p><p>The few major-defect flags have relatively high agreement with Sol: precision is {100*sft['per_class']['major_defect']['precision']:.1f}% for SFT and {100*dolci['per_class']['major_defect']['precision']:.1f}% for Dolci. But low recall leaves many material defects among the accepted records. Minor-defect and uncertainty decisions also agree poorly.</p><p><strong>The probability scores show a more useful ranking signal.</strong> Major-defect AUROC is {sft['major_vs_nonmajor_excluding_sol_uncertain']['AUROC']:.3f} on SFT and {dolci['major_vs_nonmajor_excluding_sol_uncertain']['AUROC']:.3f} on Dolci. This supports a separate calibration and held-out routing experiment, not deployment of a threshold chosen from this evaluation. These findings apply to this checkpoint and fixed rubric; they do not establish a general limit of System One models.</p></section>'''
    body=f'''<header><div class="eyebrow">SYSTEM ONE / CLASSIFIER EXPERIMENT</div><h1>Can a single-pass classifier<br>reproduce a reasoning reviewer?</h1><p>Cloudflare Clef-Flash compared with frozen GPT-6.1 Sol judgments on 6,600 SFT and Dolci records. Four labels, one fixed rubric, complete inputs, and no tuning on the evaluation samples.</p></header><section><h2>Agreement with Sol</h2>{table(headers,rows)}<p class="note">Primary scores weight sampled records equally: 100 per source across 34 SFT sources, and 200 per source across 16 Dolci sources. Population-weighted estimates appear separately below. MCC is a correlation measure for categorical predictions: +1 means perfect agreement, 0 indicates no correlation, and negative values indicate inverse association. Kappa adjusts agreement for class marginals. Accuracy alone is sensitive to label prevalence. These scores measure agreement with Sol, not verified dataset quality.</p></section>'''
    body=body.replace('<section><h2>Agreement with Sol</h2>',interpretation+'<section><h2>Agreement with Sol</h2>')
    body+='<section><h2>Where the labels differ</h2><div class="charts">'+''.join('<figure>'+s+'</figure>' for s in figures)+'</div><p>Rows are Sol labels and columns are Clef predictions. Each cell shows its record count and percentage of that Sol class. Uncertain remains its own class throughout the primary analysis.</p></section>'
    keep_rows=[]
    for dataset in ['sft','dolci']:
        subset=result['cohorts'][dataset]['clef_no_defect_subset']
        keep_rows.append([dataset,subset['n'],
            f"{100*subset['sol_no_defect_fraction']:.1f}%" if subset['n'] else '—',
            f"{100*subset['sol_major_fraction']:.1f}%" if subset['n'] else '—'])
    body+='<section><h2>If we kept Clef’s “no defect” records</h2>'+table(['Dataset','Clef no-defect decisions','Sol also found no defect','Sol found a major defect'],keep_rows)+'<p>This describes the fixed four-class argmax decisions; no acceptance threshold was selected. A low major-defect fraction here would support further validation of a filtering policy. It would not establish correctness, because Sol’s judgments have not been independently adjudicated.</p></section>'
    rank_rows=[]
    for dataset in ['sft','dolci']:
        rank=result['cohorts'][dataset]['major_vs_nonmajor_excluding_sol_uncertain']
        rank_rows.append([dataset,rank['n'],number(rank['AUROC']),number(rank['average_precision']),number(rank['positive_prevalence'])])
    body+='<section><h2>Supplement: ranking material defects</h2>'+table(['Dataset','N, excluding Sol-uncertain','Major-defect AUROC','Average precision','Major-defect prevalence'],rank_rows)+'<p>This secondary check ranks records by the predicted major-defect probability. AUROC 0.5 indicates chance-level ranking; average precision should be compared with the positive prevalence. Sol-uncertain records are excluded only from this binary supplement. All four labels remain in the primary analysis. No operating threshold was selected.</p></section>'
    for dataset in ['sft','dolci']:
        m=result['cohorts'][dataset]
        perclass=[[DISPLAY[i],int(m['per_class'][l]['support']),number(m['per_class'][l]['precision']),number(m['per_class'][l]['recall']),number(m['per_class'][l]['F1'])] for i,l in enumerate(LABELS)]
        body+=f'<section><h2>{dataset.upper()} · class-specific agreement</h2>'+table(['Sol class','Support','Precision','Recall','F1'],perclass)+'</section>'
    length_rows=[]
    for dataset in ['sft','dolci']:
        for group,m in result['cohorts'][dataset]['length_groups'].items():length_rows.append([dataset,group,m['n'],number(m['multiclass_MCC']),number(m['macro_F1']),f"{100*m['accuracy']:.1f}%"])
    body+='<section><h2>Full context, including long inputs</h2>'+table(['Dataset','Input length','N','MCC','Macro-F1','Accuracy'],length_rows)+f'''<p>The checkpoint config permits 262,144 positions, but the pinned vLLM Clef encoder defaults to 16,384. This experiment raises only that encoder cap and rejects oversize requests. The server limit is 98,304; the largest actual input is {experiment['max_input_tokens']:,} tokens. All {experiment['over_16384']:,} inputs above 16k were read in full. Exact encoded input-token counts were checked for every response.</p><p>This tests an explicit extension of the reference encoder settings. The backbone’s position limit does not itself establish decision-head accuracy at long lengths. Length-group differences are descriptive and confounded by task and source mix.</p></section>'''
    dimension_rows=[]
    for dataset in ['sft','dolci']:
        for name,m in result['cohorts'][dataset]['major_defect_dimensions'].items():
            dimension_rows.append([dataset,name,m['n'],f"{100*m['clef_major_recall']:.1f}%",f"{100*m['clef_no_defect_rate']:.1f}%"])
    body+='<section><h2>Which material defects are detected?</h2>'+table(['Dataset','Sol major-issue dimension','N','Clef major-defect recall','Clef no-defect rate'],dimension_rows)+'<p>These groups use Sol’s existing dimension assessments. A record may occur in more than one group. Clef produces one overall class, so this table measures detection of records with each defect type, not Clef’s ability to diagnose the individual dimension. No new adjudication was performed.</p></section>'
    weighted=[]
    for dataset in ['sft','dolci']:
        m=result['cohorts'][dataset]['population_weighted'];weighted.append([dataset,number(m['multiclass_MCC']),number(m['Cohen_kappa']),number(m['macro_F1']),f"{100*m['accuracy']:.1f}%"])
    body+='<section><h2>Population-weighted agreement</h2>'+table(['Dataset','MCC','Kappa','Macro-F1','Accuracy'],weighted)+'<p>Each sampled record is weighted by its source population divided by that source’s sample count. These results reflect the original source mixtures. Pooled results above use equal record weights and do not imply a shared population.</p></section>'
    body+='<section><h2>Agreement by source</h2><p>Source-level estimates use 100 or 200 records each and can be noisy. Macro-F1 always averages the same four labels, including labels absent from an individual source. These tables are descriptive, not a ranking of source quality.</p>'
    for dataset in ['sft','dolci']:
        source_rows=sorted((r for r in result['by_source'] if r['dataset']==dataset),key=lambda r:r['multiclass_MCC'],reverse=True)
        body+=f'<details><summary>{dataset.upper()} · {len(source_rows)} sources</summary><div class="source-table">'+table(['Source','N','MCC','Macro-F1','Accuracy'],[[r['source'],r['n'],number(r['multiclass_MCC']),number(r['macro_F1']),f"{100*r['accuracy']:.1f}%"] for r in source_rows])+'</div></details>'
    body+='</section>'
    body+=f'''<section><h2>Measured execution cost</h2><ul><li>Model: Clef-Flash 9B, BF16 eager mode, one AMD MI325X GPU at a time on hissu, across two nodes after a restart. This is not an optimized inference benchmark.</li><li>Bulk run: {completion['new_records']:,} newly classified records in {completion['wall_seconds']/60:.1f} minutes; {completion['cached']} pilot records reused.</li><li>Observed throughput: {completion['records_per_second']:.2f} records/s and {completion['input_tokens_per_second']:,.0f} input tokens/s.</li><li>Client concurrency: {completion['concurrency']}; pooling output tokens: zero. Initial server startup, model download, and pilot execution are excluded. Bulk wall time includes the Docker stop, failed requests, relocation, and restart. Extra server-side token validation and prefix caching are enabled; reported input tokens include cached prefix tokens.</li></ul><p>The first server received a Docker stop signal after 4,067 accepted predictions; its initiator was not identified. Remaining records were resumed on another GPU with the same checkpoint, rubric, encoding, and precision. No records were discarded because of infrastructure failures. The resumed client used loopback HTTP instead of the login-node connection. This is a recovered run, not an uninterrupted timing benchmark.</p><p>No free-form reasoning was generated. The Sol runs used a remote service with unknown hardware and token-level compute cost; this experiment cannot establish a like-for-like FLOP, dollar, or energy saving. Compare agreement and measured local throughput before treating this as a replacement.</p></section>'''
    body+='''<section><h2>What to test next</h2><p>Keep Sol as the reviewer for now. On a new calibration set, test routing by major-defect probability rather than using the four-class argmax as an automatic acceptance rule. Select a target for missed material defects, measure the resulting review coverage and workload reduction, then evaluate that fixed policy on an untouched set. Report uncertain and minor cases separately.</p><p>Human adjudication of a blinded disagreement sample would help distinguish Clef errors from Sol errors. Any revised prompt or trained decision head also needs a fresh held-out evaluation. No such adjudication, model training, threshold fitting, or post-hoc prompt optimization was performed here.</p></section>'''
    body+='''<section><h2>Reproduction and evidence</h2><p>All artifacts live in <code>smoke/systemone_classifier/</code>. Model revision, input hashes, questions, runtime configuration, encoder override, per-record probabilities, exact token counts, and timing records are preserved. Source identities, Sol labels, and Sol explanations were excluded from classifier requests.</p><p>Primary metrics retain all four labels. Supplemental major-versus-nonmajor AUROC and average precision exclude Sol-uncertain cases and appear in metrics.json. Confidence intervals use 1,000 within-source bootstrap replicates and condition on the observed model labels; they omit systematic reviewer uncertainty.</p><p><a href="metrics.json">Complete metrics</a> · <a href="source_agreement.csv">Per-source agreement</a> · <a href="record_comparison.csv">Every record and probability</a></p><p>References: <a href="https://docs.vllm.ai/en/latest/models/structured_decisions_models/">vLLM structured decisions</a> · <a href="https://huggingface.co/Cloudflare/clef-flash">Clef-Flash model</a>.</p></section>'''
    css='''body{margin:0;background:#f5f4ef;color:#1d343b;font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:50px 32px}h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.13}h1{font-size:clamp(36px,5vw,64px);margin:20px 0}h2{font-size:32px}header p{font-size:20px;max-width:900px;color:#566c70}.eyebrow{font-size:12px;letter-spacing:.16em;color:#287568}section{padding:32px 0;border-bottom:1px solid #d6ded8}.scroll{overflow:auto;background:white;border:1px solid #d6ded8;border-radius:10px;padding:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e6e9e4;white-space:nowrap}th{color:#5b736e;font-size:12px}.note{background:#e6eee8;border-left:4px solid #287568;padding:20px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;padding:12px;background:white;border:1px solid #d6ded8;border-radius:10px}svg{width:100%;height:auto}a{color:#287568}code{overflow-wrap:anywhere}@media(max-width:800px){main{padding:25px 18px}.charts{grid-template-columns:1fr}h2{font-size:27px}}@media print{main{padding:0}section{break-inside:avoid}body{background:white}}'''
    css+='details{margin:18px 0}summary{cursor:pointer;font-weight:600;padding:10px 0}.source-table td:first-child{white-space:normal;min-width:240px;max-width:420px;overflow-wrap:anywhere}'
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clef vs Sol — four-class SFT quality experiment</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    (OUT/'index.html').write_text(document)


if __name__=='__main__':main()
