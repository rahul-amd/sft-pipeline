"""Render paired raw and held-out calibration results without mixing fit and test."""
import html
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=['no_defect_found','minor_defect','major_defect','uncertain']
DISPLAY=['No defect','Minor defect','Major defect','Uncertain']
NAMES={'flash_raw':'Flash · raw','flash_selected':'Flash · selected calibration','clef_raw':'Clef 27B · raw','clef_selected':'Clef 27B · selected calibration'}


def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def render(result):
    out=ROOT/'report';out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for i,dataset in enumerate(['sft','dolci']):
        for j,variant in enumerate(['clef_raw','clef_selected']):
            cm=np.array(result['heldout'][dataset]['variants'][variant]['confusion_matrix']);den=cm.sum(axis=1,keepdims=True)
            shares=np.divide(cm,den,out=np.zeros_like(cm),where=den>0)
            ax=axes[i,j];ax.imshow(shares,cmap='Blues',vmin=0,vmax=1)
            for a in range(4):
                for b in range(4):ax.text(b,a,f'{int(cm[a,b])}\n{100*shares[a,b]:.1f}%',ha='center',va='center',color='white' if shares[a,b]>.5 else '#203b43')
            ax.set_xticks(range(4),DISPLAY,rotation=20,ha='right');ax.set_yticks(range(4),DISPLAY)
            ax.set_xlabel('Clef prediction');ax.set_ylabel('Sol reference');ax.set_title(dataset.upper()+' · '+('raw' if j==0 else 'selected calibration'))
    fig.tight_layout();fig.savefig(out/'heldout-confusion.svg',metadata={'Date':None});fig.savefig(out/'heldout-confusion.png',dpi=140);plt.close(fig)
    svg=(out/'heldout-confusion.svg').read_text();svg=svg[svg.index('<svg'):]
    fmt=lambda x:f'{x:.3f}'
    pct=lambda x:'—' if x is None else f'{100*x:.1f}%'
    body='<header><div class="eyebrow">CLEF 27B / FLASH / SOL</div><h1>Does a larger decision model<br>make a better SFT reviewer?</h1><p>The same 6,600 frozen records, four labels, full inputs, and a separate calibration experiment. Sol is the reference reviewer, not independently verified truth.</p></header>'
    findings=[]
    for dataset in ['sft','dolci']:
        variants=result['heldout'][dataset]['variants'];cal=variants['clef_selected'];flash=variants['flash_selected'];raw=variants['clef_raw']
        findings.append(f'<p><strong>{dataset.upper()}:</strong> calibrated Clef reaches MCC {fmt(cal["multiclass_MCC"])} versus {fmt(flash["multiclass_MCC"])} for calibrated Flash. Calibration raises Clef’s major-defect recall from {pct(raw["major_defect_recall"])} to {pct(cal["major_defect_recall"])}. Among its {cal["no_defect_decisions"]} no-defect decisions, {pct(cal["major_fraction_in_no_defect"])} are Sol-major; only {pct(cal["per_class"]["no_defect_found"]["precision"])} match Sol’s no-defect label.</p>')
    body+='<section><h2>A stronger reviewer signal, with important gaps</h2>'+''.join(findings)+'<p>The larger model improves agreement both before and after applying the same calibration procedure to Flash. Both paired 95% intervals for the calibrated MCC gain exclude zero. However, calibration reduces minor-defect recall and misses every Sol-uncertain case in the held-out set. It is useful evidence for a review-routing experiment, not validation of an automatic acceptance filter.</p></section>'
    rows=[]
    for dataset in ['sft','dolci']:
        for model in ['flash','clef']:
            m=result['raw_full'][dataset][model]
            rows.append([dataset,model,m['n'],fmt(m['multiclass_MCC']),fmt(m['macro_F1']),pct(m['accuracy']),pct(m['major_defect_recall'])])
    body+='<section><h2>Raw decisions on every record</h2>'+table(['Audit','Model','N','MCC','Macro-F1','Accuracy','Major-defect recall'],rows)+'<p>This directly repeats the original experiment. No calibration is applied to these full-audit scores. MCC measures categorical correlation; uncertainty remains a fourth class rather than an ordinal severity. Raw decisions preserve the API’s returned choice, including exact probability ties. A label-order argmax would differ on 4 Flash and 39 Clef records; these are audited in the metrics file. Calibrated decisions use the largest calibrated probability.</p></section>'
    fits=[]
    for name,f in result['fitted_calibrators'].items():fits.append([name,f['fit_records'],fmt(f['raw_calibration_log_loss']),fmt(f['out_of_fold_log_loss']),pct(f['relative_oof_log_loss_improvement']),'adopted' if f['calibration_adopted'] else 'raw retained'])
    body+='<section><h2>Was calibration warranted?</h2>'+table(['Model','Fitting N','Raw log loss','Out-of-fold log loss','Relative reduction','Gate decision'],fits)+'<p>The predeclared gate requires at least a 1% relative improvement in five-fold out-of-fold log loss on the calibration half. The calibrator is one pooled, regularized multinomial logistic regression over four log probabilities, with fixed C=1 and no source or dataset features. It can correct class bias; it is more than temperature scaling. Held-out labels were not used to fit, choose hyperparameters, or decide whether to adopt calibration.</p></section>'
    rows=[]
    for dataset in ['sft','dolci']:
        cohort=result['heldout'][dataset]
        for name in ['flash_raw','flash_selected','clef_raw','clef_selected']:
            m=cohort['variants'][name];ci=cohort['bootstrap']['metrics_95'][name]['multiclass_MCC']
            rows.append([dataset,NAMES[name],m['n'],fmt(m['multiclass_MCC']),' – '.join(map(fmt,ci)),fmt(m['macro_F1']),pct(m['accuracy']),fmt(m['log_loss']),fmt(m['Brier']),fmt(m['ECE_15_bins'])])
    body+='<section><h2>The fair comparison: identical held-out records</h2>'+table(['Audit','Method','N','MCC','MCC 95% CI','Macro-F1','Accuracy','Log loss','Brier','ECE'],rows)+'<p>The held-out half contains 1,700 SFT records and 1,600 Dolci records, selected 50/50 within every source before new inference. Both models receive the same calibration procedure and split. This separates improvements due to model size from improvements due to post-processing. Lower log loss, Brier score, and 15-bin top-label ECE are better. These are held out from calibration fitting, not a newly collected benchmark: the previous Flash aggregate results were already known.</p></section>'
    changes=[]
    for dataset in ['sft','dolci']:
        h=result['heldout'][dataset]
        for a,b,label in [('clef_raw','flash_raw','Larger model, raw'),('clef_selected','flash_selected','Larger model, after selected calibration'),('clef_selected','clef_raw','Calibration gain, Clef'),('flash_selected','flash_raw','Calibration gain, Flash')]:
            diff=h['variants'][a]['multiclass_MCC']-h['variants'][b]['multiclass_MCC'];ci=h['bootstrap']['paired_differences_95'][f'{a}_minus_{b}']['multiclass_MCC']
            changes.append([dataset,label,fmt(diff),' – '.join(map(fmt,ci))])
    body+='<section><h2>Paired changes in agreement</h2>'+table(['Audit','Comparison','MCC difference','Paired 95% CI'],changes)+'<p>Intervals use 1,000 paired bootstrap replicates within dataset/source strata. Models are evaluated on the same resampled records. Intervals condition on the fitted calibrators and Sol labels; they do not include label uncertainty or uncertainty from refitting calibration.</p></section>'
    decisions=[]
    for dataset in ['sft','dolci']:
        for name,m in result['heldout'][dataset]['variants'].items():
            decisions.append([dataset,NAMES[name],pct(m['major_defect_recall']),m['no_defect_decisions'],pct(m['major_fraction_in_no_defect']),fmt(m['major_AUROC_excluding_sol_uncertain'])])
    body+='<section><h2>What would a quality filter miss?</h2>'+table(['Audit','Method','Major-defect recall','No-defect decisions','Sol-major among no-defect','Major-defect AUROC'],decisions)+'<p>These use the four-class argmax, without a separately tuned acceptance threshold. The AUROC supplement excludes Sol-uncertain records only for that binary ranking calculation. A stronger correlation alone does not establish a safe automatic filter; missed material defects and review workload matter.</p></section>'
    classes=[]
    for dataset in ['sft','dolci']:
        for variant in ['flash_selected','clef_raw','clef_selected']:
            m=result['heldout'][dataset]['variants'][variant]
            for label,display in zip(LABELS,DISPLAY):
                c=m['per_class'][label]
                classes.append([dataset,NAMES[variant],display,pct(c['precision']),pct(c['recall']),fmt(c['F1'])])
    body+='<section><h2>Check every class, including uncertainty</h2><details><summary>Held-out precision, recall, and F1 by class</summary>'+table(['Audit','Method','Class','Precision','Recall','F1'],classes)+'</details><p>Overall agreement can improve while a rare class becomes less useful. Minor defects and uncertainty remain separate targets; uncertainty is not treated as an intermediate severity. A zero score is used when a class has no predictions.</p></section>'
    body+='<section><h2>How calibration changes the decisions</h2><figure>'+svg+'</figure><p>Only held-out records appear here. Rows are Sol labels; columns are Clef predictions. Cells show counts and row percentages. A selected method retains raw probabilities if its calibration gate did not pass.</p></section>'
    source_rows=[]
    for row in result['by_source']:
        variants=row['variants'];raw=variants['clef_raw'];cal=variants['clef_selected'];flash=variants['flash_selected']
        source_rows.append([row['dataset'],row['source'],cal['n'],fmt(raw['multiclass_MCC']),fmt(cal['multiclass_MCC']),fmt(flash['multiclass_MCC']),pct(cal['major_defect_recall']),pct(cal['major_fraction_in_no_defect'])])
    body+='<section><h2>Does agreement vary by source?</h2><details><summary>All 50 source-level comparisons</summary>'+table(['Audit','Source','Held-out N','Clef raw MCC','Clef selected MCC','Flash selected MCC','Clef major recall','Sol-major among Clef no-defect'],source_rows)+'</details><p>Each source contributes only 50 SFT or 100 Dolci held-out records. These descriptive estimates can be noisy and are not a source-ranking benchmark. No source-specific calibrators or thresholds were fitted.</p></section>'
    completion=json.loads((ROOT/'completion.json').read_text());runtime=json.loads((ROOT/'runtime-manifest.json').read_text());manifest=json.loads((ROOT/'experiment-manifest.json').read_text())
    body+=f'<section><h2>Execution and full-context checks</h2><p>Cloudflare/clef at revision <code>{manifest["model_revision"]}</code>, BF16 eager mode, {runtime["replicas"]} independent replicas on {runtime["gpu_count"]} MI325X GPUs. The native runtime class is named ClefFlashForJointSchema, but it is the shared Qwen3.5/Qwen3.8 joint-head implementation: this run loads the requested 27B checkpoint and its head configuration.</p><p>All 6,600 records were validated against prepared payload hashes and exact encoded input-token counts. The largest input was {manifest["max_input_tokens"]:,} tokens; {manifest["over_16384"]:,} exceeded the reference encoder’s 16k default. The experiment-local adapter lifts that cap and rejects truncation. Native token IDs matched the checkpoint reference encoder on {manifest["reference_encoder_parity_records"]} deterministic checks, including the longest inputs.</p><p>The parallel bulk interval took {completion["wall_seconds"]/60:.1f} minutes, processing {completion["new_input_tokens"]:,} new accepted input tokens at {completion["input_tokens_per_second"]:,.0f} tokens/s; {completion["cached"]} cached predictions were reused. Startup, model download, and the pilot are excluded. This hardware arrangement differs from the Flash run, so wall time is not a like-for-like compute comparison.</p><p>The user’s temporary hissu authorization expires at 00:30 UTC on 28 October 2026; Slurm limits and an independent container-cleanup watchdog enforce an earlier stop. This supersedes the initial non-hissu resource plan, without changing the statistical protocol.</p></section>'
    release_path=ROOT/'resource-release.json'
    if release_path.exists():
        release=json.loads(release_path.read_text())
        body+='<section><h2>Resources released</h2><p>Cleanup verified at '+html.escape(release['verified_utc'])+'. No experiment containers remained on the allocated node, and the experiment allocation had ended before the hissu deadline. The unused fallback allocation was canceled.</p><p><a href="../resource-release.json">Resource release evidence</a></p></section>'
    body+='<section><h2>Limits and next decision</h2><p>All results describe agreement with one frozen model reviewer. Human adjudication of a blinded disagreement sample is still needed to distinguish reviewer errors from classifier errors. The calibrator targets the balanced audit mixture; probabilities may need adjustment when deployed on a different source mixture. No new threshold should be chosen from this held-out set and then reported as independently validated. The backbone’s position limit and successful full-input processing do not establish decision-head accuracy at long lengths; lifting the encoder’s 16k default is an explicit experimental intervention.</p><p>Use these results to decide whether a new calibration-and-routing study is justified. Choose its missed-defect target and review budget in advance, then evaluate the selected policy on fresh records. No pipeline filter or production behavior was changed in this experiment.</p><p><a href="metrics.json">All metrics and source results</a> · <a href="heldout_predictions.csv">Paired held-out probabilities</a> · <a href="../protocol.json">Frozen statistical protocol</a> · <a href="../calibration-split.json">Split membership</a> · <a href="../resource-authorization.json">Resource authorization</a> · <a href="https://huggingface.co/Cloudflare/clef">Model card</a></p></section>'
    css='body{margin:0;background:#f5f4ef;color:#20383f;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:45px 30px}h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.15}h1{font-size:clamp(34px,5vw,62px)}h2{font-size:30px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#287568}header p{font-size:20px;color:#536b70;max-width:960px}section{padding:28px 0;border-bottom:1px solid #d4ded7}.scroll{overflow:auto;background:white;border:1px solid #d4ded7;border-radius:10px;padding:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px;text-align:left;border-bottom:1px solid #e3e8e2;white-space:nowrap}th{font-size:12px;color:#58716a}figure{margin:0;background:white;padding:16px;border-radius:12px}svg{width:100%;height:auto}a{color:#287568}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:22px 16px}}@media print{body{background:white}section{break-inside:avoid}}'
    (out/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clef 27B vs Flash — raw and calibrated SFT review</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
