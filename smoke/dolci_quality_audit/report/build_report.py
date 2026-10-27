"""Build the offline Dolci audit report from frozen results; no model calls."""
from collections import Counter
import base64
import gzip
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("dolci_reviewer", ROOT / "review.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
LABELS = ["major_defect", "minor_defect", "uncertain", "no_defect_found"]
NAMES = ["Major defect", "Minor defect", "Uncertain", "No defect found"]
DIMS = dict(zip(reviewer.base.DIMENSIONS, ["Prompt quality", "Answer correctness", "Reasoning quality", "Instruction following", "Parsing / generation", "Training suitability"]))
SHORT = {"OpenThoughts3-full-filtered-math": "OpenThoughts3 · math", "correct-python": "Correct Python",
    "persona-precise": "Persona Precise IF", "if_qwq": "QwQ verified IF", "nemotron-post": "Nemotron subset",
    "SYNTHETIC-2": "SYNTHETIC-2", "OpenThoughts3-full-filtered-science": "OpenThoughts3 · science",
    "aya-100k": "Aya", "OpenThoughts3-full-filtered-code": "OpenThoughts3 · code", "tulu_v3.9": "Tülu WildChat",
    "wildjailbreak": "WildJailbreak", "wildguardmix": "WildGuardMix", "wildchat-r1-p2": "WildChat · part 2",
    "coconot": "CoCoNot", "oasst1": "OpenAssistant", "tablegpt": "TableGPT"}


def short(source):
    return next((v for k, v in SHORT.items() if k in source), source)


def read_samples():
    path = ROOT / "samples.jsonl"
    raw = path.read_bytes() if path.exists() else gzip.decompress((ROOT / "samples.jsonl.gz").read_bytes())
    return raw, [json.loads(line) for line in raw.splitlines()]


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def section(id, title, body):
    return f'<section id="{id}"><div class="eyebrow">Dolci / source quality audit</div><h2>{title}</h2>{body}</section>'


def save(fig, name):
    path = HERE / "figures" / name
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', metadata={"Date": None})
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    svg = path.with_suffix('.svg').read_text()
    svg = '\n'.join(line.rstrip() for line in svg.splitlines())+'\n'
    path.with_suffix('.svg').write_text(svg)
    return svg[svg.index('<svg'):]


def style(ax):
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis='x', color='#e9edea')
    ax.set_axisbelow(True)


def main():
    raw, samples = read_samples()
    verdict_bytes = (ROOT / "verdicts.jsonl").read_bytes()
    reviews = [json.loads(line) for line in verdict_bytes.splitlines()]
    a = json.loads((ROOT / "analysis.json").read_text())
    manifest = json.loads((ROOT / "sample-manifest.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == a['samples_sha256'] == manifest['samples_sha256']
    assert hashlib.sha256(verdict_bytes).hexdigest() == a['verdicts_sha256']
    originals = {s['audit_id']: s for s in samples}
    assert len(originals) == len(samples) == len(reviews) == len({r['audit_id'] for r in reviews}) == 3200
    assert set(originals) == {r['audit_id'] for r in reviews}
    assert Counter(r['overall'] for r in reviews) == a['counts'] == {'major_defect':1337,'minor_defect':1044,'uncertain':157,'no_defect_found':662}
    assert Counter(r['source'] for r in reviews) == Counter({s['source']:200 for s in a['sources']})
    packets = []
    for r in reviews:
        s = originals[r['audit_id']]
        view = reviewer.to_review(s)
        reviewer.validate({'reviews':[{k:r[k] for k in ('audit_id','overall','summary','uncertainty_reason','dimensions','issues')}]}, [view])
        spans = []
        for m in view['messages']:
            if m['role'] != 'assistant':
                spans.append(None)
                continue
            content = m['content']
            start = len(content)-len(content.lstrip())+7 if content.lstrip().startswith('<think>') else 0
            end = start + len(m['reasoning'])
            answer_start = len(content)-len(m['answer'])
            assert content[start:end] == m['reasoning'] and content[answer_start:] == m['answer']
            # JavaScript slices UTF-16 code units. Store offsets instead of
            # duplicate text, retaining exact parsed views for every language.
            units = lambda n: len(content[:n].encode('utf-16-le'))//2
            spans.append({'reasoning_start':units(start),'reasoning_end':units(end),
                          'answer_start':units(answer_start),'parser_notes':m['parser_notes']})
        packets.append({**r, 'record':s['record'], 'record_sha256':s['record_sha256'], 'parsed':spans})
    sources = [{**s, 'weight':s['population']/a['population'], 'counts':{l:s[l] for l in LABELS}, 'major_interval':[s['major_wilson_lower'],s['major_wilson_upper']]} for s in a['sources']]
    a['sources'] = sources
    turn_rows = []
    for multi, label in [(False,'One assistant turn'),(True,'Multiple assistant turns')]:
        selected = [r for r in reviews if (sum(m['role']=='assistant' for m in originals[r['audit_id']]['record']['messages'])>1)==multi]
        k = sum(r['overall']=='major_defect' for r in selected)
        turn_rows.append({'label':label,'n':len(selected),'major':k,'rate':k/len(selected)})
    without_answer = sum(r['overall']=='major_defect' and r['dimensions']['answer_correctness']!='major_issue' for r in reviews)
    ot = [s for s in sources if 'OpenThoughts3' in s['source']]
    ot_impact = sum(s['major_contribution_pp'] for s in ot)
    weighted = a['population_weighted']; major = weighted['major_defect']
    baseline_path = ROOT.parent/'source_quality_audit/report/analysis.json'
    baseline = json.loads(baseline_path.read_text())
    a['report_analysis'] = {'turn_groups':turn_rows,'major_without_major_answer_flag':without_answer,
        'openthoughts_population':sum(s['population'] for s in ot),'openthoughts_contribution_pp':ot_impact,
        'openthoughts_share_of_major_flags':ot_impact/(100*major['estimate']),
        'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'baseline_weighted_major':baseline['weighted']['major_defect']['estimate']}
    examples = [
        ('021cd16244bb0bb5b6456084','An invented condition fixes an underdetermined problem','The reviewer flags an unjustified symmetry assumption used to select b = 3. Inspect the prompt and retained argument, not only the boxed result.'),
        ('02d7e0f5ecb04cb3713733fc','A contradiction is noticed, then ignored','The reviewer identifies incompatible leading coefficients in a polynomial problem. The response acknowledges the conflict but still returns an unconditional numerical answer.'),
        ('06bbede1d9cf356ef7235402','A concrete code counterexample','The reviewer manually traces input 3 and identifies a faulty state transition. This is static reasoning; the code was not executed.'),
        ('020e3f28333355861cef0b9c','A correct final answer can still receive a major flag','The dielectric calculation ends at the correct value according to the reviewer, but a long repeated calculation is flagged under parsing / generation integrity.'),
        ('00e657c92b71d942e7b5d8e1','A clarification can be appropriate supervision','The reviewer flags missing team identity in the prompt while describing the assistant’s clarification as appropriate. This illustrates why a major overall label is not an automatic deletion rule.'),
        ('03eb7e572f73752fc6e83f14','Table semantics expose a wrong mapping','The reviewer flags headers that imply more wins than games played and proposes a mapping consistent with the supplied rows.'),
        ('04589a70a98af83c369cdcc2','A bounded no-defect finding','The reviewer finds a factorial implementation correct within the stated range. This is useful contrasting evidence, not a certification of the entire source.'),
        ('07ee0b4313ece50fa7d16214','Uncertainty remains visible','The reviewer identifies a factual problem and an unverifiable television credit. The overall uncertain label preserves the unresolved claim rather than treating it as a pass.')]
    assert all(e[0] in originals for e in examples)
    a['report_analysis']['examples'] = [{'audit_id':i,'title':t,'commentary':c} for i,t,c in examples]
    (HERE/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    (HERE/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'dolci-audit'})
    fig,ax=plt.subplots(figsize=(10,6));style(ax)
    ax.barh(range(16),[s['major_contribution_pp'] for s in sources],color=['#bd4b3c' if 'OpenThoughts3' in s['source'] else '#637e7c' for s in sources])
    ax.set_yticks(range(16),[short(s['source']) for s in sources]);ax.invert_yaxis();ax.set_xlim(0,34)
    for i,s in enumerate(sources):ax.text(s['major_contribution_pp']+.2,i,f"{s['major_contribution_pp']:.2f}",va='center',fontsize=9)
    ax.set_xlabel('Contribution to population-weighted major flags (percentage points)')
    fig.tight_layout();impact_svg=save(fig,'source-impact')
    fig,ax=plt.subplots(figsize=(10,4));style(ax);left=[0]*6
    for state,color,label in [('major_issue','#bd4b3c','Major'),('minor_issue','#d5a44a','Minor'),('uncertain','#7c8193','Uncertain'),('pass','#348477','Pass / not applicable')]:
        values=[100*(a['dimensions'][d].get(state,0)+(a['dimensions'][d].get('not_applicable',0) if state=='pass' else 0))/3200 for d in DIMS]
        ax.barh(range(6),values,left=left,color=color,label=label);left=[x+y for x,y in zip(left,values)]
    ax.set_yticks(range(6),list(DIMS.values()));ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Percent of the equal-source sample')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False);fig.tight_layout();dimension_svg=save(fig,'dimension-profile')
    ranked=sorted(sources,key=lambda s:-s['major_rate']);fig,ax=plt.subplots(figsize=(10,6));style(ax)
    for i,s in enumerate(ranked):
        p=s['major_rate']*100;lo,hi=[x*100 for x in s['major_interval']]
        ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',color='#bd4b3c',ecolor='#a6b9b0',capsize=3)
    ax.set_yticks(range(16),[short(s['source']) for s in ranked]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Major-flag rate (%) · 95% Wilson intervals');fig.tight_layout();interval_svg=save(fig,'source-intervals')
    esc=html.escape
    mix=table(['Verdict','Sample count','Sample share','Population-weighted','95% sampling interval'],[[name,f"{a['counts'][label]:,}",f"{100*a['counts'][label]/3200:.1f}%",f"{100*weighted[label]['estimate']:.1f}%",'–'.join(f'{100*x:.1f}' for x in weighted[label]['interval_95'])+'%'] for label,name in zip(LABELS,NAMES)])
    source_rows=[]
    for s in sources:
        profile=''.join(f'<span class="seg {l}" style="width:{100*s[l]/200}%" title="{name}: {s[l]}/200"></span>' for l,name in zip(LABELS,NAMES))
        source_rows.append(f'<tr data-source="{esc(s["source"],quote=True)}" data-major="{s["major_rate"]}" data-impact="{s["major_contribution_pp"]}" data-pop="{s["population"]}"><td><button class="source-link" data-open-source="{esc(s["source"],quote=True)}">{esc(s["source"])}</button></td><td>{s["population"]:,}<small>{100*s["weight"]:.2f}% of data</small></td><td><div class="mini-stack">{profile}</div><small>{s["major_defect"]} major · {s["minor_defect"]} minor · {s["uncertain"]} uncertain · {s["no_defect_found"]} no defect found</small></td><td><strong>{100*s["major_rate"]:.1f}%</strong><small>{100*s["major_interval"][0]:.1f}–{100*s["major_interval"][1]:.1f}%</small></td><td>{s["major_contribution_pp"]:.2f} pp</td></tr>')
    nav=[('overview','The finding'),('concentration','Where flags concentrate'),('dimensions','Failure patterns'),('sources','All 16 sources'),('comparison','Compare carefully'),('examples','Read the evidence'),('explorer','Explore 3,200 reviews'),('next','Recommended actions'),('method','Method and limits')]
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"><title>Dolci-Think-SFT-7B — source quality audit</title><style>'''+(HERE/'report.css').read_text()+'''</style></head><body><a class="skip" href="#main">Skip to report</a><aside class="rail"><a class="brand" href="#overview"><span class="brand-mark">D</span><span>DATA QUALITY<br><b>Research notes</b></span></a><div class="rail-label">DOLCI AUDIT / 02</div><nav aria-label="Report sections">'''+''.join(f'<a href="#{id}"><span>{n:02d}</span>{title}</a>' for n,(id,title) in enumerate(nav,1))+'''</nav><div class="rail-foot"><span class="status-dot"></span>Review complete<small>27 October 2026<br>One model · no human adjudication</small></div></aside><main id="main"><div class="topline">DOLCI-THINK-SFT-7B / QUALITY AUDIT<button id="printButton" class="quiet">Print / save PDF</button></div>'''
    doc+=f'''<section id="overview" class="hero"><div class="eyebrow">16 sources · 3,200 complete conversations</div><h1>A concentrated quality signal.<br>A result to investigate.</h1><p class="dek">A source-stratified review of Dolci-Think-SFT-7B finds substantial model-flagged defects, with the largest estimated contribution in its OpenThoughts3 math component. The evidence supports targeted validation, not a blanket judgment about the dataset.</p><div class="scope"><span><strong>2,268,178</strong> population records</span><span><strong>200</strong> samples per source</span><span><strong>GPT-6.1 Sol</strong> · high effort</span></div><div class="metrics"><div class="metric emphasis"><span>Population-weighted major flags</span><strong>{100*major['estimate']:.1f}%</strong><small>{100*major['interval_95'][0]:.1f}–{100*major['interval_95'][1]:.1f}% sampling interval</small></div><div class="metric"><span>Major flags in the audit sample</span><strong>41.8%</strong><small>1,337 / 3,200 conversations</small></div><div class="metric"><span>OpenThoughts3 math</span><strong>184<em>/ 200</em></strong><small>92% flagged major in this source</small></div><div class="metric"><span>Validated review coverage</span><strong>100%</strong><small>All 1,226 batches complete</small></div></div><div class="callout"><strong>These are reviewer labels, not verified error rates.</strong> One model assessed all assistant turns without execution, external fact-checking, or human adjudication. A major flag can concern the prompt, reasoning, or formatting even when the final answer is correct.</div>{mix}<p class="caption">Equal source sampling gives small sources the same review count as large ones. Population weighting restores their actual sizes. Intervals describe sampling uncertainty conditional on model labels; they do not account for reviewer error.</p></section>'''
    doc+=section('concentration','Most estimated flags come from a few components.',f'''<div class="three-col"><article class="finding"><span class="tag">Largest contribution</span><h3>Math: 30.54 percentage points</h3><p>OpenThoughts3 math represents 33.2% of the population and has 184 major flags in 200 reviews. Its weighted contribution alone exceeds half of all estimated major flags.</p></article><article class="finding"><span class="tag">Shared family, different tasks</span><h3>OpenThoughts3: {ot_impact:.2f} points</h3><p>Math, science, and code together comprise 41.5% of records and contribute {100*ot_impact/(100*major['estimate']):.1f}% of the weighted major flags. Science has 143/200 major labels; code has 169/200.</p></article><article class="finding"><span class="tag neutral">Large-source exposure</span><h3>Python: 57 / 200 major</h3><p>Correct Python has a lower sampled major rate (28.5%) but supplies 20.6% of the dataset. That makes it the second-largest contributor, at 5.86 percentage points.</p></article></div><figure class="chart">{impact_svg}<figcaption>Population share × observed major-flag rate. Coral marks the three OpenThoughts3 components. These contributions sum to the weighted estimate; they are not removal-effect or training-impact estimates.</figcaption></figure>''')
    dim_table=table(['Dimension','Major labels','Sample share'],[[name,a['dimensions'][d].get('major_issue',0),f"{100*a['dimensions'][d].get('major_issue',0)/3200:.1f}%"] for d,name in DIMS.items()])
    doc+=section('dimensions','Separate wrong answers from other defects.',f'''<p>Answer correctness has the most major labels (983), followed by reasoning quality (770). The dimensions overlap: their counts must not be added to estimate affected records. <strong>{without_answer} of the 1,337 major-labeled records have no major answer-correctness flag.</strong></p><figure class="chart">{dimension_svg}<figcaption>Unweighted proportions across 3,200 sampled conversations. “Pass / not applicable” combines those two states only for the chart.</figcaption></figure>{dim_table}<div class="callout quiet-callout"><strong>The math finding warrants close inspection.</strong> Among its 200 samples, the reviewer marks 124 prompts, 149 answers, and 158 reasoning traces with major issues. Some illustrative cases involve inconsistent premises followed by a forced numerical answer. These counts overlap and do not establish the prevalence of that specific mechanism without a separate issue taxonomy.</div><p>Parsing / generation flags also include repetition and visibly damaged outputs, not just incorrect extraction of reasoning. No <code>&lt;think&gt;</code> tag, or an empty reasoning field, is not automatically a defect in this audit.</p>''')
    doc+=section('sources','Inspect every exact source label.','''<p>Each row contains 200 reviewed physical conversations. Click a source to open its records. The initial order is contribution to the population-weighted major estimate.</p><div class="controls"><label>Find a source<input id="sourceSearch" type="search" placeholder="Dataset name…"></label><label>Order by<select id="sourceSort"><option value="impact">Dataset impact</option><option value="major">Major-flag rate</option><option value="pop">Population size</option><option value="name">Source name</option></select></label><button class="button secondary" id="downloadSources">Download source CSV</button></div><div class="legend"><span><i class="dot major_defect"></i>Major</span><span><i class="dot minor_defect"></i>Minor</span><span><i class="dot uncertain"></i>Uncertain</span><span><i class="dot no_defect_found"></i>No defect found</span></div><div class="table-scroll source-table"><table id="sourceTable"><thead><tr><th>Exact dataset_source label</th><th>Population</th><th>200-review profile</th><th>Major rate / interval</th><th>Contribution</th></tr></thead><tbody>'''+''.join(source_rows)+f'''</tbody></table></div><details class="reproduce"><summary>View all source intervals</summary><figure class="chart">{interval_svg}<figcaption>95% Wilson binomial approximations; no finite population correction or multiplicity adjustment. Overlapping intervals and different task mixes limit source rankings.</figcaption></figure></details>''')
    comparison=table(['Audit','Records / sources','Weighted major labels','Review unit'],[['Our retained SFT data','3,400 / 34',f"{100*baseline['weighted']['major_defect']['estimate']:.1f}%",'Stored prompt, reasoning, answer, raw response'],['Dolci-Think-SFT-7B','3,200 / 16',f"{100*major['estimate']:.1f}%",'Complete conversation; every assistant turn']])
    turns_table=table(['Conversation group','Sampled','Major labels','Unweighted rate'],[[r['label'],r['n'],r['major'],f"{100*r['rate']:.1f}%"] for r in turn_rows])
    doc+=section('comparison','A descriptive comparison, not a quality leaderboard.',f'''{comparison}<p>The reviewer model, high reasoning effort, six dimensions, and overall severity rule are shared. The Dolci prompt explicitly adapts the input format to ordered conversations and evaluates all assistant turns. The earlier audit evaluates each stored SFT response with its prompt.</p><p>Dolci’s sample contains 2,958 records with one assistant turn and 242 with multiple assistant turns (up to 20). Longer conversations expose more opportunities for a record-level defect, but source and task difficulty also vary. The subgroup rates below are descriptive and do not isolate a length effect.</p>{turns_table}<div class="callout"><strong>Do not read 55.4% versus 27.2% as proof that one pipeline produces better training data.</strong> The datasets differ in source mix, task difficulty, formatting, and reviewed supervision units. The same model can also make correlated errors. Neither audit measures downstream training benefit.</div><p>A stronger comparison would use matched task families and length strata, equivalent assistant-turn units with their full preceding context, blinded expert adjudication of a stratified subset, and a controlled training evaluation. The audits here remain unchanged; no second review was performed for this report.</p>''')
    cards=''.join(f'<article class="case"><div class="eyebrow">Selected evidence</div><h3>{esc(title)}</h3><p>{esc(comment)}</p><button class="text-button" data-open-record="{aid}">Inspect original conversation ↗</button></article>' for aid,title,comment in examples)
    doc+=section('examples','Read the original evidence.',f'<p>These deliberately selected cases illustrate mechanisms and interpretation limits. They are not an additional random sample or an independent verification of the reviewer’s claims.</p><div class="case-grid">{cards}</div>')
    doc+=(HERE/'explorer_template.html').read_text()
    doc+=section('next','Validate the signal before changing the mixture.','''<div class="action-list"><article><span>1</span><div><h3>Adjudicate the concentrated OpenThoughts3 flags</h3><p>Take a blinded, stratified subset of major, minor, uncertain, and no-defect records across math, science, and code. Use domain reviewers to measure false positives and missed defects. Include cases where a contradictory prompt is correctly rejected.</p></div></article><article><span>2</span><div><h3>Separate prompt ambiguity from harmful supervision</h3><p>A well-handled clarification can be valuable training data. Review ambiguous-prompt labels jointly with the assistant’s response before treating them as a removal criterion.</p></div></article><article><span>3</span><div><h3>Check candidate mechanisms with targeted tests</h3><p>Use available references, executable code tests in isolation, symbolic checks where applicable, and deterministic format or repetition diagnostics. Validate each check against both flagged examples and no-defect controls. These checks are proposed follow-up work, not part of this audit.</p></div></article><article><span>4</span><div><h3>Measure training value under a controlled budget</h3><p>Compare the original mixture with validated filtering or repair, holding the student, token budget, optimization, and evaluation suite fixed. Report retained diversity and task coverage alongside downstream performance.</p></div></article></div>''')
    doc+=section('method','Reproducible samples. Bounded conclusions.',f'''<div class="method-grid"><article><h3>Sampling and provenance</h3><p>All 156 original Parquet shards were inventoried at dataset revision <code>{manifest['revision']}</code>. There are 16 nonempty source labels and 2,268,178 rows. Seed <code>20261027</code> selects 200 distinct source-relative positions per source using independent RNGs. Only the 151 shards containing selected positions were needed for retrieval.</p><p>The unit is a physical conversation record. Duplicate IDs are not silently removed. Original messages, source labels, IDs, shard positions, selected ranks, and canonical record hashes are preserved. No whitespace normalization or text truncation is applied.</p></article><article><h3>Single-model review</h3><p>GPT-6.1 Sol at high reasoning effort assessed all assistant turns against the context available at each turn. Source labels and original IDs were omitted from review payloads. The same six dimensions and severity rule as the prior audit were retained; the conversation adaptation is saved in the prompt.</p><p>All 1,226 batches completed. Technical failures were retried; the earliest technically valid whole review per record was retained. There was no ensemble, second reviewer, human adjudication, code execution, or external lookup. Exact quote validation checks text presence, not whether the judgment is correct.</p></article><article><h3>Severity and intervals</h3><p>Any major dimension makes the record major; otherwise uncertainty takes precedence over minor, and no flagged dimension yields no defect found. Dataset estimates use <code>Σ (Nₛ/N) p̂ₛ</code>. Estimated variance is <code>Σ (Nₛ/N)² (1 − nₛ/Nₛ) p̂ₛ(1−p̂ₛ)/(nₛ−1)</code>; aggregate intervals use ±1.96 standard errors.</p><p>Source intervals use Wilson approximations. All intervals condition on model labels, exclude reviewer error, and are not adjusted for multiple comparisons. “No defect found” is not proof of correctness.</p></article><article><h3>Scope and reproduction</h3><p>This audits the published snapshot, not a fresh generation run or an evaluation of the upstream model. It cannot locate where a defect was introduced or demonstrate training gains. Full conversations and their labels are embedded in this offline HTML; dataset content is rendered as inert text.</p><p>Read the <a href="https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B/tree/{manifest['revision']}">pinned dataset snapshot</a>. Local files include the compressed sample, final reviews, protocol, manifests, aggregate CSV, and report builder. Intermediate requests, retry caches, and execution logs are excluded from the Git commit.</p></article></div><details class="reproduce"><summary>Rebuild this report</summary><p>From the repository root, with Python and Matplotlib installed:</p><pre>PYTHONDONTWRITEBYTECODE=1 python smoke/dolci_quality_audit/report/build_report.py</pre><p>The builder reads either samples.jsonl or its lossless gzip archive, checks input hashes and all evidence excerpts, and creates figures and this document without new model calls. The prior SFT analysis in this repository supplies the comparison baseline. The report JSON records that file’s hash.</p></details>''')
    packed=base64.b64encode(gzip.compress(json.dumps({'analysis':a,'reviews':packets},ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()
    doc+='''<footer><b>Dolci-Think-SFT-7B · Source quality audit</b><span>Frozen evidence · model judgments · 27 October 2026</span></footer></main><dialog id="recordDialog" aria-labelledby="dialogTitle"><div class="dialog-top"><h2 id="dialogTitle">Record evidence</h2><button id="closeDialog" class="button secondary">Close</button></div><div id="dialogBody"></div></dialog><script id="packedData" type="application/octet-stream">'''+packed+'</script><script>'+(HERE/'report.js').read_text()+'</script></body></html>'
    (HERE/'index.html').write_text('\n'.join(line.rstrip() for line in doc.splitlines())+'\n')
    print(json.dumps({'records':len(reviews),'sources':len(sources),'html_bytes':(HERE/'index.html').stat().st_size,'weighted_major':major,'turn_groups':turn_rows},indent=2))


if __name__ == '__main__':
    main()
