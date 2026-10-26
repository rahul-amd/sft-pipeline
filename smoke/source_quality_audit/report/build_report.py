"""Reproduce the audit analysis, shareable figures and standalone HTML document."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import base64
import csv
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["major_defect", "minor_defect", "uncertain", "no_defect_found"]
NAMES = {"major_defect": "Major defect", "minor_defect": "Minor defect", "uncertain": "Uncertain", "no_defect_found": "No defect found"}
COLORS = {"major_defect": "#bd4b3c", "minor_defect": "#d5a44a", "uncertain": "#7c8193", "no_defect_found": "#348477"}
DIMS = {"prompt_quality": "Prompt quality", "answer_correctness": "Answer correctness", "reasoning_quality": "Reasoning quality", "instruction_following": "Instruction following", "parsing_integrity": "Parsing / generation", "training_suitability": "Training suitability"}


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, centre-half), min(1, centre+half)]


def weighted_interval(groups, populations, predicate):
    total = sum(populations.values())
    estimate = variance = 0.0
    for source, rows in groups.items():
        n, population = len(rows), populations[source]
        p = sum(predicate(r) for r in rows)/n
        w = population/total
        estimate += w*p
        variance += w*w*(1-n/population)*p*(1-p)/(n-1)
    half = 1.959963984540054*math.sqrt(variance)
    return {"estimate": estimate, "interval": [max(0,estimate-half), min(1,estimate+half)]}


def chart_style(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelcolor="#334249", pad=8)
    ax.grid(axis="x", color="#e9edea", linewidth=.7)
    ax.set_axisbelow(True)


def save_chart(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "figures" / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=160)
    plt.close(fig)


def main():
    reviews, samples = read_rows("verdicts.jsonl"), read_rows("samples.jsonl")
    originals = {r["audit_id"]: r for r in samples}
    assert len(reviews) == len(samples) == 3400
    assert {r["audit_id"] for r in reviews} == set(originals)
    assert len({r["audit_id"] for r in reviews}) == len(reviews)
    inventory = json.loads((ROOT / "sources.json").read_text())
    populations = {r["source"]: r["population"] for r in inventory}
    groups = defaultdict(list)
    for r in reviews: groups[r["source"]].append(r)
    assert set(groups) == set(populations) and all(len(rs)==100 for rs in groups.values())
    total = sum(populations.values())
    manifest = json.loads((ROOT / "sample-manifest.json").read_text())
    assert manifest["samples_sha256"] == hashlib.sha256((ROOT / "samples.jsonl").read_bytes()).hexdigest()
    counts = Counter(r["overall"] for r in reviews)
    # The editorial narrative is specific to this frozen audit. Fail rather than
    # silently combine new numeric results with old written interpretations.
    assert dict(counts) == {"major_defect":888,"minor_defect":889,"no_defect_found":1547,"uncertain":76}
    weighted = {label: weighted_interval(groups,populations,lambda r,l=label:r["overall"]==l) for label in LABELS}
    sources = []
    for source, rows in groups.items():
        c = Counter(r["overall"] for r in rows)
        dimensions = {d:dict(Counter(r["dimensions"][d] for r in rows)) for d in DIMS}
        sources.append({"source":source,"population":populations[source],"weight":populations[source]/total,
                        "sampled":len(rows),"counts":{l:c[l] for l in LABELS},
                        "major_rate":c["major_defect"]/len(rows),"major_interval":wilson(c["major_defect"],len(rows)),
                        "major_contribution_pp":100*populations[source]/total*c["major_defect"]/len(rows),"dimensions":dimensions})
    sources.sort(key=lambda r:r["major_contribution_pp"],reverse=True)
    dimensions = {d:{"counts":dict(Counter(r["dimensions"][d] for r in reviews)),
                     "weighted_major":weighted_interval(groups,populations,lambda r,d=d:r["dimensions"][d]=="major_issue")} for d in DIMS}
    major_sets = {d:{r["audit_id"] for r in reviews if r["dimensions"][d]=="major_issue"} for d in DIMS}
    overlaps = {"prompt_and_answer":len(major_sets["prompt_quality"] & major_sets["answer_correctness"]),
                "prompt_without_answer":len(major_sets["prompt_quality"]-major_sets["answer_correctness"]),
                "answer_without_prompt":len(major_sets["answer_correctness"]-major_sets["prompt_quality"]),
                "parsing_and_answer":len(major_sets["parsing_integrity"] & major_sets["answer_correctness"])}
    feature_tests = {
        "no_prompt_newlines":lambda r:"\n" not in r["prompt"],
        "fenced_prompt_without_newlines":lambda r:"```" in r["prompt"] and "\n" not in r["prompt"],
        "thought_marker_in_answer":lambda r:"<|channel>thought" in r["answer"],
        "tool_marker_in_answer":lambda r:"<|tool_call>" in r["answer"],
        "empty_answer":lambda r:not r["answer"].strip(),
        "missing_teacher_model":lambda r:not r.get("teacher_model"),
        "missing_finish_reason":lambda r:"finish_reason" not in r}
    features = {}
    for name, predicate in feature_tests.items():
        selected = [r for r in reviews if predicate(originals[r["audit_id"]]["record"])]
        features[name] = {"n":len(selected),"major":sum(r["overall"]=="major_defect" for r in selected),
                          "by_source":dict(Counter(r["source"] for r in selected))}
    assert features["fenced_prompt_without_newlines"]["n"] == 535
    assert features["thought_marker_in_answer"]["n"] == 35
    assert features["tool_marker_in_answer"]["n"] == 81
    lengths = []
    for low,high,label in [(0,2000,"Under 2,000"),(2000,5000,"2,000–4,999"),(5000,10000,"5,000–9,999"),(10000,10**9,"10,000 or more")]:
        chosen=[r for r in reviews if low<=len(originals[r["audit_id"]]["record"]["reasoning"])<high]
        k=sum(r["overall"]=="major_defect" for r in chosen)
        lengths.append({"label":label,"n":len(chosen),"major":k,"rate":k/len(chosen)})
    repaired = [json.loads(f.read_text()) for f in (ROOT / "completed").glob("*.json") if "recovery_policy" in json.loads(f.read_text())]
    examples = [
        ("2989839b6172fc5e223d25cb","Following a bad sample over the stated task","The candidate identifies the valid result, then hardcodes the contradictory sample output. This is a concrete example of prompt defects propagating into the answer.","Answer and prompt"),
        ("0d5de41fce7e2b494e8f2b68","A missing physical parameter becomes invented code","The prompt omits drag strength. The reviewer flags a response that substitutes constant velocities fitted to the example instead of the requested trajectory model.","Underspecified prompt"),
        ("00d76a5f2673079540dabd16","A tool action is exported as the final answer","A repository listing can be a valid next action in an agent trajectory. In this flattened standalone record, it does not deliver the requested code fix.","Training format mismatch"),
        ("01ea937d23130fbd1911e5f3","A correct conclusion can still sit in a flagged record","The candidate notices that two planes intersect in a line, not a solid. The reviewer nevertheless flags the contradictory premise and repetitive trace. Major flags are not synonymous with wrong final answers.","Interpret the labels"),
        ("1127ed7a2d5a5bcc143742d3","A concrete counterexample to generated code","The reviewer identifies a minus sign applied to only one regex alternative, and supplies a negative-range counterexample. This is static reviewer reasoning, not an executed test.","Code semantics"),
        ("1338c3cd371b6a1b8c7a119f","An outline replaces the requested deliverable","The Chinese prompt asks for a 3,000-character literature review. The reviewer flags the response for supplying an outline and expansion advice instead.","Task completion"),
        ("02c64d27af7ed4e46b84cef9","A straightforward usable example","The chair-count question gives all necessary inputs. The response computes 2 × 6 + 2 × 7 = 26 and the reviewer finds no defect. This example is illustrative, not evidence of broader learning gains.","No defect found"),
        ("0154be2b45c6bc6722679ce9","Unverifiable bibliography remains uncertain","The reviewer cannot establish the asserted historical publications. It records uncertainty rather than treating plausible specificity as evidence.","Uncertainty")]
    for aid,*_ in examples: assert aid in originals
    analysis = {"population":total,"sample_size":len(reviews),"sources_count":len(sources),"counts":dict(counts),
                "weighted":weighted,"sources":sources,"dimensions":dimensions,"overlaps":overlaps,"features":features,"lengths":lengths,
                "top_two_major_burden_share":sum(s["major_contribution_pp"] for s in sources[:2])/(weighted["major_defect"]["estimate"]*100),
                "recovered_batches":len(repaired),"recovered_records":sum(len(r["result"]["reviews"]) for r in repaired),
                "examples":[dict(audit_id=a,title=b,commentary=c,category=d) for a,b,c,d in examples],
                "method":{"reviewer":"gpt-6.1-sol","reasoning_effort":"high","sample_seed":manifest["seed"],
                          "weighted_interval":"95% normal interval using stratified SRS without replacement variance; finite population correction in each source",
                          "source_interval":"95% Wilson binomial approximation; no finite-population correction",
                          "caveat":"Intervals quantify record sampling uncertainty conditional on model labels; they exclude reviewer error and are not familywise-adjusted comparisons."},
                "input_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ["samples.jsonl","verdicts.jsonl","sources.json","review_prompt.txt"]}}
    (HERE / "analysis.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+"\n")
    with (HERE / "source_summary.csv").open("w",newline="") as stream:
        fields=["source","population","sampled","population_share_pct","major_n","minor_n","uncertain_n","no_defect_n","major_rate_pct","major_ci_low_pct","major_ci_high_pct","major_contribution_pp"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for s in sources:writer.writerow(dict(zip(fields,[s["source"],s["population"],s["sampled"],100*s["weight"],s["counts"]["major_defect"],s["counts"]["minor_defect"],s["counts"]["uncertain"],s["counts"]["no_defect_found"],100*s["major_rate"],100*s["major_interval"][0],100*s["major_interval"][1],s["major_contribution_pp"]])))
    (HERE / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"svg.fonttype":"none"})
    impact=sources[:8]
    fig,ax=plt.subplots(figsize=(10.5,4.5));chart_style(ax)
    ax.barh(range(len(impact)),[s["major_contribution_pp"] for s in impact],color=["#bd4b3c"]*2+["#637e7c"]*6,height=.58)
    ax.set_yticks(range(len(impact)),[s["source"] for s in impact],fontsize=10);ax.invert_yaxis();ax.set_xlim(0,9)
    for i,s in enumerate(impact):ax.text(s["major_contribution_pp"]+.12,i,f'{s["major_contribution_pp"]:.2f}',va="center",fontsize=11)
    ax.set_xlabel("Contribution to dataset-wide major-flag estimate (percentage points)",labelpad=12)
    fig.tight_layout();save_chart(fig,"source-impact")
    fig,ax=plt.subplots(figsize=(10.5,4.2));chart_style(ax)
    left=[0.0]*len(DIMS)
    for state,color,label in [("major_issue","#bd4b3c","Major"),("minor_issue","#d5a44a","Minor"),("uncertain","#7c8193","Uncertain"),("pass","#348477","Pass / not applicable")]:
        vals=[100*(dimensions[d]["counts"].get(state,0)+(dimensions[d]["counts"].get("not_applicable",0) if state=="pass" else 0))/len(reviews) for d in DIMS]
        ax.barh(range(len(DIMS)),vals,left=left,color=color,height=.56,label=label)
        left=[a+b for a,b in zip(left,vals)]
    ax.set_yticks(range(len(DIMS)),list(DIMS.values()));ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel("Percent of the equal-source audit sample",labelpad=10)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.03),ncol=4,frameon=False,fontsize=10)
    fig.tight_layout();save_chart(fig,"dimension-profile")
    ranked=sorted(sources,key=lambda s:s["major_rate"],reverse=True)
    fig,ax=plt.subplots(figsize=(10,13));chart_style(ax)
    for i,s in enumerate(ranked):
        p=s["major_rate"]*100;lo,hi=[x*100 for x in s["major_interval"]]
        ax.errorbar(p,i,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt="o",markersize=4,color="#bd4b3c",ecolor="#b4bbb8",capsize=2)
    ax.set_yticks(range(len(ranked)),[s["source"] for s in ranked],fontsize=9);ax.invert_yaxis();ax.set_xlim(-2,104)
    ax.set_xlabel("Major flags per 100 sampled records · 95% Wilson intervals")
    fig.tight_layout();save_chart(fig,"source-major-intervals")
    def svg(name):
        text=(HERE/"figures"/f"{name}.svg").read_text()
        return text[text.index("<svg"):]
    esc=html.escape
    source_rows=[]
    for s in sources:
        n=s["counts"];lo,hi=s["major_interval"]
        share_display=f'{100*s["weight"]:.4f}' if 100*s["weight"]<.01 else f'{100*s["weight"]:.2f}'
        impact_display='&lt;0.01' if s["major_contribution_pp"]<.01 else f'{s["major_contribution_pp"]:.2f}'
        segments=''.join(f'<span class="seg {l}" style="width:{n[l]}%" title="{NAMES[l]}: {n[l]}/100"></span>' for l in LABELS)
        top=max(DIMS,key=lambda d:s["dimensions"][d].get("major_issue",0))
        source_rows.append(f'<tr data-source="{esc(s["source"],quote=True)}" data-major="{s["major_rate"]}" data-impact="{s["major_contribution_pp"]}" data-pop="{s["population"]}"><td><button class="source-link" data-open-source="{esc(s["source"],quote=True)}">{esc(s["source"])}</button><small>{DIMS[top]}: {s["dimensions"][top].get("major_issue",0)} major flags</small></td><td>{s["population"]:,}<small>{share_display}% of data</small></td><td><div class="mini-stack">{segments}</div><small>{n["major_defect"]} major · {n["minor_defect"]} minor · {n["uncertain"]} uncertain · {n["no_defect_found"]} clear</small></td><td><strong>{n["major_defect"]}%</strong><small>{100*lo:.1f}–{100*hi:.1f}% interval</small></td><td>{impact_display}<small>percentage points</small></td></tr>')
    mix_rows=''.join(f'<tr><td><span class="dot {l}"></span>{NAMES[l]}</td><td>{counts[l]:,}</td><td>{100*counts[l]/len(reviews):.1f}%</td><td><strong>{100*weighted[l]["estimate"]:.1f}%</strong></td><td>{100*weighted[l]["interval"][0]:.1f}–{100*weighted[l]["interval"][1]:.1f}%</td></tr>' for l in LABELS)
    dimension_rows=''.join(f'<tr><td>{DIMS[d]}</td><td>{dimensions[d]["counts"].get("major_issue",0)}</td><td>{100*dimensions[d]["counts"].get("major_issue",0)/len(reviews):.1f}%</td><td>{100*dimensions[d]["weighted_major"]["estimate"]:.1f}%</td></tr>' for d in DIMS)
    length_rows=''.join(f'<tr><td>{r["label"]}</td><td>{r["n"]:,}</td><td>{r["major"]}</td><td>{100*r["rate"]:.1f}%</td></tr>' for r in lengths)
    cards=''.join(f'<article class="case"><div class="eyebrow">{esc(d)}</div><h3>{esc(b)}</h3><p>{esc(c)}</p><button class="text-button" data-open-record="{a}">Inspect the record <span aria-hidden="true">↗</span></button></article>' for a,b,c,d in examples)
    packets=[{**r,"record":originals[r["audit_id"]]["record"],"raw_line_sha256":originals[r["audit_id"]]["raw_line_sha256"]} for r in reviews]
    packed=base64.b64encode(gzip.compress(json.dumps({"analysis":analysis,"reviews":packets},ensure_ascii=False,separators=(",",":")).encode(),mtime=0)).decode()
    replacements={"@@CSS@@":(HERE/"report.css").read_text(),"@@JS@@":(HERE/"report.js").read_text(),"@@DATA@@":packed,
                  "@@SOURCE_ROWS@@":"\n".join(source_rows),"@@MIX_ROWS@@":mix_rows,"@@DIMENSION_ROWS@@":dimension_rows,"@@LENGTH_ROWS@@":length_rows,
                  "@@CASES@@":cards,"@@IMPACT_SVG@@":svg("source-impact"),"@@DIMENSION_SVG@@":svg("dimension-profile"),
                  "@@WEIGHTED_MAJOR@@":f'{100*weighted["major_defect"]["estimate"]:.1f}',
                  "@@TOP_TWO@@":f'{100*analysis["top_two_major_burden_share"]:.1f}'}
    document=(HERE/"report_template.html").read_text()
    for key,value in replacements.items():document=document.replace(key,value)
    assert "@@" not in document
    (HERE/"index.html").write_text(document)
    print(json.dumps({"records":len(reviews),"sources":len(sources),"population":total,"html_bytes":len(document.encode()),"weighted_major_rate":weighted["major_defect"]},indent=2))


if __name__ == "__main__":
    main()
