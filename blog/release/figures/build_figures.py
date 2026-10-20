"""Build original SVG diagrams; standard library only. No experimental results implied."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
INK = '#172b42'; MUTED = '#52677d'; BLUE = '#e6f0ff'; TEAL = '#e1f4ee'; AMBER = '#fff0d5'; LINE = '#62798d'

def text(x, y, label, size=18, color=INK, weight='normal', anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(label)}</text>'

def lines(x,y,labels,size=17,dy=24,**kwargs):
    return ''.join(text(x,y+i*dy,s,size,**kwargs) for i,s in enumerate(labels))

def box(x,y,w,h,labels,fill=BLUE,size=17):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#bbcad8"/>'+lines(x+18,y+28,labels,size)

def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{LINE}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'

def frame(name,w,h,title,subtitle,body,desc):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{LINE}"/></marker></defs>
<rect width="{w}" height="{h}" fill="#ffffff"/><g font-family="DejaVu Sans, Arial, sans-serif">
{text(32,43,title,27,weight='bold')}{text(32,74,subtitle,16,MUTED)}{body}</g></svg>'''
    (OUT/f'{name}.svg').write_text(svg)

# Topology: independent repositories, no inferred cross-repository merger.
b = text(38,120,'SFT: curate the question, then author the answer',19,weight='bold')
b += box(38,141,250,72,['1 · Collect prompts','Normalize; exact dedup'],size=16)
b += box(308,141,270,72,['2 · Generate from sources','Grounding + evidence'],size=16)
b += arrow(163,213,163,237)+arrow(443,213,443,237)
b += '<path d="M163 237 L443 237 M303 237 L303 256" stroke="#62798d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
sft=[(260,68,['Configured benchmark decontamination']), (353,68,['3 · Embed → cluster → annotate']), (446,72,['4 · Quota selection','Optional semantic dedup: default off']), (543,72,['5 · Teacher inference','OpenAI-compatible API']), (640,82,['6 · Structural checks, code checks, judge','Judge sampling is configurable']), (747,62,['Final SFT JSONL export'])]
for i,(y,h,labels) in enumerate(sft):
    b+=box(38,y,540,h,labels,TEAL if i==5 else BLUE,17)
    if i<len(sft)-1: b+=arrow(303,y+h,303,sft[i+1][0]-5)
b+=lines(38,846,['Evidence and decisions stay in audit artifacts.','Large runs may export/import external inference.'],16,color=MUTED)
b+=text(638,120,'LC: construct distinct long-context tasks',19,weight='bold')
b+=box(638,141,550,76,['Synthetic task definitions','RULER++: 22 deterministic tasks'],TEAL)
b+=box(638,245,550,76,['Quality-filtered source documents','Single docs, passage pools, or embedding clusters'],TEAL)
b+=arrow(913,321,913,348)
b+=box(638,353,550,95,['Deterministic corpus tasks','Untie: reconstruction / permutation','MRCR: retrieval · IDK: answer / abstain'],TEAL)
b+=box(638,478,550,120,['Model-authored corpus tasks','QA: sampled chunks → request → assemble','Summary: whole document → request → assemble','Inference: built-in API or external engine'],AMBER,size=16)
b+='<path d="M1188 283 L1210 283 L1210 538 L1194 538" stroke="#62798d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
b+=box(638,653,550,95,['Per-generator PT text or SFT messages','Complete-record length checks','Deterministic golds or model-generated answers'],TEAL)
b+='<path d="M638 181 L613 181 L613 700 L632 700" stroke="#62798d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
b+='<path d="M638 403 L624 403 L624 677 L632 677" stroke="#62798d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
b+=arrow(913,598,913,648)
b+=lines(638,793,['Clustering supports Untie clustered sources;','it is not required by every corpus task.','No shared SFT–LC training merger is implied.'],16,color=MUTED)
frame('pipeline-overview',1250,916,'Two repositories, two complementary data recipes','IMPLEMENTATION DIAGRAM · Paths describe the frozen code; output quality requires separate evaluation.',b,'Left: SFT collection and source generation converge before decontamination, embedding, selection, inference and filtering. Right: deterministic synthetic, deterministic corpus and model-authored corpus generators emit separate PT or SFT datasets.')

b=lines(40,126,['A training budget covers the complete serialized example, including its answer.','The same maximum length can imply very different supervision.'],21,dy=31)
for y,title,parts in [(230,'Untie reconstruction',[(80,'Template',AMBER),(400,'Shuffled source',BLUE),(400,'Reconstructed text',TEAL)]),(407,'Short-answer retrieval / abstention',[(80,'Template',AMBER),(690,'Long context + query',BLUE),(110,'Answer',TEAL)])]:
    b+=text(40,y,title,21,weight='bold')
    x=40
    for w,label,c in parts:
        b+=f'<rect x="{x}" y="{y+26}" width="{w}" height="66" fill="{c}" stroke="#ffffff" stroke-width="3"/>'
        if w<120:
            b+=text(x+w/2,y+65,label,13,anchor='middle')
        else: b+=text(x+w/2,y+65,label,18,anchor='middle')
        x+=w
    b+=text(40,y+122,'Complete serialized record ≤ configured token limit',17,MUTED)
b+=lines(40,590,['ILLUSTRATIVE · Segment widths are conceptual, not measurements or fixed ratios.','Count with the intended tokenizer and training chat template.','PT text and SFT messages can have different measured lengths.','128K smoke coverage established serialization fit, not 128K model understanding.'],17,dy=28,color=MUTED)
frame('context-budget',980,745,'Long input and long output spend the same budget','IMPLEMENTATION CONCEPT · Why input-length limits alone are insufficient.',b,'Reconstruction uses substantial input and output space, while short-answer retrieval spends most of its sequence on context. Segment widths are illustrative.')

b=''
for x,title,labels,c in [(35,'1 · Does it run?',['Integrity and recovery','IDs, joins, token budgets','Real API and sandbox paths','Evidence: integration tests'],BLUE),(415,'2 · Is supervision sound?',['Human-reviewed examples','Support, correctness, coverage','Judge calibration + uncertainty','Evidence: held-out labels'],TEAL),(795,'3 · Does a student improve?',['Matched training comparison','Same base and token budget','Paired seeds; held-out tasks','Evidence: downstream results'],AMBER)]:
    b+=text(x,139,title,22,weight='bold')+box(x,162,350,164,labels,c,size=17)
b+=box(35,382,1110,113,['Controls shared across proposed comparisons','Freeze source splits, templates, model versions, serialization and evaluation before running.','Separate public-dataset comparisons from internal causal ablations.'], '#f2f5f8',18)
b+=lines(35,548,['MEASURED: software integration evidence exists for particular configurations.','PROPOSED: representative human quality estimates and controlled student comparisons.','A passing test suite does not supply the missing quality or learning evidence.'],19,dy=32,color=MUTED)
frame('evaluation-design',1180,680,'Three claims need three kinds of evidence','EVIDENCE DESIGN · Read each result at the scope it actually tests.',b,'Three independent claims require integration tests, held-out human labels and controlled downstream student comparisons respectively.')

b=''
for x,title,subtitle in [(40,'General SFT transfer','Per-task score by dataset arm'),(630,'Long-context transfer','Per-task score by context length')]:
    b+=text(x,131,title,23,weight='bold')+text(x,162,subtitle,17,MUTED)
    b+=f'<rect x="{x}" y="190" width="530" height="320" rx="12" fill="#f8fafc" stroke="#a6b7c7" stroke-width="2" stroke-dasharray="8 6"/>'
    b+=text(x+265,328,'RESULTS NOT YET MEASURED',22,MUTED,weight='bold',anchor='middle')
    b+=text(x+265,364,'No points, bars or implied ordering',17,MUTED,anchor='middle')
b+=lines(40,563,['PLANNED · Fill only from frozen experiment artifacts; do not reuse published-model scores.','Show all selected arms, paired-seed variation and task-level confidence intervals.','Record tokenizer-measured context lengths, train tokens, target tokens and GPU-hours.'],18,dy=29,color=MUTED)
frame('comparison-placeholder',1200,690,'Does the generated data improve learning?','PLANNED FIGURE · No downstream training comparisons have been run for this release sketch.',b,'Two deliberately empty frames reserve space for general SFT and long-context transfer results. No experimental values are present.')
print('Built four SVG assets.')
