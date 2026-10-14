"""Measured counts from the earlier medium SFT validation; not a release inventory."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with (ROOT / 'sft-medium-retention.csv').open() as f:
    rows = list(csv.DictReader(f))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(12.5, 6.5))
fig.subplots_adjust(left=.22, right=.94, top=.76, bottom=.25)
values = [int(r['records']) for r in rows]
y = list(range(len(rows)))
ax.barh(y, values, color=['#c3d7ed', '#aecce7', '#76a5ce', '#76a5ce', '#439985'], height=.65)
ax.set_yticks(y, [r['stage'] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 12300)
ax.set_xticks([0, 3000, 6000, 9000, 12000], ['0', '3,000', '6,000', '9,000', '12,000'])
ax.set_xlabel('Records in this validation run')
ax.xaxis.grid(True, color='#e5eaf0')
ax.set_axisbelow(True)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(axis='both', length=0)
for i, n in enumerate(values): ax.text(n+120, i, f'{n:,}', va='center', weight='bold', color='#172b42')
fig.text(.04,.94,'An earlier SFT run completed unattended',fontsize=22,weight='bold',color='#172b42')
fig.text(.04,.885,'MEASURED · 2026-10-08 medium validation · 51 min 14 s',fontsize=13,color='#52677d')
fig.text(.04,.83,'Before the latest grounding and judge changes; not the released dataset inventory.',fontsize=12,color='#52677d')
fig.text(.04,.13,'Inputs: 9,924 collected + 1,002 generated. Selection: 3,805 / 4,000 target; scarce hard examples.',fontsize=11,color='#52677d')
fig.text(.04,.09,'Selection is budget allocation, not a quality rejection. Export counts do not measure answer correctness.',fontsize=11,color='#52677d')
fig.text(.04,.05,'Source: sft-pipeline/smoke/freeze-validation.md. All 3,805 inference requests completed; filtering removed 674.',fontsize=10,color='#52677d')
fig.savefig(ROOT / 'sft-medium-retention.svg')
fig.savefig(ROOT / 'sft-medium-retention.png', dpi=160)
plt.close(fig)
