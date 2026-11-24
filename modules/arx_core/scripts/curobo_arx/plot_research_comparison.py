"""Plot the latest completed ARX comparison; never queries or controls simulation."""
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0,'/home/zhaoyechong/.codex/skills/autoresearch-skill/scripts')
from style_presets import rcparams
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();root=Path(a.campaign);report=json.loads((root/'comparison.json').read_text());rows=[(k,v) for k,v in report['replicate_counts'].items() if v['candidate']['attempts']]
rcparams();fig,ax=plt.subplots(figsize=(max(7,len(rows)*1.55),4.7))
for i,(case,values) in enumerate(rows):
 for off,arm,color in [(-.18,'baseline','#65748b'),(.18,'candidate','#0072B2')]:
  n=values[arm]['attempts'];s=values[arm]['successes'];rate=s/n if n else 0
  ax.bar(i+off,rate,width=.32,color=color,label=arm if i==0 else None);ax.text(i+off,rate+.035,f'{s}/{n}',ha='center',fontsize=11)
ax.set_xticks(range(len(rows)),[k.rsplit('/',1)[-1] for k,v in rows],rotation=20,ha='right');ax.set_ylim(0,1.18);ax.set_ylabel('Successful held grasp / attempts');ax.set_title('Selected ARX cases; all repetitions retained');ax.legend();fig.tight_layout();fig.savefig(root/'progress.png',dpi=160);plt.close(fig)
print(root/'progress.png')
