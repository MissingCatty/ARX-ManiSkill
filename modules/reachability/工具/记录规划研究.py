"""Append a completed mechanical result; keep only improvements with no regression."""
from 公共 import EXPERIMENTS
import argparse
import csv
from pathlib import Path
import sys

from 公共 import ROOT,read,save,now
from 评估规划研究 import evaluate

CAMPAIGN=EXPERIMENTS / '自动研究_20261120_01'


def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--轮次',type=int,required=True);p.add_argument('--说明',required=True);args=p.parse_args()
    directory=CAMPAIGN/'轮次'/args.name
    baseline=CAMPAIGN/'轮次/00_基线';result=evaluate(directory,baseline)
    state=read(CAMPAIGN/'状态.json')
    baseline_score=evaluate(baseline)['score'];best=state.get('best_score',baseline_score)
    status='baseline' if args.轮次==0 else 'kept' if result['pass'] and result['score']>best else 'reverted'
    if args.轮次==0 or status=='kept':state.update(best_score=result['score'],best_name=args.name)
    state.update(completed_iterations=max(state.get('completed_iterations',0),args.轮次),latest_evaluation=result)
    save(CAMPAIGN/'状态.json',state);save(directory/'机械评估.json',{**result,'decision':status})
    delta=result['score']-baseline_score
    with (CAMPAIGN/'迭代结果.tsv').open('a',newline='') as stream:
        csv.writer(stream,delimiter='\t').writerow([args.轮次,result['score'],delta,f'{delta/baseline_score:.1%}' if baseline_score else 'n/a',status,args.说明,'mechanical',now()])
    with (CAMPAIGN/'研究日志.md').open('a') as stream:
        stream.write(f"\n## 第 {args.轮次} 轮：{args.说明}\n\n{result['passes']}/{result['total']} 通过；新增 {len(result['gains'])}，退步 {len(result['regressions'])}。决定：{status}。原始产物：轮次/{args.name}/。\n")
    with (CAMPAIGN/'研究计划.md').open('a') as stream:
        stream.write(f"| {args.轮次} | {args.说明} | {result['passes']}/{result['total']} | {len(result['gains'])}/{len(result['regressions'])} | {status} |\n")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,'/home/zhaoyechong/.codex/skills/autoresearch-skill/scripts')
    from style_presets import rcparams
    rcparams()
    with (CAMPAIGN/'迭代结果.tsv').open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
    x=[int(r['iteration']) for r in rows];y=[float(r['metric_value'])*100 for r in rows]
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(x,y,'o-',color='#0072B2')
    ax.axhline((baseline_score+4/12)*100,ls='--',color='#D55E00',label='Development target (+4/12)')
    ax.set(xlabel='Iteration',ylabel='Initial planning pass (%)',ylim=(-3,103),xticks=x)
    ax.legend(fontsize=10);fig.savefig(CAMPAIGN/'进度.png',dpi=150);plt.close(fig)
    print(result,status)


if __name__=='__main__':main()
