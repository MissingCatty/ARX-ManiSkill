"""Publish the existing V7 offline comparison with independent resource copies."""
import json
import re
import shutil

from arx_project.layout import ROOT,now,read_json,sha256,write_json


def table(summaries,raw_total,cases,priority_successes):
    labels={'control':'原候选','nms':'仅NMS','geometry_nms':'几何＋NMS','geometry_empty_nms':'几何＋空区＋NMS'}
    lines=['## 完整离线比较进度','',f'已处理{cases}/69份观测。下表的优先候选数为每场景中位数，耗时为CPU后处理耗时。','',
        '| 方案 | 位置/朝向去重阈值 | 优先候选数 | 优先候选减少 | 原成功抓法在原手臂上保留 | 原成功抓法仍为优先代表 | 耗时中位数 / P90 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,v in summaries.items():
        prefix,threshold=(name.rsplit('_p',1) if name!='control' else ('control',None))
        if threshold:
            mm,degree=threshold.split('_r');description=f'{int(mm)}毫米 / {int(degree)}°'
        else:description='—'
        reduced=1-v['representatives_total']/raw_total if raw_total else 0
        lines.append(f"| {labels[prefix]} | {description} | {v['representatives_median']:.0f} | {reduced:.1%} | "
            f"{v['original_successful_same_arm_available']}/{v['original_successful_cases']} | "
            f"{priority_successes[name]}/{v['original_successful_cases']} | "
            f"{v['postprocessing_median_ms']:.1f} / {v['postprocessing_p90_ms']:.1f}毫秒 |")
    lines+=['','被NMS合并的候选另存为后备，几何筛选拒绝的候选另记原因。'
        '表中的“保留”包含后备候选，并核查历史上选中的同一只手。','']
    return '\n'.join(lines)


def publish():
    website=ROOT/'website';cfg=read_json(ROOT/'configs/experiments/graspgenx_postprocessing.json')
    run=(ROOT/cfg['offline']['active_run']).resolve()
    if not run.is_relative_to(ROOT/'tmp/graspgenx_postprocessing'):raise ValueError('Wrong V7 run root')
    progress=read_json(run/'progress.json');reg=read_json(run/'registration.json')
    accepted=read_json(run/'acceptance.json') if (run/'acceptance.json').exists() else None
    checked=read_json(run/'independent_review.json') if (run/'independent_review.json').exists() else None
    completed=progress['completed_cases'];rows=[]
    for case in reg['cases'][:completed]:
        rows.append(json.loads((run/'cases'/case['id']/'result.json').read_text()))
    site_path=website/'内容/网站.json';registry_path=website/'内容/experiments.json'
    site,registry=read_json(site_path),read_json(registry_path)
    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('V7 attachment must be an independent copy')
        shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise ValueError('V7 attachment copy failed')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    doc=ROOT/'modules/pipeline/docs/mainline_versions/V7_GraspGenX候选后处理实验方案.md'
    attach(doc,'sam3-v7-graspgenx-postprocessing-plan.md')
    attach(ROOT/'configs/experiments/graspgenx_postprocessing.json','sam3-v7-graspgenx-postprocessing-config.json')
    attach(ROOT/cfg['offline']['component_preflight']/'acceptance.json','sam3-v7-component-equivalence.json')
    snapshot={'at':now(),'source_run':str(run.relative_to(ROOT)),'phase':progress['phase'],
        'completed_cases':completed,'total_cases':len(reg['cases']),
        'total_raw_candidates':sum(row['raw_count'] for row in rows),
        'summaries':progress.get('summaries',{}),'cases':rows,'acceptance':accepted,
        'independent_review':checked,'planning_requests':0,'physical_grasps':0}
    snapshot_path=run/'publication_snapshot.json';write_json(snapshot_path,snapshot)
    data_url=attach(snapshot_path,'sam3-v7-offline69-results.json')
    status=('离线比较与记录核验已完成' if checked and checked['passed'] else
        '离线比较已结束，记录核验中' if progress['phase']=='complete' else '69份离线比较进行中')
    body=doc.read_text().split('\n',1)[1].lstrip()
    body=body.replace('(V7_候选后处理离线对照结果.md)','(/media/sam3-v7-offline69-report.md)')
    body=re.sub(r'2026年11月26日：[^\n]+',f'当前进度：{status}，已完成{completed}/69份观测。',body,count=1)
    priority_successes={name:sum(row['original_physical_grasp_success'] and
        row['variants'][name]['retention']['selected_is_priority_representative'] and
        row['variants'][name]['retention']['original_selection_same_arm_available'] for row in rows)
        for name in snapshot['summaries']}
    section=table(snapshot['summaries'],snapshot['total_raw_candidates'],completed,priority_successes) if rows else ''
    if checked:
        section+=('\n逐候选记录、来源关系和汇总数字已独立核对通过。'
            '下一轮优先比较15毫米/15°的仅NMS方案；几何硬筛选误删3条原成功抓法，先保留为诊断。\n\n')
    section+=f'[69份逐例数据与筛选原因]({data_url}) · [CPU批量计算等值核验](/media/sam3-v7-component-equivalence.json)\n\n'
    body=body.replace('## 本轮输入与分组',section+'[[流程卡片]]\n\n## 本轮输入与分组',1)
    report=ROOT/'modules/pipeline/docs/mainline_versions/V7_候选后处理离线对照结果.md'
    if report.exists():
        url=attach(report,'sam3-v7-offline69-report.md')
        body=body.replace('## 本轮输入与分组',f'[完整结果与误删分析]({url})\n\n## 本轮输入与分组',1)
    diagnosis=ROOT/'tmp/graspgenx_postprocessing/diagnosis_20261126_01/acceptance.json'
    if diagnosis.exists():
        url=attach(diagnosis,'sam3-v7-selected-grasp-diagnosis.json')
        body=body.replace('## 本轮输入与分组',f'[原抓法被筛除的逐例定位]({url})\n\n## 本轮输入与分组',1)
    body+='\n## 参考方法\n\n[ZeroGrasp](/papers/zerograsp/) · [GraspGenX](/papers/graspgenx/)\n'
    (website/'内容/迭代/sam3-graspgenx-v7.md').write_text(body)
    group=next(v for v in registry['experiments'] if v['id']=='sam3-graspgenx')
    version=next(v for v in group['iterations'] if v['code']=='V7')
    version.update(status=status,resultSummary=f'离线比较{completed}/69份；候选来源、成功抓法保留与CPU耗时分组记录。',
        scenes='最终V6的69份冻结候选观测，共166008个原始候选。')
    version['settings']=[['候选分组','原候选、仅NMS、真实ARX几何＋NMS、独立空夹持区组。'],
        ['去重阈值','5毫米/5°、10毫米/10°、15毫米/15°、30毫米/30°。'],
        ['抓法保留','保留原位姿、开度、评分和编号；记录代表与后备，并核查原选中手臂。'],
        ['本轮测量','候选数、原成功抓法保留和CPU后处理耗时。']]
    batch={'id':run.name,'title':'69份冻结观测的候选后处理离线比较','date':'2026-11-26',
        'status':status,'summary':version['resultSummary'],'outputs':[str(run.relative_to(ROOT))],
        'artifacts':[{'label':'逐例候选与耗时','url':data_url}]}
    version['batches']=[v for v in version['batches'] if v['id']!=run.name]+[batch]
    write_json(site_path,site);write_json(registry_path,registry)
    print(f'Published V7 offline comparison: {completed}/69; current version remains {group["current"]}.')


if __name__=='__main__':publish()
