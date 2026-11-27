"""Publish the existing V7 offline comparison with independent resource copies."""
import json
import re
import shutil

from arx_project.layout import ROOT,now,read_json,sha256,write_json


def planning_section(website,site):
    active=ROOT/'tmp/graspgenx_postprocessing/planning_active.json'
    if not active.exists():return '',None,[]
    pointer=read_json(active);run=ROOT/pointer['run']
    if not run.resolve().is_relative_to(ROOT/'tmp/graspgenx_postprocessing'):
        raise ValueError('Wrong V7 planning publication root')
    reg=read_json(run/'registration.json');progress=read_json(run/'progress.json')
    if progress.get('results') and not progress.get('summaries'):
        from modules.pipeline.experiments.graspgenx_postprocessing.planning import summaries
        progress={**progress,'summaries':summaries(progress['results'])}
    review=read_json(run/'planning_review.json') if (run/'planning_review.json').exists() else None
    status=('规划对照已完成并核验' if review and review['passed'] else
        '本批规划停止，记录保留' if (run/'STOP').exists() else
        '规划结果核验中' if progress['phase']=='complete' else
        '规划模型预热中' if progress['phase']=='warming' else
        '等待通过CUDA核验的显卡' if progress['phase']=='waiting_for_gpu' else '规划对照进行中')
    name='sam3-v7-'+run.name.replace('_','-')+'-results.json'
    target=website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
        raise ValueError('V7 planning results must be a standalone resource')
    snapshot={'at':now(),'run':pointer['run'],'status':status,'registration':{k:reg[k] for k in
        ['phase','gpu','planning_requests','variants','same_backend','same_scheduler','geometry_hard_filter_enabled']},
        'progress':progress,'independent_review':review,'physical_grasps':0}
    write_json(target,snapshot);site['localAssets'][name]='../../resources/media/'+name
    phase='五场景先导' if reg['phase']=='pilot' else '完整69场景'
    lines=['## NMS候选顺序的实际规划对照','',f"{phase}：{status}，完成{progress['completed_requests']}/{reg['planning_requests']}次规划。",'',
        '两组使用同一批量16 IK、同一卡、同一个常驻模型池和相同16/64双臂轮换。'
        '原候选组按原评分顺序搜索；NMS组先检查15毫米/15°代表，其余原候选作为后备。',
        '几何硬筛选关闭，抓点、评分、开度、碰撞规则与成功判据保持。','']
    if progress.get('summaries'):
        lines+=['| 方法 | 规划通过 | 实际检查候选中位数 | 后处理耗时中位数 | 完整规划耗时中位数 / P90 | 后处理＋规划中位数 / P90 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for mode,v in progress['summaries'].items():
            plan=v['planning_request_seconds'];total=v['postprocessing_plus_planning_seconds']
            label='原候选，批量16' if mode=='control' else 'NMS15＋后备，批量16'
            lines.append(f"| {label} | {v['successes']}/{v['requests']} | {v['candidate_checks']['median']:.0f} | "
                f"{v['postprocessing_seconds']['median']*1000:.1f}毫秒 | {plan['median']:.2f} / {plan['p90']:.2f}秒 | "
                f"{total['median']:.2f} / {total['p90']:.2f}秒 |")
        lines+=['','表格按已经完成的配对场景统计。完整规划耗时包含开爪检查、双臂搜索、IK、碰撞和路径。','']
    if review:
        lines.append(f"已核查{review['checked_endpoints']}个预抓取、抓取和抬升终点，以及{review['checked_original_candidate_rows']:,}行原候选来源。")
    lines += ['',f'[本批逐例规划与延时](/media/{name})','']
    from website.tools.graspgenx_postprocessing_planning_report import write_report
    report=write_report(run,reg,progress,review)
    if report:
        report_name='sam3-v7-'+run.name.replace('_','-')+'-report.md';target=website/'resources/media'/report_name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Planning report must be an independent copy')
        shutil.copy2(report,target);site['localAssets'][report_name]='../../resources/media/'+report_name
        lines+=[f'[本批完整规划报告](/media/{report_name})','']
    plan=ROOT/'modules/pipeline/docs/mainline_versions/V7_候选后处理规划对照方案.md'
    if plan.exists():
        plan_name='sam3-v7-NMS-planning-plan.md';target=website/'resources/media'/plan_name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Planning method must be an independent copy')
        shutil.copy2(plan,target);site['localAssets'][plan_name]='../../resources/media/'+plan_name
        lines+=[f'[规划对照实施方案](/media/{plan_name})','']
    batch={'id':run.name,'title':phase+'NMS规划对照','date':'2026-11-26','status':status,
        'summary':f"完成{progress['completed_requests']}/{reg['planning_requests']}次纯规划，同批量16后端。",
        'outputs':[pointer['run']],'artifacts':[{'label':'逐例规划与延时','url':'/media/'+name}]}
    return '\n'.join(lines),status,[batch]


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
    planning,planning_status,planning_batches=planning_section(website,site)
    if planning:body=planning+'\n'+body
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
    for item in planning_batches:
        version['batches']=[v for v in version['batches'] if v['id']!=item['id']]+[item]
    if planning_status:
        version.update(status=planning_status,resultSummary=planning_batches[-1]['summary'])
    from website.tools.graspgenx_native_postprocessing_section import section as native_section
    native,native_status,native_batch=native_section(website,site)
    if native:
        page=website/'内容/迭代/sam3-graspgenx-v7.md'
        page.write_text(native+'\n'+page.read_text())
        version.update(status=native_status,resultSummary=native_batch['summary'])
        version['settings']=[['当前候选处理','从原始GraspGenX数组接入空抓检测、宽深修正、碰撞复检和NMS，真正删除淘汰候选。'],
            ['宽度对应','候选宽度、GPU碰撞模型和规划结果中的开爪命令一致；双臂三种宽度先预检。'],
            ['方案分组','原版逻辑移植、修正后重新验碰撞、ARX实际指腹适配。'],
            ['历史对照','上一轮仅NMS代表优先＋原候选后备的69场景结果继续保留。']]
        version['batches']=[v for v in version['batches'] if v['id']!=native_batch['id']]+[native_batch]
    write_json(site_path,site);write_json(registry_path,registry)
    print(f'Published V7 offline comparison: {completed}/69; current version remains {group["current"]}.')


if __name__=='__main__':publish()
