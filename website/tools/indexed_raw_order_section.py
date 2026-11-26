"""Publish original-order computation trials while preserving earlier V8 results."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT,sha256,write_json

DOCUMENT=ROOT/'modules/pipeline/docs/mainline_versions/V8_保留原顺序的计算优化对照.md'


def raw(path):return json.loads(Path(path).read_text())


def state():
    active=raw(ROOT/'tmp/indexed_grasp_planning/funnel_active.json')
    if not active.get('raw_order_run'):return None
    planning=ROOT/active['raw_order_run'];physical=ROOT/active['raw_order_physical_run'] if active.get('raw_order_physical_run') else None
    p=raw(planning/'progress.json');q=raw(physical/'progress.json') if physical else None
    proof=raw(planning/'independent_review.json') if (planning/'independent_review.json').exists() else None
    result=raw(physical/'acceptance.json') if physical and (physical/'acceptance.json').exists() else None
    status=f"原序计算优化：规划{p['completed_requests']}/138次"
    if (planning/'STOP').exists():status+='，故障停止'
    elif proof:status+='，规划核验通过' if proof['timing_gate_passed'] else '，规划完成、时延未达标'
    if q:
        status+=f"；实抓{q['completed_requests']}/20次"
        if (physical/'STOP').exists():status+='，故障停止'
        elif result:status+='，验收通过' if result['optimization_accepted'] else '，完成、未通过组合启用门槛'
    return planning,physical,p,q,proof,result,status


def document(website):
    values=state()
    if not values:return None
    planning,physical,p,q,proof,result,status=values
    lines=['# V8保留原顺序的计算优化对照','',status+'。','',
        '这一轮保持原始抓法评分顺序，只比较计算加速。原来的NMS组合及20次实抓结果完整保留在上一轮章节。','',
        '| 项目 | 当前链路 | 本轮优化组 |','| --- | --- | --- |',
        '| 候选选择 | 原始评分顺序 | 相同，NMS关闭 |',
        '| 双臂搜索 | 首轮16、后续64交替 | 相同 |',
        '| 抓取IK | 逐候选 | 每批16个 |',
        '| 配置准备 | 原YAML写出 | C版YAML写出 |',
        '| 网格查询 | 原排列 | 同一网格连续查询 |','',
        '先用69份既有观测做138次配对规划，核对原抓法、选臂、关节终点和候选原值。'
        '正确性通过，且累计规划省时至少5%、中位数下降、P90增幅不超过5%后，再执行原10个场景的20次实抓对照。','',
        '每例使用相同的冻结感知与候选。实抓恢复相同的底盘、升降、物体及关节状态，'
        '保持871球、TCP43、V2指尖余量、物理和进入开度。计划抬升3厘米，最低保持抬升1厘米、持续3秒，'
        '并满足双指接触及禁止接触检查。初始规划计时包括候选读取、共享内存交接与规划。','']
    if proof:
        lines+=['## 69场景规划对照','', '| 方法 | 规划通过 | 累计耗时 | 中位耗时 | P90耗时 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for mode,label in [('baseline','当前链路'),('optimized','本轮优化组')]:
            v=proof['groups'][mode]
            lines.append(f"| {label} | {v['successes']}/{v['requests']}（{v['successes']/v['requests']:.2%}） | "
                f"{v['total_seconds']:.2f}秒 | {v['median_seconds']:.2f}秒 | {v['p90_seconds']:.2f}秒 |")
        lines+=['',f"累计省时{proof['planning_total_time_reduction']:.2%}；"
            f"{proof['checked_requests']}次请求、{proof['checked_original_candidate_rows']}条候选原值和{proof['checked_endpoints']}个关节终点已独立核验。",'',
            f"原抓法与选臂{'全部保持' if proof['original_grasps_and_arms_retained'] else '发生变化'}；"
            f"关节终点原值{'完全相同' if proof['joint_endpoints_exactly_retained'] else '有差异，已按原精度独立核验'}。",'']
    else:
        lines+=['## 当前进度','',f"已完成{p['completed_requests']}/138次规划，当前阶段：{p['phase']}。",'']
    if result:
        lines+=['## 10场景实际抓取对照','', '| 方法 | 规划通过 | 实际成功 | 初始规划中位数 | 初始规划P90 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for mode,label in [('baseline','当前链路'),('optimized','本轮优化组')]:
            v=result['groups'][mode]
            lines.append(f"| {label} | {v['planning_passes']}/10（{v['planning_passes']/10:.0%}） | "
                f"{v['successes']}/10（{v['successes']/10:.0%}） | {v['planning_median_seconds']:.2f}秒 | {v['planning_p90_seconds']:.2f}秒 |")
        lines+=['',f"初始规划累计省时{result['planning_total_time_reduction']:.2%}，"
            f"原成功变失败{len(result['baseline_success_losses'])}例，原失败变成功{len(result['optimized_success_gains'])}例。",'']
    elif physical:
        lines+=['## 实际抓取进度','',f"已完成{q['completed_requests']}/20次，当前阶段：{q['phase']}。",'']
    else:
        lines+=['实抓阶段等待规划核验结果。','']
    lines+=['## 任务与记录','',
        '规划阶段逐项核对领取、候选、实际求解配置和独立FK。实抓按实际发生的接近、闭合、抬升与保持动作核对命令、回执和视频。'
        '普通失败如实记录，资源、程序或记录故障停止，已领取任务不重试。默认配置保持。','']
    DOCUMENT.write_text('\n'.join(lines))
    return values


def section(website,site,active):
    values=document(website)
    if not values:return None
    planning,physical,p,q,proof,result,status=values
    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Website evidence must be an independent file')
        if not target.exists() or sha256(target)!=sha256(source):shutil.copy2(source,target)
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    url=attach(DOCUMENT,'sam3-v8-raw-order.md')
    snapshot=website/'resources/media/sam3-v8-raw-order.json'
    if snapshot.is_symlink() or snapshot.exists() and snapshot.stat().st_nlink!=1:
        raise ValueError('Website snapshot must be independent')
    brief={k:v for k,v in p.items() if k!='results'}
    brief['results']=[{'id':row['id'],'comparison':row['comparison'],'variants':{
        mode:{k:entry[k] for k in ['success','selected_arm','selected_original_source_index',
            'candidate_checks','initial_planning_with_postprocessing_seconds']}
        for mode,entry in row['variants'].items()}} for row in p['results']]
    write_json(snapshot,{'planning_run':str(planning.relative_to(ROOT)),'physical_run':str(physical.relative_to(ROOT)) if physical else None,
        'planning_progress':brief,'planning_review':proof,'physical_progress':q,'physical_acceptance':result})
    site['localAssets'][snapshot.name]='../../resources/media/'+snapshot.name
    body=DOCUMENT.read_text().replace('# V8保留原顺序的计算优化对照','## V8当前实验：保留原顺序的计算优化',1)
    if physical:
        labels={'success':'成功','no_bilateral_target_contact':'闭合后未形成双指夹持',
            'insufficient_hold_height':'抬升或保持不足','approach_not_converged':'接近动作未到位',
            'no_arm_has_executable_candidate':'规划不出来'}
        rows=[]
        for case in raw(physical/'registration.json')['cases']:
            cells=[]
            for mode in ['baseline','optimized']:
                directory=physical/'trials'/case['id']/mode
                claim=physical/'claims'/(case['id']+'_'+mode+'.json')
                if not claim.exists() or raw(claim)['state']!='complete':cells.append('待执行');continue
                outcome=raw(directory/'result.json');text=labels.get(outcome['reason'],outcome['reason']);links=[]
                prefix='sam3-v8-raw-order-'+case['id'].split('_')[0].lower()+'-'+mode
                for name,suffix,label in [('execution.mp4','overview.mp4','第三人称'),('head.mp4','head.mp4','头部'),
                                          ('paired_actual_head.png','initial.png','初始图')]:
                    if (directory/name).exists():links.append(f'[{label}]({attach(directory/name,prefix+"-"+suffix)})')
                cells.append(text+' · '+' · '.join(links))
            rows.append(f"| {case['id']} | {cells[0]} | {cells[1]} |")
        body+='\n### 逐场景结果和录像\n\n| 场景 | 当前链路 | 本轮优化组 |\n| --- | --- | --- |\n'+'\n'.join(rows)+'\n'
        if result:
            failures=sorted(set(result['groups']['baseline']['failure_reasons'])|set(result['groups']['optimized']['failure_reasons']))
            body+='\n### 失败分布\n\n| 原因 | 当前链路 | 本轮优化组 |\n| --- | ---: | ---: |\n'
            for reason in failures:
                counts=[result['groups'][m]['failure_reasons'].get(reason,0) for m in ['baseline','optimized']]
                body+=f'| {labels.get(reason,reason)} | {counts[0]} | {counts[1]} |\n'
    body+=f'\n[本轮说明与结果]({url}) · [逐例统计和独立核验](/media/sam3-v8-raw-order.json)\n'
    batches=[{'id':planning.name,'title':'保留原顺序的69场景规划对照','date':raw(planning/'registration.json')['at'].split('T')[0],
        'status':status,'summary':status,'outputs':[str(planning.relative_to(ROOT))],
        'artifacts':[{'label':'本轮核验数据','url':'/media/sam3-v8-raw-order.json'}]}]
    if physical:batches.append({**batches[0],'id':physical.name,'title':'保留原顺序的20次实抓对照','outputs':[str(physical.relative_to(ROOT))]})
    return {'body':body,'status':status,'summary':status,'batches':batches}
