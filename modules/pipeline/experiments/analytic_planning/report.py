"""Report complete RPC results, keeping terminal checks and paths separate."""
import argparse
from pathlib import Path

from arx_project.layout import ROOT,now,read_json,write_json
from .run import review


NAMES={'coarse_numeric':'30毫米／30° NMS＋原IK','fine_numeric':'15毫米／15° NMS＋原IK',
       'fine_analytic':'15毫米／15° NMS＋解析批量'}


def create(root):
    review(root)
    accepted=read_json(root/'acceptance.json');offline=read_json(root/'offline_review.json')
    paired=read_json(root/'paired_review.json');proof=read_json(root/'independent_review.json')
    progress=read_json(root/'progress.json');rows=progress['results'];groups=accepted['summaries']
    windows=[w for r in rows if r['mode']=='fine_analytic' for w in r['analytic_windows']]
    counts={}
    for window in windows:
        for e in window['events']:
            key=e['stage']+':'+e['kind'];counts[key]=counts.get(key,0)+1
    snapshot={'at':now(),'run':str(root.relative_to(ROOT)),'acceptance':accepted,
        'offline':offline,'paired_review':paired,'independent_review':proof,
        'analytic_endpoint_events':counts,'startup':read_json(root/'resident/startup.json'),
        'scene_update_preflight':read_json(root/'resident/analytic_preflight/acceptance.json'),
        'results':rows,'services_stopped':read_json(root/'services_stopped.json')}
    reg=read_json(root/'registration.json')
    previous=read_json(ROOT/reg['previous_run']/'publication_snapshot.json') if reg.get('previous_run') else None
    if previous:snapshot['previous_complete_comparison']=previous
    write_json(root/'publication_snapshot.json',snapshot)
    lines=['# V8 细化NMS与解析端点完整规划对照','','本轮把NMS从30毫米／30°调到15毫米／15°，'
        '在相同评分顺序下保留更细的抓法，再将GPU解析IK与批量碰撞接入完整规划的终点检查。','',
        '## 候选变化','','69份已有观测沿用相同点云、宽深修正和硬筛结果，仅改变NMS阈值。'
        f"每场景候选数中位数由{offline['coarse_median']:.0f}增加到{offline['fine_median']:.0f}。"
        '所有输出姿态、宽度、评分和原始编号逐项核对；被NMS删除的候选不作为原始池后备。','',
        '## 第七项如何接入','','1. 按双臂原16/64窗口取候选，同时生成抓取、预抓取和抬升三个原终点。',
        '2. 使用固定256目标的常驻Graph求解析分支，并按各候选实际开度批量检查871球。',
        '3. 保持评分顺序，选择离当前关节状态较近的合格分支，再用原独立FK和当前碰撞器复验。',
        '4. 抓取终点保留V2真实网格检查，严格球模型碰桌不会提前删除可恢复的指尖接触。',
        '5. 预抓取和抬升端点通过后，继续原完整路径和接近检查；解析方案失败，先在同一候选上尝试完整原方法，再推进下一候选。','',
        '每臂会话在启动预热时创建，直接复用原规划器的场景缓存。场景由原规划器更新，核对捕获张量地址不变；'
        '预检加入障碍后确认Graph能发现碰撞，再恢复原场景并逐项比对结果。','',
        '## 完整规划结果','',f"五个固定场景，每组重复两次，本轮共{accepted['planning_requests']}次完整规划。各组共用相同常驻规划器、"
        'C版YAML、原生批量开合爪、逐采样FK及共享内存。下表包含双臂准备、终点检查、完整路径、留档和交接。','',
        '| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,name in NAMES.items():
        if mode not in groups:continue
        g=groups[mode];lines.append(f"| {name} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | "
            f"{g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 | {g['total_seconds']:.3f}秒 |")
    a,b=groups['fine_numeric'],groups['fine_analytic']
    lines+=['',f"相同细NMS下，解析批量组累计耗时变化为{(b['total_seconds']/a['total_seconds']-1)*100:+.2f}%。",'',
        '## 分项平均耗时','','分项采用互斥的主机耗时，便于区分解析批量计算、原方法回退和其余路径成本。','',
        '| 环节 | 细NMS＋原IK | 细NMS＋解析批量 |','| --- | ---: | ---: |']
    stages=[('world_and_robot_bind','机器人与世界绑定'),('gripper_opening','开爪检查（扣除子项）'),
        ('gripper_narrowing','原地收爪检查（扣除子项）'),('analytic_scene_update','共享场景缓存校验'),
        ('analytic_endpoint_batch','解析端点整批查询'),('analytic_branch_original_gate','解析分支的原FK与碰撞复核'),
        ('grasp_IK','原抓取IK'),('pregrasp_IK','原预抓取IK'),('lift_IK','原抬升IK'),
        ('other_IK','其它原IK回退'),('contact_validator_update','接触几何更新'),('path_PRM','路径搜索'),
        ('trajectory_pregrasp','预抓取轨迹优化'),('trajectory_approach','接近轨迹优化')]
    for stage,label in stages:
        values=[groups[m]['stages'].get(stage,{}).get('exclusive_seconds',0.)/groups[m]['requests'] for m in ['fine_numeric','fine_analytic']]
        lines.append(f'| {label} | {values[0]:.4f}秒 | {values[1]:.4f}秒 |')
    lines+=['','## 原方法回退与结果变化','','| 解析组事件 | 次数 |','| --- | ---: |']
    event_names={'grasp:original_IK_fallback':'抓取终点回退原IK',
        'pregrasp:original_IK_fallback':'预抓取终点回退原IK','lift:original_IK_fallback':'抬升终点回退原IK',
        'pregrasp:analytic_world_endpoint':'预抓取终点使用解析解','lift:analytic_world_endpoint':'抬升终点使用解析解',
        'complete_candidate:original_full_candidate_fallback':'解析路径未过，完整回退同一候选'}
    for key,value in sorted(counts.items()):lines.append(f'| {event_names.get(key,key)} | {value} |')
    fallback=counts.get('grasp:original_IK_fallback',0)
    lines+=['',f"解析组共检查{b['candidate_checks']}次候选，其中{fallback}次抓取终点仍回到原数值IK。"
        '因此毫秒级批量检查没有消除主要的失败候选求解；逐候选绑定模型和场景仍是完整请求的大头。'
        '本轮完成了功能接入，完整耗时没有改善，解析方案保留为独立可选入口，常用入口继续原IK。']
    lines+=['',f"细NMS原规划成功是否全部保留：{'是' if paired['fine_original_successes_retained'] else '否'}。",
        ('粗NMS沿用前一轮完整结果，本轮只比较相同细NMS下的原IK与解析批量。' if 'coarse_numeric' not in groups else
         f"粗NMS原规划成功是否全部保留：{'是' if paired['coarse_original_successes_retained'] else '否'}。"),'',
        '| 场景／重复 | 粗NMS原IK | 细NMS原IK | 细NMS解析 | 细NMS两组是否选中同一抓法与手臂 |',
        '| --- | --- | --- | --- | --- |']
    for pair in paired['pairs']:
        values=[('通过' if pair['success'][m] else '规划不出来') if m in pair['success'] else '本轮未重跑' for m in NAMES]
        lines.append('| '+pair['case']+'/'+str(pair['repeat'])+' | '+' | '.join(values)+
            ' | '+('是' if pair['same_fine_selected_grasp_and_arm'] else '否')+' |')
    if previous:
        lines+=['','## 前一轮三组完整结果','','前一轮30次完整规划全部通过。解析组有独立场景副本，平均重复更新约0.206秒，'
            '因此平均4.374→4.367秒基本持平。本轮改为直接共享原规划器场景缓存，另做20次同期配对；前一轮记录全部保留。','',
            '| 前一轮方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 |','| --- | ---: | ---: | ---: | ---: |']
        for mode,name in NAMES.items():
            g=previous['acceptance']['summaries'][mode]
            lines.append(f"| {name} | {g['successes']}/{g['requests']} | {g['mean_seconds']:.3f}秒 | {g['median_seconds']:.3f}秒 | {g['p90_seconds']:.3f}秒 |")
    lines+=['','## 验收与入口','','| 项目 | 结果 |','| --- | --- |',
        f"| 任务结果 | NMS细化与第七项完整规划接入完成，本轮{accepted['planning_requests']}个规划槽完成 |",
        f"| 记录完整性 | {proof['checked_requests']}条回执和{proof['checked_endpoints']}个物理臂终点独立核验通过 |",
        '| 事件适用性 | 本轮测完整规划；实际抓取成功率继续使用原物理试验记录 |',
        '| 关键事件覆盖 | 双臂、不同开度、场景更换、指尖网格复核、同候选原方法回退和完整路径 |','',
        '解析入口：`modules/curobo/analytic_ik/runtime.py`。现有`gripper_sweep.runtime.create`支持配置开关，当前关闭自动解析接入；'
        'NMS默认15毫米／15°。主线全局注册的默认适配器和机器人、场景物理保持。',
        f"\n登记：`{root.relative_to(ROOT)}`。本轮新增仿真和物理抓取均为0。",'']
    doc=ROOT/'modules/pipeline/docs/mainline_versions/V8_细化NMS与解析端点完整规划对照.md';doc.write_text('\n'.join(lines))
    write_json(ROOT/'tmp/analytic_planning/active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'complete':True,'report':str(doc.relative_to(ROOT))})
    print('Complete planning report:',doc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args();create(Path(args.root).resolve())
