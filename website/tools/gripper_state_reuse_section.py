"""Independent website snapshot for exact state and sweep geometry reuse."""
import shutil

from arx_project.layout import ROOT,read_json,sha256


def section(website,site):
    pointer=ROOT/'tmp/gripper_state_reuse/active.json'
    if not pointer.exists() or read_json(pointer)['phase']!='complete':return '',None,[]
    active=read_json(pointer);root=(ROOT/active['geometry_run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/gripper_state_reuse'):raise ValueError('Unexpected reuse root')
    evidence=read_json(root/'publication_snapshot.json')
    if not evidence['task_complete'] or not evidence['record_integrity']:raise ValueError('Reuse review incomplete')
    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise RuntimeError('Reuse publication requires independent files')
        shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise RuntimeError('Reuse attachment differs')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    data=attach(root/'publication_snapshot.json','sam3-v7-state-reuse-results.json')
    report=attach(ROOT/'modules/pipeline/docs/mainline_versions/V7_场景绑定与开合爪计算优化结果.md',
        'sam3-v7-state-reuse-report.md')
    from modules.pipeline.experiments.gripper_state_reuse.report import timing_table
    from modules.pipeline.experiments.gripper_sweep_optimization.report import average
    a,b=evidence['phases']
    current=next(p for p in evidence['phases'] if p['kind']==evidence['current_phase_kind'])
    mode=evidence['current_mode'];s=current['summaries'][mode];mean=s['total_seconds']/s['requests']
    post=read_json(website/'resources/media/sam3-v7-postprocessing-runtime-results.json')
    values=[r['new_complete_postprocessing_seconds'] for r in post['cases']]
    low,high=min(values),max(values)
    lines=['## 最新：场景绑定与开合爪计算优化','',
        '本轮逐项比较场景绑定复用、碰撞球缓存，以及开合爪模型读取和逐采样参数更新。'
        '全部候选顺序和采样保留，碰撞按当前场景重新检查。','',
        '### 从抓姿输出到准备执行','',
        '| 环节 | 最新耗时 |','| --- | ---: |',
        f'| 候选后处理，三个样本实测 | {low:.2f}～{high:.2f}秒 |',
        f'| 候选交接与完整规划，平均值 | {mean:.2f}秒 |',
        f"| 候选交接与完整规划，中位数 / P90 | {s['median_seconds']:.2f} / {s['p90_seconds']:.2f}秒 |",
        f'| 原始抓姿到准备执行，按分项平均耗时估算 | {mean+low:.2f}～{mean+high:.2f}秒 |','',
        '### 机器人与场景绑定优化','']
    lines+=timing_table(a)
    change=a['review']['total_saving_fraction']
    change_text=f'累计耗时减少{change:.1%}' if change>=0 else f'累计耗时增加{-change:.1%}'
    lines+=['',change_text+f"，本轮结论：{'采用' if a['kept'] else '保留原实现'}。",'',
        '### 开合爪模型读取与逐采样计算','']
    lines+=timing_table(b)
    lines+=['',f"累计耗时减少{b['review']['total_saving_fraction']:.1%}，本轮结论：{'采用' if b['kept'] else '保留原实现'}。",'',
        '### 最新规划分项耗时','',
        '下表使用最新采用实现的平均值，各行可相加。','',
        '| 环节 | 平均耗时 |','| --- | ---: |']
    measured=0.
    for label,names in [
        ('机器人与场景绑定',['world_and_robot_bind']),
        ('开合爪模型读取、FK与验碰撞',['gripper_opening','gripper_narrowing']),
        ('抓取IK与终点几何检查',['grasp_IK','grasp_endpoint']),
        ('接触几何更新',['contact_validator_update']),
        ('路径搜索＋轨迹优化',['path_PRM','trajectory_approach','trajectory_pregrasp'])]:
        value=sum(average(current,mode,n,'exclusive_seconds') for n in names)
        measured+=round(value,2);lines.append(f'| {label} | {value:.2f}秒 |')
    lines+=[f'| 其余IK、交接、调度与记录 | {round(mean,2)-measured:.2f}秒 |',
        f'| 合计 | {mean:.2f}秒 |','',
        '四种尝试共80次完整规划，选中抓姿、关节终点及预抓取/接近完整轨迹逐位一致。'
        '本次开合爪计算优化保留871球，258个采样的球坐标、半径和碰撞判断完全一致。','',
        f'[完整配对数据与核验]({data}) · [实施方法与逐场景结果]({report})','']
    status='开合爪计算优化已验证' if evidence.get('native_sweep_adopted') else '秒级耗时优化对照已完成'
    batch={'id':root.name,'title':'场景绑定与原生开合爪计算优化','date':'2026-11-27','status':status,
        'summary':f"本轮完整对照结束，最新规划平均{mean:.2f}秒，新增物理0次。",
        'outputs':[p['run'] for p in evidence['phases']],
        'artifacts':[{'label':'配对结果','url':data},{'label':'方法与分析','url':report}]}
    return '\n'.join(lines),status,[batch]
