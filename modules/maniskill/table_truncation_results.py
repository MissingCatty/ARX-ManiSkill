"""Read-only experimental verification and explicit execution-stage statistics."""
from collections import Counter
from datetime import datetime
import argparse
from pathlib import Path
import shutil

import numpy as np
import graspcl_pipeline as common

ARMS = [('fixed_zerograsp','固定朝向 · ZeroGrasp'),('fixed_graspcl','固定朝向 · GraspCL'),
        ('free_zerograsp','放开朝向 · ZeroGrasp'),('free_graspcl','放开朝向 · GraspCL')]


def analyze(root):
    reg = common.read(root / 'registration.json')
    profile = Path(reg['action_tcp_profile'])
    assert common.sha(profile)==reg['source_hashes'][str(profile)]
    shutil.copy2(profile,root / 'source/action_tcp_43mm.json')
    action = np.asarray(common.read(reg['action_tcp_profile'])['T_eef_action'])
    output = {};checks = []
    ready = common.read(root / 'resident/all_ready.json') if (root / 'resident/all_ready.json').exists() else None
    for arm,label in ARMS:
        directory = root / arm;rows = [];selected = []
        for file in sorted((directory / 'claims').glob('*.json')):
            claim = common.read(file)
            if ready:assert datetime.fromisoformat(claim['claimed_at'])>=datetime.fromisoformat(ready['ready_at'])
            if claim['state']!='complete':continue
            trial = directory / 'trials' / file.stem
            result = common.read(trial / 'result.json')
            assert common.read(trial / 'independent_evaluation.json')['passed']
            assert common.sha(trial / 'result.json')==claim['result_sha256']
            rows.append(result)
            plan_path = trial / 'plan_result.json'
            if plan_path.exists() and common.read(plan_path).get('selected'):
                plan = common.read(plan_path)['selected'];metadata = plan['candidate_metadata']
                eef = np.asarray(plan['grasp_world']);contact = np.asarray(plan['contact_pose_world'])
                assert np.allclose(eef @ action,plan['action_tcp_world'],atol=1e-9)
                assert np.allclose((eef @ action)[:3,3],contact[:3,3],atol=1e-9)
                assert 0<=metadata['insertion_retraction_m']<=metadata['original_source_insertion_m']+1e-8
                original = np.asarray(metadata['original_contact_pose'])
                assert np.allclose(contact[:3,3],original[:3,3]-metadata['insertion_retraction_m']*original[:3,2],atol=1e-9)
                assert metadata['observed_target_points_in_closing_sweep']>0
                assert metadata['gripper_table_clearance_after_m']>=0
                if arm.startswith('fixed'):assert metadata['orientation_change_rad']<1e-8
                pool = common.read(trial / 'graspcl/candidate_pool.json')
                if arm.endswith('graspcl'):assert pool['refined_candidates_rescored']
                else:assert pool['graspcl_invoked'] is False
                selected.append(metadata);checks.append(file.stem)
        def stage_done(result,name):return bool(result.get('stages',{}).get(name,{}).get('settled'))
        n = len(rows);total = reg['attempt_budget_by_arm'][arm]
        closes = sum(stage_done(r,'approach') and stage_done(r,'close') for r in rows)
        success = sum(r['success'] for r in rows)
        retractions = [m['insertion_retraction_m'] for m in selected]
        output[arm] = {'label':label,'registered':total,'completed':n,
            'initial_plan_passes':sum(r['planning_success'] for r in rows),
            'execution_entries':sum(r['physical_attempt'] for r in rows),
            'pregrasp_completed':sum(stage_done(r,'pregrasp') for r in rows),
            'approach_completed':sum(stage_done(r,'approach') for r in rows),
            'approach_and_closure_completed':closes,
            'bilateral_contact_passes':sum(len(r.get('bilateral_fingers',[]))==2 for r in rows),
            'lift_completed':sum(stage_done(r,'lift') for r in rows),
            'hold_completed':sum(stage_done(r,'hold') for r in rows),
            'successes':success,'closure_execution_rate':closes/n if n else None,
            'success_rate_per_scene':success/n if n else None,
            'success_rate_per_completed_closure':success/closes if closes else None,
            'positive_retraction_selected':sum(v>0 for v in retractions),
            'mean_selected_retraction_m':float(np.mean(retractions)) if retractions else None,
            'selected_orientation_changes':sum(m['orientation_change_rad']>1e-8 for m in selected),
            'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success'])),
            'results_by_id':{r['id']:{'success':r['success'],'reason':r['reason']} for r in rows}}
    pairs = {}
    for profile in ['zerograsp','graspcl']:
        fixed = output['fixed_'+profile]['results_by_id'];free = output['free_'+profile]['results_by_id']
        both = sorted(set(fixed)&set(free))
        pairs[profile] = {'matched_completed':len(both),
            'fixed_failure_free_success':[i for i in both if not fixed[i]['success'] and free[i]['success']],
            'fixed_success_free_failure':[i for i in both if fixed[i]['success'] and not free[i]['success']]}
    clocks = {name:common.read(root / name / 'round_timing.json') for name,_ in ARMS if (root / name / 'round_timing.json').exists()}
    suite_clock = common.read(root / 'suite_timing.json')
    startup = next(iter(clocks.values())).get('shared_suite_startup_seconds') if clocks else None
    result = {'updated_at':datetime.now().astimezone().isoformat(),'action_tcp_eef_z_m':.043,
        'arms':output,'paired_orientation_comparison':pairs,'selected_tcp_and_depth_checks_passed':len(checks),
        'timing':{'shared_startup_seconds':startup,'rounds':clocks,'suite':suite_clock},
        'all_completed':all(r['completed']==r['registered'] for r in output.values()),
        'scope':'Previously table-blocked subset only. Executability is separately recorded as initial plan, motion entry, and approach+closure completion. Physical success uses the original independent evaluator.'}
    common.save(root / 'execution_metrics.json',result)
    def percent(x):return '—' if x is None else f'{x:.1%}'
    lines = ['# 桌面拦截场景：执行与抓取统计','',
        '| 组别 | 完成/场景数 | 规划通过 | 开始执行 | 完成接近和闭爪 | 双指接触 | 抓取成功 | 执行到闭爪率 | 抓取成功率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in ARMS:
        x = output[arm]
        lines.append(f"| {label} | {x['completed']}/{x['registered']} | {x['initial_plan_passes']} | {x['execution_entries']} | {x['approach_and_closure_completed']} | {x['bilateral_contact_passes']} | {x['successes']} | {percent(x['closure_execution_rate'])} | {percent(x['success_rate_per_scene'])} |")
    scope = '四组全部完成；各链路只统计此前桌面拦截的 72 例或 55 例，不代表全部 200 例。' if result['all_completed'] else '分母为本组已完成的桌面拦截场景；全部结束后分别为 72 例和 55 例。未完成时的比率是临时结果。'
    lines += ['', scope,'',
        '“规划通过”只表示初始抓取 IK、预抓取/接近轨迹与抬升 IK 通过；“开始执行”表示开始预抓取运动；“完成接近和闭爪”由实际执行阶段收敛记录统计。',
        '抓取成功：双指接触目标、无禁止接触、动作收敛，抬升超过 6 cm 并连续保持至少 3 秒。',
        '两种朝向设置均采用 [43 mm TCP 配置](source/action_tcp_43mm.json) 和插入截断；碰撞几何、桌面及闭爪 8 mm 余量保持。与历史 40 mm 结果比较时，不能将变化全部归因于截断。',
        '放开朝向采用 32 种子位置 IK 在真实 TCP43 处提议朝向，再通过观测夹持范围、原完整位姿 IK、碰撞、轨迹和实际抓取验证；有限搜索不证明穷尽全部朝向。',
        'ZeroGrasp 分支新增朝向沿用其父候选的网络分数；GraspCL 分支对修改后的位姿重新编码和评分。',
        '逐例数据见各组 `trials/<场景>/`；位姿图在 `zg_outputs/`。详细统计见 [execution_metrics.json](execution_metrics.json)。','']
    def duration(seconds):
        seconds = round(seconds);return f'{seconds//3600} 小时 {seconds%3600//60} 分 {seconds%60} 秒' if seconds>=3600 else f'{seconds//60} 分 {seconds%60} 秒'
    for profile,label in [('zerograsp','ZeroGrasp'),('graspcl','GraspCL')]:
        paired = pairs[profile]
        if paired['matched_completed']:
            lines += ['',f"{label} 同场景朝向对照：{paired['matched_completed']} 例已配对；放开朝向后新增成功 {len(paired['fixed_failure_free_success'])} 例，原成功变失败 {len(paired['fixed_success_free_failure'])} 例。"]
    if startup is not None:lines += ['','共同加载和预热：'+duration(startup)+'。','']
    for name,label in ARMS:
        if clocks.get(name,{}).get('round_wall_seconds') is not None:lines.append('- '+label+'：'+duration(clocks[name]['round_wall_seconds'])+'。')
    if suite_clock.get('total_suite_wall_seconds') is not None:
        lines += ['','含共同加载与关闭的整批耗时：'+duration(suite_clock['total_suite_wall_seconds'])+'。']
        lines.append('常驻服务全部退出：'+('是' if suite_clock['all_owned_services_exited'] else '否')+'。')
    (root / '执行与抓取统计.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args = parser.parse_args();result = analyze(args.root.resolve())
    print({name:{k:r[k] for k in ['completed','registered','execution_entries','approach_and_closure_completed','successes']} for name,r in result['arms'].items()})
