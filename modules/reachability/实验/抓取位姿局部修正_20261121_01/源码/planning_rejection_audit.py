"""Saved ZeroGrasp planning rejections: stage census and deferred IK attribution."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import textwrap
import time

CODE = Path(__file__).resolve().parent
SOURCE = CODE/'outputs/arx_zerograsp_lift30_20261117_01'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(result, plan):
    if result['planning_success']:
        return 'initial_plan_passed'
    if result['reason'] == 'independent_scene_admission_failed':
        return 'admission_failed'
    if plan is None:
        return result['reason']
    n, rejected = plan.get('count'), plan.get('rejected', {})
    if n == 0:
        return 'empty_after_truncation'
    if n and rejected.get('grasp_ik') == n:
        return 'grasp_ik_or_collision'
    if n and sum(rejected.get(k,0) for k in ['grasp_ik','closed_collision']) == n:
        return 'nominal_closed_collision'
    if n and sum(rejected.get(k,0) for k in ['grasp_ik','closed_collision','pre_or_lift_ik']) == n:
        return 'pre_or_lift_ik'
    if rejected.get('path'):
        return 'path'
    return plan.get('reason', 'other')


def prepare(out):
    source_reg = read(SOURCE/'registration.json')
    assert read(SOURCE/'final_verification.json')['passed']
    assert read(SOURCE/'progress.json')['completed'] == 200
    out.mkdir(parents=True, exist_ok=False)
    sources = dict(source_reg['source_hashes'])
    counts = Counter()
    rows, cases = [], []
    replan_counts = Counter()
    for claim_path in sorted((SOURCE/'fixed_zerograsp/claims').glob('*.json')):
        claim = read(claim_path)
        assert claim['state'] == 'complete'
        trial = SOURCE/'fixed_zerograsp/trials'/claim_path.stem
        result = read(trial/'result.json')
        assert sha(trial/'result.json') == claim['result_sha256']
        assert read(trial/'independent_evaluation.json')['passed']
        plan = read(trial/'plan_result.json') if (trial/'plan_result.json').exists() else None
        kind = category(result, plan)
        counts[kind] += 1
        rows.append({'id':claim_path.stem, 'category':kind, 'layout':result['layout'],
                     'reason':result['reason'], 'candidate_count':None if plan is None else plan.get('count')})
        if kind in ['grasp_ik_or_collision','nominal_closed_collision','pre_or_lift_ik']:
            cases.append({'id':claim_path.stem, 'type':kind, 'trial':str(trial)})
        for stage in ['approach','lift']:
            p = trial/(stage+'_plan_result.json')
            if p.exists() and not read(p).get('success'):
                replan_counts[stage+':'+read(p).get('reason','unknown')] += 1
                cases.append({'id':claim_path.stem, 'type':'measured_'+stage+'_replan', 'trial':str(trial)})
        for name in ['result.json','state.json','graspcl/candidate_pool.json','plan_result.json',
                     'closed_state.json','lift_plan_result.json','approach_plan_result.json']:
            p = trial/name
            if p.exists():
                sources[str(p)] = sha(p)
    assert counts == {'initial_plan_passed':129, 'grasp_ik_or_collision':35,
                     'pre_or_lift_ik':19, 'nominal_closed_collision':1,
                     'empty_after_truncation':9, 'admission_failed':7}
    (out/'source').mkdir()
    for name in ['planning_rejection_audit.py','ik_workspace_audit.py','benchmark_after.py',
                 'fast_resident_planner.py','resident_planner.py','arx_pipeline/scene_tools.py']:
        p = CODE/name
        sources[str(p)] = sha(p)
        shutil.copy2(p, out/'source'/p.name)
    save(out/'registration.json', {'created_at':datetime.now().astimezone().isoformat(),
        'source_campaign':str(SOURCE), 'warmup_trial':source_reg['warmup_trial'],
        'source_hashes':sources, 'arms':[], 'total_slots':0, 'physical_attempts':0,
        'audit_cases':cases, 'stage_counts':dict(counts), 'replanning_failures':dict(replan_counts),
        'full_pose_optimization_seeds':128, 'solutions_retained_per_goal':32,
        'same_world_grasp_offset_m':.043, 'same_world_lift_m':.03,
        'scope':'Saved poses only; no robot execution, no candidate/inference regeneration, no feedback writes, no source result edits. Pure IK ignores environment collision for diagnosis; every witness is rechecked against unchanged scene checkers. Finite failed searches never prove global unreachability.'})
    save(out/'stage_census.json', {'source':str(SOURCE), 'counts':dict(counts),
        'replanning_failures':dict(replan_counts), 'cases':rows})
    save(out/'progress.json', {'phase':'waiting_for_graspcl', 'completed_cases':0, 'total_cases':len(cases)})
    (out/'run_notes.md').write_text('# 离线规划核查\n')
    write_report(out)
    print('STAGE CENSUS SAVED; DEFERRED AUDIT CASES', len(cases), flush=True)


def make_pure(model, position_only=False):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    cfg = InverseKinematicsCfg.create(robot=str(model), scene_model=None,
        optimizer_configs=['ik/lbfgs_ik.yml'], self_collision_check=False,
        num_seeds=128, seed_solver_num_seeds=128, position_tolerance=.002,
        orientation_tolerance=1e6 if position_only else .03,
        optimizer_collision_activation_distance=.003, max_batch_size=64,
        use_cuda_graph=True, load_collision_spheres=False)
    for rollout in [cfg.core_cfg.metrics_rollout_config, *cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():
            cost.disable_self_collision()
    solver = InverseKinematics(cfg)
    if position_only:
        solver.update_tool_pose_criteria({'right_eef_link':ToolPoseCriteria.track_position()})
    return solver


def assess(context, solver, fk, state, worlds, *, nominal=False, position_only=False, table_margin=0.):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    import ik_workspace_audit as helper
    q, solved, pe, re = helper.solve(solver, context, worlds, context.measured(state))
    open_clear = torch.zeros_like(solved)
    closed_clear = torch.zeros_like(solved)
    pose_valid = torch.zeros_like(solved)
    table_clear = torch.zeros_like(solved)
    ground_clear = torch.zeros_like(solved)
    examples = []
    if solved.any():
        good = q[solved]
        locations = solved.nonzero()
        for (candidate, branch), joints in zip(locations.cpu().tolist(), good.cpu().numpy()):
            goal = context.invbase @ worlds[candidate]
            actual = fk.fk(joints)
            p = float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
            r = float(Rotation.from_matrix(goal[:3,:3].T@actual[:3,:3]).magnitude())
            bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in fk_names()])
            valid = p <= .002001 and (position_only or r <= .030001) and bool(((joints>=bounds[:,0]-1e-6)&(joints<=bounds[:,1]+1e-6)).all())
            pose_valid[candidate,branch] = valid
            if valid and len(examples)<4:
                examples.append({'candidate':candidate, 'branch':branch, 'q':joints.tolist(),
                                 'cpu_position_error_m':p, 'cpu_rotation_error_rad':r})
        detail = helper.collision_details(context, state, good, raise_z=table_margin)
        open_clear[solved] = detail['full_scene_valid']
        table_clear[solved] = detail['joint_bounds_valid'] & (detail['table_penetration']<=0)
        ground_clear[solved] = detail['joint_bounds_valid'] & (detail['ground_penetration']<=0)
        if nominal:
            closed = helper.collision_details(context, state, good, context.closed_checker, raise_z=.008)
            closed_clear[solved] = closed['full_scene_valid']
        else:
            closed_clear[solved] = detail['full_scene_valid']
    masks = {'solver_success':solved, 'cpu_pose_valid':pose_valid,
             'open_scene_clear':open_clear & pose_valid, 'open_and_closed_scene_clear':open_clear & closed_clear & pose_valid,
             'table_clear':table_clear & pose_valid, 'ground_clear':ground_clear & pose_valid}
    summary = {key+'_candidates':int(value.any(-1).sum()) for key,value in masks.items()}
    summary.update(goals=len(worlds), witnesses=examples)
    arrays = {'q':q.cpu().numpy(), 'position_error':pe.cpu().numpy(), 'rotation_error':re.cpu().numpy(),
              **{key:value.cpu().numpy() for key,value in masks.items()}}
    return summary, arrays


def fk_names():
    return ['right_joint'+str(i) for i in range(21,27)]


def run(out):
    import sys
    import numpy as np
    import torch
    import graspcl_pipeline as common
    from graspcl_execute import guard
    import resident_planner
    from postprocess_planner import PostprocessPlanner
    from fast_resident_planner import FastPlannerPool
    import ik_workspace_audit as helper
    sys.path.insert(0, str(CODE/'arx_pipeline'))
    from scene_tools import ArmFK
    reg = read(out/'registration.json')
    guard(reg)
    with (out/'run_started.json').open('x') as stream:
        json.dump({'at':datetime.now().astimezone().isoformat(), 'physical_attempts':0}, stream)
    started = time.perf_counter()
    resident_planner.GraspCLPlanner = PostprocessPlanner
    pool = FastPlannerPool(out/'resident_planner', Path(reg['warmup_trial']))
    pure = make_pure(pool.contexts['open'].model)
    position = make_pure(pool.contexts['open'].model, True)
    fk = ArmFK()
    helper.probe(pool, pure, position, out)
    source = textwrap.dedent(inspect.getsource(helper.original_trace))
    assert source.count('-.040*') == 0  # The frozen function uses an in-place translation.
    assert source.count('-=.040*') == 1 and source.count('+=.12') == 1
    source = source.replace('-=.040*','-=.043*').replace('+=.12','+=.03')
    scope = dict(helper.__dict__)
    exec(compile(source, __file__+':tcp43_lift30_trace', 'exec'), scope)
    traced_ik = scope['original_trace']
    save(out/'components_loaded.json', {'all_loaded_before_cases':True, 'model_load_counts':pool.loads,
        'pure_full_pose_loads':1, 'position_only_loads':1, 'optimization_seeds':128,
        'solutions_retained_per_goal':32, 'physical_attempts':0, 'startup_seconds':time.perf_counter()-started})
    (out/'cases').mkdir()
    results = []
    for case in reg['audit_cases']:
        guard(reg)
        trial = Path(case['trial'])
        destination = out/'cases'/(case['id']+'_'+case['type'])
        destination.mkdir()
        if case['type'].startswith('measured_'):
            state = read(trial/'closed_state.json')
            collision = None
            try:
                context = pool.bind(destination/'model', state, closed=True)
            except AssertionError as error:
                assert str(error)=='Measured state collides in unchanged ARX model'
                collision = str(error)
                context = pool.contexts['closed']
            detail = helper.collision_details(context, state, context.measured(state)[None], raise_z=.008)
            result = {**case, 'replayed_start_collision':collision is not None,
                'full_scene_valid':bool(detail['full_scene_valid'][0]),
                'table_penetration_m':float(detail['table_penetration'][0]),
                'ground_penetration_m':float(detail['ground_penetration'][0]),
                'interpretation':'Measured lift start validity is separate from goal IK reachability.'}
        else:
            state = read(trial/'state.json')
            context = pool.bind(destination/'model', state)
            measured = np.array([state['qpos'][state['joint_names'].index(n)] for n in fk_names()])
            expected = context.invbase@np.asarray(state['links']['right_eef_link'])
            actual = fk.fk(measured)
            assert np.linalg.norm(actual[:3,3]-expected[:3,3])<2e-5
            candidates = read(trial/'graspcl/candidate_pool.json')['candidates']
            trace = traced_ik(context,state,candidates)
            original = read(trial/'plan_result.json')
            matches = all(trace['rejected'].get(k,0)==v for k,v in original['rejected'].items())
            grasp = []
            for candidate in candidates:
                world = np.array(candidate['pose']);world[:3,3]-=.043*world[:3,2]
                grasp.append(world)
            pure.reset_seed()
            summary, arrays = assess(context,pure,fk,state,grasp,nominal=True)
            np.savez_compressed(destination/'grasp_solutions.npz',goals=grasp,**arrays)
            result = {**case, 'candidate_count':len(grasp), 'original_replay':trace,
                      'original_rejection_counts_reproduced':matches, 'grasp':summary}
            # These are additional endpoint diagnoses, not motion/grasp replays.
            if case['type']=='pre_or_lift_ik':
                for stage in ['pregrasp','lift']:
                    worlds = []
                    for world in grasp:
                        pose = world.copy()
                        if stage=='pregrasp':pose[:3,3]-=.08*pose[:3,2]
                        else:pose[2,3]+=.03
                        worlds.append(pose)
                    pure.reset_seed()
                    result[stage], stage_arrays = assess(context,pure,fk,state,worlds)
                    np.savez_compressed(destination/(stage+'_solutions.npz'),goals=worlds,**stage_arrays)
            if summary['cpu_pose_valid_candidates'] < len(grasp):
                position.reset_seed()
                result['position_only'], position_arrays = assess(context,position,fk,state,grasp,position_only=True)
                np.savez_compressed(destination/'position_only_solutions.npz',goals=grasp,**position_arrays)
            result['interpretation'] = ('A found full-scene-valid original pose proves that endpoint is reachable under the saved model. '
                'Kinematic-only solutions may collide. Failure of 128 finite seeds does not prove global unreachability. '
                'Position-only diagnostic refers to the original EEF position, not a guaranteed executable grasp at the same action TCP.')
        save(destination/'audit.json',result)
        results.append(result)
        save(out/'audit.json', {'cases':results, 'physical_attempts':0, 'source_results_unchanged':True})
        save(out/'progress.json', {'phase':'running','completed_cases':len(results),'total_cases':len(reg['audit_cases'])})
        print('AUDIT CASE', len(results), case['id'], case['type'], flush=True)
    guard(reg)
    save(out/'progress.json', {'phase':'complete','completed_cases':len(results),
        'total_cases':len(reg['audit_cases']),'physical_attempts':0,'wall_seconds':time.perf_counter()-started})
    write_report(out)


def write_report(out):
    reg = read(out/'registration.json')
    counts = reg['stage_counts']
    lines = ['# ZeroGrasp 规划失败阶段与 IK 核查','',
        '本轮 200 例，129 例通过初始完整规划，71 例未通过。场景数按最远通过阶段互斥统计。','',
        '| 初始规划阶段 | 场景数 |','|---|---:|',
        '| 初始化未通过 | 7 |','| 截断后无候选 | 9 |','| 抓取 IK／场景碰撞合并拒绝 | 35 |',
        '| 名义闭爪碰撞 | 1 |','| 预抓取或抬升 IK／碰撞 | 19 |','| 路径搜索或轨迹优化 | 0 |','',
        '另有 1 例初始方案通过，但实际闭爪后抬升重规划时，测量起始状态被原碰撞模型拒绝；不能归为目标点 IK 不可达。','',
        '原 ik() 把求解器返回无有效解和独立碰撞检查拒绝合并为 None。预抓取与抬升也合并记数；因此现有日志不足以判定为数学不可达。',
        '离线复核保留 43 mm TCP、3 cm 抬升、原关节限制和场景。独立纯运动学分支使用 128 初值，保留 32 个解，再经 CPU 原 URDF 正向运动学及原碰撞检查器复核；另记录原条件重求解和仅位置诊断。原测试不修改、不增加实际抓取、不回灌反馈。','']
    if (out/'audit.json').exists():
        cases = read(out/'audit.json')['cases']
        lines += ['| 场景 | 原拒绝阶段 | 原条件重放一致 | 纯运动学可解候选 | 原完整场景可解候选 | 开／名义闭爪均有效候选 |',
                  '|---|---|---|---:|---:|---:|']
        for case in cases:
            if 'grasp' not in case:continue
            g = case['grasp']
            lines.append(f"| {case['id']} | {case['type']} | {case['original_rejection_counts_reproduced']} | {g['cpu_pose_valid_candidates']} | {g['open_scene_clear_candidates']} | {g['open_and_closed_scene_clear_candidates']} |")
    else:
        lines += ['离线重求解尚未开始：按用户要求等待当前 GraspCL 200 例结束并释放服务。当前只能确认失败阶段，不能把求解失败称为已证明不可达。']
    lines += ['', '有限初值未找到解只能记为“本次未找到”，不能证明所有 IK 分支均不存在；找到符合原约束的解则构成可达性的正证据。',
              '', '[逐例阶段数据](stage_census.json) · [核查进度](progress.json) · [登记](registration.json)','']
    (out/'规划失败阶段与IK核查.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run','report'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='run':run(out)
    else:write_report(out)
