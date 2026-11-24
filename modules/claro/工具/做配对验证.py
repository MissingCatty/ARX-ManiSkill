"""对 13 个登记遮挡场景执行固定挡物移除与原目标重抓。"""
from arx_project.layout import canonical_path

import argparse
from contextlib import ExitStack, contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

import 验证绿杯移障 as removal
import 重新观测绿杯移障场景 as observation
import 重抓原目标 as regrasp
from benchmark_lifecycle import mark_ready, require_ready
from resident_backend import serve
from short_lift_protocol import HOLD_SECONDS, MIN_HOLD_LIFT_M, install_verifier, validate_selected

ROOT, MS = removal.ROOT, removal.MS
read, save, sha256, call = removal.read, removal.save, removal.sha256, removal.call
OUT = ROOT / 'docs/06_做配对验证/resources'
DIRECT = MS / 'outputs/arx_physical_table_parallel_20261118_01'
WITNESS = MS / 'outputs/arx_planning_rejection_audit_20261118_01/endpoint_witness_summary.json'
CASE_IDS = ('I0015', 'I0018', 'I0038', 'I0044', 'I0048', 'I0055', 'I0063', 'I0069',
            'I0100', 'I0119', 'I0130', 'I0167', 'I0190')
BLOCKER = 'distractor_00'


def planner(directory, address):
    import torch
    import resident_planner
    import table_truncation_model as truncation
    from checked_native_services import checked_pool
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner

    spec = importlib.util.spec_from_file_location('physical_table_policy', removal.POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    placement_methods = {'placement': removal.placement, 'measured_lower': removal.plan_measured_lower,
                         'retreat': removal.plan_retreat}

    class BatchPlanner(truncation.make_pool(checked_pool(FastPlannerPool))):
        def request(self, directory, **kwargs):
            action = kwargs.pop('action', None)
            if action is None:
                return super().request(directory, **kwargs)
            if action in placement_methods:
                return placement_methods[action](self, directory)
            if action != 'validate':
                raise ValueError('Unsupported planning request')
            state = read(directory / 'state.json')
            samples = removal.dense_path(kwargs['path'])
            if np.max(np.abs(samples[0] - removal.arm_q(state))) >= .005:
                return {'success': False, 'reason': 'measured_path_start_differs'}
            context = self.bind(directory / '全场景轨迹复核', state, include_target=True)
            q = torch.as_tensor(samples, dtype=torch.float32, device='cuda')
            ok = all(bool(context.checker.validate(batch[:, None]).all()) for batch in q.split(64))
            return {'success': ok, 'points': len(samples), 'reason': None if ok else 'full_scene_path_collision'}

    resident_planner.PlannerPool = BatchPlanner
    serve(SimpleNamespace(kind='curobo', root=directory, socket=address, warmup_trial=removal.WARMUP))


@contextmanager
def ready(directory, devices):
    service = directory / 'resident'
    service.mkdir()
    env_script = MS.parent / 'arx_core/scripts/curobo_arx/environment.sh'
    with tempfile.TemporaryDirectory(prefix='claro_pair_') as tmp, ExitStack() as stack:
        sockets = {kind: Path(tmp) / (kind + '.sock') for kind in devices}
        children = {}
        for kind, gpu in devices.items():
            s = sockets[kind]
            command = ([canonical_path(MS.parent / 'tmp/arx-fullchain/env/bin/python'), MS / 'postprocess_backend.py',
                        '--kind', kind, '--socket', s, '--root', service, '--warmup-trial', removal.WARMUP]
                       if kind == 'zerograsp' else
                       [MS.parent / 'curobo/.venv/bin/python', str(Path(__file__).resolve()), 'worker', service, s])
            wrapped = ['bash', '-c', 'source "$1"; shift; export TORCH_EXTENSIONS_DIR="$CLARO_TORCH_CACHE"; exec "$@"',
                       'claro-pair', str(env_script), *map(str, command)]
            env = {**removal.planner_env(gpu), 'CLARO_TORCH_CACHE': str(ROOT / '缓存/torch_extensions'),
                   'ARX_NEURAL_GPU': gpu}
            log = stack.enter_context((service / (kind + '.log')).open('x'))
            process = subprocess.Popen(wrapped, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children[kind] = process
            stack.callback(regrasp.stop_worker, process, s)
        deadline = time.monotonic() + 480
        while not all((service / (kind + '_ready.json')).exists() for kind in devices):
            ended = [kind for kind, process in children.items() if process.poll() is not None]
            if ended or time.monotonic() >= deadline:
                raise RuntimeError('Warmup failed or timed out: ' + ', '.join(ended or list(devices)))
            time.sleep(2)
        mark_ready(service, list(devices), {'sockets': {n: str(p) for n, p in sockets.items()}}, children)
        require_ready(service, list(devices))
        yield sockets


def register(root, devices):
    metrics_file = DIRECT / 'independent_case_metrics.json'
    metrics = read(metrics_file)
    witnesses = {row['id'] for row in read(WITNESS)['collision_cases']
                 if row['has_joint_bounds_table_ground_clear_but_full_world_blocked_branch']}
    schedule = []
    files = [Path(__file__), Path(removal.__file__), Path(observation.__file__), Path(regrasp.__file__),
             removal.POLICY, metrics_file, WITNESS, regrasp.CHECKPOINT,
             ROOT / 'docs/02_选择一个挡物/选择一个挡物.md']
    for case in CASE_IDS:
        rows = [row for row in metrics if row['id'].split('_')[0] == case]
        if (len(rows) != 2 or any(row['success'] or row['planning_success'] or row['closed'] for row in rows)
                or {row['arm'] for row in rows} != {'fixed_zerograsp', 'fixed_graspcl'}):
            raise ValueError('Expected both direct baseline failures for ' + case)
        reference, = [row for row in rows if row['arm'] == 'fixed_zerograsp']
        path = DIRECT / 'zerograsp/fixed_zerograsp/trials' / reference['id']
        old = read(path / 'result.json')
        if (old['layout'] != 'S5_occluded' or old['id'] not in witnesses or old['id'] != reference['id']
                or sha256(path / 'result.json') != reference['result_sha256']):
            raise ValueError('Source result or blocker witness differs for ' + case)
        blocker, = [x for x in old['scene']['actors'] if x['name'] == BLOCKER]
        schedule.append({'case': case, 'id': old['id'], 'target_model': old['model_id'],
                         'blocker': BLOCKER, 'blocker_model': blocker['model_id'],
                         'direct': reference, 'source_trial': str(path)})
        files.extend(path / name for name in ('result.json', 'state.json', 'initial_conditions.json',
                        'visibility.json', 'observation_request.json', 'observation_plan_result.json', 'plan_result.json'))
    required = list(devices)
    return {'schedule': schedule, 'cases': len(schedule), 'required_components': required, 'devices': devices,
            'sim_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'registered_scope': '固定挡物移除配对验证',
            'blocker_selection': '步骤02登记的历史碰撞实例与仿真实例掩码',
            'one_clearance_attempt_per_case': True, 'closed_loop_same_simulator_instance': True,
            'baseline': '已验收的同场景单 ZeroGrasp 直接抓取记录',
            'source_hashes': {str(p): sha256(p) for p in files}}


def choose_actor(env, name):
    env.obj = env.objects[name]
    info = env.object_info[name]
    env.object_bbox, env.object_scale = info['bbox'], info.get('scales', [1.])[0]
    for actor, geometry in env.exported_geometry.items():
        geometry['target'] = actor == name


def capture(env, directory):
    from run import snapshot

    obs = env.get_obs()
    state = snapshot(env, obs)
    data = obs['sensor_data']['right_wrist']
    rgb = data['rgb'][0].cpu().numpy()
    depth = data['depth'][0, :, :, 0].cpu().numpy()
    mask = data['segmentation'][0, :, :, 0].cpu().numpy() == env.obj._objs[0].per_scene_id
    if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > np.iinfo(np.uint16).max:
        raise ValueError('Malformed wrist depth')
    meta = state['camera_alignment']['camera_definition']['depth_valid_range_m']
    count = int(np.count_nonzero(mask & (depth * .001 > meta[0]) & (depth * .001 <= meta[1])))
    h, w = mask.shape
    k = np.asarray(state['intrinsics'])
    camera = directory / 'camera'
    camera.mkdir()
    Image.fromarray(rgb).save(camera / 'color.png')
    Image.fromarray(depth.astype(np.uint16)).save(camera / 'depth.png')
    Image.fromarray(mask.astype(np.uint8)).save(camera / 'mask.png')
    save(camera / 'params.json', {'depth_scale': .001, 'intrinsics': {'width': w, 'height': h, 'fx': k[0, 0],
         'fy': k[1, 1], 'ppx': k[0, 2], 'ppy': k[1, 2]}})
    save(directory / 'state.json', state)
    save(directory / '观测回执.json', {'raw_mask_pixels': int(mask.sum()), 'valid_depth_points': count,
         'mask_source': 'fresh simulator instance segmentation'})
    if count < 50:
        raise removal.StageStopped('insufficient_visible_depth_for_current_object')
    return state


@contextmanager
def recorded(env, directory):
    from run import Execution
    execution = Execution(env, directory)
    try:
        yield execution
    except RuntimeError as error:
        text = str(error)
        if text == 'forbidden_robot_contact' or text.endswith('_not_converged'):
            raise removal.StageStopped(text) from error
        raise
    finally:
        execution.finish()


def segment(execution, label, path, *, goal=None, finger=.044):
    execution.segment(label, path['q'], path['times'], finger=finger, goal_world=goal)


def observe(env, directory, socket, request, archived=None):
    from run import snapshot

    directory.mkdir()
    save(directory / 'state.json', snapshot(env))
    save(directory / 'observation_request.json', request)
    path = archived
    if path is None:
        path = call(socket, directory, '观察规划', observe=True)['paths']['observe']
    call(socket, directory, '观察轨迹复核', action='validate', path=path)
    with recorded(env, directory) as execution:
        segment(execution, 'observe', path, goal=request['goal_world'])
    state = snapshot(env, env.get_obs())
    save(directory / 'observed_state.json', state)
    return state


def grasp(env, directory, sockets):
    from run import snapshot
    from native_pair_timing import native_candidates
    from grasp_visuals import render
    import arx_pipeline.baseline as verifier

    directory.mkdir()
    result = {'success': False, 'physical_attempt': False, 'close_started': False, 'planning_success': False}
    begun = time.monotonic()
    try:
        state = capture(env, directory)
        inference = call(sockets['zerograsp'], directory, '原生推理')
        pool = native_candidates(directory)['counts']
        result.update(counts={'raw': inference['count'], **pool}, inference_seconds=inference['wall_seconds'])
        if not pool['scored']:
            raise removal.StageStopped('no_native_candidates_fit_arx_opening')
        render(directory / 'candidates', directory / 'zg_outputs')
        socket = sockets['curobo']
        refinement = call(socket, directory, '插入复核', refine=True, orientation_mode='fixed_orientation')
        result['retained_candidates'] = refinement['retained']
        plan = call(socket, directory, '抓取规划', selector='graspcl')
        selected = plan['selected']
        result['planning_seconds'] = plan['wall_seconds']
        validate_selected(selected)
        if set(read(directory / 'plan/scene.json')['mesh']) != {
                name for name, mesh in state['scene_meshes'].items() if not mesh['target']}:
            raise ValueError('Planner scene omits an obstacle')
        path = plan['paths']['pregrasp']
        call(socket, directory, '进场复核', action='validate', path=path)
        result['planning_success'] = True
        with recorded(env, directory) as execution:
            result['physical_attempt'] = True
            try:
                segment(execution, 'pregrasp', path, goal=selected['pregrasp_world'])
                save(directory / 'pregrasp_state.json', snapshot(env))
                path = call(socket, directory, '实测接近', approach=True)['paths']['approach']
                segment(execution, 'approach', path, goal=selected['grasp_world'])
                result['close_started'] = True
                segment(execution, 'close', removal.until_still(selected['q_grasp'], 3.), finger=0.,
                        goal=selected['grasp_world'])
                result['bilateral_fingers'] = sorted(execution.bilateral)
                if len(execution.bilateral) != 2:
                    raise removal.StageStopped('no_bilateral_object_contact')
                save(directory / 'closed_state.json', snapshot(env))
                path = call(socket, directory, '实测短抬', lift=True)['paths']['lift']
                segment(execution, 'lift', path, finger=0., goal=selected['lift_world'])
                segment(execution, 'hold', removal.until_still(path['q'][-1], HOLD_SECONDS), finger=0.,
                        goal=selected['lift_world'])
                result['minimum_hold_lift_m'] = min(execution.heights) - execution.initial_z
                result['hold_seconds'] = len(execution.heights) / 120
                result['success'] = result['minimum_hold_lift_m'] >= MIN_HOLD_LIFT_M and result['hold_seconds'] >= HOLD_SECONDS
                result['reason'] = 'success' if result['success'] else 'insufficient_hold_height'
            finally:
                result.update(stages=execution.stages, forbidden_contact_events=len(execution.forbidden))
    except removal.StageStopped as error:
        result['reason'] = str(error)
    finally:
        result['wall_seconds'] = time.monotonic() - begun
        save(directory / 'result.json', result)
    install_verifier(verifier)
    verifier.verify_trial(directory)
    return result


def place(env, directory, socket, initial, result):
    from run import snapshot

    directory.mkdir()
    held = snapshot(env)
    save(directory / 'held_state.json', held)
    vertices = removal.mesh_vertices(held['scene_meshes'][BLOCKER])
    carry = call(socket, directory, '搬运规划', action='placement')
    path = carry['paths']['carry']
    with recorded(env, directory) as execution:
        segment(execution, 'carry', path, finger=0., goal=path['goal'])
        observer = removal.PlacementObservation(env, execution, vertices)
        samples = observer.segment('carry_settle', removal.until_still(path['q'][-1], 2.), finger=0.)
        settled = snapshot(env)
        save(directory / 'carry_settled_state.json', settled)
        grip = removal.stability(samples, 'in_gripper')
        if (not grip['stable'] or not removal.grip_confirmed(samples)
                or min(row['cup_bottom_m'] for row in samples) < removal.RELEASE_HEIGHT
                or np.linalg.norm(np.asarray(settled['object_world'])[:2, 3] - carry['planned_object_xy_m']) > .02):
            raise removal.StageStopped('blocker_unstable_or_outside_landing_area')
        relative = np.linalg.inv(settled['links']['right_eef_link']) @ settled['object_world']
        lower = call(socket, directory, '下降规划', action='measured_lower')
        path = lower['path']
        samples = observer.segment('lower', path, finger=0., goal_world=lower['goal'])
        shift, angle = np.max([removal.pose_errors(relative, row['in_gripper']) for row in samples], axis=0)
        if (not removal.grip_confirmed(samples) or shift > removal.MAX_LOWER_SHIFT or angle > removal.MAX_LOWER_ANGLE
                or min(row['cup_bottom_m'] for row in samples) < removal.CARRY_CLEARANCE):
            raise removal.StageStopped('blocker_shifted_or_touched_during_lowering')
        samples = observer.segment('release', removal.until_still(path['q'][-1], 3.), finger=.044)
        released = snapshot(env)
        save(directory / 'released_state.json', released)
        z0 = np.asarray(initial['scene_meshes'][BLOCKER]['world'])[:3, 2]
        cup = np.asarray(released['object_world'])
        drift_target = removal.pose_errors(initial['scene_meshes']['target']['world'],
                                           released['scene_meshes']['target']['world'])
        if (not removal.stability(samples, 'world')['stable']
                or not all(row['table_contact'] and not row['fingers'] and not row['other_object_contact'] for row in samples)
                or np.linalg.norm(cup[:2, 3] - lower['planned_object_xy_m']) > .02
                or np.degrees(np.arccos(np.clip(z0 @ cup[:3, 2], -1., 1.))) > 10.
                or abs(removal.transformed(vertices, cup)[:, 2].min()) > .004
                or drift_target[0] > removal.TARGET_DRIFT or drift_target[1] > .03):
            raise removal.StageStopped('placement_unstable_or_target_disturbed')
        result['placed'] = True
        retreat = call(socket, directory, '撤离规划', action='retreat')
        segment(execution, 'retreat', retreat['path'], goal=retreat['goal'])
        final = snapshot(env)
        change = observation.actor_errors(released, final)
        if not all(shift <= .005 and angle <= .03 for shift, angle in change.values()):
            raise removal.StageStopped('scene_disturbed_after_retreat')
        result['removed'] = True
        result['retreat_height_m'] = retreat['selected_vertical_distance_m']
        save(directory / 'final_state.json', final)
    return final


def run_case(directory, sockets, entry, result):
    import torch
    from run import make
    from scene_tools import visibility

    source = Path(entry['source_trial'])
    old = read(source / 'result.json')
    original = read(source / 'state.json')
    request = read(source / 'observation_request.json')
    env = make(old['model_id'], perception_camera='right_wrist', scene_spec=old['scene'])
    begun = time.monotonic()
    try:
        env.reset(seed=old['seed'])
        removal.model_matches(env, original)
        removal.initial_match(env, directory / '初始条件.json', old['seed'], source)
        for _ in range(80):
            env._step_action(torch.tensor(env.initial_q[None], device=env.device))
        socket = sockets['curobo']
        initial = observe(env, directory / '00_进场', socket, request,
                          read(source / 'observation_plan_result.json')['paths']['observe'])
        removal.require_view_match(directory, initial, original)
        result['visibility_before'] = visibility(env)
        if not result['visibility_before']['admitted_view']:
            raise removal.StageStopped('initial_wrist_observation_inadmissible')
        result['phase'] = '挡物抓取'
        choose_actor(env, BLOCKER)
        result['blocker_grasp'] = grasp(env, directory / '01_抓挡', sockets)
        if not result['blocker_grasp']['success']:
            raise removal.StageStopped(result['blocker_grasp']['reason'])
        result['phase'] = '挡物放置和撤离'
        after_place = place(env, directory / '02_放置', socket, initial, result)

        result['phase'] = '重新观测'
        choose_actor(env, 'target')
        observed = observe(env, directory / '03_重观测', socket, request)
        errors = observation.actor_errors(after_place, observed)
        camera = removal.pose_errors(initial['camera_world_optical'], observed['camera_world_optical'])
        if not observation.poses_within_tolerance({**errors, 'camera': camera}):
            raise removal.StageStopped('view_or_objects_changed_during_reobserve')
        result['visibility_after'] = visibility(env)
        if not result['visibility_after']['admitted_view']:
            raise removal.StageStopped('original_target_not_visible_after_removal')
        result['reobserved'] = True
        result['phase'] = '原目标重抓'
        result['target_grasp'] = grasp(env, directory / '04_重抓', sockets)
        result['success'] = result['target_grasp']['success']
        result['reason'] = result['target_grasp']['reason']
        if result['success']:
            result['phase'] = '完成'
    except removal.StageStopped as error:
        result['reason'] = str(error)
    finally:
        env.close()
        result['wall_seconds'] = time.monotonic() - begun


def report(root, registration):
    cases = []
    for entry in registration['schedule']:
        trial = root / '试次' / entry['case']
        path = trial / '配对结果.json'
        if path.is_file():
            cases.append(read(path))
        else:
            cases.append({**entry, 'status': 'unexecuted'})
    finished = [row for row in cases if row['status'] == 'completed']
    summary = {'total': len(cases), 'completed': len(finished),
               'unexecuted': sum(row['status'] == 'unexecuted' for row in cases),
               'infrastructure_faults': sum(row['status'] == 'infrastructure_fault' for row in cases),
               'blocker_lift_passed': sum(row.get('blocker_grasp', {}).get('success', False) for row in finished),
               'blocker_removed': sum(row.get('removed', False) for row in finished),
               'reobserved': sum(row.get('reobserved', False) for row in finished),
               'original_target_attempted': sum(row.get('target_grasp', {}).get('physical_attempt', False) for row in finished),
               'rescued': sum(row.get('success', False) for row in finished),
               'case_wall_seconds': sum(row.get('wall_seconds', 0.) for row in finished),
               'rows': cases}
    save(root / '配对汇总.json', summary)
    return summary


def batch(root, planner_gpu, neural_gpu):
    devices = {'zerograsp': neural_gpu, 'curobo': planner_gpu}
    if root.parent != OUT or not all(gpu.startswith('GPU-') for gpu in (*devices.values(), os.environ.get('CUDA_VISIBLE_DEVICES', ''))):
        raise ValueError('Use stage 06 resources and physical GPU UUIDs')
    registration = register(root, devices)
    root.mkdir(exist_ok=False)
    (root / '试次').mkdir()
    save(root / '配对登记.json', registration)
    os.environ['MS_ASSET_DIR'] = str(MS / 'assets')
    os.environ.setdefault('SAPIEN_CACHE_DIR', str(ROOT / '缓存/sapien'))
    started = time.monotonic()
    try:
        with ready(root, devices) as sockets:
            for file, digest in registration['source_hashes'].items():
                if sha256(Path(file)) != digest:
                    raise ValueError('Frozen source changed: ' + file)
            print('组件就绪，共登记', len(registration['schedule']), '例', flush=True)
            for row in registration['schedule']:
                require_ready(root / 'resident', list(devices))
                directory = root / '试次' / row['case']
                directory.mkdir()
                save(directory / '试次领取.json', row)
                result = {**row, 'status': 'completed', 'phase': '进场观测', 'placed': False,
                          'removed': False, 'reobserved': False, 'success': False}
                save(root / '当前进度.json', {'case': row['case'], 'state': 'executing'})
                print('开始', row['case'], row['target_model'], row['blocker_model'], flush=True)
                try:
                    run_case(directory, sockets, row, result)
                except Exception as error:
                    result.update(status='infrastructure_fault', reason=str(error), error=traceback.format_exc())
                    raise
                finally:
                    save(directory / '配对结果.json', result)
                    summary = report(root, registration)
                    print(row['case'], result['phase'], result.get('reason'), summary['completed'],
                          '/', summary['total'], flush=True)
    except Exception as error:
        save(root / '批次故障.json', {'reason': str(error), 'error': traceback.format_exc()})
        raise
    finally:
        summary = report(root, registration)
        save(root / '批次耗时.json', {'wall_seconds_including_warmup': time.monotonic() - started})
        save(root / '当前进度.json', {'state': 'completed' if summary['completed'] == summary['total'] else 'stopped',
             'completed': summary['completed'], 'rescued': summary['rescued']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('output', type=Path)
    parser.add_argument('gpu_or_socket')
    parser.add_argument('--neural-gpu')
    args = parser.parse_args()
    if args.mode == 'worker':
        planner(args.output.resolve(), Path(args.gpu_or_socket))
    elif args.neural_gpu is None:
        parser.error('run 需要 --neural-gpu')
    else:
        batch(args.output.resolve(), args.gpu_or_socket, args.neural_gpu)
