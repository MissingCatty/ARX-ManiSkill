"""用 I0069 的移障观测重抓积木，沿用单 ZeroGrasp 基线。"""
from arx_project.layout import canonical_path

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
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
from benchmark_lifecycle import mark_ready, require_ready
from resident_backend import serve
from short_lift_protocol import HOLD_SECONDS, MIN_HOLD_LIFT_M, validate_selected

ROOT, MS = removal.ROOT, removal.MS
read, save, sha256 = removal.read, removal.save, removal.sha256
RESOURCE = ROOT / 'docs/05_重抓原目标/resources'
HANDOFF = observation.RESOURCE / 'I0069_重新观测01'
CHECKPOINT = MS.parent / ('arx_core/servers/zerograsp/vendor/zerograsp_mainline_minimal/'
                          'checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt')


def serve_planner(directory, address):
    import torch
    import resident_planner
    import table_truncation_model as truncation
    from checked_native_services import checked_pool
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner

    spec = importlib.util.spec_from_file_location('frozen_table_policy', removal.POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner

    class RegraspPool(truncation.make_pool(checked_pool(FastPlannerPool))):
        def request(self, directory, **kwargs):
            path = kwargs.pop('full_scene_path', None)
            if path is None:
                return super().request(directory, **kwargs)
            state = read(directory / 'state.json')
            samples = removal.dense_path(path)
            if np.max(np.abs(samples[0] - removal.arm_q(state))) >= .005:
                return {'success': False, 'reason': 'pregrasp_start_differs'}
            context = self.bind(directory / '预抓取全场景复核', state, include_target=True)
            q = torch.as_tensor(samples, dtype=torch.float32, device='cuda')
            passed = all(bool(context.checker.validate(batch[:, None]).all()) for batch in q.split(64))
            return {'success': passed, 'points': len(samples), 'reason': None if passed else 'pregrasp_full_scene_collision'}

    resident_planner.PlannerPool = RegraspPool
    serve(SimpleNamespace(kind='curobo', root=directory, socket=address, warmup_trial=removal.WARMUP))


def stop_worker(process, address):
    if process.poll() is not None:
        return
    try:
        removal.rpc(address, {'command': 'shutdown'}, timeout=6)
        process.wait(timeout=12)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def ready(directory, devices):
    service = directory / 'resident'
    service.mkdir()
    env_script = MS.parent / 'arx_core/scripts/curobo_arx/environment.sh'
    with tempfile.TemporaryDirectory(prefix='claro_grasp_') as tmp, ExitStack() as stack:
        addresses, children = {}, {}
        for kind, gpu in devices.items():
            address = Path(tmp) / (kind + '.sock')
            addresses[kind] = address
            if kind == 'zerograsp':
                command = [canonical_path(MS.parent / 'tmp/arx-fullchain/env/bin/python'), MS / 'postprocess_backend.py',
                           '--kind', kind, '--socket', address, '--root', service, '--warmup-trial', removal.WARMUP]
            else:
                command = [MS.parent / 'curobo/.venv/bin/python', Path(__file__).resolve(),
                           'worker', service, address]
            log = stack.enter_context((service / (kind + '.log')).open('x'))
            wrapped = ['bash', '-c', 'source "$1"; shift; export TORCH_EXTENSIONS_DIR="$CLARO_TORCH_CACHE"; exec "$@"',
                       'claro-regrasp', str(env_script), *map(str, command)]
            env = {**removal.planner_env(gpu), 'CLARO_TORCH_CACHE': str(ROOT / '缓存/torch_extensions'),
                   'ARX_NEURAL_GPU': gpu}
            proc = subprocess.Popen(wrapped, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children[kind] = proc
            stack.callback(stop_worker, proc, address)
        deadline = time.monotonic() + 480
        while not all((service / (name + '_ready.json')).exists() for name in devices):
            ended = [kind for kind, child in children.items() if child.poll() is not None]
            if ended or time.monotonic() >= deadline:
                raise RuntimeError('Warmup failed or timed out: ' + ', '.join(ended or list(devices)))
            time.sleep(2)
        required = list(devices)
        mark_ready(service, required, {'sockets': {n: str(p) for n, p in addresses.items()}}, children)
        require_ready(service, required)
        yield addresses


def scene_for_reset(original, reference):
    scene = deepcopy(original['scene'])
    for actor in scene['actors']:
        measured = reference['objects'][actor['name']]
        if measured['model_id'] != actor['model_id']:
            raise ValueError('Unexpected actor model')
        actor['pose'] = measured['pose']
    return scene


def capture(directory, env, reference):
    from run import snapshot

    obs = env.get_obs()
    state = snapshot(env, obs)
    cam_error = removal.pose_errors(reference['camera_world_optical'], state['camera_world_optical'])
    state_errors = observation.actor_errors(reference, state)
    q_error = float(np.max(np.abs(np.asarray(state['qpos']) - reference['qpos'])))
    matching = (observation.poses_within_tolerance({**state_errors, 'camera': cam_error})
                and q_error < .005 and np.allclose(reference['intrinsics'], state['intrinsics'], atol=1e-6, rtol=0))
    data = obs['sensor_data']['right_wrist']
    rgb = data['rgb'][0].cpu().numpy()
    depth = data['depth'][0, :, :, 0].cpu().numpy()
    mask = data['segmentation'][0, :, :, 0].cpu().numpy() == env.obj._objs[0].per_scene_id
    prior_mask = np.asarray(Image.open(HANDOFF / 'camera/mask.png')) > 0
    overlap = np.count_nonzero(mask & prior_mask) / max(1, np.count_nonzero(mask | prior_mask))
    if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > np.iinfo(np.uint16).max:
        raise ValueError('Depth values outside uint16 millimeters')
    depth_mm = depth.astype(np.uint16)
    params = read(HANDOFF / 'camera/params.json')
    xyz, *_ = observation.points_in_base(depth_mm, mask, rgb, params, state)
    admitted = bool(matching and overlap >= .99 and len(xyz) >= 50)
    save(directory / '重抓观测核验.json', {'passed': admitted, 'camera_error_m_rad': cam_error,
         'joint_error_rad': q_error, 'actor_errors_m_rad': state_errors, 'stage04_mask_iou': overlap,
         'pixels': int(mask.sum()), 'depth_points': len(xyz), 'target_actor': 'target',
         'target_model': env.model_id, 'fresh_same_frame': True})
    if not admitted:
        raise removal.StageStopped('post_removal_capture_does_not_match_handoff')
    camera = directory / 'camera'
    camera.mkdir()
    Image.fromarray(rgb).save(camera / 'color.png')
    Image.fromarray(depth_mm).save(camera / 'depth.png')
    Image.fromarray(mask.astype(np.uint8)).save(camera / 'mask.png')
    save(camera / 'params.json', params)
    save(directory / 'state.json', state)
    return state


def validate_plan_scene(directory, state):
    scene = read(directory / 'plan/scene.json')
    invbase = np.linalg.inv(state['links']['body4'])
    others = set(state['scene_meshes']) - {'target'}
    if set(scene['mesh']) != others or 'distractor_00' not in others:
        raise ValueError('Updated scene lost a distractor')
    for name in others:
        wanted = invbase @ np.asarray(state['scene_meshes'][name]['world'])
        if not np.allclose(scene['mesh'][name]['pose'][:3], wanted[:3, 3], atol=1e-7, rtol=0):
            raise ValueError('Obstacle position differs from the new observation: ' + name)


def perform(directory, sockets, reference, original):
    import imageio.v2 as imageio
    from arx_pipeline.run import Execution, make, snapshot
    from native_pair_timing import native_candidates
    from grasp_visuals import render

    started = time.monotonic()
    result = {'id': original['id'], 'model_id': original['model_id'], 'success': False, 'planning_success': False,
              'physical_attempt': False, 'close_started': False, 'placement_changed_by_removal': True}
    execution = None
    env = None
    try:
        env = make(original['model_id'], perception_camera='right_wrist', scene_spec=scene_for_reset(original, reference))
        env.reset(seed=original['seed'])
        removal.model_matches(env, reference)
        observation.restore(env, reference)
        state = capture(directory, env, reference)
        imageio.imwrite(directory / '抓取前.png', env.render()[0].cpu().numpy())
        print('I0069 移障状态复核通过，开始计算新抓点', flush=True)

        infer = removal.call(sockets['zerograsp'], directory, '原生推理')
        pool = native_candidates(directory)
        result['candidate_counts'] = {'raw': infer['count'], **pool['counts']}
        result['inference_seconds'] = infer['wall_seconds']
        if not pool['counts']['scored']:
            raise removal.StageStopped('no_native_candidates_fit_arx_opening')
        render(directory / 'candidates', directory / 'zg_outputs')
        refined = removal.call(sockets['curobo'], directory, '插入检查', refine=True, orientation_mode='fixed_orientation')
        result['retained_candidates'] = refined['retained']
        print('I0069 候选', pool['counts']['scored'], '→', refined['retained'], flush=True)
        plan = removal.call(sockets['curobo'], directory, '重抓规划', selector='graspcl')
        result['planning_seconds'] = plan['wall_seconds']
        selected = plan['selected']
        validate_selected(selected)
        validate_plan_scene(directory, state)
        path = plan['paths']['pregrasp']
        check = removal.call(sockets['curobo'], directory, '预抓取复核', full_scene_path=path)
        result['pregrasp_checked_points'] = check['points']
        result['planning_success'] = True
        print('I0069 规划通过，开始执行预抓取和闭爪', flush=True)

        execution = Execution(env, directory)
        result['physical_attempt'] = True
        execution.segment('pregrasp', path['q'], path['times'], goal_world=selected['pregrasp_world'])
        save(directory / 'pregrasp_state.json', snapshot(env))
        path = removal.call(sockets['curobo'], directory, '实测接近', approach=True)['paths']['approach']
        execution.segment('approach', path['q'], path['times'], goal_world=selected['grasp_world'])
        q = selected['q_grasp']
        result['close_started'] = True
        execution.segment('close', [q, q], [0, 3], finger=0., goal_world=selected['grasp_world'])
        result['bilateral_fingers'] = sorted(execution.bilateral)
        if len(execution.bilateral) != 2:
            raise removal.StageStopped('no_bilateral_target_contact')
        save(directory / 'closed_state.json', snapshot(env))
        path = removal.call(sockets['curobo'], directory, '实测短抬', lift=True)['paths']['lift']
        execution.segment('lift', path['q'], path['times'], finger=0., goal_world=selected['lift_world'])
        q = path['q'][-1]
        execution.segment('hold', [q, q], [0, HOLD_SECONDS], finger=0., goal_world=selected['lift_world'])
        result['minimum_hold_lift_m'] = min(execution.heights) - execution.initial_z
        result['hold_seconds'] = len(execution.heights) / 120
        result['final_lift_m'] = execution.heights[-1] - execution.initial_z
        result['success'] = result['minimum_hold_lift_m'] >= MIN_HOLD_LIFT_M and result['hold_seconds'] >= HOLD_SECONDS
        result['reason'] = 'success' if result['success'] else 'insufficient_hold_height'
        imageio.imwrite(directory / '抬升结果.png', env.render()[0].cpu().numpy())
    except removal.StageStopped as error:
        result['reason'] = str(error)
    except Exception as error:
        result.update(reason=str(error), error=traceback.format_exc(), infrastructure_failure=True)
    finally:
        try:
            if execution is not None:
                execution.finish()
                result.update(stages=execution.stages, forbidden_contact_events=len(execution.forbidden))
            if env is not None and getattr(env, 'exported_geometry', None):
                save(directory / 'final_state.json', snapshot(env))
        finally:
            if env is not None:
                env.close()
            result['wall_seconds'] = time.monotonic() - started
            save(directory / 'result.json', result)
    import arx_pipeline.baseline as verifier
    from short_lift_protocol import install_verifier
    install_verifier(verifier)
    verifier.verify_trial(directory)
    print({k: v for k, v in result.items() if k != 'stages'}, flush=True)
    if result.get('infrastructure_failure'):
        raise RuntimeError(result['reason'])


def run(directory, planner, neural):
    if directory.parent != RESOURCE:
        raise ValueError('Place output in the stage 05 resources')
    sim = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not all(gpu.startswith('GPU-') for gpu in (sim, neural, planner)):
        raise ValueError('Bind physical GPU UUIDs')
    previous = read(HANDOFF / '重新观测结果.json')
    if not previous['success'] or not read(HANDOFF / '重新观测独立复核.json')['passed']:
        raise ValueError('Stage 04 observations have not passed verification')
    for file, digest in previous['handoff_sha256'].items():
        if sha256(HANDOFF / file) != digest:
            raise ValueError('Stage 04 output hash differs')
    source = Path(read(observation.HANDOFF / '验证登记.json')['source_trial'])
    reference = read(HANDOFF / 'state.json')
    original = read(source / 'result.json')
    if (original['id'].split('_')[0] != 'I0069' or original['success']
            or reference['objects']['target']['model_id'] != original['model_id']
            or not reference['scene_meshes']['target']['target']):
        raise ValueError('Unexpected source target')
    files = [Path(__file__), Path(observation.__file__), Path(removal.__file__), removal.POLICY, CHECKPOINT,
             HANDOFF / '重新观测结果.json', HANDOFF / '重新观测独立复核.json', source / 'result.json',
             source / 'plan_result.json', *(HANDOFF / p for p in previous['handoff_sha256'])]
    hashes = {str(f): sha256(f) for f in files}
    directory.mkdir(exist_ok=False)
    devices = {'zerograsp': neural, 'curobo': planner}
    save(directory / '重抓登记.json', {'scene': original['id'], 'attempt_budget': 1, 'sim_gpu': sim,
         'required_components': list(devices), 'gpu_by_component': devices,
         'source_trial': str(source), 'validated_post_removal_observation': str(HANDOFF),
         'scope': '静止场景接续，单 ZeroGrasp，固定朝向，执行一次原目标重抓验证',
         'minimum_lift_m': MIN_HOLD_LIFT_M, 'hold_seconds': HOLD_SECONDS, 'frozen_inputs': hashes})
    os.environ['MS_ASSET_DIR'] = str(MS / 'assets')
    os.environ.setdefault('SAPIEN_CACHE_DIR', str(ROOT / '缓存/sapien'))
    began = time.monotonic()
    try:
        with ready(directory, devices) as sockets:
            for file, digest in hashes.items():
                if sha256(Path(file)) != digest:
                    raise ValueError('Input changed after registration: ' + file)
            save(directory / '试次领取.json', {'scene': original['id'], 'claim': 1})
            perform(directory, sockets, reference, original)
    except Exception as error:
        save(directory / '流程故障.json', {'reason': str(error), 'error': traceback.format_exc()})
        raise
    finally:
        save(directory / '批次耗时.json', {'wall_seconds_including_warmup': time.monotonic() - began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('output', type=Path)
    parser.add_argument('planner_or_socket')
    parser.add_argument('--neural-gpu')
    args = parser.parse_args()
    if args.mode == 'worker':
        serve_planner(args.output.resolve(), Path(args.planner_or_socket))
    elif args.neural_gpu is None:
        parser.error('run 需要 --neural-gpu')
    else:
        run(args.output.resolve(), args.planner_or_socket, args.neural_gpu)
