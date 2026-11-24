"""从 I0069 的移障交接状态重新采集原目标观测。"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import tempfile
import time
import traceback

import numpy as np
from PIL import Image

import 验证绿杯移障 as removal
from 定位挡物 import points_in_base
from benchmark_lifecycle import mark_ready, require_ready

ROOT, MS = removal.ROOT, removal.MS
read, save, sha256 = removal.read, removal.save, removal.sha256
RESOURCE = ROOT / 'docs/04_重新观测并更新场景/resources'
HANDOFF = removal.RESOURCES / 'I0069_移障验证04'
EXPECTED_ACTOR = 'target'


@contextmanager
def resident(directory, gpu):
    service = directory / 'resident'
    service.mkdir()
    environment = MS.parent / 'arx_core/scripts/curobo_arx/environment.sh'
    python = MS.parent / 'curobo/.venv/bin/python'
    worker = Path(removal.__file__).resolve()
    with tempfile.TemporaryDirectory(prefix='claro_obs_') as tmp, (service / 'curobo.log').open('x') as log:
        address = Path(tmp) / 'worker.sock'
        cmd = ['bash', '-c', 'source "$1"; shift; export TORCH_EXTENSIONS_DIR="$CLARO_TORCH_CACHE"; exec "$@"',
               'claro-observe', str(environment), str(python), str(worker), 'worker', str(service), str(address)]
        process = subprocess.Popen(cmd, env={**removal.planner_env(gpu),
                                   'CLARO_TORCH_CACHE': str(ROOT / '缓存/torch_extensions')},
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            deadline = time.monotonic() + 420
            while not (service / 'curobo_ready.json').exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('cuRobo warmup failed')
                time.sleep(2)
            mark_ready(service, ['curobo'], {'sockets': {'curobo': str(address)}}, {'curobo': process})
            require_ready(service, ['curobo'])
            yield address
        finally:
            if process.poll() is None:
                try:
                    removal.rpc(address, {'command': 'shutdown'}, timeout=5)
                    process.wait(timeout=12)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def actor_errors(reference, measured):
    if reference['scene_meshes'].keys() != measured['scene_meshes'].keys():
        raise ValueError('Scene instances differ')
    return {n: removal.pose_errors(item['world'], measured['scene_meshes'][n]['world'])
            for n, item in reference['scene_meshes'].items()}


def poses_within_tolerance(errors):
    return all(p <= removal.POSITION_MATCH and a <= removal.ROTATION_MATCH for p, a in errors.values())


def restore(env, released):
    import torch
    from run import snapshot

    robot = env.agent.robot
    q = np.asarray(released['qpos'], dtype=np.float32)
    if [joint.name for joint in robot.active_joints] != released['joint_names']:
        raise ValueError('Handoff joint list differs')
    limits = robot.qlimits.cpu().numpy()[0]
    if (not np.isfinite(q).all() or np.any(q < limits[:, 0] - 1e-5)
            or np.any(q > limits[:, 1] + 1e-5)):
        raise ValueError('Measured handoff joints outside tolerances')
    q = np.clip(q, limits[:, 0], limits[:, 1])
    if not np.allclose(q[[released['joint_names'].index(n) for n in removal.FINGERS]], .044, atol=1e-4):
        raise ValueError('Expected released and open gripper')
    env.agent.reset(q)
    command = torch.tensor(q[None], device=env.device)
    for _ in range(80):
        env._step_action(command)
    return snapshot(env, env.get_obs())


def capture(directory, state, obs, env, reference_trial):
    from scene_tools import visibility

    data = obs['sensor_data']['right_wrist']
    rgb = data['rgb'][0].cpu().numpy()
    depth = data['depth'][0, :, :, 0].cpu().numpy()
    seg = data['segmentation'][0, :, :, 0].cpu().numpy()
    target_id = env.objects[EXPECTED_ACTOR]._objs[0].per_scene_id
    mask = seg == target_id
    camera = directory / 'camera'
    camera.mkdir()
    k = np.asarray(state['intrinsics'])
    h, w = mask.shape
    params = {'depth_scale': .001, 'intrinsics': {'width': w, 'height': h, 'fx': k[0, 0], 'fy': k[1, 1],
                                                'ppx': k[0, 2], 'ppy': k[1, 2]}}
    Image.fromarray(rgb).save(camera / 'color.png')
    if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > np.iinfo(np.uint16).max:
        raise ValueError('Unexpected range of wrist depth in millimeters')
    depth = depth.astype(np.uint16)
    Image.fromarray(depth).save(camera / 'depth.png')
    Image.fromarray(mask.astype(np.uint8)).save(camera / 'mask.png')
    save(camera / 'params.json', params)
    xyz, optical, uv, pixels = points_in_base(depth, mask, rgb, params, state)
    np.savez_compressed(directory / '目标点云.npz', xyz_base_m=xyz, xyz_optical_m=optical, uv=uv, rgb=pixels)
    latest = visibility(env, obs)
    prior = read(reference_trial / 'visibility.json')
    if int(mask.sum()) != latest['target_mask_pixels'] or not latest['admitted_view']:
        raise removal.StageStopped('target_wrist_view_not_admitted_after_removal')

    original_camera = reference_trial / 'camera'
    before_rgb = np.asarray(Image.open(original_camera / 'color.png').convert('RGB'))
    before_mask = np.asarray(Image.open(original_camera / 'mask.png')) > 0
    prior_state = read(reference_trial / 'state.json')
    before_depth = np.asarray(Image.open(original_camera / 'depth.png'))
    before_point_count = len(points_in_base(before_depth, before_mask, before_rgb,
                                            read(original_camera / 'params.json'), prior_state)[0])
    if int(before_mask.sum()) != prior['target_mask_pixels']:
        raise ValueError('Before-mask counts differ from baseline')
    frames = []
    for image, region in [(before_rgb, before_mask), (rgb, mask)]:
        shaded = image.copy()
        shaded[region] = np.rint(.6 * image[region] + .4 * np.asarray([255, 196, 40])).astype(np.uint8)
        frames.append(shaded)
    Image.fromarray(np.concatenate(frames, axis=1)).save(directory / '移障前后腕部对照.png')
    result = {'target_actor': EXPECTED_ACTOR, 'target_model': env.object_models[EXPECTED_ACTOR],
              'mask_source': 'SAPIEN instance segmentation, same as direct benchmark',
              'instance_id': int(target_id), 'depth_valid_points_before': before_point_count,
              'depth_valid_points_after': len(xyz), 'before': prior, 'after': latest,
              'robot_state_and_camera_same_frame': True, 'body_frame': 'body4'}
    save(directory / '重新观测记录.json', result)
    return result


def validate_world(directory, state, blocker, baseline):
    scene = read(directory / 'updated_复核/scene.json')
    if set(scene['mesh']) != set(state['scene_meshes']) or set(scene['cuboid']) != set(state['scene_cuboids']):
        raise ValueError('Collision scene lost a physical actor or support')
    invbase = np.linalg.inv(state['links']['body4'])
    for name, item in state['scene_meshes'].items():
        planner = scene['mesh'][name]
        expected = invbase @ np.asarray(item['world'])
        if (Path(planner['file_path']) != Path(item['file_path'])
                or not np.allclose(planner['pose'][:3], expected[:3, 3], atol=1e-7, rtol=0)):
            raise ValueError('Collision pose differs from observation: ' + name)
    target, = (name for name, item in state['scene_meshes'].items() if item['target'])
    if target != EXPECTED_ACTOR or state['object_world'] != state['scene_meshes'][target]['world']:
        raise ValueError('Temporary grasp target was not restored')
    return {'includes_target': True, 'target_actor': target, 'includes_moved_blocker': blocker in scene['mesh'],
            'meshes': list(scene['mesh']), 'cuboids': list(scene['cuboid']),
            'actor_pose_changes_vs_before_m_rad': actor_errors(baseline, state)}


def experiment(directory, address, released, baseline, original, source_trial):
    from run import Execution, make, snapshot

    post_scene = deepcopy(original['scene'])
    start_geometry = deepcopy(released)
    for item in post_scene['actors']:
        name = item['name']
        if item['model_id'] != released['objects'][name]['model_id']:
            raise ValueError('Actor model differs from handoff')
        item['pose'] = released['objects'][name]['pose']
        start_geometry['scene_meshes'][name]['target'] = name == EXPECTED_ACTOR

    env = make(original['model_id'], perception_camera='right_wrist', scene_spec=post_scene)
    execution = None
    try:
        env.reset(seed=original['seed'])
        removal.model_matches(env, start_geometry)
        restored = restore(env, released)
        object_errors = actor_errors(released, restored)
        robot_base = removal.pose_errors(released['links']['base_link'], restored['links']['base_link'])
        qerr = float(np.max(np.abs(np.asarray(restored['qpos']) - released['qpos'])))
        passed = poses_within_tolerance({**object_errors, 'robot_base': robot_base}) and qerr < .005
        save(directory / '重建状态核验.json', {'passed': passed, 'approximation': 'restore settled pose and zero velocities',
             'actor_error_m_rad': object_errors, 'base_error_m_rad': robot_base, 'joint_error_rad': qerr,
             'settling_control_steps': 80})
        if not passed:
            raise removal.StageStopped('measured_handoff_not_reconstructed')

        prep = directory / '观察规划'
        prep.mkdir()
        save(prep / 'state.json', restored)
        observation = read(source_trial / 'observation_request.json')
        save(prep / 'observation_request.json', observation)
        path = removal.call(address, prep, '回到原观察位', observe=True)['paths']['observe']
        check = removal.call(address, prep, '全场景观察轨迹复核', action='validate', label='observe',
                             state='state.json', path=path)
        print('I0069 场景重建与观察轨迹复核通过，回到同一腕部观测位', flush=True)
        execution = Execution(env, directory)
        execution.segment('observe', path['q'], path['times'], goal_world=observation['goal_world'])
        obs = env.get_obs()
        current = snapshot(env, obs)
        save(directory / 'state.json', current)
        camera_error = removal.pose_errors(baseline['camera_world_optical'], current['camera_world_optical'])
        same_intrinsics = bool(np.allclose(baseline['intrinsics'], current['intrinsics'], rtol=0, atol=1e-6))
        current_errors = actor_errors(restored, current)
        passed = same_intrinsics and poses_within_tolerance({**current_errors, 'camera': camera_error})
        save(directory / '同视角核验.json', {'passed': passed, 'camera_error_m_rad': camera_error,
             'same_intrinsics': same_intrinsics, 'actor_motion_m_rad': current_errors})
        if not passed:
            raise removal.StageStopped('camera_or_actors_mismatch_after_observe')

        view = capture(directory, current, obs, env, source_trial)
        q = removal.arm_q(current).tolist()
        removal.call(address, directory, '刷新碰撞场景', action='validate', label='updated', state='state.json',
                     path={'q': [q, q], 'times': [0, .025]})
        blocker = read(HANDOFF / '验证结果.json')['blocker']
        scene = validate_world(directory, current, blocker, baseline)
        return {'success': True, 'fresh_wrist_rgbd': True, 'observation_executed': True, 'physical_grasp_attempts': 0,
                'observation_path_checks': check['points'], 'camera_error_m_rad': camera_error,
                'target_visible_before': view['before']['target_mask_pixels'],
                'target_visible_after': view['after']['target_mask_pixels'],
                'target_occlusion_before': view['before']['occlusion_fraction'],
                'target_occlusion_after': view['after']['occlusion_fraction'],
                'depth_points_before': view['depth_valid_points_before'],
                'depth_points_after': view['depth_valid_points_after'],
                'collision_scene': scene}
    finally:
        try:
            if execution is not None:
                execution.finish()
        finally:
            env.close()


def run(directory, gpu):
    sim_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or not sim_gpu.startswith('GPU-'):
        raise ValueError('Bind physical GPU UUIDs')
    if directory.parent != RESOURCE:
        raise ValueError('Place output in the stage 04 resources')
    result_file = HANDOFF / '验证结果.json'
    audit_file = HANDOFF / '轨迹独立复核.json'
    released_file = HANDOFF / 'final_state.json'
    handoff = read(result_file)
    if not handoff['success'] or not read(audit_file)['check_passed']:
        raise ValueError('Step 03 handoff not validated')
    record = read(HANDOFF / '验证登记.json')
    source_trial = Path(record['source_trial'])
    original = read(source_trial / 'result.json')
    baseline_path = source_trial / 'state.json'
    if handoff['scene'] != original['id'] or original['id'].split('_')[0] != 'I0069':
        raise ValueError('Handoff scene differs from I0069')
    released = read(released_file)
    baseline = read(baseline_path)
    files = [Path(__file__), Path(removal.__file__), removal.POLICY, result_file, audit_file, released_file, baseline_path,
             source_trial / 'result.json', source_trial / 'observation_request.json', source_trial / 'visibility.json',
             *(source_trial / 'camera' / name for name in ('color.png', 'depth.png', 'mask.png', 'params.json'))]
    digests = {str(path): sha256(path) for path in files}
    directory.mkdir(exist_ok=False)
    save(directory / '重新观测登记.json', {'scene': original['id'], 'seed': original['seed'],
         'required_components': ['curobo'], 'sim_gpu': sim_gpu, 'planner_gpu': gpu, 'hand_off': str(released_file),
         'restore_note': '独立实例重建静止状态，关节和物体采用上一步实测记录，动态速度从零开始',
         'view': '原基线腕部观察位', 'scene_assets': str(MS / 'assets'), 'frozen_inputs': digests})
    os.environ['MS_ASSET_DIR'] = str(MS / 'assets')
    os.environ.setdefault('SAPIEN_CACHE_DIR', str(ROOT / '缓存/sapien'))
    begin = time.monotonic()
    try:
        with resident(directory, gpu) as address:
            for path, digest in digests.items():
                if sha256(Path(path)) != digest:
                    raise ValueError('Registered input changed: ' + path)
            result = experiment(directory, address, released, baseline, original, source_trial)
    except Exception as error:
        save(directory / '重新观测结果.json', {'success': False, 'reason': str(error),
             'error': traceback.format_exc(), 'physical_grasp_attempts': 0})
        raise
    result['wall_seconds'] = time.monotonic() - begin
    files = ['camera/color.png', 'camera/depth.png', 'camera/mask.png', 'camera/params.json', 'state.json',
             '目标点云.npz', 'updated_复核/scene.json']
    result['handoff_sha256'] = {str(name): sha256(directory / name) for name in files}
    save(directory / '重新观测结果.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('planner_gpu')
    args = parser.parse_args()
    run(args.output.resolve(), args.planner_gpu)
