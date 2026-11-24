"""Build five balanced layouts from the actual fixed head-camera frustum."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS = PROJECT.parent / 'ManiSkill'
SOURCE = MS / 'outputs/arx_selected_height_pair_20261120_01'
DEFAULT = HERE / 'datasets/head200_20261122_02'
LAYOUTS = ['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter', 'S5_occluded']
LABELS = ['开阔单物体', '桌边', '稀疏干扰', '密集杂物', '部分遮挡']
TABLE_MIN, TABLE_MAX = np.array([-.7245, -1.209]), np.array([.4845, 1.209])


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode) + '\n'); temp.replace(path)


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now().astimezone().isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_class(model):
    number = int(model.split('_')[0].split('-')[0])
    if number in [53, 54, 63]: return 'ball'
    if number in [51, 52]: return 'clamp'
    return {2: 'coffee_can', 28: 'lid', 29: 'plate', 32: 'knife', 36: 'wood_block',
        44: 'screwdriver', 59: 'chain', 65: 'cup', 70: 'wood_block',
        71: 'pegboard', 72: 'airplane_part', 73: 'lego'}.get(number, model)


def target_text(model):
    return {'coffee_can': 'coffee can', 'lid': 'pan lid', 'plate': 'plate', 'knife': 'knife',
        'wood_block': 'wooden block', 'screwdriver': 'flathead screwdriver', 'clamp': 'clamp',
        'chain': 'metal chain', 'cup': 'plastic cup', 'pegboard': 'pegboard',
        'airplane_part': 'toy airplane part', 'lego': 'LEGO brick', 'ball': 'ball'}[semantic_class(model)]


def prepare(root):
    assert not root.exists()
    source = read(SOURCE / 'registration.json')
    rows = deepcopy(source['schedule'])
    assert len(rows) == 200 and Counter(e['layout'] for e in rows) == dict.fromkeys(LAYOUTS, 40)
    root.mkdir(parents=True)
    for name in ['cases', 'source', 'preflight', 'gallery']: (root / name).mkdir()
    for i, row in enumerate(rows):
        row.update(id=f'H{i+1:04d}_{row["layout"]}_{row["model_id"]}',
            seed=2026112200 + i * 100003, target_text=target_text(row['model_id']),
            semantic_class=semantic_class(row['model_id']))
        for key in ['observation', 'admission', 'admission_failed', 'scene', 'historical_trial', 'source_admission']:
            row.pop(key, None)
    reg = {'at': now(), 'id': 'head_perception_200_v1', 'schedule': rows,
        'source_schedule': str(SOURCE / 'registration.json'), 'source_sha256': sha(SOURCE / 'registration.json'),
        'target_models': sorted({e['model_id'] for e in rows}),
        'target_quotas_by_layout': {l: dict(Counter(e['model_id'] for e in rows if e['layout'] == l)) for l in LAYOUTS},
        'layouts': dict.fromkeys(LAYOUTS, 40), 'common_initial_conditions': read(SOURCE / '高度初态/common_initial_conditions.json'),
        'camera_profile': read(PROJECT / 'configs/head_camera_mount.json'), 'lift_joint_m': .452,
        'observation': 'fixed head camera at initial robot state, no wrist preparation motion',
        'distractor_counts': {'S1_open': [0, 0], 'S2_edge': [0, 0], 'S3_sparse': [2, 3], 'S4_clutter': [4, 6], 'S5_occluded': [2, 4]},
        'admission': {'border_px': 12, 'minimum_visible_target_pixels': 64,
            'valid_target_depth_fraction': .98, 'occlusion_range_S5': [.20, .40],
            'maximum_occlusion_other': .15, 'table_edge_clearance_m': [.015, .035],
            'target_unique_semantic_class': True, 'maximum_scene_variants': 80,
            'all_actors_fully_in_head_frame': True,
            'grasp_success_or_IK_used_for_selection': False, 'SAM3_success_used_for_scene_selection': False},
        'physics_changed': False, 'physical_grasp_budget': 0,
        'asset_sha256': source['asset_sha256'],
        'source_hashes': {str(p): sha(p) for p in [Path(__file__),
            PROJECT / 'configs/head_camera_mount.json', MS / 'arx_pipeline/environment.py',
            MS / 'arx_pipeline/baseline.py', MS / 'arx_pipeline/scene_tools.py']}}
    save(root / 'registration.json', reg)
    (root / 'source/scene_builder.py').write_bytes(Path(__file__).read_bytes())
    save(root / 'progress.json', {'phase': 'registered', 'completed': 0, 'registered': 200, 'physical_grasps': 0})
    print('REGISTERED', root, '200 scenes /38 target models', flush=True)


def runtime():
    os.environ.setdefault('MS_ASSET_DIR', str(MS / 'assets'))
    sys.path[:0] = [str(MS), str(MS / 'arx_pipeline'), str(PROJECT / 'experiments/pointcloud_benchmark')]
    import arx_runtime_imports
    arx_runtime_imports.install()
    import environment, baseline, run, sapien, torch
    import gymnasium as gym
    from head_camera_config import head_configuration
    from camera_imaging import camera_profiles, sensor_and_model_frames, lens_maps
    environment.INITIAL_JOINTS = {**environment.INITIAL_JOINTS, 'joint4': .452}
    override, record = head_configuration(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS = deepcopy(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS['camera']['definitions']['head'] = record['camera_definition']
    keeper = []
    def make(model, scene):
        env = gym.make('ARXPickYCB-v1', model_id=model, perception_camera='head', scene_spec=scene,
            num_envs=1, obs_mode='rgb+depth+segmentation', control_mode='pd_joint_pos', reward_mode='none',
            render_mode='rgb_array', sim_backend='cpu', render_backend='sapien_cuda', sensor_configs=override).unwrapped
        if not keeper: keeper.append(sapien.render.SapienRenderer(env._render_device))
        return env
    return environment, baseline, run, sapien, torch, make, camera_profiles()['head'], sensor_and_model_frames, lens_maps


def project(spec, baseline, camera, k):
    p = np.asarray(spec['pose']); rot = Rotation.from_quat([*p[4:], p[3]]).as_matrix()
    world = baseline.corners(baseline.DB[spec['model_id']]) @ rot.T + p[:3]
    xyz = (world - camera[:3, 3]) @ camera[:3, :3]
    if xyz[:, 2].min() <= .10 or xyz[:, 2].max() >= 2.: return None
    uv = xyz @ k.T; uv = uv[:, :2] / uv[:, 2:]
    lo, hi = uv.min(0), uv.max(0)
    if np.any(lo < [12, 12]) or np.any(hi > [628, 468]): return None
    return {'lo': lo, 'hi': hi, 'area': float(np.prod(hi - lo)), 'depth_m': float(np.median(xyz[:, 2]))}


def nonoverlap(spec, actors, baseline, gap=.012):
    return baseline.separate(spec, actors, gap=gap)


def target_candidates(entry, baseline, camera, k, rng):
    candidates = []
    for _ in range(500):
        yaw = float(rng.uniform(-np.pi, np.pi))
        if entry['layout'] == 'S2_edge':
            temp = baseline.actor_spec('target', entry['model_id'], [0, 0], yaw)
            lo, hi = baseline.bounds(temp); half = (hi[:2] - lo[:2]) / 2
            edge = int(rng.integers(4)); gap = float(rng.uniform(.015, .035))
            xy = rng.uniform(TABLE_MIN + half + .015, TABLE_MAX - half - .015)
            axis, side = edge // 2, edge % 2
            xy[axis] = (TABLE_MIN[axis] + half[axis] + gap) if side == 0 else (TABLE_MAX[axis] - half[axis] - gap)
        else:
            xy = rng.uniform([-.62, -.55], [.30, .70])
        spec = baseline.actor_spec('target', entry['model_id'], xy, yaw)
        if not nonoverlap(spec, [], baseline): continue
        pr = project(spec, baseline, camera, k)
        if pr is not None and pr['area'] >= 100: candidates.append((spec, pr))
    if not candidates: raise RuntimeError('No fully in-frame target placement for ' + entry['id'])
    # Prefer the closer, larger visible part of the frustum without using a
    # segmentation result, grasp candidate or robot reachability test.
    if entry['layout'] == 'S5_occluded':
        # Leave actual visible table space between the target and head camera
        # for a physical blocker, rather than packing the target at the near FOV edge.
        candidates.sort(key=lambda x: abs(x[1]['depth_m'] - .80))
    else:
        candidates.sort(key=lambda x: -x[1]['area'])
    n = max(1, min(len(candidates), 80))
    order = rng.permutation(n)
    return [candidates[int(i)][0] for i in order]


def select_models(entry, reg, baseline, rng):
    lo, hi = reg['distractor_counts'][entry['layout']]
    count = int(rng.integers(lo, hi + 1)) if hi else 0
    available = [m for m in reg['target_models'] if semantic_class(m) != entry['semantic_class']]
    rng.shuffle(available)
    if entry['layout'] == 'S5_occluded':
        tall = [m for m in available if np.ptp(baseline.corners(baseline.DB[m]), axis=0)[2] >= .075]
        target_height = np.ptp(baseline.corners(baseline.DB[entry['model_id']]), axis=0)[2]
        adequate = [m for m in tall if np.ptp(baseline.corners(baseline.DB[m]), axis=0)[2] >= .8 * target_height]
        if adequate: tall = adequate
        def compactness(m):
            size = np.ptp(baseline.corners(baseline.DB[m]), axis=0)
            return size[2] / np.sqrt(size[0] * size[1])
        tall.sort(key=compactness, reverse=True)
        first = tall[0]
        available.remove(first); return [first, *available[:count-1]]
    return available[:count]


def layout_scene(entry, target, models, baseline, camera, k, rng):
    actors = [deepcopy(target)]; center = np.asarray(target['pose'][:2])
    for i, model in enumerate(models):
        name = f'distractor_{i:02d}'
        placed = None
        for _ in range(1200):
            xy = center + rng.uniform([-.42, -.45], [.42, .45])
            spec = baseline.actor_spec(name, model, xy, float(rng.uniform(-np.pi, np.pi)))
            if nonoverlap(spec, actors, baseline, .05 if entry['layout'] == 'S3_sparse' else .012) and project(spec, baseline, camera, k) is not None:
                placed = spec; break
        if placed is None: raise RuntimeError('Cannot place nonoverlapping distractor')
        actors.append(placed)
    return {'layout': entry['layout'], 'actors': actors, 'perception_camera': 'head'}


def set_scene(env, scene, baseline, sapien, torch):
    env.scene_spec = deepcopy(scene)
    for spec in scene['actors']:
        actor = env.objects[spec['name']]; p = spec['pose']
        actor.set_pose(sapien.Pose(p[:3], p[3:])); actor._bodies[0].linear_velocity = np.zeros(3)
        actor._bodies[0].angular_velocity = np.zeros(3)
    env.agent.reset(env.initial_q)
    baseline.settle(env)


def head_quality(env, baseline):
    from scene_tools import visibility
    # Visibility renders an isolated target internally; snapshot all sensor
    # tensors before that render can reuse the underlying GPU buffers.
    obs = deepcopy(env.get_obs()); quality = visibility(env, obs)
    mask = obs['sensor_data']['head']['segmentation'][0, :, :, 0].cpu().numpy() == env.obj._objs[0].per_scene_id
    depth = obs['sensor_data']['head']['depth'][0, :, :, 0].cpu().numpy() * .001
    valid = mask & np.isfinite(depth) & (depth > .10) & (depth < 2.)
    quality['valid_target_depth_fraction'] = float(valid.sum() / max(1, mask.sum()))
    quality['target_depth_median_m'] = float(np.median(depth[valid])) if valid.any() else None
    quality['visible_pixels'] = int(mask.sum())
    assert quality['visible_pixels'] == quality['target_mask_pixels']
    uv = np.asarray(quality['projected_bbox_uv'])
    quality['border_clear'] = bool((uv[0] >= [12, 12]).all() and (uv[1] <= [628, 468]).all())
    return quality, obs, mask


def accepted(q, layout):
    common = q['border_clear'] and q['visible_pixels'] >= 64 and q['valid_target_depth_fraction'] >= .98
    return bool(common and (.20 <= q['occlusion_fraction'] <= .40 if layout == 'S5_occluded' else q['occlusion_fraction'] <= .15))


def adjust_blocker(env, scene, baseline, camera, k, rng):
    """Place a real opaque distractor along the head camera's line of sight."""
    target = next(a for a in scene['actors'] if a['name'] == 'target')
    lo, hi = baseline.bounds(target); center = (lo + hi) / 2
    direction = camera[:2, 3] - center[:2]
    distance = np.linalg.norm(direction); forward = direction / distance
    side = np.array([-forward[1], forward[0]])
    blocker = scene['actors'][1]; others = [a for a in scene['actors'] if a['name'] != blocker['name']]
    sizes = np.ptp(baseline.corners(baseline.DB[blocker['model_id']]), axis=0)
    yaw = float(np.arctan2(forward[1], forward[0]) + (0 if sizes[0] < sizes[1] else np.pi/2))
    positions = []
    target_projection = project(target, baseline, camera, k)
    for along in [.05, .10, .16, .23, .32, .42]:
        for lateral in np.linspace(-.26, .26, 53):
            point = center[:2] + along * forward + lateral * side
            spec = baseline.actor_spec(blocker['name'], blocker['model_id'], point, yaw)
            if not nonoverlap(spec, others, baseline, .008): continue
            pr = project(spec, baseline, camera, k)
            if pr is None: continue
            overlap = np.maximum(0., np.minimum(pr['hi'], target_projection['hi']) - np.maximum(pr['lo'], target_projection['lo']))
            fraction = float(np.prod(overlap) / target_projection['area'])
            if .08 <= fraction <= .90: positions.append((abs(fraction - .30), spec))
    # Search from one side of the silhouette toward the other. Only the
    # rendered geometric occlusion fraction decides the layout, never SAM3.
    import sapien
    positions.sort(key=lambda item: item[0])
    for _, spec in positions:
        p = spec['pose']; actor = env.objects[spec['name']]
        actor.set_pose(sapien.Pose(p[:3], p[3:])); actor._bodies[0].linear_velocity = np.zeros(3)
        actor._bodies[0].angular_velocity = np.zeros(3)
        q, _, _ = head_quality(env, baseline)
        if accepted(q, 'S5_occluded'):
            scene['actors'][1] = spec
            return True
    return False


def capture(root, entry, env, scene, quality, obs, mask, run_module, profile, convert, maps):
    out = root / 'cases' / entry['id']; out.mkdir()
    camera = out / 'camera'; camera.mkdir()
    data = obs['sensor_data']['head']; rgb = data['rgb'][0].cpu().numpy(); z = data['depth'][0, :, :, 0].cpu().numpy() * .001
    frames = convert(rgb, z, profile, maps)
    Image.fromarray(frames['rgb']).save(camera / 'color.png')
    Image.fromarray(np.rint(frames['depth_m'] / .0001).astype(np.uint16)).save(camera / 'depth.png')
    import cv2
    to_raw, to_rectified = maps
    raw_mask = cv2.remap(mask.astype(np.uint8), to_raw[..., 0], to_raw[..., 1], cv2.INTER_NEAREST)
    model_mask = cv2.remap(raw_mask, to_rectified[..., 0], to_rectified[..., 1], cv2.INTER_NEAREST)
    Image.fromarray(model_mask * 255).save(out / 'target_gt.png')
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / 'target_gt_native.png')
    state = run_module.snapshot(env, obs); save(out / 'state.json', state)
    save(camera / 'params.json', {'depth_scale': .0001, 'intrinsics': state['intrinsics'], 'camera_world_optical': state['camera_world_optical']})
    Image.fromarray(env.render().cpu().numpy()[0]).save(out / 'overview.jpg', quality=88)
    Image.fromarray(frames['rgb']).resize((320, 240)).save(out / 'head_thumb.jpg', quality=85)
    actual = deepcopy(scene)
    for item in actual['actors']: item['pose'] = env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
    bound = {**entry, 'scene': actual, 'admission': quality, 'camera': 'head',
        'robot_initial_q': env.initial_q.tolist(), 'base_pose': env.agent.robot.pose.raw_pose[0].cpu().numpy().tolist(),
        'observation': None, 'physical_grasp_attempts': 0}
    save(out / 'entry.json', bound)
    save(out / 'admission.json', {'passed': True, 'entry': bound, 'camera_input_sha256': {
        f.name: sha(f) for f in camera.iterdir()}, 'ground_truth_mask_is_evaluation_only': True})
    return bound


def galleries(root, entries):
    for layout in LAYOUTS:
        selected = [e for e in entries if e['layout'] == layout]
        sheet = Image.new('RGB', (320 * 5, 264 * 8), 'white'); draw = ImageDraw.Draw(sheet)
        for i, entry in enumerate(selected):
            image = Image.open(root / 'cases' / entry['id'] / 'head_thumb.jpg')
            x, y = i % 5 * 320, i // 5 * 264
            sheet.paste(image, (x, y)); draw.text((x + 5, y + 241), entry['id'].split('_')[0] + ' ' + entry['model_id'], fill='black')
        sheet.save(root / 'gallery' / (layout + '.jpg'), quality=88)


def build(root, continue_draft=False):
    reg = read(root / 'registration.json')
    assert continue_draft or not list((root / 'cases').iterdir()), 'Use an explicit continuation for an existing dataset'
    for file, expected in reg['source_hashes'].items(): assert sha(file) == expected, file
    environment, baseline, run_module, sapien, torch, make, profile, convert, get_maps = runtime()
    maps = get_maps(profile)
    probe = make('014_lemon', {'actors': [baseline.actor_spec('target', '014_lemon', [-.4, 0], 0.)]})
    try:
        probe.reset(seed=0); baseline.settle(probe)
        obs = probe.get_obs(); state = run_module.snapshot(probe, obs)
        camera, k = np.asarray(state['camera_world_optical']), np.asarray(state['intrinsics'])
        assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
        save(root / 'preflight/head_camera.json', state)
    finally: probe.close()
    completed = []; started = time.perf_counter()
    try:
        for index, entry in enumerate(reg['schedule']):
            if (root / 'STOP').exists(): raise RuntimeError('User STOP')
            existing = root / 'cases' / entry['id'] / 'admission.json'
            if continue_draft and existing.exists():
                record = read(existing); assert record['passed']
                assert accepted(record['entry']['admission'], entry['layout'])
                completed.append(record['entry']); continue
            rng = np.random.default_rng(entry['seed'])
            candidates = target_candidates(entry, baseline, camera, k, rng)
            models = select_models(entry, reg, baseline, rng)
            env = None; failures = []; bound = None
            print('BUILD', index + 1, entry['id'], 'target placements', len(candidates), flush=True)
            try:
                for variant, target in enumerate(candidates[:80]):
                    try:
                        scene = layout_scene(entry, target, models, baseline, camera, k, rng)
                        if env is None:
                            env = make(entry['model_id'], scene); env.reset(seed=entry['seed'])
                        set_scene(env, scene, baseline, sapien, torch)
                        if entry['layout'] == 'S5_occluded':
                            if not adjust_blocker(env, scene, baseline, camera, k, rng):
                                raise RuntimeError('No head-view 20-40% occlusion in this layout variant')
                            set_scene(env, scene, baseline, sapien, torch)
                        quality, obs, mask = head_quality(env, baseline)
                        if not accepted(quality, entry['layout']): raise RuntimeError('Head admission: ' + str(quality))
                        for item in scene['actors']:
                            actual_item = {**item, 'pose': env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()}
                            if project(actual_item, baseline, camera, k) is None: raise RuntimeError('Settled actor outside head frame: ' + item['name'])
                        bound = capture(root, entry, env, scene, quality, obs, mask, run_module, profile, convert, maps)
                        save(root / 'cases' / entry['id'] / 'construction_attempts.json', {'accepted_variant': variant, 'rejected': failures})
                        break
                    except (RuntimeError, AssertionError) as exc:
                        if any(token in str(exc).lower() for token in ['cuda', 'memory', 'cache integrity']): raise
                        failures.append({'variant': variant, 'reason': str(exc)})
                if bound is None: raise RuntimeError('No admitted layout: ' + entry['id'] + ' ' + str(failures[-2:]))
            finally:
                if env is not None: env.close()
            completed.append(bound)
            save(root / 'progress.json', {'phase': 'building', 'completed': len(completed), 'registered': 200,
                'by_layout': dict(Counter(e['layout'] for e in completed)), 'last_case': entry['id'],
                'elapsed_seconds': time.perf_counter() - started, 'physical_grasps': 0, 'at': now()})
            print('ACCEPTED', entry['id'], quality['visible_pixels'], quality['occlusion_fraction'], flush=True)
        galleries(root, completed)
        save(root / 'manifest.json', {'registration': str(root / 'registration.json'), 'registration_sha256': sha(root / 'registration.json'),
            'schedule': completed, 'camera': 'head', 'physical_grasps': 0, 'completed': 200})
        save(root / 'progress.json', {'phase': 'geometry_complete', 'completed': 200, 'registered': 200,
            'by_layout': dict(Counter(e['layout'] for e in completed)), 'physical_grasps': 0, 'at': now()})
    except BaseException:
        save(root / 'halt.json', {'at': now(), 'error': traceback.format_exc(), 'completed': len(completed)})
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'build'])
    p.add_argument('--root', type=Path, default=DEFAULT); p.add_argument('--continue-draft', action='store_true'); args = p.parse_args()
    if args.stage == 'prepare': prepare(args.root.resolve())
    else: build(args.root.resolve(), args.continue_draft)


if __name__ == '__main__': main()
