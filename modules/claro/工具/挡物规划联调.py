"""用第 02 步的挡物定位结果进行离线 ZeroGrasp 和 cuRobo 联调。"""

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

from 定位挡物 import points_in_base, read, sha256


ROOT = Path(__file__).resolve().parents[1]
MS = ROOT.parent / 'maniskill'
WARMUP = MS / 'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
POLICY = MS / 'tmp/physical_table_pair_20261118_01.py'


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def instance_binding(xyz, state):
    """离线记录没有完整实例图，按同帧网格核对挡物的场景编号。"""
    import trimesh

    if len(xyz) < 1024:
        raise ValueError('挡物有效点不足以做本次离线实例绑定')
    base = np.asarray(state['links']['body4'])
    world = xyz[np.linspace(0, len(xyz) - 1, 1024, dtype=int)] @ base[:3, :3].T + base[:3, 3]
    names, distances = [], []
    for name, actor in state['scene_meshes'].items():
        asset = Path(actor['file_path'])
        if sha256(asset) != actor['file_sha256']:
            raise ValueError('历史碰撞网格与记录不符')
        t = np.asarray(actor['world'])
        local = (world - t[:3, 3]) @ t[:3, :3]
        mesh = trimesh.load_mesh(asset, process=False)
        _, dist, _ = trimesh.proximity.closest_point(mesh, local)
        names.append(name)
        distances.append(dist)
    distances = np.asarray(distances)
    median = np.median(distances, axis=1)
    best, runner_up = np.argsort(median)[:2]
    fraction = float(np.mean(np.argmin(distances, axis=0) == best))
    actor = names[best]
    if (state['scene_meshes'][actor]['target'] or fraction < .90 or median[best] > .010
            or median[runner_up] - median[best] < .025):
        raise ValueError('无法唯一绑定一个非原目标的场景实例')
    return actor, {'sample_count': len(world), 'nearest_share': fraction,
                   'median_surface_distance_m': dict(zip(names, median.tolist()))}


def prepare(trial, localization, selection, output):
    original = read(trial / 'state.json')
    locator = read(localization / '定位记录.json')
    vlm = read(selection)
    scene = trial.name.split('_')[0]
    if scene != locator['scene'] or scene != vlm['scene'] or vlm['decision']['tool'] != 'locate_object':
        raise ValueError('挡物定位与待联调场景不一致')
    if locator['object_name'] != vlm['decision']['object_name'] or locator['base_frame'] != 'body4':
        raise ValueError('选挡记录的实例或点云坐标系不一致')
    if read(trial / 'plan_result.json')['success']:
        raise ValueError('直接抓取可规划，跳过移障')
    camera = trial / 'camera'
    originals = [camera / p for p in ('color.png', 'depth.png', 'params.json')]
    sources = [*originals, trial / 'state.json', selection]
    for file in sources:
        if sha256(file) != locator['source_sha256'].get(file.name):
            raise ValueError(f'原定位记录与 RGB-D、标定或选择回执不一致：{file.name}')
    mask_file = localization / '挡物掩码.png'
    mask = np.asarray(Image.open(mask_file)) > 0
    x, y = locator['point_xy']
    if mask.ndim != 2 or not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]):
        raise ValueError('定位点不在挡物掩码内')
    rgb = np.asarray(Image.open(originals[0]).convert('RGB'))
    depth = np.asarray(Image.open(originals[1]))
    xyz, _, uv, _ = points_in_base(depth, mask, rgb, read(originals[2]), original)
    cloud_file = localization / '挡物点云.npz'
    with np.load(cloud_file, allow_pickle=False) as cloud:
        if (not np.array_equal(cloud['uv'], uv) or not np.allclose(cloud['xyz_base_m'], xyz, atol=1e-8, rtol=0)
                or len(xyz) != locator['depth_points']):
            raise ValueError('挡物点云无法由同帧深度和掩码复现')
    actor, diagnostics = instance_binding(xyz, original)
    old_target, = [name for name, item in original['scene_meshes'].items() if item['target']]
    state = deepcopy(original)
    for name, item in state['scene_meshes'].items():
        item['target'] = name == actor
    state['object_world'] = state['scene_meshes'][actor]['world']
    base = np.asarray(state['links']['body4'])
    state['object_center'] = (xyz.mean(0) @ base[:3, :3].T + base[:3, 3]).tolist()

    output.mkdir(parents=True, exist_ok=False)
    (output / 'camera').mkdir()
    for file in originals:
        shutil.copy2(file, output / 'camera' / file.name)
    Image.fromarray(mask.astype(np.uint8)).save(output / 'camera/mask.png')
    save(output / 'state.json', state)
    save(output / '移障输入登记.json', {
        'scene': scene, 'source_trial': str(trial), 'selected_object': locator['object_name'],
        'selected_scene_actor': actor, 'model_id': state['scene_meshes'][actor]['model_id'],
        'original_grasp_target': old_target, 'instance_binding': diagnostics,
        'binding_method': '仅用同帧已保存碰撞网格绑定执行实例。挡物候选仍来自 VLM 和 SAM3。',
        'source_sha256': {str(p): sha256(p) for p in [*sources, mask_file, cloud_file, localization / '定位记录.json']},
        'planner_input_sha256': {str(p.relative_to(output)): sha256(p) for p in [output / 'state.json', *(output / 'camera').iterdir()]},
        'collision_actor_poses_unchanged': True, 'robot_motion_commands': 0,
    })
    print(scene, actor, state['scene_meshes'][actor]['model_id'])


def validate_input(directory):
    reg = read(directory / '移障输入登记.json')
    for path, digest in reg['planner_input_sha256'].items():
        if sha256(directory / path) != digest:
            raise ValueError('已登记的联调输入发生变化')
    for path, digest in reg['source_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('第 02 步原始记录发生变化')
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('运行模型前须绑定一个可用物理 GPU 的 UUID')
    return reg


def infer(directories):
    from postprocess_backend import NativePostNeural

    for directory in directories:
        validate_input(directory)
    warmup = directories[0] / '推理预热'
    warmup.mkdir(exist_ok=False)
    worker = NativePostNeural(warmup, WARMUP)
    for directory in directories:
        result = worker.request(directory)
        manifest = read(directory / 'network_candidates.json')
        manifest['mask'] = 'CLARO 第 02 步 SAM3 所选挡物掩码'
        save(directory / 'network_candidates.json', manifest)
        save(directory / '挡物推理记录.json', {**result, 'model_load_count': worker.loads,
             'warmup_seconds': worker.warmup_seconds, 'shared_warmup_directory': str(warmup),
             'robot_motion_commands': 0})
        print(directory.name, result)


def plan(directories):
    import resident_planner
    from checked_native_services import checked_pool
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    import table_truncation_model as truncation

    registrations = [validate_input(directory) for directory in directories]
    for directory in directories:
        if not read(directory / '挡物推理记录.json')['success']:
            raise ValueError('挡物推理未完成')
    spec = importlib.util.spec_from_file_location('已登记的实际桌面策略', POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.install_collision_policy()
    resident_planner.GraspCLPlanner = ShortLiftPlanner
    adapter = truncation.make_pool(checked_pool(FastPlannerPool))
    warm_start = time.perf_counter()
    warm_dir = directories[0] / '规划预热'
    worker = adapter(warm_dir, WARMUP)
    warmup = time.perf_counter() - warm_start
    for directory, reg in zip(directories, registrations):
        plan_one(directory, reg, worker, warmup, warm_dir)


def plan_one(directory, reg, worker, warmup, warm_dir):
    from native_pair_timing import native_candidates
    from short_lift_protocol import validate_selected
    import torch

    pool = native_candidates(directory)
    refined = worker.request(directory, refine=True, orientation_mode='fixed_orientation')
    result = worker.request(directory, selector='graspcl')
    model = read(directory / 'plan/scene.json') if (directory / 'plan/scene.json').exists() else {}
    if model and reg['original_grasp_target'] not in model.get('mesh', {}):
        raise ValueError('原任务目标从碰撞世界中消失')
    if model and reg['selected_scene_actor'] in model.get('mesh', {}):
        raise ValueError('挡物未按临时抓取目标处理')
    transit = None
    if result.get('success'):
        validate_selected(result['selected'])
        state = read(directory / 'state.json')
        try:
            context = worker.bind(directory / '进场全场景复核', state, include_target=True)
        except AssertionError as error:
            if str(error) != 'Initial ARX state collides in cuRobo model':
                raise
            transit = {'passed': False, 'reason': str(error), 'points_checked': 0,
                       'include_selected_blocker': True}
        else:
            path = torch.as_tensor(result['paths']['pregrasp']['q'], device='cuda', dtype=torch.float32)
            fraction = torch.arange(5, device='cuda') / 5
            dense = torch.cat([(path[:-1, None] * (1 - fraction[None, :, None])
                                + path[1:, None] * fraction[None, :, None]).reshape(-1, 6), path[-1:]])
            passed = all(bool(context.checker.validate(batch[:, None]).all()) for batch in dense.split(64))
            transit = {'passed': passed, 'points_checked': len(dense), 'include_selected_blocker': True}
    save(directory / '挡物规划联调结果.json', {
        'scene': reg['scene'], 'selected_scene_actor': reg['selected_scene_actor'], 'model_id': reg['model_id'],
        'native_candidates': pool['counts'], 'refinement': refined, 'planning_passed': bool(result.get('success')),
        'planning_reason': result.get('reason'), 'rejected': result.get('rejected'), 'transit_all_scene_check': transit,
        'planner_load_count': worker.loads, 'planner_warmup_seconds': warmup,
        'shared_warmup_directory': str(warm_dir), 'plan_request_seconds': result.get('wall_seconds'),
        'robot_motion_commands': 0,
    })
    print(read(directory / '挡物规划联调结果.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'infer', 'plan'))
    parser.add_argument('--output', required=True, action='append', type=Path)
    parser.add_argument('--trial', type=Path)
    parser.add_argument('--localization', type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    outputs = [path.resolve() for path in args.output]
    if len(set(outputs)) != len(outputs):
        parser.error('同一输出目录只能出现一次')
    if args.phase == 'prepare':
        if not all((args.trial, args.localization, args.selection)) or len(outputs) != 1:
            parser.error('prepare 需要 --trial、--localization、--selection 和一个 --output')
        prepare(args.trial.resolve(), args.localization.resolve(), args.selection.resolve(), outputs[0])
        return
    sys.path.insert(0, str(MS))
    (infer if args.phase == 'infer' else plan)(outputs)


if __name__ == '__main__':
    main()
