"""Run the fresh 40-class/80-model, 200-scene head-camera grasp benchmark."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import inspect
from pathlib import Path
import shutil
import sys
import types

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEGACY = PROJECT / 'experiments/rayst3r_graspgenx'
sys.path.insert(0, str(LEGACY))
from common import MS, WORKSPACE, GPUS, paths, read, save, sha, now
import run_suite as executor

DEFAULT = HERE / 'runs/diverse200_fixed452_20261122_01'


def prepare(root):
    sys.path.insert(0,str(PROJECT/'experiments/current_configuration'))
    from configuration import load_current
    current=load_current(verify_dataset=True)
    assert read(PROJECT/'configs/scene_dataset.json')['grasp_tests_authorized'], 'New grasp batch requires current authorization'
    paths()
    from graspgenx_partial_geometry import gripper_descriptor
    assert not root.exists(), 'Use a new registration directory'
    dataset_config = read(PROJECT / 'configs/scene_dataset.json')
    dataset = Path(dataset_config['dataset'])
    verified = read(dataset / 'verification.json')
    assert verified['passed'] and verified['scenes'] == 200 and verified['categories'] == 40 and verified['models'] == 80
    manifest = read(dataset / 'manifest.json'); catalog = load_object_catalog(dataset / 'object_catalog.json')
    assert manifest['registration_sha256'] == sha(dataset / 'registration.json')
    assert len(manifest['schedule']) == 200 and len(catalog['models']) == 80
    root.mkdir(parents=True)
    for name in ['entries', 'claims', 'trials', 'source', 'resident', 'preflight']: (root / name).mkdir()
    shutil.copy2(dataset / 'object_catalog.json', root / 'object_catalog.json')
    first=manifest['schedule'][0]
    save(root / 'common_initial_conditions.json',{'base_pose':first['base_pose'],
        'initial_q':first['robot_initial_q'],'source':str(dataset/'manifest.json')})
    save(root / 'gripper_descriptor.json', gripper_descriptor())
    entries, files = [], []
    for row in manifest['schedule']:
        entry = deepcopy(row)
        assert not entry.get('observation'), 'Catalog scenes must use the head at the original initial state'
        entry.pop('observation', None)
        entry.update(campaign_registration=str(root / 'registration.json'),
            experiment_arm='head_rayst3r_graspgenx_diverse200', pipeline_profile='rayst3r_graspgenx',
            source_scene_dataset=str(dataset), source_scene_entry=str(dataset / 'cases' / row['id'] / 'entry.json'))
        save(root / 'entries' / (entry['id'] + '.json'), entry)
        files.append(root / 'entries' / (entry['id'] + '.json'))
        files += [dataset / 'cases' / entry['id'] / name for name in ['entry.json', 'admission.json', 'state.json']]
        entries.append(entry)
    assert Counter(e['layout'] for e in entries) == {layout: 40 for layout in ['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter', 'S5_occluded']}
    assert all(n == 5 for n in Counter(e['semantic_class'] for e in entries).values())
    files += list(HERE.glob('*.py')) + list(LEGACY.glob('*.py'))
    files += [PROJECT / 'configs/head_camera_mount.json', PROJECT / 'configs/camera_profiles.json',
        root / 'gripper_descriptor.json', root / 'object_catalog.json', root / 'common_initial_conditions.json',
        dataset / 'manifest.json', dataset / 'registration.json', dataset / 'verification.json']
    files += [PROJECT / 'experiments/diverse_object_scenes' / n for n in ['pipeline.py', 'asset_environment.py', 'asset_catalog.py']]
    files.append(PROJECT / 'experiments/head_perception_scenes/scene_builder.py')
    files += [MS / 'arx_pipeline' / n for n in ['run.py', 'environment.py', 'baseline.py', 'scene_tools.py', 'planner.py']]
    files += [MS / n for n in ['graspgenx_partial_geometry.py', 'resident_planner.py', 'fast_resident_planner.py',
        'checked_native_services.py', 'short_lift_protocol.py', 'short_lift_planner.py', 'postprocess_planner.py',
        'table_truncation_model.py', 'resident_backend.py', 'arx_runtime_imports.py', 'tmp/physical_table_pair_20261118_01.py']]
    files += [PROJECT / 'experiments/pointcloud_benchmark' / n for n in ['model_adapters.py', 'camera_imaging.py', 'head_camera_config.py']]
    for folder in [PROJECT / 'third_party/rayst3r', PROJECT / 'third_party/dinov2/dinov2',
        WORKSPACE / 'research_dependencies/GraspGenX/graspgenx', WORKSPACE / 'claro/依赖/SAM3源码/sam3']:
        files += list(folder.rglob('*.py'))
    for model in ['model', 'model_closed']:
        files += [WORKSPACE / 'arx_core/scripts/curobo_arx/static_pick/reference' / model / n for n in ['arx.yml', 'arx.urdf']]
    files += [WORKSPACE / 'arx_core/arx-sim/config' / n for n in ['controllers.yaml', 'runtime.yaml']]
    files += [MS / 'assets/arx/robot.urdf', MS / 'assets/arx/manifest.json']
    sources = {str(p.resolve()): sha(p) for p in set(files)}
    sources.update(read(root / 'gripper_descriptor.json')['source_sha256'])
    assets = {}
    for model in catalog['models'].values():
        for file, digest in model['source_hashes'].items():
            assert sha(file) == digest, 'Catalog asset changed: ' + file
            assets[file] = digest
    weights = [WORKSPACE / 'claro/权重/sam3.pt', PROJECT / 'weights/rayst3r/rayst3r.pth',
        PROJECT / 'weights/dinov2/dinov2_vitl14_reg4_pretrain.pth',
        WORKSPACE / 'research_dependencies/graspgenx_checkpoints/release/gen/epoch_736.pth',
        WORKSPACE / 'research_dependencies/graspgenx_checkpoints/release/dis/epoch_1056.pth']
    reg = {'at': now(), 'protocol': 'diverse200_head_sam3_rayst3r_graspgenx_fixed452_v1',
        'schedule': entries, 'total_slots': 200, 'physical_slot_budget': 200,
        'automatic_retries': 0, 'old_results_carried': 0, 'categories': 40, 'models': 80,
        'scene_dataset': str(dataset), 'scene_config': dataset_config,
        'source_hashes': sources, 'asset_sha256': assets,
        'model_files': {str(p): {'sha256': sha(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in weights},
        'common_initial_conditions': str(root / 'common_initial_conditions.json'),
        'selected_lift_m': .452, 'planned_lift_m': .03, 'success': {
            'bilateral_target_contact':True,'hold_seconds':3,'joint_tolerance_rad':.005,
            'forbidden_contacts':0,'min_continuous_hold_lift_m':.01,'minimum_lift_m_inclusive':.01,
            'minimum_height_comparison':'>=','planned_eef_world_z_lift_m':.03},
        'required_components': ['sam3', 'rayst3r', 'graspgenx', 'curobo'], 'gpus': GPUS,
        'planner_warmup_trial': str(root/'preflight/head_warmup'), 'camera': 'head',
        'environment_id': 'ARXPickCatalog-v1', 'robot_and_scene_physics': 'unchanged from verified current dataset',
        'segmentation': {'threshold': .5, 'selection': 'highest SAM3 score for manifest target text; no GT matching'},
        'rayst3r': {'unchanged_benchmark_adapter': True, 'n_pred_views_parameter': 5, 'actual_novel_views': 21},
        'graspgenx': {'planner': 'graspmoe', 'diffusion_samples': 512, 'moe_obb_density': 'dense',
            'score_threshold': -1., 'topk_cap': None, 'posthoc_topdown_filter': False, 'pose_refinement': False},
        'execution': {'fixed_lift_joint': True, 'tcp_eef_z_m': .043, 'pregrasp_m': .08,
            'nominal_closed_precheck': False, 'extra_table_margin_m': 0., 'measured_closed_lift_check': True,
            'graspcl': False, 'mpc': False, 'reachability_map': False, 'wrist_preparation_motion': False},
        'fresh_observation_each_scene': True, 'cached_dataset_rgb_or_GT_as_model_input': False,
        'ground_truth_mask': 'evaluation only; original collision assets are retained for cuRobo scene checks',
        'runtime_adaptations': ['catalog environment factory', 'head recording', '200-slot log labels', 'separate planning-stage accounting'],
        'user_authorization': '2026-11-22: scenes are ready; begin testing the current diverse-object dataset'}
    save(root / 'registration.json', reg)
    for p in HERE.glob('*.py'): shutil.copy2(p, root / 'source' / p.name)
    save(root / 'preflight/registration.json', {'passed': True, 'scenes': 200, 'categories': 40,
        'models': 80, 'layouts': dict(Counter(e['layout'] for e in entries)), 'physical_grasps': 0,
        'observation_None_removed': True, 'dataset_verification_passed': True})
    aggregate(root, 'registered')
    print('REGISTERED', root, flush=True)


def aggregate(root, phase):
    reg = read(root / 'registration.json'); rows, pending = [], []
    for p in sorted((root / 'claims').glob('*.json')):
        claim = read(p)
        if claim['state'] != 'complete': pending.append(p.stem); continue
        trial = root / 'trials' / p.stem
        assert sha(trial / 'result.json') == claim['result_sha256']
        row = read(trial / 'result.json'); row['grasp_planning_entered'] = (trial / 'plan_result.json').exists()
        rows.append(row)
    n = len(reg['schedule']); entered = sum(r['grasp_planning_entered'] for r in rows)
    planned = sum(bool(r['planning_success']) for r in rows); success = sum(bool(r['success']) for r in rows)
    summary = {'phase': phase, 'registered': n, 'completed': len(rows), 'not_executed': n-len(rows),
        'claimed_without_terminal_receipt': pending, 'grasp_planning_entered': entered,
        'before_grasp_planning_failures': sum(not r['grasp_planning_entered'] for r in rows),
        'planning_passes': planned, 'planning_rejections': entered-planned,
        'physical_attempts': sum(bool(r['physical_attempt']) for r in rows), 'successes': success,
        'failure_reasons': dict(Counter(r['reason'] for r in rows if not r['success'])),
        'planning_rate_among_entered': planned/entered if entered else None,
        'planning_rate_registered': planned/n, 'success_rate_registered': success/n,
        'success_rate_completed': success/len(rows) if rows else None, 'at': now()}
    summary['by_layout'] = {layout: {'completed': sum(r['layout'] == layout for r in rows),
        'planning_passes': sum(r['layout'] == layout and bool(r['planning_success']) for r in rows),
        'successes': sum(r['layout'] == layout and bool(r['success']) for r in rows)} for layout in reg['scene_config'].get('layouts', ['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter', 'S5_occluded'])}
    save(root / 'progress.json', summary)
    return summary


def report(root, s):
    lines = ['# 多类别200场景抓取测试', '',
        '40类、80型号，五种布局各40例。固定45.2厘米，头部RGB-D、SAM3、RaySt3R、GraspGenX及cuRobo六轴执行。', '',
        '| 指标 | 数量 |', '| --- | ---: |', f"| 完成 | {s['completed']}/200 |",
        f"| 尚未进入抓取规划就失败 | {s['before_grasp_planning_failures']} |",
        f"| 进入抓取规划 | {s['grasp_planning_entered']} |", f"| 规划通过 | {s['planning_passes']} |",
        f"| 规划拒绝 | {s['planning_rejections']} |", f"| 实际成功 | {s['successes']}/200 |",
        f"| 未执行或尚无完整回执 | {s['not_executed']} |", '',
        '成功要求目标实际抬升至少1厘米并保持3秒，同时双指接触、收敛与禁止接触检查达标。', '',
        '| 布局 | 完成 | 规划通过 | 抓取成功 |', '| --- | ---: | ---: | ---: |']
    for layout, row in s['by_layout'].items(): lines.append(f"| {layout} | {row['completed']}/40 | {row['planning_passes']} | {row['successes']} |")
    lines += ['', '| 失败原因 | 数量 |', '| --- | ---: |']
    lines += [f'| {reason} | {count} |' for reason, count in s['failure_reasons'].items()]
    lines += ['', '当前状态：' + s['phase'], '', '数据目录：`' + str(root) + '`。']
    (PROJECT / 'docs/diverse_grasp_benchmark/抓取测试结果.md').write_text('\n'.join(lines) + '\n')


def run(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location('simulation', HERE / 'catalog_simulation.py')
    module = importlib.util.module_from_spec(spec); sys.modules['simulation'] = module; spec.loader.exec_module(module)
    executor.aggregate, executor.report = aggregate, report
    source = inspect.getsource(executor.run)
    assert source.count("'/160'") == 1
    source = source.replace("'/160'", "'/200'")
    scope = dict(executor.__dict__); exec(compile(source, str(__file__) + ':catalog_executor', 'exec'), scope)
    adapted = types.FunctionType(scope['run'].__code__, executor.__dict__, 'run')
    adapted(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, default=DEFAULT); args = p.parse_args(); root = args.root.resolve()
    if args.stage == 'prepare': prepare(root)
    else:
        with (root / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run(root)
