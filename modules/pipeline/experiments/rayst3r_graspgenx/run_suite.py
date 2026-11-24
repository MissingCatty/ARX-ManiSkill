"""Register and execute 160 one-shot head-camera grasps at fixed 45.2 cm."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from common import HERE, PROJECT, WORKSPACE, MS, PRIOR, DEFAULT, GPUS, PROTOCOL
from common import paths, read, save, sha, now, check_sources, make_candidates, aggregate


def prepare(root):
    raise RuntimeError('The old 160-scene registration is retired; use experiments/repaired_grasp_benchmark/run_repaired_benchmark.py with the current configuration')
    paths()
    from graspgenx_partial_geometry import gripper_descriptor
    assert not root.exists(), 'New registration requires a new run directory'
    previous = read(PRIOR / 'registration.json')
    schedule = deepcopy(previous['schedule'])
    assert len(schedule) == 160
    assert Counter(e['layout'] for e in schedule) == dict.fromkeys(['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter'], 40)
    root.mkdir(parents=True)
    for folder in ['entries', 'claims', 'trials', 'source', 'resident', 'preflight']: (root / folder).mkdir()
    save(root / 'gripper_descriptor.json', gripper_descriptor())
    initial = PRIOR / 'reference/common_initial_conditions.json'
    save(root / 'common_initial_conditions.json', read(initial))
    source_files = [*HERE.glob('*.py'), PROJECT / 'configs/head_camera_mount.json',
        PROJECT / 'configs/camera_profiles.json', root / 'gripper_descriptor.json',
        root / 'common_initial_conditions.json', PRIOR / 'registration.json',
        MS / 'graspgenx_partial_geometry.py', MS / 'assets/arx/manifest.json', MS / 'assets/arx/robot.urdf']
    for name in ['run.py', 'environment.py', 'baseline.py', 'scene_tools.py', 'planner.py']:
        source_files.append(MS / 'arx_pipeline' / name)
    for name in ['resident_planner.py', 'fast_resident_planner.py', 'checked_native_services.py',
                 'short_lift_protocol.py', 'short_lift_planner.py', 'postprocess_planner.py',
                 'table_truncation_model.py', 'resident_backend.py', 'arx_runtime_imports.py',
                 'tmp/physical_table_pair_20261118_01.py']:
        source_files.append(MS / name)
    source_files += list((PROJECT / 'third_party/rayst3r').rglob('*.py'))
    source_files += list((WORKSPACE / 'research_dependencies/GraspGenX/graspgenx').rglob('*.py'))
    source_files += list((MS / 'assets/arx').rglob('*.ply'))
    source_files += list((MS / 'assets/arx').rglob('*.stl'))
    for name in ['model', 'model_closed']:
        model = WORKSPACE / 'arx_core/scripts/curobo_arx/static_pick/reference' / name
        source_files += [model / 'arx.yml', model / 'arx.urdf']
    source_files += [WORKSPACE / 'arx_core/arx-sim/config/controllers.yaml',
                     WORKSPACE / 'arx_core/arx-sim/config/runtime.yaml']
    for name in ['model_adapters.py', 'camera_imaging.py', 'head_camera_config.py']:
        source_files.append(PROJECT / 'experiments/pointcloud_benchmark' / name)
    entries = []
    for row in schedule:
        entry = read(PRIOR / 'entries' / (row['id'] + '.json'))
        source_files.append(PRIOR / 'entries' / (row['id'] + '.json'))
        text = re.sub(r'^\d+(?:-[a-z]|[a-z])?_', '', entry['model_id']).replace('_', ' ')
        entry.update(campaign_registration=str(root / 'registration.json'), target_text=text,
            experiment_arm='head_rayst3r_graspgenx', pipeline_profile='rayst3r_graspgenx',
            camera='head', seed=int(entry['seed']))
        save(root / 'entries' / (entry['id'] + '.json'), entry)
        source_files.append(root / 'entries' / (entry['id'] + '.json'))
        entries.append(entry)
    descriptor = read(root / 'gripper_descriptor.json')
    sources = {str(p.resolve()): sha(p) for p in source_files}
    sources.update(descriptor['source_sha256'])
    checkpoints = [WORKSPACE / 'claro/权重/sam3.pt', PROJECT / 'weights/rayst3r/rayst3r.pth',
        PROJECT / 'weights/dinov2/dinov2_vitl14_reg4_pretrain.pth',
        WORKSPACE / 'research_dependencies/graspgenx_checkpoints/release/gen/epoch_736.pth',
        WORKSPACE / 'research_dependencies/graspgenx_checkpoints/release/dis/epoch_1056.pth']
    weights = {str(p): {'sha256': sha(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in checkpoints}
    reg = {'created_at': now(), 'protocol': PROTOCOL, 'schedule': entries, 'total_slots': 160,
        'physical_slot_budget': 160, 'automatic_retries': 0, 'old_results_carried': 0,
        'source_campaign': str(PRIOR), 'source_registration_sha256': sha(PRIOR / 'registration.json'),
        'source_hashes': sources, 'model_files': weights, 'asset_sha256': previous['asset_sha256'],
        'common_initial_conditions': str(root / 'common_initial_conditions.json'),
        'selected_lift_m': .452, 'planned_lift_m': .03, 'success': previous['success'],
        'required_components': ['sam3', 'rayst3r', 'graspgenx', 'curobo'], 'gpus': GPUS,
        'planner_warmup_trial': previous['warmup_trial'], 'camera': 'head', 'downward_pitch_deg': 45.,
        'segmentation': {'threshold': .5, 'prompt': 'registered YCB target name', 'selection': 'highest model score'},
        'rayst3r': {'unchanged_benchmark_adapter': True, 'n_pred_views_parameter': 5, 'actual_novel_views': 21},
        'graspgenx': {'planner': 'graspmoe', 'diffusion_samples': 512, 'moe_obb_density': 'dense',
            'score_threshold': -1., 'topk_cap': None, 'posthoc_topdown_filter': False,
            'pose_refinement': False, 'input_voxel_m': .002, 'input_max_points': 3500,
            'actual_arx_gripper_conditioning': descriptor},
        'execution': {'fixed_lift_joint': True, 'tcp_eef_z_m': .043, 'pregrasp_m': .08,
            'nominal_closed_precheck': False, 'extra_table_margin_m': 0.,
            'measured_closed_lift_check': True, 'graspcl': False, 'mpc': False, 'reachability_map': False,
            'historical_observation_arm_motion_retained': True, 'legacy_wrist_visibility_gate_used': False},
        'ground_truth': {'mask': 'evaluation_only', 'target_mesh_not_input_to_models': True,
            'planning_scene': 'same measured actor collision meshes as fixed452 baseline'},
        'resource_fault_policy': 'stop scheduling; preserve claims, receipts and STOP; no automatic replay',
        'user_authorization': '2026-11-22: RaySt3R -> GraspGenX success test at fixed45.2cm, head camera, original160 non-occluded scenes'}
    save(root / 'registration.json', reg)
    for p in HERE.glob('*.py'): shutil.copy2(p, root / 'source' / p.name)
    save(root / 'preflight/registration.json', {'passed': True, 'scene_count': len(entries),
        'layouts': dict(Counter(e['layout'] for e in entries)), 'physical_grasps': 0,
        'old_results_carried': 0, 'no_zero_grasp_checkpoint': True})
    aggregate(root, 'registered')
    print('REGISTERED', root, '160 scenes', flush=True)


class Services:
    def __init__(self, root, reg):
        self.root, self.reg, self.children, self.streams = root, reg, {}, []
        key = sha(root / 'registration.json')[:10]
        from arx_project.layout import ROOT
        self.socket_root = ROOT / 'tmp/rpc' / key
        self.socket_root.parent.mkdir(parents=True,exist_ok=True)
        self.socket_root.mkdir(mode=0o700, exist_ok=False)
        self.sockets = {kind: self.socket_root / (kind + '.sock') for kind in reg['required_components']}

    def start(self):
        from arx_service_readiness import wait_ready
        from arx_project.layout import ROOT, module_dir
        from arx_project.registry import module_records
        records=module_records()
        python={kind:ROOT/records[kind]['environment']/'bin/python' for kind in self.reg['required_components']}
        for kind in self.reg['required_components']:
            warmup = self.reg['planner_warmup_trial'] if kind == 'curobo' else str(self.root / 'preflight/head_warmup')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[kind], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                ARX_SIM_GPU=GPUS['curobo'], ARX_PLANNER_GPU=GPUS['curobo'], LILO_GPU=GPUS['curobo'],
                GRASP_TRAJOPT_POLICY='interactive', GRASPGENX_CHECKPOINT_DIR=str(WORKSPACE / 'research_dependencies/graspgenx_checkpoints'),
                GRASPGENX_GRIPPER_CFG_DIR=str(WORKSPACE / 'research_dependencies/GraspGenX/ext/gripper_descriptions'))
            env.pop('PYTHONPATH', None)
            cmd = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'ray-ggx-service',
                str(module_dir('arx_core') / 'scripts/curobo_arx/environment.sh'), str(python[kind]),
                '-m', 'arx_project', 'serve', '--kind', kind, '--root', str(self.root / 'resident'),
                '--socket', str(self.sockets[kind]), '--warmup-trial', warmup,
                '--descriptor', str(self.root / 'gripper_descriptor.json')]
            stream = (self.root / 'resident' / (kind + '.log')).open('x'); self.streams.append(stream)
            self.children[kind] = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            health = wait_ready(self.sockets[kind], self.root / 'resident' / (kind + '_ready.json'), self.children[kind], kind, timeout=1800)
            print('READY', kind, health['pid'], flush=True)
            aggregate(self.root, 'ready_' + kind)

    def request(self, kind, directory, **flags):
        from resident_backend import rpc
        if self.children[kind].poll() is not None: raise RuntimeError(kind + ' exited')
        with (self.root / 'requests.jsonl').open('a') as stream:
            stream.write(__import__('json').dumps({'event': 'request', 'at': now(), 'kind': kind,
                'directory': str(directory), 'flags': flags}) + '\n')
        response = rpc(self.sockets[kind], {'directory': str(directory), **flags}, timeout=900)
        with (self.root / 'requests.jsonl').open('a') as stream:
            stream.write(__import__('json').dumps({'event': 'response', 'at': now(), 'kind': kind,
                'directory': str(directory), 'flags': flags, 'response': response}) + '\n')
        if response['status'] != 'ok': raise RuntimeError(response.get('error', str(response)))
        return response

    def stop(self):
        from resident_backend import rpc
        for kind, child in reversed(list(self.children.items())):
            if child.poll() is None:
                try: rpc(self.sockets[kind], {'command': 'shutdown'}, timeout=10)
                except Exception: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in self.streams: stream.close()
        save(self.root / 'services_stopped.json', {'at': now(),
            'all_owned_exited': all(p.poll() is not None for p in self.children.values()),
            'returncodes': {k: p.returncode for k, p in self.children.items()}})


def report(root, summary):
    lines = ['# RaySt3R＋GraspGenX固定45.2厘米抓取结果', '',
        '使用原160个非遮挡场景。头部向下相机、SAM3预测掩码、RaySt3R补全、GraspGenX原生生成与评分、cuRobo六轴规划。', '',
        '| 指标 | 数量／登记 | 比例 |', '| --- | ---: | ---: |',
        f"| 已完成 | {summary['completed']}/160 | {summary['completed']/160:.2%} |",
        f"| 初始规划通过 | {summary['planning_passes']}/160 | {summary['planning_rate_registered']:.2%} |",
        f"| 实际抓取成功 | {summary['successes']}/160 | {summary['success_rate_registered']:.2%} |",
        f"| 尚未执行或未形成完整回执 | {summary['not_executed']}/160 | {summary['not_executed']/160:.2%} |", '',
        '成功要求物体抬高至少1厘米并保持3秒，同时双指接触、执行收敛、禁止接触均达标。', '',
        '| 失败原因 | 数量 |', '| --- | ---: |']
    lines += [f'| {name} | {n} |' for name, n in summary['failure_reasons'].items()]
    lines += ['', '状态：' + summary['phase'] + '。', '', '原始运行目录：`' + str(root) + '`。']
    doc = PROJECT / 'docs/rayst3r_graspgenx/固定高度抓取结果.md'
    doc.write_text('\n'.join(lines) + '\n')


def run(root):
    paths()
    reg = read(root / 'registration.json'); check_sources(reg)
    assert not list((root / 'claims').glob('*.json')) and not (root / 'STOP').exists()
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in raw.strip().splitlines()}
    floors = {GPUS['rayst3r']: 20000, GPUS['sam3']: 14000, GPUS['curobo']: 12000}
    save(root / 'preflight/resources.json', {'at': now(), 'free_mib': free, 'minimum_free_mib': floors})
    assert all(free.get(gpu, 0) >= minimum for gpu, minimum in floors.items()), 'Insufficient free GPU memory'
    for name, evidence in reg['model_files'].items():
        stat = Path(name).stat()
        assert stat.st_size == evidence['size'] and stat.st_mtime_ns == evidence['mtime_ns'], name
    for name, digest in reg['asset_sha256'].items():
        assert sha(name) == digest, 'Historical asset changed: ' + name
    from simulation import install
    run_module, baseline, capture_training = install(root)
    service = None; error = None; started = time.perf_counter()
    try:
        aggregate(root, 'head_camera_preflight')
        capture_training()
        service = Services(root, reg)
        aggregate(root, 'loading_components'); service.start()
        # The development lemon tests the complete generated-candidate interface
        # without executing a grasp or consuming an evaluation slot.
        warm = root / 'preflight/head_warmup'
        make_candidates(warm)
        response = service.request('curobo', warm)
        save(root / 'preflight/full_chain_plan.json', response)
        assert response['result']['success'], 'Development planning interface did not pass'
        save(root / 'preflight/passed.json', {'passed': True, 'physical_grasps': 0,
            'components_resident': reg['required_components'], 'full_chain_development_plan': True})
        for number, entry in enumerate(reg['schedule']):
            check_sources(reg)
            if (root / 'STOP').exists(): raise RuntimeError('STOP present')
            assert shutil.disk_usage(root).free > 10 * 1024**3, 'Disk free below 10GiB'
            for kind, child in service.children.items():
                if child.poll() is not None: raise RuntimeError(kind + ' worker exited')
            slot = entry['id']; trial = root / 'trials' / slot
            claim = root / 'claims' / (slot + '.json')
            assert not claim.exists() and not trial.exists()
            save(claim, {'state': 'claimed', 'id': slot, 'at': now(), 'attempt': 1})
            events = []

            def worker(script, directory, gpu, log, **flags):
                directory = Path(directory)
                try:
                    begin = time.perf_counter()
                    if script == 'neural.py':
                        save(directory / 'entry.json', entry)
                        sam = service.request('sam3', directory)
                        if not sam['result']['target_found']:
                            save(directory / 'perception_failure.json', {'reason': 'sam3_no_target_mask'})
                            return 2
                        ray = service.request('rayst3r', directory)
                        if ray['result']['point_count'] < 20:
                            save(directory / 'perception_failure.json', {'reason': 'rayst3r_empty_pointcloud'})
                            return 2
                        response = service.request('graspgenx', directory, seed=20261122 + number)
                        count = make_candidates(directory)
                        save(directory / 'candidate_count.json', {'count': count})
                    else:
                        response = service.request('curobo', directory, **flags)
                    save(log, response)
                    events.append({'component': script, 'flags': flags, 'wall_seconds': time.perf_counter() - begin})
                    return response['returncode']
                except BaseException:
                    save(directory / 'infrastructure_error.json', {'at': now(), 'script': script, 'flags': flags, 'error': traceback.format_exc()})
                    raise
            run_module.worker = worker
            print('BEGIN', number + 1, '/160', slot, flush=True)
            result = run_module.trial(entry, trial, GPUS['rayst3r'], GPUS['curobo'], 'head')
            if (trial / 'perception_failure.json').exists():
                result['reason'] = read(trial / 'perception_failure.json')['reason']
                save(trial / 'result.json', result)
            save(trial / 'component_timing.json', events)
            baseline.verify_trial(trial)
            if (trial / 'infrastructure_error.json').exists(): raise RuntimeError('Component failure preserved in ' + slot)
            expected_failures = {'success', 'sam3_no_target_mask', 'rayst3r_empty_pointcloud',
                'no_feasible_native_post_or_recalled_candidate', 'observation_path_failed',
                'measured_start_state_in_collision', 'measured_closure_lift_unreachable',
                'measured_pregrasp_approach_unreachable', 'no_bilateral_target_contact',
                'insufficient_hold_height', 'forbidden_robot_contact',
                'Target has fewer than 50 visible pixels'}
            expected_failures.update(stage + '_not_converged' for stage in ['observe', 'pregrasp', 'approach', 'close', 'lift', 'hold'])
            if result['reason'] not in expected_failures:
                raise RuntimeError('Unclassified execution/recording fault; stop scheduling: ' + result['reason'])
            if result['physical_attempt']:
                for name in ['execution.jsonl', 'execution.mp4', 'head.mp4', 'stages.json', 'forbidden_contacts.json']:
                    assert (trial / name).exists() and (trial / name).stat().st_size > 0, 'Missing execution evidence: ' + name
            if not (trial / 'initial_conditions.json').exists() and not (trial / 'observation_plan_result.json').exists():
                raise RuntimeError('Simulation failed before producing initial state: ' + slot)
            if (trial / 'state.json').exists():
                state = read(trial / 'state.json')
                assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
            if (trial / 'plan_result.json').exists() and read(trial / 'plan_result.json').get('selected'):
                from short_lift_protocol import validate_selected
                validate_selected(read(trial / 'plan_result.json')['selected'])
            save(claim, {'state': 'complete', 'id': slot, 'at': now(), 'attempt': 1,
                'result_sha256': sha(trial / 'result.json')})
            summary = aggregate(root, 'running'); report(root, summary)
            print('RESULT', slot, result['success'], result['reason'], flush=True)
        summary = aggregate(root, 'complete'); report(root, summary)
        check_sources(reg)
        save(root / 'final_summary.json', summary)
    except BaseException:
        error = traceback.format_exc()
        save(root / 'halt.json', {'at': now(), 'error': error})
        (root / 'STOP').write_text(error)
        summary = aggregate(root, 'halted'); report(root, summary)
        print(error, flush=True)
    finally:
        if service is not None: service.stop()
        save(root / 'suite_timing.json', {'wall_seconds': time.perf_counter() - started, 'at': now()})
    if error: raise RuntimeError(error)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, default=DEFAULT)
    args = p.parse_args(); root = args.root.resolve()
    if args.stage == 'prepare': prepare(root)
    else:
        with (root / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root)


if __name__ == '__main__': main()
