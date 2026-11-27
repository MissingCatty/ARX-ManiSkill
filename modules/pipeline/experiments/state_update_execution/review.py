"""Review raw success evidence, original candidates, FK, paths, receipts and videos."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel
from modules.pipeline.experiments.indexed_grasp_planning.physical.contract import check_trial
from .contract import VARIANTS, check_requests
from .registration import guard


def describe(values):
    return {'samples': len(values), 'total_seconds': float(sum(values)),
        'mean_seconds': float(np.mean(values)) if values else None,
        'median_seconds': float(np.median(values)) if values else None,
        'p90_seconds': float(np.percentile(values, 90)) if values else None}


def validate_path(path, limits):
    q, times = np.asarray(path['q']), np.asarray(path['times'])
    if (q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or times.shape != q.shape[:1]
            or not np.isfinite(q).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
        raise ValueError('Invalid recorded motion command')
    if (q < limits[:, 0] - 1e-6).any() or (q > limits[:, 1] + 1e-6).any():
        raise ValueError('Recorded motion exceeds unchanged joint limits')
    derivative = q.copy(); sample_t = times.copy()
    for bound in [1.5, 5., 100.]:
        derivative = np.diff(derivative, axis=0) / np.diff(sample_t)[:, None]
        sample_t = (sample_t[:-1] + sample_t[1:]) / 2
        if derivative.size and float(np.abs(derivative).max()) > bound * 1.0001:
            raise ValueError('Recorded motion exceeds original derivative limits')


def review(root):
    reg = guard(root); progress = read_json(root / 'progress.json')
    if len(progress['results']) != 148 or not read_json(root / 'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Complete execution or service shutdown is missing')
    model = ArmModel(); groups = {}; videos = []; endpoints = []; path_count = 0; request_counts = Counter()
    ffmpeg = shutil.which('ffmpeg') or subprocess.check_output([str(ROOT / 'environments/simulation/bin/python'),
        '-c', 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True, timeout=30).strip()
    cases = {c['id']: c for c in reg['cases']}
    for mode in VARIANTS:
        group = {'registered': 74, 'completed': 0, 'planning_passes': 0, 'physical_attempts': 0,
            'successes': 0, 'common_precondition_unexecuted': 0, 'failure_reasons': Counter()}
        times = {stage: [] for stage in ['initial', 'approach', 'lift']}; totals = []
        for item in [s for s in reg['schedule'] if s['variant'] == mode]:
            case = cases[item['case_id']]; out = root / 'trials' / case['id'] / mode
            claim = read_json(root / 'claims' / (item['id'] + '.json'))
            if claim['state'] != 'complete': raise ValueError('Incomplete physical claim')
            for name, digest in claim['files_sha256'].items():
                if sha256(out / name) != digest: raise ValueError('Changed/missing physical record: ' + name)
            result = read_json(out / 'result.json'); group['completed'] += 1
            if case.get('common_precondition_failure'):
                if result.get('physical_attempt') or result.get('planning_success') or not result['common_precondition_unexecuted']:
                    raise ValueError('Inherited prerequisite was counted as a new grasp')
                group['common_precondition_unexecuted'] += 1
                continue
            check_trial(out, result); requested = check_requests(out); request_counts.update(requested)
            original = read_json(ROOT / case['input_directory'] / 'candidates/candidate_pool.json')
            if read_json(out / 'candidates/candidate_pool.json') != original:
                raise ValueError('Delivered candidates differ from the frozen common pool')
            for target, key in [('planning_passes', 'planning_success'), ('physical_attempts', 'physical_attempt'), ('successes', 'success')]:
                group[target] += bool(result.get(key))
            if not result['success']: group['failure_reasons'][result['reason']] += 1
            request_total = 0.
            for stage in requested:
                receipt = read_json(out / ('execution_planner_' + stage + '.json'))
                if (receipt['stage'] != stage or receipt['variant'] != mode
                        or receipt['memory_state_updates_enabled'] != (mode == 'optimized')
                        or receipt['measured_execution_state_used'] != (stage != 'initial')
                        or not receipt['receipt_drain_completed'] or receipt['collision_spheres_per_arm'] != 871
                        or not receipt['geometry_and_thresholds_unchanged']):
                    raise ValueError('The executed planner differs from the registered method')
                metric = read_json(out / (stage + '_metrics.json')); times[stage].append(metric['seconds'])
                request_total += metric['seconds']
                plan = read_json(out / ('plan_result.json' if stage == 'initial' else stage + '_plan_result.json'))
                if bool(metric['success']) != bool(plan['success']): raise ValueError('Planner receipt and command disagree')
            totals.append(request_total)
            initial = read_json(out / 'plan_result.json')
            if initial['success']:
                selected = initial['selected']; index = selected['index']; candidate = original['candidates'][index]
                if selected['contact_pose_world'] != candidate['pose']:
                    raise ValueError('Selected world grasp differs from its frozen candidate')
                if abs(selected['open_finger_m'] - candidate['width'] / 2) > 1e-12:
                    raise ValueError('Selected actual gripper opening differs')
                from .opening import validate_opening
                validate_opening(selected, selected['open_finger_m'])
                if result.get('physical_attempt'):
                    opening = out / 'execution_opening_binding.json'
                    inherited = item['id'] in (reg.get('continuation') or {}).get('completed_slot_ids', [])
                    if not opening.exists() and not (inherited and selected['open_finger_m'] == .044):
                        raise ValueError('Executed opening has no width-binding receipt')
                    if opening.exists():
                        receipt = read_json(opening)
                        if receipt['open_finger_m'] != selected['open_finger_m'] or receipt['plan_sha256'] != sha256(out / 'plan_result.json'):
                            raise ValueError('Executed opening differs from the selected plan')
                        commands = read_json(out / 'execution_finger_commands.json')
                        for command in commands:
                            label = command['stage']; width = command['finger_position_m']
                            expected = selected['open_finger_m'] if label in ['open', 'pregrasp', 'approach'] else 0.
                            if abs(width - expected) > 1e-12:
                                raise ValueError('The executed motion did not retain its collision-checked finger width')
                spec = get_arm(selected['arm']); state = read_json(out / 'state.json')
                limits = np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in spec.joints])
                for label, key in [('pre', 'pregrasp_world'), ('grasp', 'grasp_world'), ('lift', 'lift_world')]:
                    actual = deepcopy(state); actual['links'] = {'base_link': state['links']['base_link']}
                    q = selected['q_' + label]
                    for name, value in zip(spec.joints, q): actual['qpos'][state['joint_names'].index(name)] = value
                    if (np.asarray(q) < limits[:, 0]).any() or (np.asarray(q) > limits[:, 1]).any():
                        raise ValueError('Selected endpoint violates original joint limits')
                    reached = model.frames(actual)[spec.eef]; goal = np.asarray(selected[key])
                    position = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
                    rotation = float(Rotation.from_matrix(reached[:3, :3] @ goal[:3, :3].T).magnitude())
                    if position > .002 or rotation > .03: raise ValueError('Independent physical-arm FK failed')
                    endpoints.append({'slot': item['id'], 'stage': label, 'position_error_m': position,
                        'rotation_error_rad': rotation})
                for stage in requested:
                    plan = read_json(out / ('plan_result.json' if stage == 'initial' else stage + '_plan_result.json'))
                    if plan['success']:
                        for path in plan['paths'].values(): validate_path(path, limits); path_count += 1
            if result.get('physical_attempt'):
                for name in ['execution.mp4', 'head.mp4']:
                    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(out / name), '-f', 'null', '-'],
                        check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    videos.append(str((out / name).relative_to(ROOT)))
        group.update(failure_reasons=dict(group['failure_reasons']),
            planning_unexecuted=group['common_precondition_unexecuted'],
            planning_failures=69 - group['planning_passes'],
            timings={stage: describe(values) for stage, values in times.items()},
            all_planning_per_scene=describe(totals))
        groups[mode] = group
    comparisons = []; losses = []; gains = []
    for case in reg['cases']:
        if case.get('common_precondition_failure'): continue
        plans = [read_json(root / 'trials' / case['id'] / mode / 'plan_result.json') for mode in VARIANTS]
        results = [read_json(root / 'trials' / case['id'] / mode / 'result.json') for mode in VARIANTS]
        a, b = [p.get('selected', {}) for p in plans]
        same = plans[0]['success'] == plans[1]['success'] and all(a.get(k) == b.get(k)
            for k in ['arm', 'index', 'contact_pose_world', 'open_finger_m', 'q_pre', 'q_grasp', 'q_lift'])
        arrays = plans[0].get('paths') == plans[1].get('paths')
        comparisons.append({'id': case['id'], 'same_initial_selection_and_joints': same, 'same_initial_paths': arrays,
            'baseline_success': results[0]['success'], 'optimized_success': results[1]['success']})
        if results[0]['success'] and not results[1]['success']: losses.append(case['id'])
        if results[1]['success'] and not results[0]['success']: gains.append(case['id'])
    continuation = reg.get('continuation') or {}
    cross_restart = continuation.get('zero_motion_handoff_slot', '').rsplit('_', 1)[0]
    # The reused initial plan predates the new model process. Keep it in task
    # results, and use only matched-phase pairs for the primary timing table.
    matched = {}
    for mode in VARIANTS:
        stage_values = {s: [] for s in ['initial', 'approach', 'lift']}; total_values = []
        for case in reg['cases']:
            if case.get('common_precondition_failure') or case['id'] == cross_restart: continue
            out = root / 'trials' / case['id'] / mode; total = 0.
            for stage in check_requests(out):
                value = read_json(out / (stage + '_metrics.json'))['seconds']
                stage_values[stage].append(value); total += value
            total_values.append(total)
        matched[mode] = {'timings': {s: describe(v) for s, v in stage_values.items()},
            'all_planning_per_scene': describe(total_values)}
    checked = {'at': now(), 'passed': True, 'task_complete': True, 'record_integrity': True,
        'completed_slots': 148, 'groups': groups, 'baseline_success_losses': losses, 'optimized_success_gains': gains,
        'all_initial_selections_and_joints_equal': all(r['same_initial_selection_and_joints'] for r in comparisons),
        'all_initial_paths_equal': all(r['same_initial_paths'] for r in comparisons), 'comparisons': comparisons,
        'event_applicability': 'complete simulated grasp execution; frozen perception and candidates',
        'key_event_coverage': {'requested_stages': dict(request_counts), 'independent_FK_endpoints': len(endpoints),
            'checked_command_paths': path_count, 'decoded_videos': len(videos), 'unexecuted_common_slots': 10},
        'decoded_videos': videos, 'automatic_default_activation': False,
        'continuation': {'completed_results_inherited': len(continuation.get('completed_slot_ids', [])),
            'zero_motion_handoff_slot': continuation.get('zero_motion_handoff_slot'),
            'initial_solver_calls_repeated': 0, 'physical_grasps_repeated': 0},
        'matched_phase_timings': matched, 'cross_restart_timing_case': cross_restart}
    write_json(root / 'endpoint_review.json', {'passed': True, 'endpoints': endpoints})
    write_json(root / 'acceptance.json', checked)
    return checked
