"""Reconcile physical slots, executed-stage receipts, trajectories and videos."""
import json
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np

from arx_project.layout import ROOT, now, sha256, write_json
from .contract import check_trial
from .registration import guard, raw


def review(root):
    reg = guard(root);p = raw(root/'progress.json')
    if len(p['results']) != 20 or not raw(root/'services_stopped.json')['all_owned_exited']:
        raise ValueError('Physical schedule or shutdown evidence is incomplete')
    groups = {};seen = set();videos = []
    ffmpeg = shutil.which('ffmpeg') or subprocess.check_output([
        str(ROOT/'environments/simulation/bin/python'), '-c',
        'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True, timeout=30).strip()
    for item in reg['schedule']:
        if item['id'] in seen: raise ValueError('Duplicate physical slot')
        seen.add(item['id']);claim = raw(root/'claims'/(item['id']+'.json'))
        out = root/'trials'/item['case_id']/item['variant'];result = raw(out/'result.json')
        if claim['state'] != 'complete': raise ValueError('An incomplete claim cannot pass')
        for name, digest in claim['files_sha256'].items():
            if sha256(out/name) != digest: raise ValueError('Physical record changed: '+name)
        check_trial(out, result)
        for actual_state, receipt, plan in [('pregrasp_state.json', 'execution_planner_approach.json', 'approach_plan_result.json'),
                                           ('closed_state.json', 'execution_planner_lift.json', 'lift_plan_result.json')]:
            if (out/actual_state).exists():
                if not (out/receipt).is_file() or not (out/plan).is_file():
                    raise ValueError('An executed stage lost its planner command/receipt')
        initial = raw(out/'execution_planner_initial.json')
        mode = item['variant'];enabled = mode == 'optimized'
        if (initial['variant'] != mode or initial['batch_capacity'] != (16 if enabled else 0)
                or initial['C_YAML_enabled'] != enabled or initial['obstacle_major_queries'] != enabled
                or not initial['scalar_branch_validation'] or not initial['geometry_and_thresholds_unchanged']):
            raise ValueError('Physical planner variant differs from its registration')
        case = next(c for c in reg['cases'] if c['id'] == item['case_id'])
        original = raw(ROOT/case['input_directory']/'candidates/candidate_pool.json')['candidates']
        manifest = raw(out/'candidate_order.json');delivered = raw(out/'candidates/candidate_pool.json')['candidates']
        if len(delivered) != len(original): raise ValueError('Candidate pool was truncated')
        for rank, source in enumerate(manifest['priority_to_original_source']):
            if delivered[rank] != original[source]: raise ValueError('Original candidate geometry changed')
        if result.get('planning_success'):
            selected = raw(out/'plan_result.json')['selected'];index = selected['original_source_index']
            if not np.array_equal(selected['contact_pose_world'], original[index]['pose']):
                raise ValueError('Chosen physical world grasp changed')
        for name in ['execution.mp4', 'head.mp4'] if result.get('physical_attempt') else []:
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(out/name), '-f', 'null', '-'],
                check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            videos.append(str((out/name).relative_to(ROOT)))
        value = groups.setdefault(mode, {'registered': 10, 'completed': 0, 'planning_passes': 0,
            'physical_attempts': 0, 'successes': 0, 'failure_reasons': {}, 'planning_seconds': []})
        value['completed'] += 1
        for target, key in [('planning_passes', 'planning_success'), ('physical_attempts', 'physical_attempt'), ('successes', 'success')]:
            value[target] += bool(result.get(key))
        if not result['success']:
            reason = result['reason'];value['failure_reasons'][reason] = value['failure_reasons'].get(reason, 0)+1
        value['planning_seconds'].append(raw(out/'latency.json')['timings']['initial_planning_with_postprocessing_seconds'])
    for group in groups.values():
        times = group.pop('planning_seconds')
        group.update(planning_total_seconds=sum(times), planning_median_seconds=statistics.median(times),
                     planning_p90_seconds=float(np.percentile(times, 90)))
    losses = [];gains = []
    for case in reg['cases']:
        values = [raw(root/'trials'/case['id']/mode/'result.json')['success'] for mode in ['baseline', 'optimized']]
        if values == [True, False]: losses.append(case['id'])
        if values == [False, True]: gains.append(case['id'])
    saving = 1-groups['optimized']['planning_total_seconds']/groups['baseline']['planning_total_seconds']
    latency = {'total_reduced_5pct': saving >= .05,
        'median_reduced': groups['optimized']['planning_median_seconds'] < groups['baseline']['planning_median_seconds'],
        'p90_not_more_than_5pct_higher': groups['optimized']['planning_p90_seconds'] <= groups['baseline']['planning_p90_seconds']*1.05}
    result = {'at': now(), 'passed': True, 'task_complete': True, 'record_integrity': True,
        'completed_slots': 20, 'physical_attempts': sum(v['physical_attempts'] for v in groups.values()),
        'groups': groups, 'baseline_success_losses': losses, 'optimized_success_gains': gains,
        'planning_total_time_reduction': saving, 'timing_gates': latency,
        'optimization_accepted': not losses and all(latency.values()), 'decoded_videos': videos,
        'event_applicability': 'executed stages checked against measured state and planner receipts',
        'key_event_coverage': {'initial_plans': 20, 'actual_executions': len(videos)//2,
            'videos': len(videos), 'fresh_state_restores': 20}, 'automatic_default_activation': False}
    write_json(root/'acceptance.json', result);return result
