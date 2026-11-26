"""Matched V8 validation pilot, with all original candidates and no simulation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport, close_owned_transports
from arx_project.client import Client
from modules.graspgenx.postprocessing.ordering import make_queue
from modules.pipeline.experiments.graspgenx_postprocessing.planning import raw, verify

CONFIG = ROOT/'configs/experiments/indexed_grasp_validation.json'


def request_order(variants, ordinal):
    if set(variants) == {'control_1', 'vectorized_1', 'vectorized_2', 'control_2'}:
        return ['control_1', 'vectorized_1', 'vectorized_2', 'control_2'] if ordinal%2 == 0 else \
               ['vectorized_1', 'control_1', 'control_2', 'vectorized_2']
    if set(variants) == {'control_1', 'vectorized_1'}:
        return ['control_1', 'vectorized_1'] if ordinal%2 == 0 else ['vectorized_1', 'control_1']
    raise ValueError('Unsupported validation comparison schedule')


def prepare(root, config=CONFIG):
    if root.exists(): raise ValueError('Keep previous V8 registrations; choose a fresh directory')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    from modules.graspgenx.postprocessing.nms import native_backend
    from .validation_launch import monitor_policy
    current = load_current();cfg = read_json(config)
    source = ROOT/cfg['source_run'];source_reg = raw(source/'registration.json')
    if not raw(source/'planning_review.json')['passed']:
        raise ValueError('The full NMS+B16 source was not independently accepted')
    full = cfg.get('phase') == 'full'
    cases = source_reg['cases'] if full else [c for c in source_reg['cases'] if c['id'] in cfg['pilot_cases']]
    if (full and len(cases) != 69) or (not full and (len(cases) != 5 or
            {c['id'] for c in cases} != set(cfg['pilot_cases']))):
        raise ValueError('Validation cohort differs from its registered phase')
    if cfg['planning_requests'] != len(cases)*len(cfg['variants']):
        raise ValueError('Validation request budget differs from its exact schedule')
    request_order(cfg['variants'], 0)
    inputs = {str(source/name): sha256(source/name) for name in ['registration.json', 'planning_review.json']}
    if full:
        pilot = ROOT/cfg['pilot_source'];checked = raw(pilot/'independent_review.json')
        accepted = raw(pilot/'acceptance.json')
        if (pilot/'STOP').exists() or not all(checked[k] for k in ['passed', 'record_integrity',
                'control_successes_retained', 'all_original_selections_and_branches_retained', 'timing_gate_passed']):
            raise ValueError('The independently accepted pilot did not meet the full-comparison gates')
        if not accepted['task_complete'] or accepted['planning_requests'] != 20:
            raise ValueError('The required pilot budget was not completed')
        for name in ['registration.json', 'independent_review.json', 'acceptance.json']:
            inputs[str(pilot/name)] = sha256(pilot/name)
        # Completed source snapshots remain the authority after runner changes.
        for name, digest in raw(pilot/'registration.json')['source_sha256'].items():
            archived = pilot/'source_registered'/Path(name).relative_to(ROOT)
            if sha256(archived) != digest: raise ValueError('Pilot source archive was changed')
            relative = str(Path(name).relative_to(ROOT))
            if relative.startswith(('modules/curobo/', 'modules/maniskill/', 'arx_project/',
                                    'modules/graspgenx/postprocessing/')) or relative.endswith('/validation_worker.py'):
                if sha256(name) != digest:
                    raise ValueError('The full comparison changed a pilot planning algorithm: '+relative)
    for case in cases:
        directory = Path(case['source_directory'])
        for name in ['state.json', 'candidates/candidate_pool.json', 'plan_result.json', 'result.json']:
            inputs[str(directory/name)] = sha256(directory/name)
        expected = ROOT/cfg['offline_source']/'cases'/case['id']/'nms_p015_r15.json'
        inputs[str(expected)] = sha256(expected)
    sources = {config, ROOT/'configs/modules.json', ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/policies/fingertip_table.json', native_backend()[1],
        ROOT/'configs/experiments/dual_arm_interleaving.json', ROOT/'modules/curobo/arx_adapter.py',
        ROOT/'modules/curobo/scene_cache.py', ROOT/'modules/maniskill/resident_planner.py',
        ROOT/'modules/maniskill/fast_resident_planner.py', ROOT/'modules/maniskill/arx_pipeline/planner.py',
        ROOT/'modules/maniskill/checked_native_services.py', ROOT/'modules/pipeline/monitoring/gpu_auto_start.py'}
    for folder in ['modules/curobo/dual_arm', 'modules/curobo/fingertip_contact', 'modules/curobo/candidate_io',
                   'modules/curobo/indexed_planning', 'modules/graspgenx/postprocessing',
                   'modules/pipeline/experiments/graspgenx_postprocessing',
                   'modules/pipeline/experiments/indexed_grasp_planning', 'arx_project']:
        sources.update((ROOT/folder).glob('*.py'))
    sources.update(Path(p) for p in current['configuration_hashes'])
    sources.update(ROOT/'modules/curobo/source/curobo/_src/solver'/p for p in ['solver_ik.py', 'solver_ik_cfg.py'])
    root.mkdir(parents=True);(root/'claims').mkdir()
    write_json(root/'monitor_config.json', monitor_policy(root))
    inputs[str(root/'monitor_config.json')] = sha256(root/'monitor_config.json')
    warm = ROOT/'tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01/methods/zerograsp/preflight/head_warmup'
    for name in ['state.json', 'closed_state.json']:
        target = root/'warmup'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(warm/name, target);inputs[str(warm/name)] = sha256(warm/name)
        inputs[str(target)] = sha256(target)
    for path in sorted(sources):
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
    reg = {'at': now(), **cfg, 'cases': cases, 'gpu': 'pending_resource_selection',
        'device_assignment_pending': True, 'variants': list(cfg['variants']), 'variant_modes': cfg['variants'],
        'source_sha256': {str(p): sha256(p) for p in sorted(sources)}, 'input_sha256': inputs,
        'case_selection': 'all_69_frozen_observations_no_outcome_filtering' if full else 'five_posthoc_engineering_fixtures_before_full69_decision',
        'geometry_hard_filter_enabled': False, 'new_candidate_poses': False,
        'physical_grasps': 0, 'automatic_default_activation': False}
    write_json(root/'registration.json', reg);write_json(root/'registration_pending.json', reg)
    write_json(root/'progress.json', {'at': now(), 'phase': 'waiting_for_gpu',
        'completed_requests': 0, 'total_requests': reg['planning_requests'], 'results': []})


def compare_pair(pair):
    return {'control_success_retained': not pair['control']['success'] or pair['vectorized']['success'],
        'same_original_selection': all(pair['control'][k] == pair['vectorized'][k] for k in
            ['success', 'selected_arm', 'selected_original_source_index', 'candidate_checks']),
        'same_joint_endpoints': pair['control']['joint_endpoints'] == pair['vectorized']['joint_endpoints'],
        'same_candidate_event_decisions': pair['control']['event_decisions'] == pair['vectorized']['event_decisions']}


def summarize(rows, modes):
    summary = {}
    for group in ['control', 'vectorized']:
        entries = [v for row in rows for mode, v in row['variants'].items() if modes[mode] == group]
        times = [v['planning_request_seconds'] for v in entries]
        summary[group] = {'requests': len(entries), 'successes': sum(v['success'] for v in entries),
            'median_seconds': float(np.median(times)), 'p90_seconds': float(np.percentile(times, 90)),
            'total_seconds': float(sum(times)), 'candidate_checks': sum(v['candidate_checks'] for v in entries)}
        stages = {}
        for entry in entries:
            for name, value in entry['stage_summary'].items():
                out = stages.setdefault(name, {'calls': 0, 'wall_seconds': 0., 'exclusive_seconds': 0.})
                for key in out: out[key] += value[key]
        summary[group]['stage_totals'] = stages
    return summary


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Do not repeat an already claimed V8 pilot')
    reg = raw(root/'registration.json');verify(reg)
    if reg['device_assignment_pending']: raise ValueError('CUDA-verified assignment is required')
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu']})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=reg['gpu'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    socket_path = root/'rpc.sock'
    if len(str(socket_path).encode()) >= 108: raise ValueError('V8 socket path exceeds UNIX socket limit')
    command = [str(ROOT/'environments/curobo/bin/python'), '-m',
        'modules.pipeline.experiments.indexed_grasp_planning.validation_worker', '--root', str(root/'resident'),
        '--socket', str(socket_path), '--warmup-trial', str(root/'warmup'),
        '--descriptor', str(ROOT/'configs/modules.json')]
    log = (root/'resident.log').open('a')
    worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_json(root/'worker.json', {'at': now(), 'pid': worker.pid, 'command': command, 'gpu': reg['gpu']})
    client = Client(socket_path, timeout=1800, candidate_transport=True)
    rows, completed = [], 0
    try:
        write_json(root/'progress.json', {'at': now(), 'phase': 'warming', 'completed_requests': 0,
            'total_requests': reg['planning_requests'], 'results': []})
        deadline = time.monotonic()+1200
        while not (root/'resident/curobo_indexed_planning_ready.json').exists():
            if worker.poll() is not None: raise RuntimeError('V8 worker exited during startup; see resident.log')
            if time.monotonic() > deadline: raise TimeoutError('V8 validation warmup exceeded deadline')
            time.sleep(1)
        write_json(root/'startup.json', {'at': now(), **raw(root/'resident/startup.json')})
        for ordinal, case in enumerate(reg['cases']):
            source = Path(case['source_directory']);payload = raw(source/'candidates/candidate_pool.json')
            expected = raw(ROOT/reg['offline_source']/'cases'/case['id']/'nms_p015_r15.json')
            order = request_order(reg['variants'], ordinal)
            pair = {}
            for variant in order:
                directory = root/'plans'/case['id']/variant;directory.mkdir(parents=True)
                claim = root/'claims'/(case['id']+'_'+variant+'.json')
                with claim.open('x') as stream: json.dump({'at': now(), 'state': 'claimed'}, stream)
                shutil.copy2(source/'state.json', directory/'state.json')
                write_json(root/'progress.json', {'at': now(), 'phase': 'planning', 'completed_requests': completed,
                    'total_requests': reg['planning_requests'], 'current': {'id': case['id'], 'variant': variant}, 'results': rows})
                print('V8_VALIDATION_BEGIN', case['id'], variant, flush=True)
                began = time.perf_counter();queue, manifest = make_queue(payload, 'nms15')
                post_seconds = time.perf_counter()-began
                if any(manifest[k] != expected[k] for k in ['representatives', 'deferred']):
                    raise ValueError('Native NMS differs from the accepted original source')
                write_json(directory/'candidate_order.json', manifest)
                descriptor = publish_candidate_pool(directory, queue)
                response = client.request(directory, candidate_transport=descriptor,
                    validation_variant=reg['variant_modes'][variant])
                if response['status'] != 'ok': raise RuntimeError('V8 planner RPC fault: '+str(response))
                finish_candidate_transport(directory)
                io = raw(directory/'candidate_io.json')
                if io['source'] != 'shared_memory' or io['decode_count'] != 1 or io['full_arm_pool_copies'] != 0:
                    raise RuntimeError('V8 lost the single-decode shared-memory contract')
                metrics = raw(directory/'validation_metrics.json')
                plan = raw(directory/'plan_result.json')
                metrics['joint_endpoints'] = {k: plan.get('selected', {}).get(k) for k in ['q_pre', 'q_grasp', 'q_lift']}
                decisions = {}
                for side in ['right', 'left']:
                    arm = raw(directory/'arms'/side/'plan_result.json')
                    decisions[side] = [{k: e[k] for k in ['candidate_index', 'kind', 'kinematic_solutions', 'reasons'] if k in e}
                                       for e in arm.get('candidate_events', [])]
                metrics.update(event_decisions=decisions, postprocessing_seconds=post_seconds,
                    candidate_decode_seconds=io['load_seconds'],
                    postprocessing_plus_planning_seconds=post_seconds+metrics['planning_request_seconds'])
                write_json(directory/'metrics.json', metrics);pair[variant] = metrics;completed += 1
                write_json(claim, {'at': now(), 'state': 'complete',
                    'arm_results_sha256': {side: sha256(directory/'arms'/side/'plan_result.json')
                        for side in ['right', 'left']},
                    **{key: sha256(directory/name) for key, name in [
                        ('metrics_sha256', 'metrics.json'), ('plan_sha256', 'plan_result.json'),
                        ('order_sha256', 'candidate_order.json'), ('timing_sha256', 'stage_timings.json'),
                        ('validation_metrics_sha256', 'validation_metrics.json')]}})
                print('V8_VALIDATION_END', case['id'], variant, metrics['success'],
                      round(metrics['planning_request_seconds'], 3), metrics['candidate_checks'], flush=True)
            repeats = [1, 2] if 'control_2' in reg['variants'] else [1]
            comparisons = [compare_pair({mode: pair[mode+'_'+str(n)] for mode in ['control', 'vectorized']}) for n in repeats]
            rows.append({'id': case['id'], 'variants': pair, 'comparison': comparisons})
            write_json(root/'progress.json', {'at': now(), 'phase': 'comparing', 'completed_requests': completed,
                'total_requests': reg['planning_requests'], 'results': rows, 'summaries': summarize(rows, reg['variant_modes'])})
            if not all(all(c.values()) for c in comparisons):
                raise RuntimeError('Original decisions, first branch or control success changed; stop scheduling')
        verify(reg)
        summary = summarize(rows, reg['variant_modes'])
        saving = 1-summary['vectorized']['total_seconds']/summary['control']['total_seconds']
        write_json(root/'acceptance.json', {'at': now(), 'task_complete': completed == reg['planning_requests'],
            'record_integrity': True, 'planning_requests': completed, 'summaries': summary, 'cases': rows,
            'control_successes_retained': all(c['control_success_retained'] for r in rows for c in r['comparison']),
            'all_original_selections_and_branches_retained': all(all(c.values()) for r in rows for c in r['comparison']),
            'planning_total_time_reduction': saving,
            'timing_gate_passed': saving >= reg['minimum_total_time_reduction_for_full_comparison'],
            'physical_grasps': 0, 'automatic_full_comparison': False, 'automatic_default_activation': False})
        write_json(root/'progress.json', {'at': now(), 'phase': 'complete', 'completed_requests': completed,
            'total_requests': reg['planning_requests'], 'results': rows, 'summaries': summary})
    finally:
        close_owned_transports()
        if worker.poll() is None:
            try: client.call({'command': 'shutdown'})
            except Exception: worker.terminate()
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: worker.kill();worker.wait(timeout=10)
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args();root = ensure_run_path(args.root)
    if args.stage == 'prepare': prepare(root, args.config.resolve())
    else:
        try: run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
            raise
