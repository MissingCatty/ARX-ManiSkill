"""Run registered control/parallel4/parallel8 requests and review their records."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from arx_project.artifacts import publish_candidate_pool, finish_candidate_transport, close_owned_transports
from arx_project.client import Client
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard


def summarize(rows):
    result = {}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        group = [r for r in rows if r['mode'] == mode]
        if not group: continue
        values = [r['RPC_seconds'] for r in group]
        batches = [b for r in group for w in r.get('parallel_windows', []) for b in w['batches']]
        result[mode] = {'requests': len(group), 'successes': sum(r['success'] for r in group),
            'total_seconds': sum(values), 'mean_seconds': float(np.mean(values)),
            'median_seconds': float(np.median(values)), 'p90_seconds': float(np.percentile(values, 90)),
            'candidate_checks': sum(r['candidate_checks'] for r in group),
            'actual_parallel_calls': len(batches), 'real_parallel_rows': sum(b['real_rows'] for b in batches),
            'speculative_endpoint_checks': sum(w['extra_endpoint_evaluations'] for r in group for w in r.get('parallel_windows', []))}
    return result


def run(root):
    reg = guard(root)
    if reg['gpu'] == 'pending':
        raise RuntimeError('Assign an admitted GPU before starting this run')
    if (root / 'STOP').exists() or (root / 'started.json').exists():
        raise ValueError('Do not retry a started run')
    from modules.pipeline.experiments.indexed_grasp_planning.physical.controller import environment, wait_ready
    write_json(root / 'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': reg['gpu']})
    socket = ROOT / 'tmp/rpc' / ('parallel' + sha256(root / 'registration.json')[:10] + '.sock')
    socket.parent.mkdir(exist_ok=True, parents=True)
    client = Client(socket, timeout=1800, candidate_transport=True)
    rows = []; cases = {c['id']: c for c in reg['cases']}
    worker_module = reg['config'].get('worker_module', 'modules.pipeline.experiments.parallel_planning.worker')
    cmd = [str(ROOT / 'environments/curobo/bin/python'), '-u', '-m', worker_module, '--root', str(root / 'resident'),
        '--socket', str(socket), '--warmup-trial', str(root / 'warmup'),
        '--descriptor', str(ROOT / 'configs/modules.json')]
    if reg['config'].get('require_idle_device'):
        cmd.append('--strict-idle')
    with (root / 'planner.log').open('x') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=environment(reg['gpu']), stdout=log, stderr=subprocess.STDOUT)
        write_json(root / 'planner_process.json', {'pid': child.pid, 'command': cmd})
        try:
            wait_ready(child, root / 'resident/curobo_dual_arm_ready.json')
            for item in reg['schedule']:
                guard(root)
                out = root / 'plans' / item['case_id'] / item['variant']
                claim = root / 'claims' / (item['id'] + '.json')
                if out.exists() or claim.exists(): raise RuntimeError('Planning slot already claimed')
                out.mkdir(parents=True)
                shutil.copy2(Path(cases[item['case_id']]['source_directory']) / 'state.json', out / 'state.json')
                payload = read_json(root / 'offline' / item['case_id'] / 'fine.json')
                write_json(claim, {'at': now(), 'state': 'claimed', **item})
                write_json(root / 'progress.json', {'at': now(), 'phase': 'planning', 'completed_requests': len(rows),
                    'total_requests': len(reg['schedule']), 'current': item, 'results': rows, 'summaries': summarize(rows)})
                began = time.perf_counter()
                transport = publish_candidate_pool(out, payload)
                response = client.request(out, variant=item['mode'], candidate_transport=transport)
                elapsed = time.perf_counter() - began
                finish_candidate_transport(out)
                if response['status'] != 'ok': raise RuntimeError('Planner infrastructure error: ' + str(response))
                row = {**response['result'], **item, 'RPC_seconds': elapsed, 'input_candidates': len(payload['candidates'])}
                write_json(out / 'metrics.json', row)
                write_json(claim, {'at': now(), 'state': 'complete', **item,
                    'plan_sha256': sha256(out / 'plan_result.json'), 'metrics_sha256': sha256(out / 'metrics.json'),
                    'candidate_pool_sha256': sha256(out / 'candidates/candidate_pool.json')})
                rows.append(row)
                write_json(root / 'progress.json', {'at': now(), 'phase': 'planning', 'completed_requests': len(rows),
                    'total_requests': len(reg['schedule']), 'results': rows, 'summaries': summarize(rows)})
                print('PARALLEL_PLAN', len(rows), item['case_id'], item['mode'], row['success'], round(elapsed, 4), flush=True)
                if row.get('stop_after_request'):
                    raise RuntimeError('New foreign GPU process appeared; completed request retained and later scheduling stopped')
            guard(root)
            write_json(root / 'acceptance.json', {'at': now(), 'task_complete': True, 'planning_requests': len(rows),
                'summaries': summarize(rows), 'physical_grasps': 0, 'simulation_instances': 0, 'default_activated': False})
        finally:
            close_owned_transports()
            if child.poll() is None:
                try: client.call({'command': 'shutdown'})
                except Exception: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
            write_json(root / 'services_stopped.json', {'at': now(), 'all_owned_exited': child.poll() is not None,
                'returncode': child.returncode})
    review(root)


def review(root):
    from modules.pipeline.experiments.graspgenx_native_postprocessing.planning import review as independent_review
    independent_review(root)
    reg = guard(root)
    comparisons = []
    for case in reg['cases']:
        for repeat in [1, 2]:
            plans = {mode: read_json(root / 'plans' / case['id'] / (mode + '_' + str(repeat)) / 'plan_result.json')
                     for mode in reg['config']['variants']}
            identity = lambda p: {key: p.get('selected', {}).get(key) for key in
                ['arm', 'original_native_rank', 'source_generator_index']}
            comparisons.append({'case': case['id'], 'repeat': repeat,
                'success': {m: p['success'] for m, p in plans.items()},
                'selected': {m: identity(p) for m, p in plans.items()},
                'same_grasp_and_arm': all(identity(p) == identity(plans['control']) for p in plans.values())})
    write_json(root / 'paired_review.json', {'at': now(), 'passed': True, 'pairs': comparisons,
        'original_successes_retained': all(not p['success']['control'] or
            all(p['success'][m] for m in reg['config']['variants'] if m != 'control') for p in comparisons)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = ensure_run_path(p.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root / 'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
