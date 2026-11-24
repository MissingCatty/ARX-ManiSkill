"""Bounded fresh ARX validation of the ZeroGrasp + GraspCL demonstration route."""
from arx_project.layout import canonical_path
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from graspcl_pipeline import ROOT, BASELINE, read, save, sha

BACKEND = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'
SCENE = 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169'


def now():
    return datetime.now().astimezone().isoformat()


@contextmanager
def locked(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def guard(reg):
    for path, digest in reg['source_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source/model changed: ' + path)


def prepare(out, bank):
    bankreg = read(bank / 'registration.json')
    complete = read(bank / 'bank_complete.json')
    offline = read(bank / 'offline_summary.json')
    if offline['completed'] != offline['registered_queries'] or any(r['status'] != 'inference_completed' for r in offline['results']):
        raise RuntimeError('Offline integration must finish without errors first')
    out.mkdir(parents=True, exist_ok=False)
    sources = dict(bankreg['source_hashes'])
    sources.update(complete['model_sha256'])
    for path in [Path(__file__), CODE / 'graspcl_planner.py', CODE / 'grasp_visuals.py', bank / 'registration.json', bank / 'bank_complete.json']:
        sources[str(path.resolve())] = sha(path)
    ids = ['B0196_S4_clutter_042_adjustable_wrench', 'B0069_S3_sparse_014_lemon']
    entries, original = [], {}
    for slot in ids:
        query = next(e for e in bankreg['query_trials'] if e['id'] == slot)
        historical = Path(query['trial'])
        result = read(historical / 'result.json')
        entry = {k: result[k] for k in ['id', 'model_id', 'family', 'layout', 'seed', 'scene', 'observation']}
        entry.update(campaign_registration=str(out / 'registration.json'),
                     historical_trial=str(historical), original_success=query['original_success'])
        entries.append(entry)
        original[slot] = {str(historical / name): sha(historical / name) for name in
                          ['result.json', 'initial_conditions.json', 'state.json', 'independent_evaluation.json']}
    reg = {'created_at': now(), 'chain': bankreg['chain'], 'bank': str(bank),
        'attempt_budget': 2, 'one_attempt_only': True, 'entries': entries,
        'source_hashes': sources, 'historical_evidence_sha256': original,
        'scope': 'Two fresh ManiSkill/SAPIEN ARX functional smoke tests, one historical failure and one historical success. Not a success-rate comparison or promotion decision.',
        'candidate_policy': bankreg['candidate_policy'],
        'selection_change': 'GraspCL posterior mean over ZeroGrasp top256 with finger symmetry plus all admitted recall matches. Direct canonical contact pose to original EEF via unchanged 40mm offset. Original downward-approach gate retained; baseline GT center-radius/height-shift heuristic omitted explicitly in this branch.',
        'observation': 'Replay each historical scene and already registered wrist observation joint goal; fresh images after motion. No new GT-centered observation search.',
        'frozen': 'Original ARX robot, physics, scene poses, scales, camera mount/K/extrinsics, cuRobo robot/collision/optimizer and all physical success gates.',
        'success': read(BASELINE / 'registration.json')['success'],
        'gpu': {'scene': SCENE, 'neural_and_planner': BACKEND, 'graspcl': 'CPU, four threads'},
        'backend_lock': str(BASELINE / 'backend.lock'),
        'unknown_policy': 'Never replay a claimed slot. A crash/timeout without terminal result remains unknown.',
    }
    guard(reg)
    save(out / 'registration.json', reg)
    for folder in ['trials', 'claims', 'entries', 'backend_requests']:
        (out / folder).mkdir()
    for entry in entries:
        save(out / 'entries' / (entry['id'] + '.json'), entry)
    shutil_source = BASELINE / 'storage_policy.json'
    (out / 'storage_policy.json').write_bytes(shutil_source.read_bytes())
    print(json.dumps({'prepared': str(out), 'physical_budget': 2, 'ids': ids}), flush=True)


def infer_case(trial, bank):
    from graspcl_pipeline import seed, load_model, infer, write_inference
    seed()
    module = load_model(bank)
    result = infer(module, trial, read(bank / 'registration.json')['candidate_policy'])
    destination = trial / 'graspcl'
    destination.mkdir(exist_ok=False)
    (destination / 'zg_outputs').mkdir()
    write_inference(result, trial, destination)
    print(json.dumps(result['counts']), flush=True)


def execute(out, entry_path):
    sys.path.insert(0, str(CODE / 'arx_pipeline'))
    import run
    import baseline
    entry = read(entry_path)
    campaign = Path(entry['campaign_registration']).parent
    reg = read(entry['campaign_registration'])
    guard(reg)
    original_worker = run.worker

    def worker(script, directory, gpu, log, **flags):
        guard(reg)
        if flags.get('observe'):
            current = read(directory / 'state.json')
            old = read(Path(entry['historical_trial']) / 'initial_conditions.json')
            # Validate original reset before moving the arm. Observation q itself is
            # copied verbatim into the frozen new entry.
            if np.max(np.abs(np.array(current['qpos']) - np.array(old['initial_q']))) > .005:
                raise RuntimeError('Historical initial robot joints differ')
            for obj in entry['scene']['actors']:
                if current['objects'][obj['name']]['model_id'] != obj['model_id']:
                    raise RuntimeError('Historical object model differs')
        request = campaign / 'backend_requests' / (uuid.uuid4().hex + '.json')
        receipt = {'slot': entry['id'], 'script': script, 'flags': flags, 'gpu': gpu,
                   'queued_at': now(), 'state': 'queued'}
        save(request, receipt)
        with locked(reg['backend_lock']):
            receipt.update(state='running', started_at=now())
            save(request, receipt)
            try:
                actual = str(CODE / 'graspcl_planner.py') if script == 'planner.py' and not any(flags.values()) else script
                status = original_worker(actual, directory, gpu, log, **flags)
                receipt.update(state='complete', returncode=status)
            except BaseException:
                receipt.update(state='failed', error=traceback.format_exc())
                raise
            finally:
                receipt['finished_at'] = now()
                save(request, receipt)
        if script == 'neural.py' and status == 0:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            env.pop('PYTHONPATH', None)
            command = [str(canonical_path(ROOT / 'tmp/arx-graspcl-env/bin/python')), str(Path(__file__)),
                       'infer', '--output', str(directory), '--bank', reg['bank']]
            with (directory / 'graspcl_inference.log').open('x') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
            from grasp_visuals import render
            render(directory / 'graspcl', directory / 'zg_outputs')
        return status

    run.worker = worker
    run.trial(entry, out, BACKEND, BACKEND, 'right_wrist')
    if (out / 'initial_conditions.json').exists():
        old = read(Path(entry['historical_trial']) / 'initial_conditions.json')
        new = read(out / 'initial_conditions.json')
        save(out / 'reset_comparison.json', {
            'spawn_equal': old['spawn'] == new['spawn'], 'base_pose_equal': old['base_pose'] == new['base_pose'],
            'initial_q_max_difference': float(np.max(np.abs(np.array(old['initial_q'])-np.array(new['initial_q'])))),
            'historical_observation_reused_exactly': entry['observation'] == read(Path(entry['historical_trial']) / 'result.json')['observation']})
    baseline.verify_trial(out)


def run_campaign(out):
    reg = read(out / 'registration.json')
    guard(reg)
    with (out / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for entry in reg['entries']:
            guard(reg)
            slot = entry['id']
            trial = out / 'trials' / slot
            if (out / 'claims' / (slot + '.json')).exists():
                raise RuntimeError('Claimed slot must not be replayed: ' + slot)
            save(out / 'progress.json', {'phase': 'executing', 'current': slot, 'completed': len(results), 'budget': 2})
            with (out / 'claims' / (slot + '.json')).open('x') as stream:
                json.dump({'slot': slot, 'claimed_at': now()}, stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=SCENE, ARX_SIM_GPU=SCENE)
            command = ['/bin/bash', str(CODE / 'run.sh'), str(Path(__file__)), 'execute',
                       '--output', str(trial), '--entry', str(out / 'entries' / (slot + '.json'))]
            with (out / (slot + '.log')).open('x') as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if not (trial / 'result.json').exists() or not (trial / 'independent_evaluation.json').exists():
                save(out / 'STOP', {'reason': 'claimed_slot_missing_terminal_evidence', 'slot': slot, 'returncode': process.returncode})
                raise RuntimeError('Unknown slot; stopped without replay: ' + slot)
            result = read(trial / 'result.json')
            if not read(trial / 'independent_evaluation.json')['passed']:
                raise RuntimeError('Independent evaluation disagreement')
            results.append({'id': slot, 'historical_success': entry['original_success'],
                            'success': result['success'], 'reason': result['reason'],
                            'physical_attempt': result['physical_attempt'],
                            'planning_success': result['planning_success']})
            save(out / 'summary.json', {'registered': 2, 'completed': len(results),
                'physical_attempts': sum(r['physical_attempt'] for r in results),
                'successes': sum(r['success'] for r in results), 'results': results,
                'scope': 'Functional smoke test only; not a controlled success-rate comparison'})
        save(out / 'progress.json', {'phase': 'complete', 'completed': len(results), 'budget': 2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'infer'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage == 'prepare':
        prepare(out, args.bank.resolve())
    elif args.stage == 'run':
        run_campaign(out)
    elif args.stage == 'execute':
        execute(out, args.entry.resolve())
    else:
        infer_case(out, args.bank.resolve())
