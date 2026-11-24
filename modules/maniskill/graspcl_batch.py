"""Matched 200-scene ARX benchmark with a frozen, per-case-excluded GraspCL bank.

Separate scene processes share the original ZeroGrasp/cuRobo backend lock.
Historical scene admissions are bound only after their baseline slot is terminal.
"""
from arx_project.layout import canonical_path
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback
import uuid

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from graspcl_pipeline import ROOT, BASELINE, read, save, sha
from graspcl_execute import locked, guard

BANK = CODE / 'outputs/arx_zerograsp_graspcl_demo_20261117_02'
SCENES = ['GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169', 'GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e']
BACKEND = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'


def now():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def original_modules():
    sys.path.insert(0, str(CODE / 'arx_pipeline'))
    import run
    import baseline
    return run, baseline


def prepare(out):
    source = read(BASELINE / 'registration.json')
    bankreg = read(BANK / 'registration.json')
    complete = read(BANK / 'bank_complete.json')
    assert len(source['schedule']) == source['attempt_budget'] == 200
    assert len({(e['model_id'], e['layout']) for e in source['schedule']}) == 200
    assert len({e['model_id'] for e in source['schedule']}) == 40
    hashes = {**bankreg['source_hashes'], **complete['model_sha256']}
    for path in [Path(__file__), CODE / 'graspcl_execute.py', CODE / 'graspcl_planner.py',
                 CODE / 'grasp_visuals.py', BANK / 'registration.json', BANK / 'bank_complete.json',
                 BASELINE / 'registration.json']:
        hashes[str(path.resolve())] = sha(path)
    guard({'source_hashes': hashes})
    references = []
    for e in source['schedule']:
        path = BASELINE / 'trials' / e['id'] / 'initial_conditions.json'
        if path.is_file():
            references.append((path, read(path)))
    if not references:
        raise RuntimeError('No historical initial-state receipt')
    common = references[0][1]
    if any(r['initial_q'] != common['initial_q'] or r['base_pose'] != common['base_pose'] for _, r in references):
        raise RuntimeError('Historical cases do not share a common declared robot reset')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ['trials', 'claims', 'entries', 'workers', 'backend_requests', 'source', 'reference']:
        (out / folder).mkdir()
    shutil.copytree(BANK / 'model', out / 'frozen_bank/model')
    for name in ['registration.json', 'bank_complete.json', 'runtime_receipt.json']:
        shutil.copy2(BANK / name, out / 'frozen_bank' / name)
    hashes.update({str(p.resolve()): sha(p) for p in (out / 'frozen_bank').rglob('*') if p.is_file()})
    shutil.copy2(references[0][0], out / 'reference/common_initial_conditions.json')
    hashes[str(out / 'reference/common_initial_conditions.json')] = sha(out / 'reference/common_initial_conditions.json')
    # The historical contract directory is absent; preserve the active baseline's
    # explicit AGENTS-derived 1 GiB policy and its missing-snapshot disclosure.
    storage = BASELINE / 'storage_policy.json'
    if read(storage)['minimum_free_bytes'] != 1073741824:
        raise RuntimeError('Storage policy differs from the declared 1 GiB reserve')
    shutil.copy2(storage, out / 'storage_policy.json')
    registration = {
        'created_at': now(), 'chain': bankreg['chain'], 'attempt_budget': 200,
        'storage_policy_source': str(storage),
        'schedule': source['schedule'], 'layouts': source['layouts'], 'one_attempt_only': True,
        'source_baseline': str(BASELINE), 'source_registration_sha256': sha(BASELINE / 'registration.json'),
        'bank': str(out / 'frozen_bank'), 'source_bank': str(BANK),
        'source_hashes': hashes, 'candidate_policy': bankreg['candidate_policy'],
        'success': source['success'], 'table_height_m': source['table_height_m'],
        'camera': source['camera'], 'recording_camera': source['recording_camera'],
        'gpu': {'scene_simulators': SCENES, 'neural': BACKEND, 'planner': BACKEND, 'graspcl': 'CPU, 4 threads per case'},
        'scene_workers': 2, 'backend_concurrency': 1, 'backend_lock': str(BASELINE / 'backend.lock'),
        'bank_frozen': True, 'online_updates_during_evaluation': False,
        'self_case_exclusion': 'Exclude this trial ID from both online scoring memory and object/patch recall templates, in memory only; other cases retain the same frozen bank. Applies to all 25 labelled source cases. No held-out outcome enters inference.',
        'source_feedback_ids_in_memory_order': complete['source_ids'],
        'scene_binding': 'Same 200 frozen allocations. Copy each original terminal slot admission scene and registered observation q; never generate a replacement scene. Wait for the original baseline to bind future scenes. Original admission failures remain explicit non-executed slots.',
        'observation': 'Fresh wrist RGB-D after the historical registered observation motion. Existing instance mask and historical GT bbox-based observation protocol remain disclosed limitations; no new viewpoint search.',
        'selection_change': 'Same smoke-tested direct canonical pose mapping and GraspCL posterior-mean ordering. Original 40mm EEF offset and downward-approach gate retained; baseline GT center radius and height shift omitted.',
        'frozen': 'Original robot, calibration, camera, scene initial conditions, physics, cuRobo model/collision/optimizer, and success gates',
        'historical_runtime_missing': 'If a baseline slot failed before initial_conditions.json, validate against the common declared robot reset and exact admitted scene; report unavailable historical runtime evidence explicitly.',
        'common_reset_source': str(references[0][0]),
        'common_reset_matching_historical_receipts_at_registration': len(references),
        'prior_two_case_smoke_test': 'Separate completed pilot; neither pilot result is reused or counted toward this new 200-slot budget.',
        'unknown_policy': 'Claim each slot exactly once. Never replay started or unknown slots; stop and preserve evidence on integrity/infrastructure errors.',
        'scope': 'Matched 200-case benchmark with per-case exclusion. Same object models occur in bank and evaluation; this is not unseen-object generalization or an isolated scoring-only ablation.',
    }
    save(out / 'registration.json', registration)
    for name in ['graspcl_batch.py', 'graspcl_execute.py', 'graspcl_pipeline.py', 'graspcl_planner.py', 'grasp_visuals.py']:
        shutil.copy2(CODE / name, out / 'source' / name)
    _, baseline = original_modules()
    baseline.summary(out, registration['schedule'])
    save(out / 'progress.json', {'phase': 'prepared', 'completed': 0, 'total': 200,
                                'successes': 0, 'scene_workers': 2, 'backend_concurrency': 1})
    print(json.dumps({'prepared': str(out), 'slots': 200, 'models': 40, 'per_layout': 40,
                      'bank_positive': complete['positive_demonstrations'],
                      'bank_negative': complete['negative_feedback']}), flush=True)


def exclude_self(module, source_ids, slot):
    """Filter only in-memory state. Never mutate the shared frozen bank."""
    from sklearn.neighbors import KDTree
    scorer = module.grasp_scoring_module_cl
    if len(source_ids) != len(scorer.X_on):
        raise RuntimeError('Online feedback provenance ordering mismatch')
    keep = np.array([identity != slot for identity in source_ids], dtype=bool)
    before = {'feedback': len(scorer.X_on), 'object': len(module.grasp_recall_module_object._db),
              'patch': len(module.grasp_recall_module_patch._db)}
    scorer.X_on, scorer.y_on, scorer.w_on = scorer.X_on[keep], scorer.y_on[keep], scorer.w_on[keep]
    scorer.tree_on = KDTree(scorer.X_on) if len(scorer.X_on) else None
    for recall in [module.grasp_recall_module_object, module.grasp_recall_module_patch]:
        recall._db = [entry for entry in recall._db
                      if all(g.get('metadata', {}).get('source_trial_id') != slot for g in entry['grasps'])]
    after = {'feedback': len(scorer.X_on), 'object': len(module.grasp_recall_module_object._db),
             'patch': len(module.grasp_recall_module_patch._db)}
    return {'query_id': slot, 'excluded_feedback_ids': [x for x in source_ids if x == slot],
            'retained_feedback_ids': [x for x in source_ids if x != slot], 'before': before, 'after': after,
            'retained_feature_sha256': hashlib.sha256(scorer.X_on.tobytes()).hexdigest(),
            'bank_mutated': False, 'application': 'in-memory filtering before any recall or scoring'}


def infer_case(trial, campaign):
    from graspcl_pipeline import seed, load_model, infer, write_inference
    reg = read(campaign / 'registration.json')
    guard(reg)
    seed()
    # Upstream loading writes scoring metadata; isolate these writes per trial.
    runtime = trial / 'graspcl_runtime'
    shutil.copytree(Path(reg['bank']) / 'model', runtime / 'model')
    module = load_model(runtime)
    exclusion = exclude_self(module, reg['source_feedback_ids_in_memory_order'], trial.name)
    save(trial / 'self_case_exclusion.json', exclusion)
    result = infer(module, trial, reg['candidate_policy'])
    for candidate in result['recalled_before_filter']:
        if candidate['metadata'].get('source_trial_id') == trial.name:
            raise RuntimeError('Self demonstration leaked into the query')
    destination = trial / 'graspcl'
    destination.mkdir(exist_ok=False)
    (destination / 'zg_outputs').mkdir()
    write_inference(result, trial, destination)
    print(json.dumps({'counts': result['counts'], 'self_case_exclusion': exclusion}), flush=True)


def execute(trial, entry_path):
    run, baseline = original_modules()
    entry = read(entry_path)
    campaign = Path(entry['campaign_registration']).parent
    reg = read(entry['campaign_registration'])
    guard(reg)
    binding = read(campaign / 'entries' / (entry['id'] + '_binding.json'))
    if sha(entry_path) != binding['entry_sha256']:
        raise RuntimeError('Bound trial entry changed')
    for path, digest in binding['source_evidence_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Historical scene evidence changed: ' + path)
    base_worker = run.worker
    initial = read(campaign / 'reference/common_initial_conditions.json')
    historical = Path(entry['historical_trial'])
    historical_initial = read(historical / 'initial_conditions.json') if (historical / 'initial_conditions.json').is_file() else None

    def worker(script, directory, gpu, log, **flags):
        guard(reg)
        if flags.get('observe'):
            state = read(directory / 'state.json')
            expected = historical_initial or initial
            error = float(np.max(np.abs(np.array(state['qpos'])-np.array(expected['initial_q']))))
            if error >= .005:
                raise RuntimeError('Declared historical robot reset differs')
            save(directory / 'reset_preflight.json', {'joint_error_rad': error,
                'historical_runtime_available': historical_initial is not None,
                'reference': str(historical / 'initial_conditions.json') if historical_initial else reg['common_reset_source']})
        request = campaign / 'backend_requests' / (uuid.uuid4().hex + '.json')
        receipt = {'slot': entry['id'], 'script': script, 'flags': flags, 'gpu': gpu,
                   'state': 'queued', 'queued_at': now(), 'pid': os.getpid()}
        save(request, receipt)
        with locked(reg['backend_lock']):
            receipt.update(state='running', started_at=now())
            save(request, receipt)
            try:
                actual = str(CODE / 'graspcl_planner.py') if script == 'planner.py' and not any(flags.values()) else script
                status = base_worker(actual, directory, gpu, log, **flags)
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
            command = [str(canonical_path(ROOT / 'tmp/arx-graspcl-env/bin/python')), str(Path(__file__)), 'infer',
                       '--output', str(directory), '--campaign', str(campaign)]
            with (directory / 'graspcl_inference.log').open('x') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
            from grasp_visuals import render
            render(directory / 'graspcl', directory / 'zg_outputs')
        return status

    run.worker = worker
    run.trial(entry, trial, BACKEND, BACKEND, 'right_wrist')
    result = read(trial / 'result.json')
    infrastructure_words = ['out of memory', 'vk_error', 'resource descriptor',
                            'frozen source', 'declared historical robot reset differs']
    if result['reason'] == 'zerograsp_failed' or any(word in result.get('error', '').lower() for word in infrastructure_words):
        raise RuntimeError('Infrastructure/integrity failure preserved: ' + entry['id'])
    if (trial / 'graspcl_inference.log').exists() and not (trial / 'zg_outputs/grasp_top10.json').exists():
        raise RuntimeError('GraspCL inference/visualization integration failed; preserve slot')
    if (trial / 'initial_conditions.json').exists():
        current = read(trial / 'initial_conditions.json')
        reference = historical_initial or initial
        comparison = {'historical_runtime_available': historical_initial is not None,
            'base_pose_equal': reference['base_pose'] == current['base_pose'],
            'initial_q_max_difference': float(np.max(np.abs(np.array(reference['initial_q'])-np.array(current['initial_q'])))),
            'spawn_equal': historical_initial['spawn'] == current['spawn'] if historical_initial else None,
            'observation_equal_to_source_admission': entry['observation'] == read(Path(entry['source_admission']))['entry']['observation']}
        save(trial / 'reset_comparison.json', comparison)
        if not comparison['base_pose_equal'] or comparison['initial_q_max_difference'] != 0. or comparison['spawn_equal'] is False or not comparison['observation_equal_to_source_admission']:
            raise RuntimeError('Matched reset check failed')
    baseline.verify_trial(trial)


class Campaign:
    def __init__(self, out):
        self.out = out
        self.reg = read(out / 'registration.json')
        guard(self.reg)
        self.mutex = threading.Lock()
        self.failed = threading.Event()
        self.states, self.errors = {}, []
        self.claimed = {p.stem for p in (out / 'claims').glob('*.json')}
        for slot in self.claimed:
            if not (out / 'trials' / slot / 'independent_evaluation.json').is_file():
                raise RuntimeError('Started/unknown slot must not be replayed: ' + slot)

    def progress(self, worker, **state):
        with self.mutex:
            self.states[worker] = {**self.states.get(worker, {}), **state, 'updated_at': now()}
            save(self.out / 'workers' / (worker + '.json'), self.states[worker])
            _, baseline = original_modules()
            summary = baseline.summary(self.out, self.reg['schedule'])
            save(self.out / 'progress.json', {'phase': 'draining_after_error' if self.failed.is_set() else 'running',
                'completed': summary['completed'], 'successes': summary['successes'], 'total': 200,
                'scene_workers': 2, 'backend_concurrency': 1, 'workers': self.states, 'updated_at': now()})

    def next_entry(self, worker, gpu):
        with self.mutex:
            for scheduled in self.reg['schedule']:
                slot = scheduled['id']
                if slot in self.claimed:
                    continue
                historical = BASELINE / 'trials' / slot
                admission = BASELINE / 'admission' / slot / 'admission.json'
                if not (historical / 'result.json').is_file() or not (historical / 'independent_evaluation.json').is_file() or not admission.is_file():
                    continue
                if not read(historical / 'independent_evaluation.json')['passed']:
                    raise RuntimeError('Baseline independent evaluation disagrees: ' + slot)
                bound = read(admission)
                source_entry = bound['entry']
                if any(source_entry[k] != scheduled[k] for k in ['id', 'model_id', 'layout', 'seed']):
                    raise RuntimeError('Matched source allocation changed')
                entry = deepcopy(source_entry)
                entry.update(campaign_registration=str(self.out / 'registration.json'),
                    historical_trial=str(historical), source_admission=str(admission),
                    original_success=read(historical / 'result.json')['success'],
                    admission_failed=not bound['passed'], parallel_worker=worker)
                evidence = [admission, historical / 'result.json', historical / 'independent_evaluation.json']
                if (historical / 'initial_conditions.json').exists():
                    evidence.append(historical / 'initial_conditions.json')
                receipt = self.out / 'entries' / (slot + '.json')
                save(receipt, entry)
                save(self.out / 'entries' / (slot + '_binding.json'), {
                    'bound_at': now(), 'source_evidence_sha256': {str(f): sha(f) for f in evidence},
                    'historical_runtime_available': (historical / 'initial_conditions.json').exists(),
                    'entry_sha256': sha(receipt),
                    'scene_and_observation_copied_exactly': True})
                claim = {'slot': slot, 'worker': worker, 'gpu': gpu, 'state': 'claimed', 'claimed_at': now()}
                with (self.out / 'claims' / (slot + '.json')).open('x') as stream:
                    json.dump(claim, stream, indent=2)
                self.claimed.add(slot)
                return entry, claim
        return None, None

    def resources(self, worker, gpu):
        while not self.failed.is_set() and not (self.out / 'STOP').exists():
            if shutil.disk_usage(self.out).free <= 1073741824:
                raise RuntimeError('Storage reserve reached')
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
            free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in raw.splitlines()}
            if free.get(gpu, 0) >= 3500 and free.get(BACKEND, 0) >= 8500:
                return True
            self.progress(worker, phase='waiting_for_resources', gpu=gpu, free_mib=free)
            self.failed.wait(15)
        return False

    def worker(self, index, gpu):
        worker = f'scene_{index}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MANISKILL_GPU_UUID=gpu, ARX_SIM_GPU=gpu,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        _, baseline = original_modules()
        try:
            while not self.failed.is_set() and not (self.out / 'STOP').exists():
                if not self.resources(worker, gpu):
                    break
                guard(self.reg)
                entry, claim = self.next_entry(worker, gpu)
                if entry is None:
                    if len(self.claimed) == 200:
                        break
                    source_progress = read(BASELINE / 'progress.json')
                    self.progress(worker, phase='waiting_for_historical_scene', gpu=gpu,
                                  original_completed=source_progress.get('completed'))
                    self.failed.wait(20)
                    continue
                slot = entry['id']
                trial = self.out / 'trials' / slot
                self.progress(worker, phase='executing', current=slot, gpu=gpu)
                if entry['admission_failed']:
                    trial.mkdir(exist_ok=False)
                    save(trial / 'result.json', {**entry, 'success': False, 'physical_attempt': False,
                        'planning_success': False, 'reason': 'matched_source_scene_admission_failed'})
                else:
                    cmd = ['/bin/bash', str(CODE / 'run.sh'), str(Path(__file__)), 'execute',
                           '--output', str(trial), '--entry', str(self.out / 'entries' / (slot + '.json'))]
                    with (self.out / 'entries' / (slot + '.log')).open('x') as stream:
                        child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        self.progress(worker, current=slot, phase='executing', gpu=gpu, child_pid=child.pid)
                        child.wait()
                    if child.returncode != 0:
                        raise RuntimeError('Execution/verification subprocess failed; preserve slot: ' + slot)
                    if not (trial / 'result.json').is_file():
                        raise RuntimeError('Unknown execution outcome; never replay: ' + slot)
                baseline.verify_trial(trial)
                result = read(trial / 'result.json')
                if 'source changed' in result.get('reason', '').lower() or 'frozen source' in result.get('reason', '').lower():
                    raise RuntimeError('Integrity failure preserved in trial: ' + slot)
                guard(self.reg)
                claim.update(state='complete', finished_at=now(), result_sha256=sha(trial / 'result.json'))
                save(self.out / 'claims' / (slot + '.json'), claim)
                self.progress(worker, phase='between_scenes', current=None, child_pid=None)
                print('RESULT', worker, slot, result['success'], result['reason'], flush=True)
        except BaseException:
            self.failed.set()
            error = {'worker': worker, 'time': now(), 'error': traceback.format_exc()}
            self.errors.append(error)
            save(self.out / 'workers' / (worker + '_halt.json'), error)
            self.progress(worker, phase='halted', error=error['error'])
        finally:
            if not self.failed.is_set():
                self.progress(worker, phase='idle', current=None, child_pid=None)

    def start(self):
        with (self.out / 'campaign.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.out / 'STOP').exists():
                raise RuntimeError('Explicit STOP preserved')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.worker, i, gpu) for i, gpu in enumerate(SCENES)]
                for future in futures:
                    future.result()
            if self.failed.is_set():
                save(self.out / 'halt.json', {'time': now(), 'errors': self.errors})
                raise RuntimeError('Campaign halted; do not replay claimed slots')
            _, baseline = original_modules()
            summary = baseline.summary(self.out, self.reg['schedule'])
            save(self.out / 'progress.json', {'phase': 'complete' if summary['completed'] == 200 else 'stopped',
                 'completed': summary['completed'], 'successes': summary['successes'], 'total': 200, 'updated_at': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'infer'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage == 'prepare':
        prepare(out)
    elif args.stage == 'run':
        Campaign(out).start()
    elif args.stage == 'execute':
        execute(out, args.entry.resolve())
    else:
        infer_case(out, args.campaign.resolve())
