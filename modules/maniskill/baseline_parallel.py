"""Parallel ARX scene worlds with one serialized ZeroGrasp/cuRobo backend.

The original frozen physical pipeline remains unchanged. A worker-local wrapper
queues its backend calls through one campaign-wide flock, while other worlds can
execute their already planned motion. Completed serial slots are carried forward.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'arx_pipeline'))
import baseline
import run


def now():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


@contextmanager
def exclusive(path):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def validate_sources(registration):
    for name, digest in registration['source_hashes'].items():
        if run.sha(Path(name)) != digest:
            raise RuntimeError('Frozen source changed: ' + name)


def execute(args):
    """One scene owns one world; all its backend requests use the shared queue."""
    entry = json.loads(args.execute_entry.read_text())
    campaign = Path(entry['campaign_registration']).parent
    original_worker = run.worker

    def queued_worker(script, out, gpu, log, **flags):
        request = campaign / 'backend_requests' / (uuid.uuid4().hex + '.json')
        receipt = {'slot': entry['id'], 'script': script, 'flags': flags,
                   'gpu': gpu, 'pid': os.getpid(), 'queued_at': now(), 'state': 'queued'}
        run.save(request, receipt)
        with exclusive(campaign / 'backend.lock'):
            receipt.update(state='running', started_at=now())
            run.save(request, receipt)
            try:
                status = original_worker(script, out, gpu, log, **flags)
                receipt.update(state='complete', returncode=status)
                return status
            except BaseException:
                receipt.update(state='failed', error=traceback.format_exc())
                raise
            finally:
                receipt['finished_at'] = now()
                run.save(request, receipt)

    run.worker = queued_worker
    run.trial(entry, args.output, args.backend_gpu, args.backend_gpu, 'right_wrist')


def prepare(args):
    """Only continue a fully drained serial service with no unknown slot."""
    previous = args.previous.resolve()
    out = args.output.resolve()
    state = subprocess.check_output([
        'systemctl', '--user', 'show', 'maniskill-arx-baseline-20261116-01.service',
        '--property=ActiveState', '--value'], text=True).strip()
    if state != 'inactive':
        raise RuntimeError('Serial service must finish its active slot first: ' + state)
    if not (previous / 'STOP').exists() or not (previous / 'parallel_handover_request.json').exists():
        raise RuntimeError('Missing explicit serial-to-parallel handover receipt')
    old_registration = previous / 'registration.json'
    reg = json.loads(old_registration.read_text())
    validate_sources(reg)
    if reg['attempt_budget'] != 200 or len(reg['schedule']) != 200:
        raise RuntimeError('Original cumulative budget mismatch')
    expected = {e['id'] for e in reg['schedule']}
    carried = {}
    for path in (previous / 'trials').iterdir():
        result = path / 'result.json'
        if path.name not in expected or not result.exists():
            raise RuntimeError('Unknown/interrupted serial slot: ' + str(path))
        evaluation = path / 'independent_evaluation.json'
        if not evaluation.exists() or not json.loads(evaluation.read_text())['passed']:
            raise RuntimeError('Serial evaluation missing or disagrees: ' + str(path))
        evidence = {str(f): run.sha(f) for f in path.rglob('*') if f.is_file()}
        carried[path.name] = {'result_sha256': run.sha(result), 'evidence_sha256': evidence}
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('trials', 'admission', 'entries', 'claims', 'workers', 'backend_requests', 'source'):
        (out / folder).mkdir()
    for name in carried:
        (out / 'trials' / name).symlink_to(previous / 'trials' / name, target_is_directory=True)
    for path in (previous / 'admission').iterdir():
        if (path / 'admission.json').exists():
            (out / 'admission' / path.name).symlink_to(path, target_is_directory=True)
    for path in (previous / 'source').iterdir():
        if path.is_file():
            shutil.copy2(path, out / 'source' / path.name)
    shutil.copy2(Path(__file__), out / 'source' / Path(__file__).name)
    reg = deepcopy(reg)
    reg['source_hashes'][str(Path(__file__).resolve())] = run.sha(Path(__file__))
    reg.update(previous_registration=str(old_registration), previous_registration_sha256=run.sha(old_registration),
               scene_workers=len(args.scene_gpus), gpu={'scene_simulators': args.scene_gpus,
               'neural': args.backend_gpu, 'planner': args.backend_gpu},
               backend_concurrency=1, backend_policy='One shared flock queue for all ZeroGrasp and cuRobo calls',
               change_scope='Scheduling and GPU placement only; original trial, model, scene admission and evaluation sources unchanged')
    run.save(out / 'registration.json', reg)
    run.save(out / 'continuation.json', {'created_at': now(), 'previous_output': str(previous),
        'original_budget': 200, 'carried_completed': len(carried), 'remaining_slots': 200-len(carried),
        'serial_service_state': state, 'unknown_serial_slots': [], 'carried_slots': carried,
        'old_STOP_preserved': True, 'no_replay': True})
    shutil.copy2(previous / 'storage_policy.json', out / 'storage_policy.json')
    baseline.summary(out, reg['schedule'])
    run.save(out / 'progress.json', {'phase': 'prepared', 'completed': len(carried), 'total': 200,
                                   'scene_workers': len(args.scene_gpus), 'backend_concurrency': 1})
    print('PREPARED', len(carried), 'completed,', 200-len(carried), 'remaining', flush=True)


class Campaign:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.reg = json.loads((self.out / 'registration.json').read_text())
        if self.reg['gpu']['scene_simulators'] != args.scene_gpus or self.reg['gpu']['planner'] != args.backend_gpu:
            raise RuntimeError('Registered GPU placement changed')
        validate_sources(self.reg)
        self.mutex = threading.Lock()
        self.admission_mutex = threading.Lock()
        self.failed = threading.Event()
        self.errors = []
        self.states = {}
        self.pending = []
        for entry in self.reg['schedule']:
            trial = self.out / 'trials' / entry['id']
            claim = self.out / 'claims' / (entry['id'] + '.json')
            if (trial / 'result.json').exists():
                continue
            if trial.exists() or claim.exists():
                raise RuntimeError('Unknown/interrupted slot must never be replayed: ' + entry['id'])
            self.pending.append(entry)
        self.next_index = 0

    def progress(self, worker, **state):
        with self.mutex:
            self.states[worker] = {**self.states.get(worker, {}), **state, 'updated_at': now()}
            run.save(self.out / 'workers' / (worker + '.json'), self.states[worker])
            result = baseline.summary(self.out, self.reg['schedule'])
            run.save(self.out / 'progress.json', {'phase': 'draining_after_error' if self.failed.is_set() else 'running',
                'completed': result['completed'], 'successes': result['successes'], 'total': 200,
                'scene_workers': len(self.args.scene_gpus), 'backend_concurrency': 1,
                'workers': self.states, 'updated_at': now()})

    def resources(self, worker, gpu):
        while not self.failed.is_set() and not (self.out / 'STOP').exists():
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                          '--format=csv,noheader,nounits'], text=True)
            free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in raw.splitlines()}
            if free.get(gpu, 0) >= 3500 and free.get(self.args.backend_gpu, 0) >= 8500:
                return True
            self.progress(worker, phase='waiting_for_resources', gpu=gpu, free_mib=free)
            self.failed.wait(15)
        return False

    def admit(self, entry, env):
        # Serialize admission receipts so same-target duplicate checks see every prior scene.
        with self.admission_mutex:
            out = self.out / 'admission' / entry['id']
            if (out / 'admission.json').exists():
                return json.loads((out / 'admission.json').read_text())['entry']
            out.mkdir(exist_ok=False)
            entry = deepcopy(entry)
            prior = []
            for receipt in (self.out / 'admission').glob('*/admission.json'):
                d = json.loads(receipt.read_text())
                if d['passed'] and d['entry']['model_id'] == entry['model_id']:
                    prior.append(d['entry']['scene'])
            entry['avoid_scenes'] = prior
            run.save(out / 'entry.json', entry)
            errors = []
            for variant in range(12):
                if self.failed.is_set() or (self.out / 'STOP').exists():
                    raise RuntimeError('Admission stopped before scene execution')
                destination = out / f'variant_{variant:02d}'
                cmd = [sys.executable, str(ROOT / 'arx_pipeline/baseline.py'), '--output', str(destination),
                       '--admit-entry', str(out / 'entry.json'), '--variant', str(variant),
                       '--neural-gpu', 'unused', '--planner-gpu', 'unused']
                with (out / f'variant_{variant:02d}.log').open('x') as stream:
                    subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
                receipt = destination / 'admission.json'
                if not receipt.exists():
                    raise RuntimeError('Admission infrastructure exited without receipt: ' + str(destination))
                d = json.loads(receipt.read_text())
                if not d['passed'] and any(any(word in e['error'].lower() for word in
                    ['out of memory', 'supported physical device', 'vk_error', 'resource descriptor'])
                    for e in d['rejected_variants']):
                    raise RuntimeError('GPU/render infrastructure failure: ' + str(destination))
                if d['passed']:
                    for name in ('wrist.png', 'overview.png', 'observation_probe_state.json'):
                        shutil.copy2(destination / name, out / name)
                    d['prior_variants'] = errors
                    run.save(out / 'admission.json', d)
                    return d['entry']
                errors += d['rejected_variants']
            failed = {**entry, 'admission_failed': True, 'admission_errors': errors}
            run.save(out / 'admission.json', {'passed': False, 'entry': failed, 'rejected_variants': errors})
            return failed

    def worker(self, index, gpu):
        worker = f'scene_{index}'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'MANISKILL_GPU_UUID': gpu}
        try:
            while not self.failed.is_set() and not (self.out / 'STOP').exists():
                if not self.resources(worker, gpu):
                    break
                with self.mutex:
                    if self.next_index >= len(self.pending):
                        break
                    entry = deepcopy(self.pending[self.next_index])
                    self.next_index += 1
                    claim = {'slot': entry['id'], 'worker': worker, 'gpu': gpu, 'claimed_at': now(),
                             'state': 'claimed', 'controller_pid': os.getpid()}
                    with (self.out / 'claims' / (entry['id'] + '.json')).open('x') as stream:
                        json.dump(claim, stream, indent=2)
                validate_sources(self.reg)
                if shutil.disk_usage(self.out).free <= 1073741824:
                    raise RuntimeError('Storage reserve reached')
                self.progress(worker, phase='admission', current=entry['id'], gpu=gpu)
                entry = self.admit(entry, env)
                if self.failed.is_set() or (self.out / 'STOP').exists():
                    break
                validate_sources(self.reg)
                trial = self.out / 'trials' / entry['id']
                self.progress(worker, phase='executing', current=entry['id'], gpu=gpu)
                if entry.get('admission_failed'):
                    trial.mkdir()
                    run.save(trial / 'result.json', {**entry, 'success': False, 'physical_attempt': False,
                                                   'planning_success': False, 'reason': 'scene_admission_failed'})
                else:
                    entry['campaign_registration'] = str(self.out / 'registration.json')
                    entry['parallel_worker'] = worker
                    receipt = self.out / 'entries' / (entry['id'] + '.json')
                    run.save(receipt, entry)
                    cmd = [sys.executable, str(Path(__file__).resolve()), '--output', str(trial),
                           '--execute-entry', str(receipt), '--backend-gpu', self.args.backend_gpu]
                    with receipt.with_suffix('.log').open('x') as stream:
                        child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        self.progress(worker, phase='executing', current=entry['id'], gpu=gpu, child_pid=child.pid)
                        child.wait(timeout=3600)
                    if not (trial / 'result.json').exists():
                        raise RuntimeError('Unknown execution outcome; never replay: ' + str(trial))
                baseline.verify_trial(trial)
                claim.update(state='complete', completed_at=now(), result_sha256=run.sha(trial / 'result.json'))
                run.save(self.out / 'claims' / (entry['id'] + '.json'), claim)
                self.progress(worker, phase='between_scenes', current=None, child_pid=None)
                result = json.loads((trial / 'result.json').read_text())
                print('RESULT', worker, entry['id'], result['success'], result.get('reason'), flush=True)
        except BaseException:
            self.failed.set()
            error = {'worker': worker, 'time': now(), 'error': traceback.format_exc()}
            self.errors.append(error)
            run.save(self.out / 'workers' / (worker + '_halt.json'), error)
            self.progress(worker, phase='halted', error=error['error'])
        finally:
            if not self.failed.is_set():
                self.progress(worker, phase='idle')

    def start(self):
        with (self.out / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.out / 'STOP').exists():
                raise RuntimeError('Campaign stopped')
            with ThreadPoolExecutor(max_workers=len(self.args.scene_gpus)) as pool:
                futures = [pool.submit(self.worker, i, gpu) for i, gpu in enumerate(self.args.scene_gpus)]
                for future in futures:
                    future.result()
            if self.failed.is_set():
                run.save(self.out / 'halt.json', {'time': now(), 'errors': self.errors})
                raise RuntimeError('Parallel scene campaign halted; unknown slots are preserved')
            result = baseline.summary(self.out, self.reg['schedule'])
            phase = 'complete' if result['completed'] == 200 else 'stopped'
            run.save(self.out / 'progress.json', {'phase': phase, 'completed': result['completed'],
                     'successes': result['successes'], 'total': 200, 'updated_at': now()})
            if phase == 'complete':
                (self.out / 'STOP').write_text('Original cumulative 200 slots completed. No retries.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--scene-gpus', nargs='+')
    parser.add_argument('--backend-gpu', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute-entry', type=Path)
    args = parser.parse_args()
    if args.execute_entry:
        execute(args)
    elif args.prepare:
        prepare(args)
    else:
        Campaign(args).start()


if __name__ == '__main__':
    main()
