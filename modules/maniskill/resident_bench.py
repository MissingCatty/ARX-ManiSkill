"""Continue the fixed ARX benchmark using benchmark-owned resident backends."""
from arx_project.layout import canonical_path
from copy import deepcopy
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import graspcl_batch as batch
from graspcl_pipeline import read, save, sha
from graspcl_execute import guard
from resident_backend import rpc


def prepare(out, previous):
    validation = CODE / 'outputs/arx_resident_backend_validation_20261117_03'
    if not read(validation / 'planner_validation.json')['passed'] or not read(validation / 'neural_validation.json')['passed']:
        raise RuntimeError('Resident backend validation is incomplete')
    validation_registration = read(validation / 'registration.json')
    for name in ['resident_backend.py', 'resident_planner.py']:
        path = CODE / name
        if sha(path) != validation_registration['source_sha256'][str(path.resolve())]:
            raise RuntimeError('Validated resident implementation changed: ' + name)
    reg = read(previous / 'registration.json')
    guard(reg)
    if read(previous / 'progress.json')['phase'] != 'stopped' or not (previous / 'resident_handover_request.json').exists():
        raise RuntimeError('Previous batch must drain under the resident handover request')
    service = subprocess.check_output(['systemctl', '--user', 'show', 'maniskill-arx-graspcl-200-20261117-01.service',
                                       '--property=ActiveState', '--value'], text=True).strip()
    if service != 'inactive':
        raise RuntimeError('Previous benchmark unit is not fully inactive')
    owned = subprocess.run(['pgrep', '-af', str(CODE / 'graspcl_batch.py')], capture_output=True, text=True)
    if owned.returncode == 0:
        raise RuntimeError('Previous benchmark processes still present: ' + owned.stdout)
    completed = {}
    for path in (previous / 'claims').glob('*.json'):
        claim = read(path)
        trial = previous / 'trials' / path.stem
        if claim['state'] != 'complete' or not (trial / 'independent_evaluation.json').exists() or not read(trial / 'independent_evaluation.json')['passed']:
            raise RuntimeError('Unknown slot must not be replayed: ' + path.stem)
        if sha(trial / 'result.json') != claim['result_sha256']:
            raise RuntimeError('Previous result changed: ' + path.stem)
        completed[path.stem] = claim
    if any(t.name not in completed for t in (previous / 'trials').iterdir()):
        raise RuntimeError('Previous trial folder lacks terminal claim')
    for path in (previous / 'backend_requests').glob('*.json'):
        if read(path)['state'] in ['queued', 'running']:
            raise RuntimeError('Old backend request is not terminal')
    out.mkdir(parents=True, exist_ok=False)
    for name in ['trials', 'claims', 'entries', 'workers', 'backend_requests', 'source', 'reference']:
        (out / name).mkdir()
    for slot, claim in completed.items():
        (out / 'trials' / slot).symlink_to(previous / 'trials' / slot, target_is_directory=True)
        save(out / 'claims' / (slot + '.json'), claim)
        for suffix in ['.json', '_binding.json']:
            path = previous / 'entries' / (slot + suffix)
            if path.exists():
                shutil.copy2(path, out / 'entries' / path.name)
    shutil.copy2(previous / 'reference/common_initial_conditions.json', out / 'reference/common_initial_conditions.json')
    shutil.copy2(previous / 'storage_policy.json', out / 'storage_policy.json')
    new = deepcopy(reg)
    new.update(previous_output=str(previous), previous_registration_sha256=sha(previous / 'registration.json'),
        resident_validation=str(validation),
        runtime_change='User-authorized resident ZeroGrasp and open/closed cuRobo models; load and warm up before claiming scenes. Original algorithm, candidate pool, state/reset, model, collision and success semantics retained.',
        resident_backends=True, resident_mesh_capacity=8, resident_primitive_capacity=8,
        gpu_backend_free_mib_floor=2000,
        backend_lifecycle='Start both subprocesses in the benchmark control group, await ready/warmup, use them until benchmark stops, then shut down. A backend failure stops the batch; no transparent replay.',
        numerical_scope='Persistent CUDA execution can retain ordinary floating-point variation. Unmodified demo decoder is used; no output candidate is substituted or reused from an old observation.')
    for path in [Path(__file__), CODE / 'resident_backend.py', CODE / 'resident_planner.py']:
        new['source_hashes'][str(path.resolve())] = sha(path)
        shutil.copy2(path, out / 'source' / path.name)
    new['source_hashes'][str(out / 'reference/common_initial_conditions.json')] = sha(out / 'reference/common_initial_conditions.json')
    save(out / 'registration.json', new)
    save(out / 'continuation.json', {'created_at': datetime.now().astimezone().isoformat(),
        'previous': str(previous), 'original_budget': 200, 'carried_completed': len(completed),
        'remaining': 200 - len(completed), 'old_service': service, 'unknown_slots': [],
        'old_STOP_preserved': True, 'no_completed_or_started_slot_replayed': True,
        'algorithm_configuration_change': False, 'backend_lifecycle_change': True,
        'carried_result_sha256': {slot: c['result_sha256'] for slot, c in completed.items()}})
    _, baseline = batch.original_modules()
    baseline.summary(out, new['schedule'])
    save(out / 'progress.json', {'phase': 'prepared', 'completed': len(completed), 'total': 200,
                                'scene_workers': 2, 'backend_concurrency': 1})
    print('RESIDENT CONTINUATION PREPARED', len(completed), 'carried;', 200-len(completed), 'remaining', flush=True)


class Backends:
    def __init__(self, campaign):
        self.campaign = campaign
        self.reg = read(campaign / 'registration.json')
        self.children = {}
        self.streams = []
        self.config = {}

    def start(self):
        root = self.campaign / 'resident'
        root.mkdir(exist_ok=False)
        key = hashlib.sha256(str(self.campaign).encode()).hexdigest()[:12]
        sockets = CODE / 'tmp' / ('resident_' + key)
        sockets.mkdir(mode=0o700, exist_ok=False)
        warmup = CODE / 'outputs/arx_zerograsp_graspcl_physical_20261117_01/trials/B0069_S3_sparse_014_lemon'
        self.config = {'gpu': batch.BACKEND, 'owner_pid': os.getpid(), 'warmup_trial': str(warmup),
                       'sockets': {kind: str(sockets / (kind + '.sock')) for kind in ['curobo', 'zerograsp']}}
        save(root / 'backend_config.json', self.config)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=batch.BACKEND, ARX_NEURAL_GPU=batch.BACKEND,
                   ARX_PLANNER_GPU=batch.BACKEND, ARX_SIM_GPU=batch.BACKEND, LILO_GPU=batch.BACKEND,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH', None)
        for kind in ['curobo', 'zerograsp']:
            python = canonical_path(batch.ROOT / ('curobo/.venv/bin/python' if kind == 'curobo' else 'tmp/arx-fullchain/env/bin/python'))
            command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'resident-worker',
                str(batch.ROOT / 'arx_core/scripts/curobo_arx/environment.sh'), str(python),
                str(CODE / 'resident_backend.py'), '--kind', kind, '--socket', self.config['sockets'][kind],
                '--root', str(root), '--warmup-trial', str(warmup)]
            stream = (root / (kind + '.log')).open('x')
            self.streams.append(stream)
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.children[kind] = process
            deadline = time.monotonic() + 240
            while not Path(self.config['sockets'][kind]).exists():
                if process.poll() is not None:
                    raise RuntimeError('Resident backend startup failed: ' + kind)
                if time.monotonic() > deadline:
                    raise RuntimeError('Resident backend startup timed out: ' + kind)
                time.sleep(.2)
            health = rpc(Path(self.config['sockets'][kind]), {'command': 'health'})
            if health['status'] != 'ready':
                raise RuntimeError('Resident backend is not ready')
            print('RESIDENT READY', kind, health['pid'], health['startup_seconds'], flush=True)
        save(root / 'all_ready.json', {'ready_at': datetime.now().astimezone().isoformat(),
                                     'pids': {k: p.pid for k, p in self.children.items()}})

    def stop(self):
        for kind, process in self.children.items():
            path = Path(self.config['sockets'][kind])
            if process.poll() is None:
                try:
                    rpc(path, {'command': 'shutdown'}, timeout=10)
                    process.wait(timeout=20)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        for stream in self.streams:
            stream.close()


class ResidentCampaign(batch.Campaign):
    def resources(self, worker, gpu):
        while not self.failed.is_set() and not (self.out / 'STOP').exists():
            if shutil.disk_usage(self.out).free <= 1073741824:
                raise RuntimeError('Storage reserve reached')
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
            free = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in raw.splitlines()}
            if free.get(gpu, 0) >= 3500 and free.get(batch.BACKEND, 0) >= self.reg['gpu_backend_free_mib_floor']:
                return True
            self.progress(worker, phase='waiting_for_resources', gpu=gpu, free_mib=free)
            self.failed.wait(15)
        return False


def execute(trial, entry):
    run, _ = batch.original_modules()
    reg_path = Path(read(entry)['campaign_registration'])
    config = read(reg_path.parent / 'resident/backend_config.json')

    def resident_worker(script, directory, gpu, log, **flags):
        guard_receipt = read(directory / 'source_guard.json')
        if sha(Path(guard_receipt['registration'])) != guard_receipt['sha256']:
            raise RuntimeError('Campaign registration changed')
        guard(read(reg_path))
        kind = 'zerograsp' if script == 'neural.py' else 'curobo'
        message = {'directory': str(directory)}
        if kind == 'curobo':
            message.update(flags, selector='graspcl')
        response = rpc(Path(config['sockets'][kind]), message, timeout=900)
        with log.open('x') as stream:
            json.dump(response, stream, indent=2)
        if response['status'] != 'ok':
            save(directory / 'resident_request_error.json', response)
            raise RuntimeError('resident_backend_failed: ' + response.get('error', 'unknown'))
        return response['returncode']

    run.worker = resident_worker
    batch.execute(trial, entry)
    if (trial / 'resident_request_error.json').exists():
        raise RuntimeError('Resident integration error; terminal evidence retained, do not replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'infer'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    # Reuse the frozen controller/GraspCL implementation, routing only its child
    # dispatch to this wrapper. No original source file is edited.
    batch.__file__ = str(Path(__file__).resolve())
    if args.stage == 'prepare':
        prepare(out, args.previous.resolve())
    elif args.stage == 'execute':
        execute(out, args.entry.resolve())
    elif args.stage == 'infer':
        batch.infer_case(out, args.campaign.resolve())
    else:
        guard(read(out / 'registration.json'))
        backends = Backends(out)
        try:
            backends.start()
            ResidentCampaign(out).start()
        finally:
            backends.stop()
