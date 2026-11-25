"""Register and supervise a bounded, isolated reconstruction comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json

CONFIG = ROOT/'configs/experiments/rayst3r_compute_optimization.json'


def input_paths(reference, segmentation, case_id):
    observation = reference/'observations'/case_id
    return [observation/'rgb.png', observation/'depth_m.npy', observation/'camera.json',
            reference/segmentation/case_id/'mask.png']


def prepare(root):
    if root.exists():
        raise ValueError('Use a new registration; do not erase STOP or replay slots')
    settings = read_json(CONFIG)
    reference = ROOT/settings['reference_run']
    old = read_json(reference/'registration.json')
    cases = old['cases']
    if len(cases) != settings['case_count'] or len({c['id'] for c in cases}) != 50:
        raise ValueError('Expected the same 50 frozen observations')
    inputs = {str(p): digest for c in cases for p, digest in c['input_hashes'].items()}
    for name, digest in inputs.items():
        if sha256(name) != digest:
            raise RuntimeError('Frozen observation changed: '+name)
    warm = input_paths(reference, old['segmentation_directory'], 'warmup')
    inputs.update({str(p): sha256(p) for p in warm})
    protected = ROOT/settings['protected_campaign']
    protected_registration = read_json(protected/'registration.json')
    allowed_devices = sorted(set(protected_registration['gpu_roles'].values()))
    weights = {name: entry for name, entry in protected_registration['weights'].items()
               if Path(name).is_relative_to(ROOT/'weights/rayst3r')
               or Path(name).is_relative_to(ROOT/'weights/dinov2')}
    if len(weights) != 2:
        raise ValueError('Expected only the existing RaySt3R and DINO checkpoints')
    for name, entry in weights.items():
        if Path(name).stat().st_size != entry['size']:
            raise RuntimeError('Existing reconstruction weights changed: '+name)
    sources = {CONFIG, ROOT/'arx_project/layout.py', ROOT/'configs/path_aliases.json',
               ROOT/'modules/rayst3r/arx_adapter.py', ROOT/'modules/rayst3r/arx_reconstruction.py'}
    for folder in ['modules/rayst3r/optimized', 'modules/rayst3r/models',
                   'modules/rayst3r/utils', 'modules/rayst3r/eval_wrapper', 'modules/dinov2']:
        sources.update((ROOT/folder).rglob('*.py'))
    sources.add(ROOT/'modules/rayst3r/engine.py')
    source_hashes = {str(p): sha256(p) for p in sorted(sources)}
    root.mkdir(parents=True)
    for name in ['preflight', 'trials', 'source_registered']:
        (root/name).mkdir()
    for name in source_hashes:
        target = root/'source_registered'/Path(name).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    write_json(root/'registration.json', {
        'at': now(), 'settings': settings, 'source_hashes': source_hashes,
        'input_hashes': inputs, 'weights': weights, 'cases': cases,
        'reference': str(reference), 'segmentation': old['segmentation_directory'],
        'protected_campaign': str(protected), 'allowed_gpu_uuids': allowed_devices,
        'protected_registration_sha256': sha256(protected/'registration.json'),
        'weight_hash_origin': 'Protected registration; payload rehashed before and after GPU execution',
        'measurement_budget': 50*len(settings['variants'])*settings['repetitions'],
        'warmup_budget': len(settings['variants']), 'physical_trials': 0,
        'planning_trials': 0, 'automatic_activation': False})
    write_json(root/'progress.json', {'at': now(), 'phase': 'registered', 'completed_cases': 0,
                                      'registered_cases': 50, 'physical_trials': 0})
    print('REGISTERED_RECONSTRUCTION_COMPARISON', root, flush=True)


def guard(root, *, include_payload=False):
    registration = read_json(root/'registration.json')
    if (root/'STOP').exists():
        raise RuntimeError('STOP exists; automatic retry is disabled')
    for name, digest in {**registration['source_hashes'], **registration['input_hashes']}.items():
        if sha256(name) != digest:
            raise RuntimeError('Registered source or observation changed: '+name)
    protected = Path(registration['protected_campaign'])
    if sha256(protected/'registration.json') != registration['protected_registration_sha256']:
        raise RuntimeError('Protected campaign registration changed')
    for name, entry in registration['weights'].items():
        if Path(name).stat().st_size != entry['size'] or (include_payload and sha256(name) != entry['sha256']):
            raise RuntimeError('Registered model payload changed: '+name)
    return registration


def dependency_ready(protected):
    """The foreground experiment and its publication must finish first."""
    if not (protected/'progress.json').is_file():
        return False, 'campaign_progress_missing'
    if read_json(protected/'progress.json')['phase'] not in ['complete', 'stopped']:
        return False, 'protected_100_scene_campaign_running'
    stop_receipt, publication = protected/'services_stopped.json', protected/'publication_complete.json'
    if not stop_receipt.is_file() or not read_json(stop_receipt)['all_owned_exited']:
        return False, 'protected_components_not_confirmed_exited'
    if not publication.is_file():
        return False, 'protected_final_checks_and_publication_pending'
    if not read_json(publication)['audit_passed']:
        raise RuntimeError('Protected campaign ended with unresolved record-integrity issues')
    # Verify that a stale exit receipt cannot coexist with a resident worker.
    for ready in (protected/'resident').glob('*/*ready.json'):
        pid = read_json(ready)['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        return False, 'protected_worker_pid_still_present'
    return True, 'protected_campaign_and_publication_finished'


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    return {uuid.strip(): int(memory) for uuid, memory in
            (line.split(',') for line in result.stdout.strip().splitlines())}


def watch(root):
    reg = guard(root)
    if (root/'started.json').exists() or list((root/'trials').iterdir()):
        raise RuntimeError('Comparison already started; do not automatically repeat it')
    start = time.monotonic()
    while True:
        ready, reason = dependency_ready(Path(reg['protected_campaign']))
        inventory = gpu_inventory()
        available = sorted((inventory.get(gpu, 0), gpu) for gpu in reg['allowed_gpu_uuids']
                           if inventory.get(gpu, 0) >= reg['settings']['minimum_gpu_free_mib'])
        state = {'at': now(), 'phase': 'waiting', 'reason': reason if not ready else 'waiting_for_gpu_memory',
                 'free_mib': inventory, 'completed_cases': 0, 'registered_cases': 50, 'physical_trials': 0}
        if ready and available:
            gpu = available[-1][1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(ROOT))
            probe = subprocess.run([str(ROOT/'environments/reconstruction/bin/python'), '-c',
                'import torch; x=torch.zeros(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'],
                env=env, capture_output=True, text=True, timeout=30)
            write_json(root/'preflight/device_allocation.json', {'at': now(), 'gpu': gpu,
                'free_mib': inventory[gpu], 'passed': probe.returncode == 0,
                'stdout': probe.stdout, 'stderr': probe.stderr[-2000:]})
            if probe.returncode:
                raise RuntimeError('Selected GPU cannot allocate CUDA memory')
            guard(root, include_payload=True)
            write_json(root/'preflight/selected_gpu.json', {'at': now(), 'uuid': gpu,
                'free_mib_before_start': gpu_inventory()[gpu], 'dependency': reason})
            with (root/'worker.log').open('a') as stream:
                result = subprocess.run([str(ROOT/'environments/reconstruction/bin/python'), '-u', '-m',
                    'modules.rayst3r.optimized.worker', '--root', str(root)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Offline comparison stopped; see worker.log')
            return
        write_json(root/'progress.json', state)
        if time.monotonic()-start > reg['settings']['wait_timeout_seconds']:
            raise TimeoutError('GPU comparison waiting budget exhausted')
        time.sleep(reg['settings']['poll_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'watch'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root)
        return
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            watch(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(),
                                    'automatic_retry': False})
            previous = read_json(root/'progress.json')
            write_json(root/'progress.json', {**previous, 'at': now(), 'phase': 'stopped'})
            raise


if __name__ == '__main__':
    main()
