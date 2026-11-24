"""Register and execute bounded V3 dual-arm integration trials."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
from itertools import permutations
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from arx_project.client import Client
from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from arx_project.registry import module_records
from arx_project.scenes import current_dataset, current_schedule

HERE = Path(__file__).resolve().parent
EXPERIMENT = ROOT/'configs/experiments/sam3_graspgenx_v3.json'
COMPONENTS = ['sam3', 'rayst3r', 'graspgenx', 'curobo_dual_arm']


def imports():
    sys.path[:0] = [str(HERE.parent/'repaired_grasp_benchmark'),
                   str(HERE.parent/'rayst3r_graspgenx'), str(ROOT/'modules/maniskill'),
                   str(ROOT/'modules/maniskill/arx_pipeline')]


def current():
    sys.path.insert(0, str(HERE.parent/'current_configuration'))
    from configuration import load_current
    return load_current(verify_dataset=True)


def gpu_inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {a.strip(): int(b) for a, b in (line.split(',') for line in text.strip().splitlines())}


def allocation_probes(gpus):
    def probe(gpu):
        result = subprocess.run([str(ROOT/'environments/reconstruction/bin/python'), '-c',
            'import torch; x=torch.zeros(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), capture_output=True, text=True, timeout=60)
        return gpu, {'passed': result.returncode == 0, 'returncode': result.returncode,
                     'stdout': result.stdout.strip(), 'stderr': result.stderr[-2000:]}
    with ThreadPoolExecutor(max_workers=4) as workers:
        return dict(workers.map(probe, gpus))


def choose_gpus():
    free = gpu_inventory()
    probes = allocation_probes(free)
    available = [gpu for gpu in free if probes[gpu]['passed']]
    # Startup estimates select a placement; the hard admission gate follows
    # actual model loading and warmup, before any physical trial is claimed.
    choices = [(min(free[r]-18000, free[s]-12000, free[p]-10000), r, s, p)
               for r, s, p in permutations(available, 3)
               if free[r] >= 18000 and free[s] >= 12000 and free[p] >= 10000]
    if not choices:
        raise RuntimeError('Need three CUDA-allocatable GPUs with 18000/12000/10000 MiB free at startup')
    _, ray, sam, plan = max(choices)
    return {'rayst3r': ray, 'sam3': sam, 'graspgenx': sam,
            'curobo_dual_arm': plan, 'simulation': plan}, {ray: 18000, sam: 12000, plan: 10000}, probes


def prepare(root, case_ids, arms, predecessor=None):
    if root.exists():
        raise ValueError('Use a new run directory; existing trials and STOP stay intact')
    if not case_ids or len(set(case_ids)) != len(case_ids) or len(case_ids) > 10:
        raise ValueError('Register 1 to 10 distinct integration scenes; no automatic full benchmark')
    if arms and len(arms) != len(case_ids):
        raise ValueError('Supply one arm choice per scene')
    config = current()
    table = {row['id']: row for row in current_schedule()}
    schedule = []
    for i, case in enumerate(case_ids):
        row = dict(table[case]); row.pop('observation', None)
        row['planner_arm'] = None if not arms or arms[i] == 'auto' else arms[i]
        schedule.append(row)
    carried = None
    if predecessor is not None:
        predecessor = ensure_run_path(predecessor)
        previous = read_json(predecessor/'registration.json')
        if not (predecessor/'STOP').is_file() or list((predecessor/'claims').glob('*.json')):
            raise ValueError('Only a stopped preflight with zero claimed trials can transfer this budget')
        if not read_json(predecessor/'services_stopped.json')['all_owned_exited']:
            raise ValueError('Previous services have not stopped')
        if [(r['id'], r['planner_arm']) for r in previous['schedule']] != [(r['id'], r['planner_arm']) for r in schedule]:
            raise ValueError('Transferred integration schedule differs')
        carried = {'run': str(predecessor), 'registration_sha256': sha256(predecessor/'registration.json'),
                   'stop_sha256': sha256(predecessor/'STOP'), 'claimed_trials': 0, 'physical_attempts': 0}
    roles, floors, probes = choose_gpus()
    root.mkdir(parents=True)
    for name in ['preflight', 'resident', 'claims', 'trials']:
        (root/name).mkdir()
    write_json(root/'preflight/device_allocation.json', probes)
    imports()
    from repair_contract import descriptor
    write_json(root/'gripper_descriptor.json', descriptor())
    shutil.copy2(current_dataset()/'object_catalog.json', root/'object_catalog.json')
    write_json(root/'common_initial_conditions.json', {'base_pose': schedule[0]['base_pose'],
        'initial_q': schedule[0]['robot_initial_q']})
    sources = {str(EXPERIMENT): sha256(EXPERIMENT), **config['configuration_hashes']}
    for folder in [ROOT/'arx_project', ROOT/'modules/curobo/dual_arm',
                   ROOT/'modules/curobo/fingertip_contact', ROOT/'modules/maniskill/arx_pipeline',
                   ROOT/'modules/maniskill/planning_common', HERE,
                   HERE.parent/'repaired_grasp_benchmark', HERE.parent/'diverse_object_scenes',
                   HERE.parent/'pointcloud_benchmark']:
        for path in folder.glob('*.py'):
            sources[str(path)] = sha256(path)
    for path in [ROOT/'configs/modules.json', ROOT/'configs/experiments/active_experiment.json',
                 ROOT/'modules/arx_core/arms.py', ROOT/'modules/curobo/arx_adapter.py',
                 ROOT/'modules/maniskill/resident_planner.py', ROOT/'modules/maniskill/fast_resident_planner.py',
                 ROOT/'modules/maniskill/short_lift_protocol.py', ROOT/'modules/maniskill/short_lift_planner.py',
                 ROOT/'modules/maniskill/checked_native_services.py',
                 ROOT/'modules/maniskill/compat_tools/physical_table_pair_20261118_01.py']:
        sources[str(path)] = sha256(path)
    weights = {}
    for kind in COMPONENTS:
        for name in module_records()[kind].get('weights', []):
            location = ROOT/name
            files = [location] if location.is_file() else [p for p in location.rglob('*') if p.is_file()]
            for file in files:
                weights[str(file)] = {'size': file.stat().st_size, 'sha256': sha256(file)}
    write_json(root/'registration.json', {'at': now(), 'version': 'V3', 'schedule': schedule,
        'physical_slot_budget': len(schedule), 'automatic_retries': False, 'predecessor': carried,
        'configuration_id': config['profile_id'], 'source_hashes': sources, 'weights': weights,
        'dataset': str(current_dataset()), 'dataset_freeze_sha256': config['dataset_freeze_sha256'],
        'gpu_roles': roles, 'minimum_free_mib': floors, 'required_components': COMPONENTS,
        'success': {'planned_lift_m': .03, 'minimum_hold_lift_m': .01, 'hold_seconds': 3.,
                    'bilateral_contact': True, 'joint_tolerance_rad': .005, 'forbidden_contacts': 0},
        'user_authorization': read_json(EXPERIMENT)['user_request'],
        'scope': 'Bounded dual-arm integration, original scene placements, one attempt per target'})
    write_json(root/'progress.json', {'phase': 'registered', 'registered': len(schedule), 'completed': 0})
    print('REGISTERED', root, roles, flush=True)


def guard(root):
    reg = read_json(root/'registration.json')
    if (root/'STOP').exists():
        raise RuntimeError('STOP exists; do not reschedule this registration')
    for name, expected in reg['source_hashes'].items():
        if sha256(name) != expected:
            raise RuntimeError('Registered source/configuration changed: '+name)
    config = current()
    if config['profile_id'] != reg['configuration_id'] or config['dataset_freeze_sha256'] != reg['dataset_freeze_sha256']:
        raise RuntimeError('Current base configuration changed')
    return reg


class Services:
    def __init__(self, root, reg):
        self.root, self.reg = root, reg
        socket_root = ROOT/'tmp/rpc'/sha256(root/'registration.json')[:12]
        socket_root.mkdir(parents=True, exist_ok=False)
        self.sockets = {name: socket_root/(name+'.sock') for name in COMPONENTS}
        self.children, self.streams = {}, []

    def start(self):
        from arx_service_readiness import wait_ready
        for kind in COMPONENTS:
            folder = self.root/'resident'/kind
            folder.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.reg['gpu_roles'][kind],
                ARX_PLANNER_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                ARX_SIM_GPU=self.reg['gpu_roles']['simulation'],
                LILO_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                GRASP_TRAJOPT_POLICY='interactive', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            env.pop('PYTHONPATH', None)
            python = ROOT/module_records()[kind]['environment']/'bin/python'
            cmd = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'v3-worker',
                str(ROOT/'modules/arx_core/scripts/curobo_arx/environment.sh'), str(python),
                '-m', 'arx_project', 'serve', '--kind', kind, '--root', str(folder),
                '--socket', str(self.sockets[kind]), '--warmup-trial', str(self.root/'preflight/head_warmup'),
                '--descriptor', str(self.root/'gripper_descriptor.json')]
            stream = (folder/'service.log').open('x'); self.streams.append(stream)
            child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.children[kind] = child
            wait_ready(self.sockets[kind], folder/(kind+'_ready.json'), child, kind, timeout=1800)
            print('READY', kind, flush=True)
        free = gpu_inventory()
        write_json(self.root/'preflight/warm_resources.json', {'free_mib': free, 'minimum_remaining_mib': 1536})
        if any(free.get(gpu, 0) < 1536 for gpu in set(self.reg['gpu_roles'].values())):
            raise RuntimeError('Insufficient GPU memory after warmup')

    def request(self, kind, directory, **flags):
        if self.children[kind].poll() is not None:
            raise RuntimeError('Service exited: '+kind)
        request = {'directory': str(directory), **flags}
        with (self.root/'requests.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps({'event': 'request', 'at': now(), 'kind': kind, **request})+'\n')
        response = Client(self.sockets[kind], timeout=1800).call(request)
        with (self.root/'requests.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps({'event': 'response', 'at': now(), 'kind': kind, 'directory': str(directory), 'response': response})+'\n')
        if response['status'] != 'ok':
            raise RuntimeError(response.get('error', str(response)))
        return response

    def stop(self):
        for kind, child in reversed(list(self.children.items())):
            if child.poll() is None:
                try: Client(self.sockets[kind], timeout=10).call({'command': 'shutdown'})
                except Exception: child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in self.streams:
            stream.close()
        write_json(self.root/'services_stopped.json', {'at': now(),
            'all_owned_exited': all(child.poll() is not None for child in self.children.values()),
            'returncodes': {name: child.returncode for name, child in self.children.items()}})


def summarize(root, phase):
    reg = read_json(root/'registration.json')
    rows = [read_json(path) for path in sorted((root/'trials').glob('*/result.json'))]
    result = {'phase': phase, 'registered': len(reg['schedule']), 'completed': len(rows),
        'not_completed': len(reg['schedule'])-len(rows),
        'planning_passes': sum(r.get('planning_success', False) for r in rows),
        'physical_attempts': sum(r.get('physical_attempt', False) for r in rows),
        'successes': sum(r.get('success', False) for r in rows),
        'selected_arms': dict(Counter(r['selected_arm'] for r in rows if 'selected_arm' in r)),
        'failure_reasons': dict(Counter(r.get('reason', 'unknown') for r in rows if not r['success']))}
    write_json(root/'progress.json', result)
    write_json(root/'summary.json', {**result, 'results': rows})
    return result


def run(root):
    reg = guard(root)
    if (root/'started.json').exists() or list((root/'claims').glob('*.json')):
        raise RuntimeError('This registration has already started; no automatic replay')
    free = gpu_inventory()
    if any(free.get(gpu, 0) < minimum for gpu, minimum in reg['minimum_free_mib'].items()):
        raise RuntimeError('Registered GPU resources are no longer available')
    probes = allocation_probes(set(reg['gpu_roles'].values()))
    write_json(root/'preflight/device_allocation_before_run.json', probes)
    if not all(probe['passed'] for probe in probes.values()):
        raise RuntimeError('Registered device cannot allocate CUDA memory')
    for file, entry in reg['weights'].items():
        if Path(file).stat().st_size != entry['size'] or sha256(file) != entry['sha256']:
            raise RuntimeError('Model weight changed: '+file)
    write_json(root/'preflight/resources.json', {'at': now(), 'free_mib': free, 'gpu_roles': reg['gpu_roles']})
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu_roles']['simulation']
    imports()
    spec = importlib.util.spec_from_file_location('v3_repaired_simulation', HERE.parent/'repaired_grasp_benchmark/simulation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    executor, evaluator, capture = module.install(root)
    from repair_contract import make_candidates
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid()})
    services = None
    try:
        capture()
        services = Services(root, reg)
        services.start()
        warm = root/'preflight/head_warmup'
        make_candidates(warm)
        response = services.request('curobo_dual_arm', warm)
        write_json(root/'preflight/full_chain.json', response)
        if not response['result']['success']:
            raise RuntimeError('Full-chain development planning did not pass')
        for number, entry in enumerate(reg['schedule']):
            guard(root)
            trial = root/'trials'/entry['id']
            claim = root/'claims'/(entry['id']+'.json')
            if trial.exists() or claim.exists():
                raise RuntimeError('Scene already claimed')
            write_json(claim, {'state': 'claimed', 'at': now(), 'scene': entry['id'], 'attempt': 1})

            def worker(script, directory, gpu, log, **flags):
                directory = Path(directory)
                try:
                    if script == 'neural.py':
                        write_json(directory/'entry.json', entry)
                        sam = services.request('sam3', directory)
                        if not sam['result'].get('target_found', True):
                            write_json(directory/'perception_failure.json', {'reason': 'sam3_no_target_mask'})
                            return 2
                        ray = services.request('rayst3r', directory)
                        if ray['result'].get('point_count', 0) < 20:
                            write_json(directory/'perception_failure.json', {'reason': 'rayst3r_empty_pointcloud'})
                            return 2
                        response = services.request('graspgenx', directory, seed=20261125+number)
                        make_candidates(directory)
                    else:
                        if not any(flags.values()):
                            flags['arm'] = entry['planner_arm']
                        response = services.request('curobo_dual_arm', directory, **flags)
                    write_json(log, response)
                    return response['returncode']
                except BaseException:
                    write_json(directory/'infrastructure_error.json', {'at': now(), 'error': traceback.format_exc()})
                    raise
            executor.worker = worker
            print('BEGIN', number+1, entry['id'], entry['planner_arm'] or 'auto', flush=True)
            result = executor.trial(entry, trial, reg['gpu_roles']['rayst3r'], reg['gpu_roles']['curobo_dual_arm'], 'head')
            if (trial/'perception_failure.json').exists():
                result['reason'] = read_json(trial/'perception_failure.json')['reason']
                write_json(trial/'result.json', result)
            evaluator.verify_trial(trial)
            normal = {'success', 'sam3_no_target_mask', 'rayst3r_empty_pointcloud',
                      'no_arm_has_executable_candidate', 'no_bilateral_target_contact',
                      'insufficient_hold_height', 'forbidden_robot_contact',
                      'measured_closure_lift_unreachable', 'measured_pregrasp_approach_unreachable',
                      'gripper_open_not_converged', 'open_not_converged',
                      'Target has fewer than 50 visible pixels'}
            normal.update(stage+'_not_converged' for stage in ['pregrasp', 'approach', 'close', 'lift', 'hold'])
            if (trial/'infrastructure_error.json').exists() or result['reason'] not in normal:
                raise RuntimeError('Execution/recording fault: '+result['reason'])
            if result.get('physical_attempt'):
                for name in ['execution.jsonl', 'execution.mp4', 'head.mp4', 'stages.json',
                             'contact_entities.json', 'independent_evaluation.json']:
                    if not (trial/name).is_file() or (trial/name).stat().st_size == 0:
                        raise RuntimeError('Missing execution evidence: '+name)
            write_json(claim, {'state': 'complete', 'at': now(), 'scene': entry['id'],
                              'result_sha256': sha256(trial/'result.json')})
            summarize(root, 'running')
            print('RESULT', entry['id'], result.get('selected_arm'), result['reason'], flush=True)
        guard(root)
        summarize(root, 'complete')
    finally:
        if services is not None:
            services.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--arms', nargs='+', choices=['auto', 'right', 'left'])
    parser.add_argument('--predecessor', type=Path)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root, args.cases, args.arms, args.predecessor)
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                run(root)
            except BaseException:
                write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()})
                summarize(root, 'stopped')
                raise


if __name__ == '__main__':
    main()
