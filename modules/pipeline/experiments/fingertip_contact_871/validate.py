"""Register and run three bounded contact motions in the frozen ARX scenes."""
import argparse
from dataclasses import asdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from arx_project.assets import load_object_catalog
from arx_project.client import Client
from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from arx_project.scenes import current_dataset, current_schedule
from modules.curobo.fingertip_contact import load_profile

HERE = Path(__file__).resolve().parent
DEFAULT = ROOT/'tmp/fingertip_contact/v2_871_20261124_02'
PLANNER_GPU = 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169'
SIMULATION_GPU = 'GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e'
SPEC = [('D0008_S1_open_teapot', -.001, [-.60, -.10]),
        ('D0036_S1_open_cell_phone', .001, [-.60, 0.]),
        ('D0037_S1_open_bar_soap', .0015, [-.60, .10])]


def current_config():
    sys.path.insert(0, str(HERE.parent/'current_configuration'))
    from configuration import load_current, CONFIG
    return load_current(verify_dataset=True), CONFIG


def prepare(root, predecessor=None):
    ensure_run_path(root)
    assert (root/'implementation_registration.json').is_file()
    assert not (root/'registration.json').exists()
    carried = None
    if predecessor is not None:
        predecessor = ensure_run_path(predecessor)
        assert (predecessor/'STOP').is_file(), 'Only a recorded stopped startup can be carried'
        assert not list((predecessor/'claims').glob('*.json')), 'Do not automatically repeat claimed motions'
        carried = {'root': str(predecessor), 'registration_sha256': sha256(predecessor/'registration.json'),
                   'stop_sha256': sha256(predecessor/'STOP'), 'claimed_motions': 0, 'physical_grasps': 0}
    config, config_file = current_config()
    entries = {r['id']: r for r in current_schedule()}
    schedule = []
    for case_id, depth, xy in SPEC:
        row = dict(entries[case_id]); row.pop('observation', None)
        row.update(fixture={'desired_depth_m': depth, 'xy_world_m': xy},
                   campaign_registration=str(root/'registration.json'))
        schedule.append(row)
    for name in ['preflight', 'resident', 'claims', 'trials', 'source']:
        (root/name).mkdir(exist_ok=True)
    shutil.copy2(current_dataset()/'object_catalog.json', root/'object_catalog.json')
    first = schedule[0]
    write_json(root/'common_initial_conditions.json', {'base_pose': first['base_pose'], 'initial_q': first['robot_initial_q']})
    write_json(root/'descriptor.json', {'scope': 'contact_component_validation', 'neural_models': False})
    sources = {**config['configuration_hashes'], str(config_file): sha256(config_file)}
    for folder in [HERE, ROOT/'modules/curobo/fingertip_contact']:
        for path in folder.glob('*.py'):
            sources[str(path)] = sha256(path)
    for path in [ROOT/'modules/maniskill/planning_common/contact_execution.py', ROOT/'modules/maniskill/arx_pipeline/run.py']:
        sources[str(path)] = sha256(path)
    for case_id, _, _ in SPEC:
        for name in ['entry.json', 'state.json']:
            path = current_dataset()/'cases'/case_id/name; sources[str(path)] = sha256(path)
    registration = {'at': now(), 'schedule': schedule,
        'gpu_roles': {'planner': PLANNER_GPU, 'simulation': SIMULATION_GPU},
        'resource_minimum_mib': {'planner': 8192, 'simulation': 4096, 'combined': 14336},
        'predecessor': carried,
        'active_configuration_id': config['profile_id'], 'source_hashes': sources,
        'contact_profile': asdict(load_profile()), 'simulation_motion_budget': 3, 'physical_grasp_budget': 0,
        'automatic_retries': False, 'old_results_carried': 0,
        'fixture_yaw_degrees': [0., 45., -45., 90., -90., 135., -135., 180.],
        'scope': 'Current production planner and execution contact interfaces; fixed tabletop fixtures, unchanged target placements',
        'user_authorization': '2026-11-24: continue integrating and verifying the scoped fingertip/table allowance in the current 871-sphere version'}
    write_json(root/'registration.json', registration)
    for folder in [HERE, ROOT/'modules/curobo/fingertip_contact']:
        for path in folder.glob('*.py'):
            name = folder.name+'_'+path.name; shutil.copy2(path, root/'source'/name)
    write_json(root/'progress.json', {'phase': 'registered', 'completed': 0, 'registered': 3, 'physical_grasps': 0})


def guard(root):
    reg = read_json(root/'registration.json')
    for path, expected in reg['source_hashes'].items():
        if sha256(path) != expected:
            raise RuntimeError('Registered source changed: '+path)
    return reg


def run(root):
    assert not (root/'STOP').exists() and not (root/'started.json').exists()
    reg = guard(root)
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    memory = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in free.strip().splitlines()}
    roles = reg['gpu_roles']; minima = reg['resource_minimum_mib']
    for role, gpu in roles.items():
        assert memory[gpu] >= minima[role], 'Insufficient free memory for '+role
    assert sum(memory[gpu] for gpu in set(roles.values())) >= minima['combined']
    probes = {}
    for role, gpu in roles.items():
        python = ROOT/('environments/curobo/bin/python' if role == 'planner' else 'environments/simulation/bin/python')
        probe = subprocess.run([str(python), '-c',
            'import torch; assert torch.cuda.is_available(); x=torch.zeros(1,device="cuda"); '
            'torch.cuda.synchronize(); print(torch.cuda.get_device_name())'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), text=True, capture_output=True, timeout=60)
        probes[role] = {'gpu_uuid': gpu, 'returncode': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr}
    write_json(root/'preflight/resources.json', {'at': now(), 'free_mib': memory, 'gpu_roles': roles,
        'minimum_mib': minima, 'cuda_allocation_probes': probes})
    assert all(p['returncode'] == 0 for p in probes.values()), 'Registered GPU cannot allocate CUDA memory'
    os.environ['CUDA_VISIBLE_DEVICES'] = roles['simulation']
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid()})
    sys.path[:0] = [str(HERE.parent/'rayst3r_graspgenx'), str(HERE.parent/'repaired_grasp_benchmark'),
                    str(ROOT/'modules/maniskill'), str(ROOT/'modules/maniskill/arx_pipeline')]
    spec = importlib.util.spec_from_file_location('contact_current_simulation', HERE.parent/'repaired_grasp_benchmark/simulation.py')
    sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
    executor, baseline, capture = sim.install(root)
    capture()
    socket = root/'planner.sock'
    args = [str(ROOT/'environments/curobo/bin/python'), str(HERE/'planning_worker.py'),
        '--root', str(root/'resident'), '--socket', str(socket),
        '--warmup-trial', str(root/'preflight/head_warmup'), '--descriptor', str(root/'descriptor.json')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=roles['planner'], ARX_SIM_GPU=roles['simulation'],
               ARX_PLANNER_GPU=roles['planner'], LILO_GPU=roles['planner'],
               GRASP_TRAJOPT_POLICY='interactive', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    command = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'contact-worker',
        str(ROOT/'modules/arx_core/scripts/curobo_arx/environment.sh'), *args]
    worker = None; writer = None; results = []
    try:
        writer = (root/'resident/planner.log').open('x')
        worker = subprocess.Popen(command, env=env, stdout=writer, stderr=subprocess.STDOUT)
        from arx_service_readiness import wait_ready
        wait_ready(socket, root/'resident/curobo_ready.json', worker, 'curobo', timeout=1800)
        client = Client(socket, timeout=1800)
        write_json(root/'preflight/all_ready.json', {'passed': True, 'at': now(), 'physical_grasps': 0})
        for entry in reg['schedule']:
            guard(root)
            case_id = entry['id']; trial = root/'trials'/case_id
            assert not (root/'STOP').exists() and not trial.exists()
            trial.mkdir()
            claim = root/'claims'/(case_id+'.json'); assert not claim.exists()
            write_json(claim, {'state': 'claimed', 'at': now(), 'id': case_id, 'attempt': 1})
            result = {'id': case_id, 'motion_executed': False, 'success': False, 'physical_grasps': 0}
            environment = execution = None
            try:
                import torch
                environment = executor.make(entry['model_id'], perception_camera='head', scene_spec=entry['scene'])
                environment.reset(seed=entry['seed'])
                for _ in range(80): environment._step_action(torch.tensor(environment.initial_q[None], device=environment.device))
                obs = environment.get_obs(); state = executor.snapshot(environment, obs)
                write_json(trial/'state.json', state); write_json(trial/'entry.json', entry)
                write_json(trial/'fixture_request.json', entry['fixture'])
                executor.Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(trial/'initial_head.png')
                executor.Image.fromarray(environment.render().cpu().numpy()[0]).save(trial/'initial_overview.png')
                response = client.request(trial, fixture=True)
                plan = response['result']
                if not plan['success']:
                    result['reason'] = plan['reason']
                else:
                    execution = executor.Execution(environment, trial)
                    selected = plan['selected']; result['selected'] = selected
                    result['motion_executed'] = True
                    path = plan['paths']['pregrasp']
                    execution.segment('pregrasp', path['q'], path['times'], goal_world=selected['pregrasp_world'])
                    write_json(trial/'pregrasp_state.json', executor.snapshot(environment))
                    approach = client.request(trial, approach=True)['result']
                    if not approach['success']: raise RuntimeError(approach['reason'])
                    path = approach['paths']['approach']
                    execution.segment('approach', path['q'], path['times'], goal_world=selected['grasp_world'])
                    q = selected['q_grasp']
                    execution.segment('close', [q, q], [0., 3.], finger=0., goal_world=selected['grasp_world'])
                    write_json(trial/'closed_state.json', executor.snapshot(environment))
                    lift = client.request(trial, lift=True)['result']
                    if not lift['success']: raise RuntimeError(lift['reason'])
                    path = lift['paths']['lift']
                    execution.segment('lift', path['q'], path['times'], finger=0., goal_world=selected['lift_world'])
                    result.update(component_motion_passed=True, reason='contact_and_release_completed')
            except RuntimeError as error:
                if str(error) not in {'forbidden_robot_contact', 'pregrasp_not_converged', 'approach_not_converged',
                    'close_not_converged', 'lift_not_converged', 'measured_closure_lift_unreachable', 'measured_pregrasp_approach_unreachable'}:
                    raise
                result.update(component_motion_passed=False, reason=str(error))
            finally:
                if execution is not None:
                    execution.finish(); result['stages'] = execution.stages
                    result['allowed_tip_events'] = len(execution.tip_contacts)
                if environment is not None:
                    write_json(trial/'final_state.json', executor.snapshot(environment)); environment.close()
            write_json(trial/'result.json', result)
            baseline.verify_trial(trial)
            write_json(claim, {'state': 'complete', 'at': now(), 'result_sha256': sha256(trial/'result.json')})
            results.append(result)
            write_json(root/'progress.json', {'phase': 'running', 'completed': len(results), 'registered': 3,
                'last': case_id, 'physical_grasps': 0})
            print('RESULT', case_id, result.get('reason'), flush=True)
        guard(root)
        write_json(root/'summary.json', {'completed': len(results), 'registered': 3, 'physical_grasps': 0,
            'motions_executed': sum(r['motion_executed'] for r in results),
            'component_motion_passes': sum(r.get('component_motion_passed', False) for r in results), 'results': results})
        write_json(root/'progress.json', {'phase': 'complete', 'completed': len(results), 'registered': 3, 'physical_grasps': 0})
    finally:
        if worker is not None and worker.poll() is None:
            try: Client(socket, timeout=10).call({'command': 'shutdown'})
            except Exception: worker.terminate()
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait()
        if writer: writer.close()
        write_json(root/'services_stopped.json', {'at': now(), 'all_owned_exited': worker is None or worker.poll() is not None,
                                                  'returncode': None if worker is None else worker.returncode})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--predecessor', type=Path)
    args = parser.parse_args(); root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root, args.predecessor)
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            try: run(root)
            except BaseException:
                write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()})
                write_json(root/'progress.json', {'phase': 'stopped', 'physical_grasps': 0})
                raise
