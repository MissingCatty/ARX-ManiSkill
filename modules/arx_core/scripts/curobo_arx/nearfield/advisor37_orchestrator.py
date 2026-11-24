"""Finite advisor37 data campaign; no training command or automatic retry."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, verify_files
from scripts.curobo_arx.data_factory.launch import devices
from .advisor31_campaign import exited
from .advisor36_progress import collect
from .advisor36_orchestrator import launch
from .advisor37_seed import prepare as prepare_seed
from .advisor37_static_launch import prepare as prepare_static


def combined(top):
    old = Path(read(top / 'scope_registration.json')['source_campaign'])
    a, b = collect(old), collect(top)
    if set(a['objects']) & set(b['objects']):
        raise RuntimeError('A previously released canonical was tested again')
    return {'objects': {**a['objects'], **b['objects']},
            'candidate_rounds': a['candidate_rounds'] + b['candidate_rounds'],
            'old_source': str(old), 'released_objects': a['released_objects'] + b['released_objects'],
            'executed': a['executed'] + b['executed'],
            'successful_seed_tests': a['successful_seed_tests'] + b['successful_seed_tests']}


def next_candidates(top, data):
    from .advisor36_geometry import local_neighbors
    old = Path(data['old_source'])
    inventory = read(top / 'source_inventory.json')
    proposed, exhausted = [], []
    for item in inventory:
        obj = item['object_id']
        if obj in data['objects']:
            continue
        history = [r for r in data['candidate_rounds'] if r['object_id'] == obj]
        if any(r['processed'] != 7 for r in history):
            raise RuntimeError('Previous canonical cases not terminal')
        if len(history) >= 8:
            exhausted.append(obj)
            continue
        original = read(old / 'geometry' / (obj + '.json'))
        if 'masher' in obj:
            file = top / 'geometry' / (obj + '.json')
            choices = [c for c in read(file)['candidates'] if c['pass_local_geometry'] and c['wrist_clearance_pass']]
        else:
            file = top / 'geometry_neighbors' / (obj + '.json')
            if not file.exists():
                write(file, local_neighbors(item, original['selected']))
            # Include closing-axis rolls now; do not exhaust all sphere yaws
            # before considering a geometrically valid approach-angle change.
            choices = [c for c in original['candidates'] + read(file)['candidates'] if c['pass_local_geometry']]
        tried = [np.asarray(r['geometry']['selected']['Gcanonical_object']) for r in history]

        def distinct(candidate):
            g = np.asarray(candidate['Gcanonical_object'])
            return all(np.linalg.norm(g[:3, 3] - p[:3, 3]) >= .001 or
                       Rotation.from_matrix(p[:3, :3].T @ g[:3, :3]).magnitude() >= np.deg2rad(.5)
                       for p in tried)

        choices = [c for c in choices if distinct(c)]
        if not choices:
            exhausted.append(obj)
            continue
        selected = min(choices, key=lambda c: (c['cost'], max(c['contact_normal_errors_degrees']), c['index']))
        row = {k: copy.deepcopy(v) for k, v in original.items() if k != 'candidates'}
        row.update(selected=selected, refinement_geometry_file=str(file),
                   prior_registered_candidate_count=len(history), old_failed_candidates_preserved=True)
        proposed.append(row)
    return proposed, exhausted


def supervisor_path(top, kind, generation):
    if generation < 1:
        raise ValueError('Positive supervisor generation required')
    suffix = '' if generation == 1 else f'_{generation:02d}'
    return top / ('data_supervisor_' + kind + suffix + '.json')


def register(top, generation=1, initial_seed=None, previous_unit=None):
    top = Path(top).resolve()
    path = supervisor_path(top, 'registration', generation)
    if path.exists():
        raise FileExistsError('Never overwrite a supervisor registration')
    authority = read(top / 'user_authority.json')
    if authority['automatic_training_authorized']:
        raise RuntimeError('This supervisor has no training authorization')
    scope = read(top / 'scope_registration.json')
    initial_seed = Path(initial_seed).resolve() if initial_seed else top / 'seed_preflight'
    predecessor_files = {}
    if generation > 1:
        if not previous_unit or read(top / 'data_supervisor_state.json')['status'] != 'scheduling_stopped':
            raise RuntimeError('Explicit stopped predecessor required')
        check = subprocess.run(['systemctl', '--user', 'show', previous_unit,
                                '-p', 'MainPID', '-p', 'ActiveState'], capture_output=True, text=True, timeout=15)
        properties = dict(line.split('=', 1) for line in check.stdout.splitlines() if '=' in line)
        if check.returncode or properties.get('MainPID') != '0' or properties.get('ActiveState') not in ('failed', 'inactive'):
            raise RuntimeError('Previous supervisor absence unconfirmed')
        for kind in ('registration', 'STARTED', 'FAULT'):
            p = supervisor_path(top, kind, generation - 1)
            predecessor_files[str(p)] = sha(p)
        old = read(supervisor_path(top, 'registration', generation - 1))
        if old['deadline_unix'] != scope['deadline_unix']:
            raise RuntimeError('Original deadline changed')
        continuation = initial_seed / 'preparation_continuation_receipt.json'
        if read(continuation)['replayed_started_trials'] != 0:
            raise RuntimeError('Started trials cannot be replayed')
        predecessor_files[str(continuation)] = sha(continuation)
        write(supervisor_path(top, 'previous_state', generation), read(top / 'data_supervisor_state.json'))
    service_state(initial_seed)
    source_files = {str(p): sha(p) for p in (ROOT / 'scripts/curobo_arx/nearfield').glob('*advisor37*.py')}
    for name in ('advisor36_export.py', 'advisor36_progress.py', 'advisor36_orchestrator.py'):
        file = ROOT / 'scripts/curobo_arx/nearfield' / name
        source_files[str(file)] = sha(file)
    seed = read(initial_seed / 'registration.json')['requested_devices']
    line = read(top / 'static_lineage.json')['household_hope_milk_carton']
    static = read(Path(line['current']) / 'registration.json')['requested_devices']
    reg = {'created_unix': time.time(), 'generation': generation, 'deadline_unix': scope['deadline_unix'],
           'authority': str(top / 'user_authority.json'), 'authority_sha256': sha(top / 'user_authority.json'),
           'initial_seed_campaign': str(initial_seed), 'initial_seed_registration_sha256': sha(initial_seed / 'registration.json'),
           'source_files': source_files, 'predecessor_files': predecessor_files,
           'previous_supervisor_unit': previous_unit,
           'seed_devices': seed, 'static_devices': static, 'maximum_candidates_per_object_including_old': 8,
           'fresh_static_proposal_cap_per_object': 4096, 'training_authorized': False,
           'ordinary_failure': 'Keep outcome and select a new normal-qualified canonical within remaining budget',
           'infrastructure_failure': 'Stop future scheduling; retain already running bounded services and all STOPs',
           'automatic_retries': 0, 'automatic_training_entry_point': None}
    write(path, reg)
    write(top / 'data_supervisor_state.json', {'status': 'prepared', 'training_started': False})
    return str(path)


def require_unstarted(root):
    import sqlite3
    root = Path(root)
    for name in ('STARTED.json', 'STOP', 'supervisor_service.json', 'workers', 'groups', 'group_registrations'):
        if (root / name).exists():
            raise RuntimeError('Prepared child has execution evidence: ' + name)
    reg = read(root / 'registration.json')
    paths = list((root / 'trial_ledgers').glob('*.sqlite3'))
    if len(paths) != len(reg['seed_trials']):
        raise RuntimeError('Missing prepared trial ledger')
    for path in paths:
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
            for table in ('attempts', 'events', 'object_leases'):
                if db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]:
                    raise RuntimeError('Prepared child ledger has execution history')


def service_state(root):
    state = read(root / 'state.json')
    if (root / 'STOP').exists() or state['status'] == 'fault':
        raise RuntimeError('Preserved child fault/STOP: ' + str(root))
    if state['status'] == 'complete':
        exited(root)
        return state
    if state['status'] == 'prepared' and not (root / 'supervisor_service.json').exists():
        require_unstarted(root)
        if not read(root / 'full_preflight.json')['pass'] or not read(root / 'container_preflight.json')['pass']:
            raise RuntimeError('Prepared child preflight incomplete')
        return state
    unit = read(root / 'supervisor_service.json')['unit']
    active = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True, timeout=15)
    if active.stdout.strip() not in ('active', 'activating'):
        raise RuntimeError('Child disappeared without a complete terminal receipt: ' + str(root))
    return {**state, 'status': 'launching'} if state['status'] == 'prepared' else state


def run(top, generation=1):
    top = Path(top).resolve()
    lock = (top / 'data_supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    regpath = supervisor_path(top, 'registration', generation)
    reg = read(regpath)
    started = supervisor_path(top, 'STARTED', generation)
    if started.exists():
        raise RuntimeError('No automatic supervisor restart')
    write(started, {'wall': time.time(), 'registration_sha256': sha(regpath)})
    current = Path(reg['initial_seed_campaign'])
    if sha(current / 'registration.json') != reg['initial_seed_registration_sha256']:
        raise RuntimeError('Initial seed registration changed')
    exhausted = []
    try:
        while True:
            if (top / 'STOP').exists() or time.time() >= reg['deadline_unix']:
                raise RuntimeError('Campaign stop/original deadline')
            verify_files(reg['source_files'])
            verify_files(reg.get('predecessor_files', {}))
            if sha(reg['authority']) != reg['authority_sha256']:
                raise RuntimeError('Execution authority changed')
            seed_state = service_state(current)
            data = combined(top)
            lineage = read(top / 'static_lineage.json')
            active_static, datasets = [], []
            for obj, entry in lineage.items():
                root = Path(entry['current'])
                if sha(root / 'registration.json') != entry['registration_sha256']:
                    raise RuntimeError('Static lineage registration changed')
                state = service_state(root)
                if state['status'] == 'complete':
                    if not (root / 'far_straight_target_sweep.json').exists():
                        from .advisor37_sweep import report
                        write(top / 'data_supervisor_state.json', {'status': 'offline_target_sweep',
                              'object_id': obj, 'training_started': False})
                        report(root)
                    datasets.append({'object_id': obj, 'campaign': str(root),
                                     **read(root / 'dataset/quota_status.json'),
                                     'far_target_sweep': {k: v for k, v in read(root / 'far_straight_target_sweep.json').items() if k != 'records'}})
                else:
                    active_static.append(root)
            if len(active_static) > 1:
                raise RuntimeError('More than one registered static World owner')
            inventory = {d['uuid']: d for d in devices()}

            def indices(roles, simulator_floor):
                compute, sim = inventory[roles['compute_uuid']], inventory[roles['sim_uuid']]
                if compute['free_mb'] < 6500 or sim['free_mb'] < simulator_floor:
                    return None
                return compute['index'], sim['index']

            if seed_state['status'] == 'prepared':
                resources = indices(read(current / 'registration.json')['requested_devices'], 9000)
                if resources:
                    child = read(current / 'registration.json')
                    verify_files({**child['execution_source_files'], **child['dependencies']})
                    require_unstarted(current)
                    launch(current, 'scripts.curobo_arx.nearfield.advisor37_seed',
                           'arx-a37-seed-' + current.name.replace('_', '-') + f'-g{generation:02d}')
                    seed_state = {'status': 'launch_requested'}
            if not active_static:
                pending = [o for o in data['objects'] if o not in lineage]
                resources = indices(reg['static_devices'], 12000)
                if pending and resources:
                    obj = pending[0]
                    source = Path(data['objects'][obj]['seed_campaign']).parent
                    root = Path(prepare_static(top, source, obj, *resources))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor37_static_launch',
                           'arx-a37-static-' + obj.replace('_', '-') + '-01')
                    active_static = [root]
            if seed_state['status'] == 'complete' and data['released_objects'] < 8:
                proposals, exhausted = next_candidates(top, data)
                resources = indices(reg['seed_devices'], 9000)
                if proposals and resources:
                    rounds = len([p for p in top.glob('seed_preflight*') if (p / 'registration.json').exists()])
                    folder = top / 'candidate_reselection' / f'{rounds + 1:02d}'
                    folder.mkdir(parents=True, exist_ok=False)
                    file = folder / 'canonical_candidates.json'
                    write(file, proposals)
                    root = Path(prepare_seed(top, *resources, 'coupled_xyz',
                        name=f'seed_preflight_{rounds + 1:02d}', candidate_file=file,
                        objects=[p['object_id'] for p in proposals], previous_campaign=current))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor37_seed', f'arx-a37-seed-{rounds + 1:02d}')
                    current = root
            state = {'status': 'running_data_only', 'wall': time.time(), 'current_seed_campaign': str(current),
                     'generation': generation, 'seed_status': seed_state['status'],
                     'seed_waiting_for_resources': seed_state['status'] == 'prepared',
                     'released_objects': data['released_objects'], 'executed_seed_tests': data['executed'],
                     'successful_seed_tests': data['successful_seed_tests'],
                     'active_static_campaign': str(active_static[0]) if active_static else None,
                     'completed_object_datasets': len(datasets),
                     'admitted_observations': sum(r['admitted_observations'] for r in datasets),
                     'objects_meeting_every_quota': sum(r['minimum_met'] for r in datasets),
                     'datasets': datasets, 'canonical_search_exhausted_objects': exhausted,
                     'training_started': False, 'training_requires_later_user_confirmation': True}
            write(top / 'data_supervisor_state.json', state)
            write(top / 'current_canonical_releases.json', data)
            if not active_static and len(datasets) == data['released_objects'] and (
                    data['released_objects'] == 8 or len(exhausted) == 8 - data['released_objects']):
                complete = data['released_objects'] == 8 and all(r['minimum_met'] for r in datasets)
                write(top / 'data_completion_receipt.json', {**state, 'pass': complete,
                      'status': 'data_ready_waiting_user_training_confirmation' if complete else 'bounded_data_stage_incomplete',
                      'finished_unix': time.time()})
                write(top / 'data_supervisor_state.json', {**state, 'status': 'complete_waiting_user' if complete else 'budget_or_geometry_incomplete'})
                return
            time.sleep(20)
    except BaseException:
        error = traceback.format_exc()
        write(supervisor_path(top, 'FAULT', generation), {'error': error, 'wall': time.time(),
              'already_running_bounded_services_retained': True, 'training_started': False})
        write(top / 'data_supervisor_state.json', {'status': 'scheduling_stopped', 'error': error, 'training_started': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'run'])
    parser.add_argument('root')
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--initial-seed')
    parser.add_argument('--previous-unit')
    args = parser.parse_args()
    print(register(args.root, args.generation, args.initial_seed, args.previous_unit)
          if args.mode == 'register' else run(args.root, args.generation))
