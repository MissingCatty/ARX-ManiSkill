"""Bounded model-screen -> seven physical cases -> static data, without training."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.launch import devices
from .advisor31_campaign import exited
from .advisor36_progress import collect
from .advisor36_orchestrator import launch
from .advisor38_prefilter import require_seven
from .advisor38_seed import prepare as prepare_seed
from .advisor38_static_launch import prepare as prepare_static


def combined(top):
    old = read(top / 'carry_forward_registry.json')
    fresh = collect(top)
    if set(old['objects']) & set(fresh['objects']):
        raise RuntimeError('Previously released objects cannot be physically retested')
    return {'objects': {**old['objects'], **fresh['objects']},
            'candidate_rounds': old['candidate_rounds'] + fresh['candidate_rounds'],
            'released_objects': old['released_objects'] + fresh['released_objects'],
            'executed': old['executed'] + fresh['executed'],
            'successful_seed_tests': old['successful_seed_tests'] + fresh['successful_seed_tests']}


def child_state(root):
    state = read(root / 'state.json')
    if (root / 'STOP').exists() or state['status'] == 'fault':
        raise RuntimeError('Preserved child STOP/fault: ' + str(root))
    if state['status'] == 'complete':
        exited(root)
        return state
    service = read(root / 'supervisor_service.json')
    active = subprocess.run(['systemctl', '--user', 'is-active', service['unit']],
                            capture_output=True, text=True, timeout=15)
    if active.stdout.strip() not in ('active', 'activating'):
        raise RuntimeError('Owned child stopped without terminal/idle evidence: ' + str(root))
    return state


def select_candidates(top, data, spec, rows, prefilter_done):
    selected, exhausted, blocked = [], [], []
    for obj, item in spec['objects'].items():
        if obj in data['objects']:
            blocked.append(obj)
            continue
        history = [r for r in data['candidate_rounds'] if r['object_id'] == obj]
        if any(r['processed'] != 7 for r in history):
            raise RuntimeError('Prior physical candidate has unresolved cases')
        if len(history) >= item['maximum_registered_candidates_cumulative']:
            exhausted.append(obj); blocked.append(obj)
            continue
        tried = {digest(r['geometry']['selected']) for r in history}
        pool = {r['index']: r for r in read(item['candidate_pool'])['candidates']}
        choices = [r for r in rows if r['object_id'] == obj and r['pass'] and r['candidate_sha256'] not in tried]
        if not choices:
            if prefilter_done:
                exhausted.append(obj); blocked.append(obj)
            continue
        choice = choices[0]
        candidate = pool[choice['candidate_index']]
        file = top / 'prefilter' / obj / str(choice['candidate_index']) / 'result.json'
        proof = read(file)
        if proof['object_id'] != obj or proof['candidate_sha256'] != choice['candidate_sha256']:
            raise RuntimeError('Qualified result does not match the registered object/pool')
        require_seven(proof, candidate)
        verify_files(proof['evidence_files'])
        row = {k: copy.deepcopy(v) for k, v in read(item['geometry']).items() if k != 'candidates'}
        row.update(selected=candidate, model_prefilter_file=str(file),
                   refinement_geometry_file=item['candidate_pool'],
                   prior_registered_candidate_count=len(history),
                   original_cumulative_candidate_cap=item['maximum_registered_candidates_cumulative'])
        selected.append(row)
    return selected, exhausted, blocked


def register(top):
    top = Path(top).resolve()
    path = top / 'data_supervisor_registration.json'
    if path.exists():
        raise FileExistsError('Never overwrite a supervisor registration')
    authority, scope = read(top / 'user_authority.json'), read(top / 'scope_registration.json')
    if authority['automatic_training_authorized'] or scope['training_authorized']:
        raise RuntimeError('Data stage cannot authorize training')
    if scope['max_canonical_candidates_per_object_cumulative'] != 15:
        raise RuntimeError('Cumulative candidate budget mismatch')
    old = read(top / 'carry_forward_registry.json')
    previous = max({r['campaign'] for r in old['candidate_rounds']},
                   key=lambda p: read(Path(p) / 'registration.json')['created_unix'])
    write(top / 'supervisor_predecessor_exit.json', exited(previous))
    roles = read(top / 'resource_plan.json')['roles']
    sources = {str(p): sha(p) for folder in ('scripts/curobo_arx/nearfield',
               'scripts/curobo_arx/data_factory') for p in (ROOT / folder).glob('*.py')}
    fixed = {str(top / f): sha(top / f) for f in ('scope_registration.json', 'user_authority.json',
             'carry_forward_registry.json', 'prefilter_registration.json', 'prefilter_service.json',
             'source_inventory.json', 'advisor38.md', 'predecessor_scheduling_stop.json')}
    for entry in old['objects'].values():
        fixed.update(entry['physical_evidence'])
    spec = read(top / 'prefilter_registration.json')
    for item in spec['objects'].values():
        fixed[item['candidate_pool']] = sha(item['candidate_pool'])
    verify_files(fixed)
    reg = {'created_unix': time.time(), 'deadline_unix': scope['deadline_unix'],
           'initial_previous_seed': previous, 'source_files': sources, 'fixed_files': fixed,
           'seed_devices': {'compute_uuid': roles['compute_uuid'], 'sim_uuid': roles['seed_sim_uuid']},
           'static_devices': {'compute_uuid': roles['compute_uuid'], 'sim_uuid': roles['static_sim_uuid']},
           'maximum_registered_candidates_per_object_including_history': 15,
           'maximum_geometry_candidates_per_pending_object': 60,
           'static_target_per_object': 1024, 'static_fresh_proposal_cap_per_object': 4096,
           'old_failures_preserved': True, 'physical_7_of_7_required': True,
           'automatic_retries': 0, 'training_authorized': False,
           'infrastructure_failure': 'stop new scheduling, preserve already running bounded children'}
    write(path, reg)
    write(top / 'data_supervisor_state.json', {'status': 'prepared', 'training_started': False})
    return str(path)


def run(top):
    top = Path(top).resolve()
    lock = (top / 'data_supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reg = read(top / 'data_supervisor_registration.json')
    if (top / 'data_supervisor_STARTED.json').exists():
        raise RuntimeError('No automatic supervisor restart')
    write(top / 'data_supervisor_STARTED.json', {'wall': time.time(),
          'registration_sha256': sha(top / 'data_supervisor_registration.json')})
    current, data, fingerprint = None, None, None
    spec = read(top / 'prefilter_registration.json')
    try:
        verify_files(reg['fixed_files'])
        while True:
            if (top / 'STOP').exists() or time.time() >= reg['deadline_unix']:
                raise RuntimeError('Campaign STOP or original deadline')
            verify_files(reg['source_files'])
            for name in ('user_authority.json', 'scope_registration.json', 'prefilter_registration.json',
                         'carry_forward_registry.json'):
                if sha(top / name) != reg['fixed_files'][str(top / name)]:
                    raise RuntimeError('Registered campaign boundary changed')
            pf = read(top / 'prefilter/state.json')
            if (top / 'prefilter/FAULT.json').exists() or pf['status'] == 'fault':
                raise RuntimeError('Model prefilter infrastructure fault')
            if pf['status'] != 'complete':
                unit = read(top / 'prefilter_service.json')['unit']
                active = subprocess.run(['systemctl', '--user', 'is-active', unit],
                                        capture_output=True, text=True, timeout=15)
                if active.stdout.strip() not in ('active', 'activating'):
                    raise RuntimeError('Model worker disappeared without a completion receipt')
            seed_state = child_state(current) if current else {'status': 'none'}
            signature = [(str(p), p.stat().st_mtime_ns) for p in sorted(top.glob('seed_preflight*/groups/*/results.json'))]
            signature += [(str(p), p.stat().st_mtime_ns) for p in sorted(top.glob('seed_preflight*/registration.json'))]
            if signature != fingerprint:
                data, fingerprint = combined(top), signature
            lineage = read(top / 'static_lineage.json')
            active_static, datasets = [], []
            for obj, entry in lineage.items():
                root = Path(entry['current'])
                if sha(root / 'registration.json') != entry['registration_sha256']:
                    raise RuntimeError('Static lineage registration changed')
                state = child_state(root)
                if state['status'] == 'complete':
                    datasets.append({'object_id': obj, 'campaign': str(root),
                                     **read(root / 'dataset/quota_status.json')})
                else:
                    active_static.append(root)
            if len(active_static) > 1:
                raise RuntimeError('Multiple static World owners')
            inventory = {r['uuid']: r for r in devices()}

            def ready(roles, floor):
                a, b = inventory[roles['compute_uuid']], inventory[roles['sim_uuid']]
                return (a['index'], b['index']) if a['free_mb'] >= 6500 and b['free_mb'] >= floor else None

            if not active_static:
                pending = [o for o in data['objects'] if o not in lineage]
                resource = ready(reg['static_devices'], 12000)
                if pending and resource:
                    obj = pending[0]
                    source = Path(data['objects'][obj]['seed_campaign']).parent
                    root = Path(prepare_static(top, source, obj, *resource))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor38_static_launch',
                           'arx-a38-static-' + obj.replace('_', '-') + '-01')
                    active_static.append(root)
            records = read(top / 'prefilter/results.json')['records'] if (top / 'prefilter/results.json').exists() else []
            exhausted, proposals = [], []
            if seed_state['status'] in ('none', 'complete'):
                proposals, exhausted, blocked = select_candidates(top, data, spec, records, pf['status'] == 'complete')
                write(top / 'prefilter_control.json', {'released_or_budget_exhausted': blocked})
                resource = ready(reg['seed_devices'], 9000)
                if proposals and resource:
                    rounds = len(list(top.glob('seed_preflight*/registration.json')))
                    file = top / 'candidate_reselection' / f'{rounds + 1:02d}' / 'canonical_candidates.json'
                    if file.exists():
                        raise RuntimeError('Candidate selection already registered without a completed child')
                    write(file, proposals)
                    write(top / 'data_supervisor_state.json', {'status': 'preparing_screened_physical_candidates',
                          'objects': [r['object_id'] for r in proposals], 'training_started': False})
                    root = Path(prepare_seed(top, *resource, 'coupled_xyz', name=f'seed_preflight_{rounds + 1:02d}',
                                candidate_file=file, objects=[r['object_id'] for r in proposals],
                                previous_campaign=current or reg['initial_previous_seed']))
                    launch(root, 'scripts.curobo_arx.nearfield.advisor38_seed', f'arx-a38-seed-{rounds + 1:02d}')
                    current, seed_state = root, {'status': 'launch_requested'}
            state = {'status': 'running_data_only', 'wall': time.time(), 'training_started': False,
                     'training_requires_later_user_confirmation': True, 'released_objects': data['released_objects'],
                     'executed_seed_tests': data['executed'], 'successful_seed_tests': data['successful_seed_tests'],
                     'current_seed_campaign': str(current) if current else None, 'seed_status': seed_state['status'],
                     'active_static_campaign': str(active_static[0]) if active_static else None,
                     'completed_object_datasets': len(datasets),
                     'admitted_observations_completed_objects': sum(d['admitted_observations'] for d in datasets),
                     'objects_meeting_every_quota': sum(d['minimum_met'] for d in datasets), 'datasets': datasets,
                     'model_prefilter_completed': len(records), 'model_prefilter_qualified': sum(r['pass'] for r in records),
                     'model_prefilter_state': pf, 'canonical_search_exhausted_objects': exhausted}
            write(top / 'data_supervisor_state.json', state)
            write(top / 'current_canonical_releases.json', data)
            if seed_state['status'] in ('none', 'complete') and not active_static and len(datasets) == data['released_objects'] and (
                    data['released_objects'] == 8 or len(exhausted) == 8 - data['released_objects']):
                complete = data['released_objects'] == 8 and all(d['minimum_met'] for d in datasets)
                write(top / 'data_completion_receipt.json', {**state, 'pass': complete, 'finished_unix': time.time(),
                      'status': 'data_ready_waiting_training_confirmation' if complete else 'bounded_data_stage_incomplete'})
                return
            time.sleep(20)
    except BaseException:
        error = traceback.format_exc()
        write(top / 'data_supervisor_FAULT.json', {'error': error, 'wall': time.time(),
              'already_running_bounded_children_retained': True, 'training_started': False})
        write(top / 'data_supervisor_state.json', {'status': 'scheduling_stopped', 'error': error, 'training_started': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('root')
    args = parser.parse_args()
    print(register(args.root) if args.mode == 'register' else run(args.root))
