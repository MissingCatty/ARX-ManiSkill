"""Finite per-layer replacement acquisition. Training has no entry point here."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices, environment
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.watchdog import process_info, exclusive_stop
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from .advisor34_launch import owned_process, wait_ready, healthy, container_command
from .advisor36_static_launch import pilot_ids
from .advisor36_progress import collect
from .advisor37_samples import generate_pool, select_wave, quota_status, PLANNED, MINIMUM
from .advisor37_export import export, load_lines
from .resources import probe_cuda
from .physical_startup import startup_expired


def prepare(top, source_top, obj, compute_index=1, sim_index=3, *, name=None, predecessor=None):
    top, source_top = Path(top).resolve(), Path(source_top).resolve()
    out = top / 'static_objects' / (name or obj)
    if out.parent != top / 'static_objects':
        raise ValueError('Invalid per-object registration name')
    if out.exists():
        raise FileExistsError('Keep every old registration and started ID')
    authority = read(top / 'user_authority.json')
    if authority['automatic_training_authorized']:
        raise RuntimeError('This data coordinator cannot train')
    releases = collect(source_top)
    if obj not in releases['objects']:
        raise RuntimeError('Current canonical has not passed seven physical cases')
    entry = releases['objects'][obj]
    verify_files(entry['physical_evidence'])
    oldroot = Path(entry['seed_campaign'])
    old = read(oldroot / 'registration.json')
    out.mkdir(parents=True)
    space(out)
    write(out / 'canonical_registry.json', {'objects': {obj: entry}, 'released_objects': 1,
          'training_authorized': False, 'source_campaign': str(source_top)})
    rows, quality = generate_pool(obj, entry['geometry']['selected']['Gcanonical_object'],
                                 entry['source']['source_split'] != 'train')
    write(out / 'candidate_quality.json', quality)
    with (out / 'candidates.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    pilot = pilot_ids([r for r in rows if r['wave'] == 0])
    phases = [{'name': 'pilot', 'ids': pilot, 'wave': 0},
              {'name': 'wave0', 'ids': [r['sample_id'] for r in rows
                                       if r['wave'] == 0 and r['sample_id'] not in set(pilot)], 'wave': 0}]
    phases += [{'name': f'wave{wave}', 'wave': wave} for wave in range(1, 4)]
    write(out / 'sample_manifest.json', {'planned_observations': 1024, 'fresh_proposal_cap': 4096,
          'phases': phases, 'planned_strata': PLANNED, 'minimum_strata': MINIMUM,
          'target': 'Fill planned quotas; stop on finite proposal/deadline exhaustion',
          'old_observations_replayed': 0, 'proposals_are_not_admitted_observations': True})
    prior = Path(old['source_snapshot'])
    source = out / 'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'):
        (source.parent / name).symlink_to(ROOT.parent / name, target_is_directory=True)
    execution = {str(source / Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in old['execution_source_files'].items()}
    module_dir = ROOT / 'scripts/curobo_arx/nearfield'
    added = sorted({p for pattern in ('advisor36_*.py', 'runtime_advisor36_*.py', 'test_advisor36*.py',
                                     'advisor37_*.py', 'runtime_advisor37_*.py', 'test_advisor37*.py')
                    for p in module_dir.glob(pattern)})
    for path in added:
        dest = source / path.relative_to(ROOT)
        shutil.copy2(path, dest)
        execution[str(dest)] = sha(dest)
    inventory = devices()
    by = {r['index']: r for r in inventory}
    roles = {'compute_uuid': by[compute_index]['uuid'], 'sim_uuid': by[sim_index]['uuid']}
    if roles['compute_uuid'] == roles['sim_uuid']:
        raise ValueError('Separate rendering and model-query devices')
    deps = {**old['dependencies'], **entry['physical_evidence']}
    info = subprocess.run([str(ROOT.parent / 'curobo/.venv/bin/python'), '-c',
        'import fcl,json,pathlib;print(json.dumps({"version":fcl.__version__,"binary":fcl.fcl.__file__,"libraries":[str(p) for p in (pathlib.Path(fcl.__file__).parent.parent/"python_fcl.libs").glob("*.so*")]}))'],
        capture_output=True, text=True, check=True, timeout=30)
    info = json.loads(info.stdout)
    if info['version'] != '0.7.0.11':
        raise RuntimeError('Unregistered geometry dependency')
    for path in [info['binary'], *info['libraries']]:
        deps[path] = sha(path)
    for path in [top / 'advisor37.md', top / 'user_authority.json', top / 'scope_registration.json',
                 out / 'canonical_registry.json', out / 'candidates.jsonl',
                 out / 'candidate_quality.json', out / 'sample_manifest.json']:
        deps[str(path)] = sha(path)
    deadline = min(read(top / 'scope_registration.json')['deadline_unix'], time.time() + 8 * 3600)
    if predecessor is not None:
        from .advisor31_campaign import exited
        prior_root = Path(predecessor).resolve()
        absence = exited(prior_root)
        previous = read(prior_root / 'registration.json')
        if (not (prior_root / 'STOP').is_file() or list(prior_root.glob('group_registrations/*.json'))
            or list(prior_root.glob('groups/*/observations/*/STARTED.json'))):
            raise RuntimeError('Only a proven pre-dispatch resource stop can be re-registered here')
        if load_lines(previous['static_observations']['candidate_table']) != rows:
            raise RuntimeError('Fresh resource placement changed the proposal table')
        deadline = min(deadline, previous['deadline_unix'])
        receipt = {'predecessor': str(prior_root), 'old_STOP_preserved': True,
                   'process_and_World_absence': absence, 'old_groups_dispatched': 0,
                   'old_observations_started': 0, 'sample_ids_replayed': 0,
                   'proposal_budget_and_original_deadline_retained': True,
                   'old_devices': previous['requested_devices'], 'new_devices': roles,
                   'reason': 'Available GPU changed before static World dispatch'}
        write(out / 'resource_continuation_receipt.json', receipt)
        for path in (prior_root / 'registration.json', prior_root / 'STOP',
                     prior_root / 'state.json', out / 'resource_continuation_receipt.json'):
            deps[str(path)] = sha(path)
    if deadline <= time.time():
        raise RuntimeError('Inherited data deadline expired')
    reg = copy.deepcopy(old)
    reg.update(stage='advisor37_tapered_static_observations', project_root=str(ROOT),
        source_snapshot=str(source), execution_source_files=execution, source_identity=digest(execution),
        dependencies=deps, runtime_module='scripts.curobo_arx.nearfield.runtime_advisor37_static',
        worker_module='scripts.curobo_arx.nearfield.advisor37_static_teacher',
        coordinator_module='scripts.curobo_arx.nearfield.advisor37_static_launch',
        created_unix=time.time(), deadline_unix=deadline, requested_devices=roles,
        max_physical_trials=0, global_physical_budget=0, seed_trials=[], training_authorized_here=False,
        no_automatic_retry=True, automatic_attempt_retries=0, automatic_startup_retries=0,
        static_observations={'object_id': obj, 'canonical_registry': str(out / 'canonical_registry.json'),
            'candidate_table': str(out / 'candidates.jsonl'), 'candidate_count': len(rows),
            'planned_observations': 1024, 'sample_phases': phases, 'prior_capture_groups': [],
            'excluded_without_replay': [], 'prior_admission_exclusions': {},
            'actual_target_surface_check_required': True, 'target_native_convex_required': True,
            'IK_and_actual_FK_fit_m': .0005, 'IK_and_actual_FK_fit_degrees': .2,
            'GRASP_original_closed_model_check': True, 'mirror_pair_admission': True,
            'training_requires_later_user_acceptance': True, 'physics_state_setters_not_trajectories': True,
            'full_physics_unchanged': True, 'sample_identity_can_advance_with_fixed_physics_time': True,
            'surface_query_dependency': info, 'mask_source': 'same-render native instance IDs'})
    Ledger(out / 'ledger.sqlite3').close()
    shutil.copy2(oldroot / 'storage_policy.json', out / 'storage_policy.json')
    write(out / 'resource_plan.json', {'inventory': inventory, 'roles': roles})
    write(out / 'registration.json', reg)
    verify_files({**execution, **deps})
    write(out / 'full_preflight.json', {'pass': True, 'checked_files': len({**execution, **deps}),
          'observations': 0, 'grasp_trials': 0})
    write(out / 'state.json', {'status': 'prepared', 'training_started': False})
    lineage = read(top / 'static_lineage.json') if (top / 'static_lineage.json').exists() else {}
    previous_paths = lineage.get(obj, {}).get('predecessors', [])
    lineage[obj] = {'current': str(out), 'registration_sha256': sha(out / 'registration.json'),
                    'predecessors': list(dict.fromkeys(previous_paths + ([str(Path(predecessor).resolve())] if predecessor else [])))}
    write(top / 'static_lineage.json', lineage)
    return str(out)


def run(root):
    root = Path(root).resolve()
    reg = read(root / 'registration.json')
    roles, static = reg['requested_devices'], reg['static_observations']
    lock = (root / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'STARTED.json').exists() or (root / 'STOP').exists():
        raise RuntimeError('No restart, replay or STOP deletion')
    write(root / 'STARTED.json', {'wall': time.time(), 'registration_sha256': sha(root / 'registration.json')})
    planner = active = name = error = latest_export = None
    unresolved, phases, records, started = False, [], [], set()
    try:
        verify_files({**reg['dependencies'], **reg['execution_source_files']})
        inventory = {r['uuid']: r for r in devices()}
        for role, floor in (('compute', 6500), ('sim', 12000)):
            d = inventory[roles[role + '_uuid']]
            if d['free_mb'] < floor:
                raise RuntimeError(role + ' GPU memory floor')
            proof = probe_cuda(d, environment(d))
            write(root / (role + '_cuda_preflight.json'), proof)
            if not proof['pass']:
                raise RuntimeError(role + ' CUDA preflight failed')
        if available_memory() < reg['minimum_available_host_memory_bytes']:
            raise RuntimeError('Host memory floor')
        directory = root / 'workers/planner'
        env = environment(inventory[roles['compute_uuid']])
        env.update(PYTHONPATH=reg['source_snapshot'], PYTHONUNBUFFERED='1')
        planner = owned_process([str(ROOT.parent / 'curobo/.venv/bin/python'), '-m', reg['worker_module'],
            '--directory', str(directory), '--registration', str(root / 'registration.json')],
            reg['source_snapshot'], env, directory)
        wait_ready(planner, directory, root)
        renderer = read(static['canonical_registry'])['objects'][static['object_id']]['source']['renderer']
        pool = load_lines(static['candidate_table'])
        for phase in static['sample_phases']:
            healthy(root, {'planner': planner})
            space(root)
            if time.time() >= reg['deadline_unix']:
                raise RuntimeError('Inherited static deadline reached')
            ids = phase.get('ids')
            if ids is None:
                ids = select_wave(pool, phase['wave'], records, started)
            if not ids:
                continue
            if started.intersection(ids):
                raise RuntimeError('Duplicate scheduled sample identity')
            if next(d['free_mb'] for d in devices() if d['uuid'] == roles['sim_uuid']) < 12000:
                raise RuntimeError('Rendering GPU memory changed')
            ident = phase['name']
            name = 'arx-a37-static-' + digest(str(root))[:10] + '-' + ident
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('Prior World absence unconfirmed')
            launch, group = root / 'group_registrations' / (ident + '.json'), root / 'groups' / ident
            write(launch, {'group_id': ident, 'renderer': renderer, 'gpu_uuid': roles['sim_uuid'],
                'env_count': 1, 'planner_directory': str(directory),
                'candidate_directory': str(root / 'no_neural_requests'), 'sample_ids': ids,
                'stop_unix': reg['deadline_unix'], 'registration_sha256': sha(root / 'registration.json'),
                'preceding_independent_quota': quota_status(records)})
            with launch.with_suffix('.log').open('x') as log:
                active = subprocess.Popen(container_command(root, reg, roles['sim_uuid'], launch, name),
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(launch.with_suffix('.process.json'), {'pid': active.pid, 'container': name,
                  'owned': True, **(process_info(active.pid) or {})})
            started.update(ids)
            begin = time.monotonic()
            while active.poll() is None:
                healthy(root, {'planner': planner})
                if (root / 'STOP').exists() or time.time() > reg['deadline_unix']:
                    raise RuntimeError('Static stopped/deadline')
                if startup_expired(group, time.monotonic() - begin, 1800):
                    raise RuntimeError('Static World startup timeout')
                current = read(group / 'state.json') if (group / 'state.json').exists() else {'processed': 0}
                write(root / 'state.json', {**current, 'status': 'capturing', 'phase': ident,
                      'independently_admitted': quota_status(records)['admitted_observations'], 'training_started': False})
                time.sleep(3)
            code, active = active.returncode, None
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                raise RuntimeError('World removal unconfirmed')
            if code or (group / 'FAULT.json').exists():
                raise RuntimeError('Static simulator/configuration/record failure')
            if not read(group / 'shutdown_receipt.json')['all_trajectories_idle']:
                raise RuntimeError('Unconfirmed static idle state')
            phases.append(ident)
            latest_export = root / 'exports' / ident
            write(root / 'state.json', {'status': 'independent_admission', 'phase': ident, 'training_started': False})
            export(root, latest_export, phases)
            records = load_lines(latest_export / 'admission.jsonl')
            write(root / 'quota_status.json', quota_status(records))
            if ident == 'pilot':
                labels = {r['category'] for r in records if r['admitted']}
                if not {'grasp', 'funnel'} <= labels:
                    raise RuntimeError('Interface probe lacks a complete admitted motion/GRASP pair')
        if latest_export is None:
            raise RuntimeError('No static observations executed')
        (root / 'dataset').symlink_to(latest_export.relative_to(root), target_is_directory=True)
        verify_files({**reg['dependencies'], **reg['execution_source_files']})
        quota = quota_status(records)
        write(root / 'completion_receipt.json', {'pass': True, 'quota_minimum_met': quota['minimum_met'],
              'registered_proposals': len(pool), 'scheduled_observations': len(started),
              'dataset': str(root / 'dataset'), 'training_started': False, 'physical_grasp_trials': 0,
              'finished_unix': time.time(), 'unmet_quotas_are_not_success': True})
    except BaseException:
        error = traceback.format_exc()
        write(root / 'coordinator_fault.json', {'error': error, 'wall': time.time()})
        exclusive_stop(root / 'STOP', {'reason': 'static_infrastructure_configuration_or_record_failure', 'error': error})
    finally:
        if active is not None and active.poll() is None:
            subprocess.run(['podman', 'kill', '--signal', 'TERM', name], capture_output=True, timeout=30)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['podman', 'kill', '--signal', 'KILL', name], capture_output=True, timeout=30)
                try:
                    active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    unresolved = True
            if subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=30).returncode != 1:
                unresolved = True
        if planner:
            write(root / 'workers/planner/STOP', {'reason': 'bounded static acquisition ended'})
            try:
                planner.wait(timeout=30)
            except subprocess.TimeoutExpired:
                planner.terminate()
                planner.wait(timeout=30)
            write(root / 'workers/planner/process_exit.json', {'pid': planner.pid, 'exit_code': planner.returncode})
        write(root / 'state.json', {'status': 'fault' if error or unresolved else 'complete',
              'error': error, 'unconfirmed_world': unresolved, 'training_started': False,
              'finished_unix': time.time(), 'independently_admitted': quota_status(records)['admitted_observations']})
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('root')
    p.add_argument('--source-top')
    p.add_argument('--object')
    p.add_argument('--name')
    p.add_argument('--predecessor')
    p.add_argument('--compute-index', type=int, default=1)
    p.add_argument('--sim-index', type=int, default=3)
    a = p.parse_args()
    print(prepare(a.root, a.source_top, a.object, a.compute_index, a.sim_index,
                  name=a.name, predecessor=a.predecessor) if a.mode == 'prepare' else run(a.root))
