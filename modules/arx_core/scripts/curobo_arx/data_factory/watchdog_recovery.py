"""Prepare a new ARX registration after externally verified process death.

Invoked only by the sidecar with the existing cuRobo Python. No services start
in this helper, and no original record/STOP is rewritten or deleted.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

try:
    from .watchdog import (container_running, group_records, manual_stop, owned_process,
                           read, sha, write, fatal_evidence, storage_ready)
except ImportError:  # Direct invocation of the immutable sidecar copy.
    from watchdog import (container_running, group_records, manual_stop, owned_process,
                          read, sha, write, fatal_evidence, storage_ready)


def inspect_records(parent):
    """Check every occurred command; interrupted execution remains unknown."""
    db = sqlite3.connect('file:' + str(parent / 'ledger.sqlite3') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Parent ledger is corrupt')
        rows = [dict(r) for r in db.execute('SELECT * FROM attempts')]
    finally:
        db.close()
    for row in rows:
        if not row['episode']:
            if row['state'] == 'committed':
                raise ValueError('Committed attempt lost its episode path')
            continue
        episode = Path(row['episode'])
        executions = {p.name.removesuffix('_execution.json') for p in episode.glob('*_execution.json')}
        requests_path = episode / 'requests.json'
        requests = read(requests_path) if requests_path.exists() else []
        labels = {r['label'] for r in requests}
        reported = set()
        if (episode / 'grasp_report.json').exists():
            reported.update(read(episode / 'grasp_report.json').get('stage_timing', {}))
        if (episode / 'physical_receipt.json').exists():
            reported.update(read(episode / 'physical_receipt.json').get('executed_stages', []))
        if executions - labels:
            raise ValueError('Execution receipt has no command request: ' + row['id'])
        if reported - labels:
            raise ValueError('Stage report has no command request: ' + row['id'])
        for label in labels | executions | reported:
            if not (episode / (label + '_command.json')).is_file():
                raise ValueError('Occurred/requested execution lost its command file: ' + row['id'] + ':' + label)
            if row['state'] in ('committed', 'failed', 'visual_rejected', 'duplicate_excluded'):
                for suffix in ('_execution.json', '_settling.json'):
                    if not (episode / (label + suffix)).is_file():
                        raise ValueError('Terminal attempt lost its execution evidence: ' + row['id'])
        if row['state'] in ('committed', 'failed', 'visual_rejected', 'duplicate_excluded'):
            for name in ('episode.json', 'physical_receipt.json', 'quality_receipt.json'):
                if not (episode / name).is_file():
                    raise ValueError('Terminal attempt lost its outcome evidence: ' + row['id'] + ':' + name)
        if row['state'] in ('committed', 'duplicate_excluded'):
            stored = json.loads(row['receipt'])
            quality = read(episode / 'quality_receipt.json')
            physical = read(episode / 'physical_receipt.json')
            meta = read(episode / 'episode.json')
            if (quality.get('pass') is not True or physical.get('pass') is not True or
                    meta.get('write_complete') is not True or meta.get('attempt_id') != row['id'] or
                    meta.get('object_id') != row['object_id']):
                raise ValueError('Successful episode lost acceptance/identity evidence')
            if (sha(episode / 'quality_receipt.json') != stored['quality_sha256'] or
                    sha(episode / 'physical_receipt.json') != stored['physical_sha256']):
                raise ValueError('Accepted receipt changed')
            for relative, expected in quality['files'].items():
                if sha(episode / relative) != expected:
                    raise ValueError('Accepted raw evidence changed: ' + str(episode / relative))
    return rows


def probe_storage(root, mounts):
    code = ("from pathlib import Path; p=Path(" + repr(str(root / 'write_probe.txt')) + "); "
            "p.write_text('watchdog-storage-probe'); assert p.read_text()=='watchdog-storage-probe'; "
            "p.unlink(); print('WRITE_PROBE_OK')")
    command = ['podman','run','--rm','--user','0','--security-opt=label=disable',
               *mounts,'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return {'pass': result.returncode == 0 and 'WRITE_PROBE_OK' in result.stdout,
            'runtime_user': 0, 'physical_trials': 0, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def prepare(request):
    request = read(request); cfg = read(request['watchdog_config'])
    home = Path(request['watchdog_config']).parent
    parent, out = Path(request['parent']).resolve(), Path(request['out']).resolve()
    project = Path(cfg['project_root']).resolve()
    if manual_stop(parent) or (home / 'STOP').exists():
        return {'status': 'manual_stop'}
    if not storage_ready(cfg):
        return {'status': 'resource_wait', 'reason': 'Registered data mount/readiness marker is unavailable'}
    if out.exists():
        raise FileExistsError('Recovery requires a fresh immutable directory')
    lock = (parent / 'coordinator.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        old = read(parent / 'registration.json')
        launch = (read(parent / 'launch.json') if (parent / 'launch.json').exists()
                  else {'stop_unix': old['deadline_unix'], 'started_unix': old['campaign_started_unix']})
        if old.get('stage') != 'graspness_teacher':
            raise ValueError('This recovery adapter is only registered for Graspness ARX teachers')
        if time.time() >= launch['stop_unix']:
            return {'status': 'complete', 'reason': 'Original production deadline expired'}
        fatal = fatal_evidence(parent)
        if fatal:
            raise ValueError('Configuration/record integrity fault: ' + fatal)
        processes = []
        process_files = [(parent / 'coordinator_process.json', 'coordinator')]
        process_files += [(p, 'worker') for p in (parent / 'workers').glob('*/process.json')]
        process_files += [(p, 'group') for _, p in group_records(parent) if p.exists()]
        for path, role in process_files:
            meta = read(path)
            if owned_process(meta, parent, role):
                raise ValueError('Old owned process is still alive: ' + str(meta['pid']))
            processes.append({'registration': str(path), 'pid': meta['pid'], 'old_owned_process_exited': True})
        containers = []
        for group, _ in group_records(parent):
            name = 'arx-data-factory-' + group['group_id']
            if container_running(name) is not None:
                raise ValueError('Old virtual World is present or its absence is unverified: ' + name)
            containers.append({'name': name, 'absent': True})
        if not containers and not (parent / 'coordinator_process.json').is_file():
            raise ValueError('Missing positive coordinator ownership evidence for pre-simulator recovery')
        # Import only the original immutable code used by this collector.
        sys.path.insert(0, old['source_snapshot'])
        from scripts.curobo_arx.data_factory.common import verify_files, space
        from scripts.curobo_arx.data_factory.launch import allocate, devices
        from scripts.curobo_arx.data_factory.resume import copy_and_recover
        from scripts.curobo_arx.data_factory.storage import register_bulk_storage, container_storage_mounts
        verify_files({**old['execution_source_files'], **old['dependencies']})
        # The coordinator runs from the workspace; its code must still match.
        for name in ('launch', 'common', 'ledger', 'report', 'storage'):
            path = project / ('scripts/curobo_arx/data_factory/' + name + '.py')
            verify_files({str(path): old['source_files'][str(path)]})
        try:
            space(project / 'runs'); space('/coolas/data')
            inventory = devices()
            compute, simulators = allocate(inventory, old['envs_per_gpu'], cfg['compute_uuid'], cfg['sim_uuids'])
        except (OSError, RuntimeError, StopIteration, subprocess.SubprocessError) as error:
            return {'status': 'resource_wait', 'reason': str(error)}
        rows = inspect_records(parent)
        proof = {'old_process_exited': True, 'environments_idle': True, 'containers_absent': True,
            'checked_unix': time.time(),
            'idle_basis': 'All registered virtual Worlds destroyed; pre-crash convergence remains unverified',
            'evidence': {'processes': processes, 'containers': containers,
                'no_simulator_group_dispatched': not bool(containers),
                'parent_state': read(parent / 'state.json'), 'parent_STOP_sha256': sha(parent / 'STOP'),
                'external_watchdog_authorized_recovery': str(home)}}
        if manual_stop(parent) or (home / 'STOP').exists():
            return {'status': 'manual_stop'}
        out.mkdir()
        recovery = copy_and_recover(parent / 'ledger.sqlite3', out / 'ledger.sqlite3', proof)
        write(out / 'exit_proof.json', proof); write(out / 'recovery_receipt.json', recovery)
        reg = copy.deepcopy(old)
        reg.update(created_unix=time.time(), deadline_unix=launch['stop_unix'],
            campaign_started_unix=old.get('campaign_started_unix', launch['started_unix']),
            lineage=old.get('lineage', []) + [{'campaign': str(parent),
                'registration_sha256': sha(parent / 'registration.json'),
                'state_sha256': sha(parent / 'state.json'), 'status': 'fault',
                'status_basis': 'External process/World exit proof; original state file retained'}],
            wall_budget_scope='Original cumulative attempt limit and original deadline retained',
            resume={'parent': str(parent), 'proof_sha256': sha(out / 'exit_proof.json'),
                'recovery_receipt_sha256': sha(out / 'recovery_receipt.json'), **recovery,
                'runtime_change': 'Automatic external-watchdog process recovery; same execution source and device UUIDs',
                'frozen_unchanged': 'Robot, physics, cameras, placements, model, success and admission criteria'},
            bulk_storage=register_bulk_storage(out, Path(cfg['bulk_parent']) / out.name))
        write(out / 'registration.json', reg); write(out / 'catalog.json', reg['catalog'])
        shutil.copy2(parent / 'storage_policy.json', out / 'storage_policy.json')
        shutil.copy2(parent / 'method_receipt.json', out / 'method_receipt.json')
        probe = probe_storage(Path(reg['bulk_storage']['root']), container_storage_mounts(out, reg))
        write(out / 'storage_preflight_probes.json', probe)
        if not probe['pass']:
            raise ValueError('New runtime-user storage mapping probe failed')
        for row in rows:
            video = parent / 'videos' / (row['id'] + '.mp4')
            if row['state'] == 'committed' and video.is_file():
                target = out / 'videos' / video.name; target.parent.mkdir(exist_ok=True)
                try:
                    os.link(video, target)
                except OSError:
                    shutil.copy2(video, target)
        write(out / 'preflight.json', {'pass': True, 'wall': time.time(), 'compute': compute,
            'simulators': simulators, 'exit_proof_verified': True, 'accepted_records_verified': True,
            'watchdog_recovery_sha256': sha(__file__),
            'source_and_dependency_hashes_verified': True, 'robot_physics_and_gates_unchanged': True})
        write(out / 'state.json', {'status': 'prepared', 'successes': len(recovery['inherited_success_ids']),
            'new_successes': 0, 'physical_attempts': recovery['inherited_physical_attempts'], 'quota_complete': False})
        return {'status': 'prepared', 'campaign': str(out), 'recovery': recovery}
    finally:
        lock.close()


if __name__ == '__main__':
    request = read(sys.argv[1])
    try:
        result = prepare(sys.argv[1])
    except Exception:
        result = {'status': 'blocked', 'error': traceback.format_exc()}
    write(request['result'], result)
    print(json.dumps(result))
    sys.exit(1 if result['status'] == 'blocked' else 0)
