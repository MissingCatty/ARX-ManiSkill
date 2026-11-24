"""Start a preregistered benchmark once its predecessor finishes and exits."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def readiness(root):
    """No inference, outcomes, physical requests or learning are involved."""
    root = Path(root)
    try:
        reg = read(root/'registration.json')
        progress = read(root/'progress.json')
        clock = read(root/'suite_timing.json')
        phase = progress['phase']
        if phase in ['halted', 'stopped', 'draining_after_error'] or (root/'STOP').exists():
            return 'blocked', 'Predecessor stopped; never resume or replay it automatically'
        if phase != 'complete':
            return 'waiting', f"Predecessor {phase}: {progress['completed']}/{reg['total_slots']}"
        if progress['completed'] != reg['total_slots']:
            return 'blocked', 'Predecessor completion count differs from its budget'
        if clock.get('phase') != 'complete' or not clock.get('all_owned_services_exited'):
            return 'waiting', 'Waiting for predecessor shutdown receipts'
        verification = root/'final_verification.json'
        if not verification.exists():
            return 'waiting', 'Waiting for predecessor final verification'
        if not read(verification).get('passed'):
            return 'blocked', 'Predecessor final verification failed'
        return 'ready', 'All predecessor cases verified and owned services exited'
    except (FileNotFoundError, json.JSONDecodeError):
        return 'waiting', 'Waiting for complete predecessor status files'


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'is-active', unit],
                            text=True, capture_output=True)
    return result.stdout.strip() or 'unknown'


def wait_and_start(root):
    root = Path(root).resolve()
    queue = read(root/'queue_registration.json')
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'queue_start.json').exists() or (root/'suite_timing.json').exists():
            raise RuntimeError('This registered successor already started; never replay')
        previous_message = None
        while True:
            if (root/'STOP').exists():
                save(root/'queue_status.json', {'phase':'cancelled', 'at':now()})
                return
            phase, reason = readiness(queue['predecessor_output'])
            units = {u:unit_state(u) for u in queue['wait_for_units']}
            if phase == 'ready' and any(s in ['active', 'activating', 'deactivating'] for s in units.values()):
                phase, reason = 'waiting', 'Waiting for predecessor controller and reporter to exit'
            if phase == 'ready' and any(s == 'failed' for s in units.values()):
                phase, reason = 'blocked', 'A predecessor unit failed; retain the registered successor without starting'
            status = {'phase':phase, 'updated_at':now(), 'reason':reason,
                      'predecessor_output':queue['predecessor_output'], 'unit_states':units,
                      'successor_physical_attempts_before_release':0}
            save(root/'queue_status.json', status)
            if reason != previous_message:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                previous_message = reason
            if phase == 'blocked':
                raise RuntimeError(reason)
            if phase == 'ready':
                break
            time.sleep(5)
        reg = read(root/'registration.json')
        for filename, digest in reg['source_hashes'].items():
            assert sha(filename) == digest, 'Frozen successor dependency changed: '+filename
        for name, _, _ in reg['arms']:
            assert not list((root/name/'claims').glob('*.json')), 'Never replay started successor slots'
        record = Path(queue['batch_record_path'])
        content = record.read_text()
        if '## '+root.name not in content:
            record.write_text(content.rstrip()+'\n\n'+queue['batch_record_section'].rstrip()+'\n')
        receipt = {'released_at':now(), 'predecessor_verified':True,
                   'predecessor_verification_sha256':sha(Path(queue['predecessor_output'])/'final_verification.json'),
                   'command':queue['command'], 'new_component_startup_required':True,
                   'test_results_imported_into_memory':0}
        with (root/'queue_start.json').open('x') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
        save(root/'queue_status.json', {'phase':'released', **receipt})
        print('PREDECESSOR COMPLETE; STARTING REGISTERED GRASPCL ROUND', flush=True)
        os.execv(queue['command'][0], queue['command'])


def self_test():
    with tempfile.TemporaryDirectory(prefix='arx_queue_gate_') as temporary:
        root = Path(temporary)
        save(root/'registration.json', {'total_slots':200})
        save(root/'progress.json', {'phase':'running', 'completed':70})
        save(root/'suite_timing.json', {'phase':'starting_components'})
        assert readiness(root)[0] == 'waiting'
        save(root/'progress.json', {'phase':'complete', 'completed':200})
        assert readiness(root)[0] == 'waiting'
        save(root/'suite_timing.json', {'phase':'complete', 'all_owned_services_exited':True})
        assert readiness(root)[0] == 'waiting'
        save(root/'final_verification.json', {'passed':True})
        assert readiness(root)[0] == 'ready'
        save(root/'final_verification.json', {'passed':False})
        assert readiness(root)[0] == 'blocked'
        save(root/'final_verification.json', {'passed':True})
        save(root/'progress.json', {'phase':'complete', 'completed':199})
        assert readiness(root)[0] == 'blocked'
        save(root/'progress.json', {'phase':'halted', 'completed':70})
        assert readiness(root)[0] == 'blocked'
    print(json.dumps({'passed':True, 'queue_gate_checks':7, 'physical_attempts':0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.output is not None
        wait_and_start(args.output)
