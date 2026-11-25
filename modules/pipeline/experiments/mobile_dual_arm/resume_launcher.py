"""Wait for verified inheritance and resource admission, then launch once."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from arx_project.layout import ROOT, now, read_json, write_json, sha256
from modules.pipeline.experiments.mobile_dual_arm.workspace_priority.resources import select_resources


def launch(root):
    with (root/'resume_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (root/'STOP').exists() or (root/'started.json').exists():
                raise RuntimeError('Continuation has already started or stopped')
            ready = root/'inheritance_verified.json'
            progress = read_json(root/'progress.json')
            evidence = read_json(ready) if ready.exists() else None
            if (not evidence or (root/'inheritance_staging').exists()
                    or progress['completed'] != evidence['complete']
                    or progress.get('interrupted') != evidence['interrupted']):
                write_json(root/'resume_launch_status.json', {'at': now(), 'state': 'waiting_for_inheritance',
                    'materialized_trials': len(list((root/'trials').iterdir()))})
                time.sleep(10)
                continue
            try:
                roles, floors, probes = select_resources(ray_minimum_mib=23000)
            except RuntimeError as error:
                write_json(root/'resume_launch_status.json', {'at': now(), 'state': 'waiting_for_resources',
                    'reason': str(error), 'services_started': False, 'new_slots_claimed': 0})
                time.sleep(30)
                continue
            break
        reg = read_json(root/'registration.json')
        from modules.pipeline.experiments.mobile_dual_arm.continuation import verify_inheritance
        verify_inheritance(root, reg)
        revision = root/'registration_revisions/launch_resources'
        revision.mkdir(parents=True)
        shutil.copy2(root/'registration.json', revision/'registration.json')
        old_roles = reg['gpu_roles']
        reg['gpu_roles'], reg['minimum_free_mib'] = roles, floors
        reg['source_hashes'][str(Path(__file__).resolve())] = sha256(__file__)
        snapshot = root/'source_registered'/Path(__file__).resolve().relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, snapshot)
        write_json(root/'registration.json', reg)
        write_json(root/'preflight/resources_after_inheritance.json', {'at': now(),
            'before_first_service_or_new_slot': True, 'previous_roles': old_roles,
            'new_roles': roles, 'minimum_free_mib': floors, 'fresh_cuda_probes': probes})
        common = ['systemd-run', '--user', '--property=WorkingDirectory='+str(ROOT),
            '--property=StandardError=inherit', '--setenv=PYTHONPATH='+str(ROOT),
            '--setenv=OMP_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1']
        python = str(ROOT/'environments/simulation/bin/python')
        controller = 'arx-'+root.name.replace('_', '-')
        watcher = controller+'-watch'
        subprocess.run(common+['--unit='+controller, '--property=StandardOutput=append:'+str(root/'controller.log'),
            python, '-u', '-m', 'modules.pipeline.experiments.mobile_dual_arm.run',
            'run', '--root', str(root)], check=True)
        subprocess.run(common+['--unit='+watcher, '--property=StandardOutput=append:'+str(root/'followup.log'),
            python, '-u', '-m', 'modules.pipeline.experiments.mobile_dual_arm.finish',
            '--root', str(root), '--watch'], check=True)
        write_json(root/'resume_launch_status.json', {'at': now(), 'state': 'started',
            'controller': controller+'.service', 'watcher': watcher+'.service',
            'gpu_roles': roles, 'pending_slots': reg['continuation']['new_execution_slot_budget']})
        print('CONTINUATION_SERVICES_STARTED',roles,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    launch(parser.parse_args().root.resolve())
