"""Advisor28 bounded four-slot coordinator; explicit resource and shutdown budgets."""
import argparse
from pathlib import Path
import subprocess
import threading
import time
import traceback
from scripts.curobo_arx.data_factory.common import read,write
from scripts.curobo_arx.data_factory.watchdog import process_info,exclusive_stop
from .learned_campaign import PYTHON,registered_launcher_project
from .physical_startup import startup_expired
from . import campaign as launcher

def run(campaign):
    campaign = Path(campaign).resolve();reg = read(campaign / 'registration.json');directory = campaign / 'workers/policy'
    if (campaign / 'STOP').exists() or directory.exists():raise RuntimeError('No service or experiment retries')
    from scripts.curobo_arx.data_factory.launch import devices, environment, allocate
    from .resources import probe_cuda
    devices_now = devices();allocate(devices_now, 1, reg['requested_devices']['compute_uuid'], [reg['requested_devices']['sim_uuid']])
    device = next(d for d in devices_now if d['uuid'] == reg['learned_policy']['gpu_uuid'])
    if device['free_mb'] < reg['policy_memory_floor_mib']:raise RuntimeError('Policy GPU lacks the registered free memory')
    proof = probe_cuda(device, environment(device));write(campaign / 'policy_cuda_preflight.json', proof)
    if not proof['pass']:raise RuntimeError('Policy CUDA initialization failed')
    directory.mkdir(parents=True);(directory / 'queue').mkdir();(directory / 'done').mkdir()
    env = environment(device);env['PYTHONPATH'] = reg['source_snapshot']
    command = [str(PYTHON), '-m', reg.get('policy_module', 'training.nearfield_arx.learned_policy'), str(campaign)]
    process = None;finished = threading.Event()
    try:
        with (directory / 'process.log').open('x') as log:
            process = subprocess.Popen(command, cwd=reg['source_snapshot'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(directory / 'process.json', {'pid': process.pid, 'command': command, 'owned': True, **(process_info(process.pid) or {})})
        deadline = time.monotonic() + 600
        while not (directory / 'ready.json').exists():
            if process.poll() is not None or time.monotonic() >= deadline or (campaign / 'STOP').exists():
                raise RuntimeError('Pinned learned policy service failed to become ready')
            time.sleep(.25)
        def monitor():
            while not finished.wait(.5):
                if process.poll() is not None:
                    exclusive_stop(campaign / 'STOP', {'reason': 'owned_learned_policy_service_exited'});return
        watcher = threading.Thread(target=monitor, daemon=True);watcher.start()
        previous = launcher.startup_expired
        launcher.startup_expired = lambda group, elapsed: startup_expired(group, elapsed, 1800)
        try:
            with registered_launcher_project(reg):
                launcher.run(campaign, reg['requested_devices']['compute_uuid'], reg['requested_devices']['sim_uuid'])
        finally:launcher.startup_expired = previous
    except BaseException:
        error = traceback.format_exc();exclusive_stop(campaign / 'STOP', {'reason': 'learned_coordinator_fault', 'error': error})
        write(campaign / 'coordinator_fault.json', {'error': error})
        if not (campaign / 'launch.json').exists():
            write(campaign / 'state.json', {'status': 'fault', 'physical_trials': 0, 'unconfirmed_world': False, 'error': error})
        raise
    finally:
        finished.set()
        if process is not None:
            write(directory / 'STOP', {'reason': 'bounded learned four-slot campaign finished'})
            try:process.wait(timeout=300)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=30)
            write(directory / 'process_exit.json', {'pid': process.pid, 'exit_code': process.returncode, 'wall': time.time()})
    from .learned_report import report
    return report(campaign)



def main(campaign):
    from training.nearfield_arx import closure_recovery as policy
    from training.nearfield_arx.closure_verification import audit_policy
    from . import learned_report
    policy.install();learned_report.decode=policy.decode;learned_report.audit_policy=audit_policy
    return run(campaign)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');main(p.parse_args().campaign)
