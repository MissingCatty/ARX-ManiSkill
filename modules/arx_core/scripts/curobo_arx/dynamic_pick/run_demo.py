"""Repeat the ARX constant-velocity obstacle demo in the existing side simulator.

Uses an archived ZeroGrasp selection and fully optimized cuRobo paths with online
collision-prediction waiting. This is not a native MPC spatial-detour controller.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
TASK = ROOT / 'runs/arx_dynamic_pick'
MAIN = ROOT / 'runs/autoresearch_arx_20261022_approach_regression'
sys.path.insert(0, str(ROOT / 'scripts/curobo_arx'))
from client import Client


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def frozen_properties(fingerprint):
    """Compare physics/model properties, ignoring only declared side-task state."""
    value = copy.deepcopy(fingerprint['invariant'])
    # The side simulator restarted with an explicit output directory. This is
    # log routing, not robot/scene physics; retain the full value in raw evidence.
    value.get('configured', {}).get('runtime', {}).pop('run_dir', None)
    for path in list(value['usd']):
        if path.startswith('/World/DynamicPickObstacle'):
            value['usd'].pop(path)
        elif path.startswith('/World/COOLASSensors/'):
            attrs = value['usd'][path]['attributes']
            value['usd'][path]['attributes'] = {
                key: item for key, item in attrs.items() if not key.startswith('xformOp')}
    for path in list(value.get('runtime_objects', {})):
        if path.startswith('/World/DynamicPickObstacle'):
            value['runtime_objects'].pop(path)
    return value


def check_sources():
    expected = read(MAIN / 'configuration_guard.json')['file_sha256']
    changed = [path for path, digest in expected.items()
               if not Path(path).is_file()
               or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError('Protected sources changed: ' + ', '.join(changed))
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=time.strftime('demo_%Y%m%d_%H%M%S'))
    parser.add_argument('--restore-view', type=Path,
                        help='Restore a saved user viewport after a simulator restart')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        parser.error('--name must be a single directory name')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT,
                               text=True).strip() != 'zhaoyechong':
        raise RuntimeError('Run this side task on zhaoyechong')
    if read(MAIN / 'state.json')['status'] != 'paused_by_user':
        raise RuntimeError('Main research must remain paused during the side task')
    if os.environ.get('ARX_RESEARCH_GUARD_FILE'):
        raise RuntimeError('Use a separate side-task shell, without the main campaign guard')
    out = TASK / args.name
    if out.exists() or (TASK / (args.name + '_setup.json')).exists():
        raise FileExistsError('Preserve existing attempts; choose a new --name')
    data = TASK.resolve().parent
    if shutil.disk_usage(data).free < 1073741824:
        raise RuntimeError('Less than the required 1 GiB reserve on artifact storage')
    lock = (TASK / '.demo.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checked = check_sources()
    client = Client(TASK / 'mailbox')
    snap = client.request('snapshot')
    if snap.get('streaming') or any(snap.get('moving_obstacle', {}).get('velocity', [])):
        raise RuntimeError('Another side controller/obstacle motion is active')
    if snap.get('moving_obstacle'):
        raise RuntimeError('Restart the side simulator before another demo; retain prior evidence. '
                           'PhysX fixture repositioning after Fabric updates is not validated.')
    setup = client.request('dynamic_setup', scenario=read(TASK / 'task_contract.json')['scenario'])
    write(TASK / (args.name + '_setup.json'), setup)
    historical = read(TASK / 'initial_setup.json')['fingerprint']
    if frozen_properties(setup['fingerprint']) != frozen_properties(historical):
        raise RuntimeError('Original robot/scene properties differ; setup evidence retained')
    if args.restore_view:
        client.request('grasp_viewport_state', restore=read(args.restore_view))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_CACHE_PATH=str(data / 'host-cuda-cache'),
               XDG_CACHE_HOME=str(data / 'host-cache'),
               TORCH_EXTENSIONS_DIR=str(data / 'torch-extensions'), TMPDIR=str(data / 'host-tmp'))
    for key in ['CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'TMPDIR']:
        Path(env[key]).mkdir(exist_ok=True, parents=True)
    # The controller creates the fresh trial directory and warms models before recording.
    with (TASK / (args.name + '.log')).open('x') as log:
        process = subprocess.run([sys.executable, str(CODE / 'online_time_pick.py'),
                                  '--name', args.name], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    if not out.exists():
        raise RuntimeError('Controller failed before creating evidence; inspect the trial log')
    shutil.copy2(CODE / 'online_time_pick.py', out / 'runner_source.py')
    shutil.copy2(CODE / 'sim_runner.py', out / 'sim_runner_source.py')
    final = client.request('grasp_research_fingerprint', timeout=180)
    write(out / 'final_fingerprint.json', final)
    write(out / 'final_snapshot.json', client.request('snapshot'))
    write(out / 'final_viewport.json', client.request('grasp_viewport_state'))
    changed = frozen_properties(setup['fingerprint']) != frozen_properties(final)
    count_after = check_sources()
    write(out / 'configuration_review.json', {
        'pass': not changed, 'protected_files_checked_before': checked,
        'protected_files_checked_after': count_after,
        'original_configuration_changed': changed,
        'nonphysical_output_directory_comparison_excluded': True,
        'main_status': read(MAIN / 'state.json')['status'],
        'method': 'Online collision-prediction waiting; no native MPC success claim'})
    if (out / 'frames/frames.json').exists() and len(read(out / 'frames/frames.json')['frames']) > 1:
        subprocess.run([sys.executable, str(CODE / 'encode_walltime.py'), str(out)],
                       cwd=ROOT, env=env, check=True)
    if changed:
        raise RuntimeError('Configuration guard failed; raw trial retained')
    if process.returncode:
        raise RuntimeError('Physical demo failed; raw data/video retained: ' + str(out))
    print(out / 'dynamic_obstacle_grasp.mp4')


if __name__ == '__main__':
    main()
