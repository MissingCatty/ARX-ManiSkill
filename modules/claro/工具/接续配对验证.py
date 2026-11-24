"""接续未开始物理执行的配对场景，每例单独启动仿真进程。"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import 做配对验证 as pairing
from benchmark_lifecycle import require_ready

read, save, sha256 = pairing.read, pairing.save, pairing.sha256
FIRST = pairing.OUT / 'S5_固定挡物配对01'


def run_in_isolated_process(directory, sockets, row):
    result = {**row, 'status': 'completed', 'phase': '进场观测', 'placed': False,
              'removed': False, 'reobserved': False, 'success': False}
    start = time.monotonic()
    try:
        pairing.run_case(directory, sockets, row, result)
    except Exception as error:
        result.update(status='infrastructure_fault', reason=str(error), error=traceback.format_exc())
        raise
    finally:
        result['isolated_scene_wall_seconds'] = time.monotonic() - start
        save(directory / '配对结果.json', result)


def registration_for_resume():
    old_reg = read(FIRST / '配对登记.json')
    receipt = read(FIRST / '配对汇总.json')
    ready = read(FIRST / 'resident/all_ready.json')
    for name in old_reg['required_components']:
        owned = read(FIRST / ('resident/' + name + '_stopped.json'))
        if (owned['pid'] != ready['components'][name]['pid'] or Path('/proc', str(owned['pid'])).exists()):
            raise ValueError('Previous workers have not verified shutdown')
    by_case = {row['case']: row for row in receipt['rows']}
    if set(by_case) != {row['case'] for row in old_reg['schedule']}:
        raise ValueError('Prior trial accounting differs from the original schedule')

    schedule, included = [], []
    files = [Path(__file__), FIRST / '配对登记.json', FIRST / '配对汇总.json', FIRST / '批次故障.json',
             FIRST / 'resident/all_ready.json']
    for row in old_reg['schedule']:
        case = row['case']
        old = by_case[case]
        trial = FIRST / '试次' / case
        if old['status'] == 'completed':
            included.append(case)
            files.append(trial / '配对结果.json')
        elif old['status'] == 'unexecuted' and not trial.exists():
            schedule.append(row)
        elif (old['status'] == 'infrastructure_fault' and old['phase'] == '进场观测'
              and old['reason'] == 'Failed to find a supported physical device "cuda:0"'
              and sorted(p.name for p in trial.iterdir()) == ['试次领取.json', '配对结果.json']):
            files.extend((trial / '试次领取.json', trial / '配对结果.json'))
            schedule.append({**row, 'preceding_setup_failure': str(trial / '配对结果.json'),
                             'preceding_commanded_robot_actions': 0})
        else:
            raise ValueError('Unsafe or unknown prior execution state: ' + case)
    if len(included) != 9 or len(schedule) != 4:
        raise ValueError('Expected nine verified results and four not yet executed')
    return {**old_reg, 'schedule': schedule, 'cases': len(schedule), 'preserved_cases': included,
            'first_epoch': str(FIRST), 'isolated_sim_process_per_case': True,
            'source_hashes': {**old_reg['source_hashes'], **{str(p): sha256(p) for p in files}}}


def run(directory, planner, neural):
    sim = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if directory.parent != pairing.OUT or not all(gpu.startswith('GPU-') for gpu in (sim, planner, neural)):
        raise ValueError('Use the stage 06 path and physical GPUs')
    registration = registration_for_resume()
    devices = {'zerograsp': neural, 'curobo': planner}
    registration.update(devices=devices, sim_gpu=sim)
    directory.mkdir(exist_ok=False)
    (directory / '试次').mkdir()
    save(directory / '配对登记.json', registration)
    os.environ['MS_ASSET_DIR'] = str(pairing.MS / 'assets')
    os.environ.setdefault('SAPIEN_CACHE_DIR', str(pairing.ROOT / '缓存/sapien'))
    start = time.monotonic()
    try:
        with pairing.ready(directory, devices) as sockets:
            for filename, digest in registration['source_hashes'].items():
                if sha256(Path(filename)) != digest:
                    raise ValueError('Frozen file differs: ' + filename)
            for row in registration['schedule']:
                require_ready(directory / 'resident', list(devices))
                trial = directory / '试次' / row['case']
                trial.mkdir()
                save(trial / '试次领取.json', row)
                save(directory / '当前进度.json', {'case': row['case'], 'state': 'executing'})
                print('开始独立仿真', row['case'], flush=True)
                argv = [sys.executable, str(Path(__file__).resolve()), 'scene', str(trial),
                        str(sockets['curobo']), str(sockets['zerograsp'])]
                with (trial / '仿真日志.txt').open('x') as log:
                    proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                                          cwd=pairing.ROOT, timeout=480)
                if proc.returncode or not (trial / '配对结果.json').is_file():
                    raise RuntimeError('Isolated simulator failed: ' + row['case'])
                receipt = read(trial / '配对结果.json')
                print(row['case'], receipt['phase'], receipt.get('reason'), flush=True)
                pairing.report(directory, registration)
    except Exception as error:
        save(directory / '批次故障.json', {'reason': str(error), 'error': traceback.format_exc()})
        raise
    finally:
        report = pairing.report(directory, registration)
        save(directory / '批次耗时.json', {'wall_seconds_including_warmup': time.monotonic() - start})
        state = 'completed' if report['completed'] == report['total'] else 'stopped'
        save(directory / '当前进度.json', {'state': state, 'completed': report['completed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'scene'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('planner')
    parser.add_argument('neural')
    args = parser.parse_args()
    path = args.directory.resolve()
    if args.mode == 'run':
        run(path, args.planner, args.neural)
    else:
        run_in_isolated_process(path, {'curobo': Path(args.planner), 'zerograsp': Path(args.neural)},
                                read(path / '试次领取.json'))
