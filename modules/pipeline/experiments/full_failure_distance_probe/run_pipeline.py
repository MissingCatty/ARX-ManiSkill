"""Own three independent workers and stop the diagnostic on any fault."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/all53_20261123_01'
WORKSPACE = HERE.parents[2]
ENVIRONMENT = WORKSPACE / 'arx_core/scripts/curobo_arx/environment.sh'
JOBS = [
    ('capture', 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834', WORKSPACE/'maniskill/.venv/bin/python', ['capture']),
    ('planner0', 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169', WORKSPACE/'curobo/.venv/bin/python', ['plan', '--worker', '0']),
    ('planner1', 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b', WORKSPACE/'curobo/.venv/bin/python', ['plan', '--worker', '1'])]


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not (ROOT/'controller_started.json').exists()
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {s.split(',')[0].strip(): int(s.split(',')[1]) for s in raw.splitlines()}
    for name, gpu, *_ in JOBS: assert free[gpu] >= (4000 if name == 'capture' else 10000), (name, free)
    save(ROOT/'resource_preflight.json', {'passed': True, 'free_mib': free, 'jobs': [(n,g) for n,g,*_ in JOBS]})
    save(ROOT/'controller_started.json', {'pid': os.getpid(), 'automatic_retries': 0})
    children = []
    try:
        for name, gpu, python, flags in JOBS:
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=gpu, ARX_SIM_GPU=gpu, ARX_PLANNER_GPU=gpu, ARX_NEURAL_GPU=gpu, LILO_GPU=gpu)
            cmd = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'failure-distance', str(ENVIRONMENT), str(python), str(HERE/'probe.py'), *flags]
            stream = (ROOT/(name+'.log')).open('x')
            child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stream.close(); children.append((name, child))
        save(ROOT/'owned_workers.json', [{'name': n, 'pid': c.pid} for n,c in children])
        while True:
            if (ROOT/'STOP').exists(): raise RuntimeError('Worker preserved a diagnostic fault')
            failed = [(n,c.returncode) for n,c in children if c.poll() is not None and c.returncode != 0]
            if failed: raise RuntimeError('Owned worker failed: '+str(failed))
            rows = []
            for p in (ROOT/'cases').glob('*/result.json'):
                r = json.loads(p.read_text()); rows.append(r)
            save(ROOT/'progress.json', {'phase': 'running', 'completed': len(rows), 'registered': 318,
                'original_cases_diagnosed': sum(r['level']==0 for r in rows),
                'planning_passes': sum(r['planning_success'] for r in rows),
                'physical_attempts': 0, 'last_update_unix': time.time()})
            if all(c.poll() is not None for _,c in children): break
            time.sleep(5)
        assert len(rows) == 318
        assert json.loads((ROOT/'capture_complete.json').read_text())['passed']
        for i in range(2): assert json.loads((ROOT/f'workers/{i}/complete.json').read_text())['passed']
        save(ROOT/'summary.json', {'completed': 318, 'registered': 318, 'source_failures': 53, 'physical_attempts': 0,
            'cases': sorted(rows, key=lambda r:r['id']), 'source_inputs_unchanged': True})
        save(ROOT/'progress.json', {'phase': 'complete', 'completed': 318, 'registered': 318, 'source_failures': 53,
            'planning_passes': sum(r['planning_success'] for r in rows), 'physical_attempts': 0})
        save(ROOT/'completion_verification.json', {'passed': True, 'completed': 318, 'source_failures': 53, 'physical_attempts': 0})
    except BaseException:
        import traceback
        if not (ROOT/'STOP').exists(): (ROOT/'STOP').write_text(traceback.format_exc())
        for name, child in children:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for name, child in children:
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        raise


if __name__ == '__main__': main()
