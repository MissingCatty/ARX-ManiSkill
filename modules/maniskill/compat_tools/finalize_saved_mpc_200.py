"""Finish reporting for the completed MPC batch using only archived results."""
import fcntl
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import arx_graspcl_mpc_200_bench as full
import arx_graspcl_mpc_200_split_bench as split

OUT = ROOT / 'outputs/arx_graspcl_mpc_200_20261119_03'
read = full.read


def main():
    with (OUT/'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = read(OUT/'progress.json')
        timing = read(OUT/'suite_timing.json')
        halt = read(OUT/'halt.json')
        assert progress['completed'] == progress['registered'] == 200
        assert timing['all_owned_services_exited']
        assert "KeyError: 'wall_seconds'" in halt['error']
        reg = read(OUT/'registration.json')
        arm = OUT / full.ARM
        expected = {e['id'] for e in reg['schedule']}
        assert expected == {p.stem for p in (arm/'claims').glob('*.json')}
        no_timing = []
        for slot in sorted(expected):
            trial = arm/'trials'/slot
            row = read(trial/'result.json')
            if 'wall_seconds' not in row:
                assert row['reason'] == 'independent_scene_admission_failed'
                no_timing.append(slot)
        assert len(no_timing) == reg['admission_failed'] == 7
        print('200 archived results, seven admission failures have no execution time', flush=True)
        base = split.configure(reg)
        base.verify(OUT)
        print('Registered final verifier passed', flush=True)

        summarize = base.report
        source = inspect.getsource(summarize)
        old = "trial_wall_seconds=result['wall_seconds']"
        assert source.count(old) == 1
        namespace = dict(summarize.__globals__)
        exec(compile(source.replace(old, "trial_wall_seconds=result.get('wall_seconds')"),
                     __file__ + ':summarize', 'exec'), namespace)
        namespace[summarize.__name__](OUT)
        full.finalize(OUT)
        full.save(OUT/'report_recovery.json', {'prior_halt': halt, 'verified': True,
            'missing_trial_timing_for_admission_failures': no_timing,
            'round_timing_and_progress_original': True, 'new_physical_attempts': 0})
        full.save(OUT/'progress.json', {**progress, 'phase':'complete', 'final_verification_passed':True,
                   'report_recovery':'report_recovery.json'})
        print(read(OUT/'batch_summary.json'), flush=True)


if __name__ == '__main__':
    main()
