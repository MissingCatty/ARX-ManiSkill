"""Check real saved pools and arm choices without CUDA or robot execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import statistics
import time

from arx_project.json_io import read_json as fast_read
from arx_project.layout import ROOT, now, read_json, sha256, write_json
from .loader import load_staged


def run(root, staged):
    modules = load_staged(staged, planning=True, client=False)
    Planner = modules['modules.curobo.dual_arm.placement_service'].PlacementPlanner
    baseline = ROOT/'tmp/mobile_dual_arm/v4_high100_resume_20261125_02'
    registration = read_json(baseline/'registration.json')
    checks, timing = [], []

    class Model:
        def bind(self, state, arm, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            return {**deepcopy(state), 'planning_arm': arm}

    class Pool:
        def __init__(self, outcomes):
            self.outcomes = outcomes
            self.seen = []

        def request(self, directory, **flags):
            from .store import candidate_pool
            directory = Path(directory)
            candidates = candidate_pool(directory)['candidates']
            self.seen.append((directory.name, len(candidates)))
            result = deepcopy(self.outcomes[directory.name])
            if result['success'] and result['selected']['index'] >= len(candidates):
                return {'success': False, 'reason': 'no_feasible_native_post_or_recalled_candidate'}
            return result

    class CheckedPlanner(Planner):
        def opening_sweep(self, work, canonical):
            return {'passed': True, 'scope': 'Saved-arm-choice verification; opening physics validated separately'}

    for entry in registration['schedule']:
        original = baseline/'trials'/entry['id']
        if not (original/'plan_result.json').exists():
            continue
        result = fast_read(original/'plan_result.json')
        if not result.get('success'):
            continue
        source = original/'candidates/candidate_pool.json'
        target = root/'cases'/entry['id']
        (target/'candidates').mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target/'candidates/candidate_pool.json')
        shutil.copy2(original/'state.json', target/'state.json')
        actual = json.loads(source.read_text())
        compatible = read_json(source)
        fast = fast_read(source)
        if compatible != fast:
            raise RuntimeError('Fast compatibility reader changed saved data: '+entry['id'])
        outcomes = {}
        for side in ['right', 'left']:
            saved = original/'arms'/side/'plan_result.json'
            outcomes[side] = fast_read(saved) if saved.exists() else {
                'success':False, 'reason':'not_examined_by_source'}
        pool = Pool(outcomes)
        planner = CheckedPlanner(pool, Model(), root/'unused')
        start = time.perf_counter()
        got = planner.request(target, arm=result.get('forced_arm'))
        elapsed = time.perf_counter()-start
        selected = got.get('selected', {})
        expected = result['selected']
        same = (got['success'] == result['success'] and selected.get('arm') == expected['arm']
            and selected.get('index') == expected['index'] and selected.get('score') == expected['score']
            and selected.get('contact_pose_world') == expected['contact_pose_world'])
        old_bounds = [(row['arm'], row['retained_prefix_count']) for row in result['acceleration']['rank_bounds']]
        same &= pool.seen == old_bounds
        proof = fast_read(target/'candidate_io.json')
        same &= proof['decode_count'] == 1 and proof['full_arm_pool_copies'] == 0
        same &= json.loads((target/'candidates/candidate_pool.json').read_text()) == actual
        checks.append({'id':entry['id'], 'passed':bool(same), 'candidate_count':len(actual['candidates']),
            'arm':selected.get('arm'), 'index':selected.get('index'),
            'same_prefix_visits': pool.seen == old_bounds, 'new_adapter_seconds':elapsed,
            'input_sha256':sha256(source)})
        if len(timing) < 3:
            start = time.perf_counter(); read_json(source); old_seconds = time.perf_counter()-start
            start = time.perf_counter(); fast_read(source); new_seconds = time.perf_counter()-start
            timing.append({'id':entry['id'], 'before_seconds':old_seconds, 'after_seconds':new_seconds})
        write_json(root/'progress.json', {'checked':len(checks),'passed':sum(c['passed'] for c in checks)})
    report = {'at':now(), 'passed':bool(checks) and all(c['passed'] for c in checks),
        'checks':checks, 'reader_timing':timing, 'physical_grasps':0, 'CUDA_calls':0,
        'scope':'Exact saved candidate values, native score order, arm selection and per-arm prefix bounds'}
    write_json(root/'verification.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--staged',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.root.resolve(),args.staged.resolve())
    print('SAVED_POOLS',len(result['checks']),result['passed'])
