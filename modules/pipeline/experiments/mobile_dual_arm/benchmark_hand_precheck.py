"""CPU geometry equivalence and timing over already-recorded map witnesses."""
import argparse
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.curobo.dual_arm.model import ArmModel
from modules.mobile_manipulation.hand_table_precheck import HandTablePrecheck


def run(root, source):
    if root.exists():
        raise ValueError('Use a new CPU comparison directory')
    root.mkdir(parents=True)
    original_file = source/'source_registered/modules/mobile_manipulation/hand_table_precheck.py'
    current_file = ROOT/'modules/mobile_manipulation/hand_table_precheck.py'
    spec = importlib.util.spec_from_file_location('reference_hand_precheck', original_file)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    registration = read_json(source/'registration.json')
    write_json(root/'registration.json', {'at': now(), 'source_replay': str(source.relative_to(ROOT)),
        'reference_code_sha256': sha256(original_file), 'current_code_sha256': sha256(current_file),
        'benchmark_code_sha256': sha256(__file__), 'physical_grasps': 0, 'new_IK_calls': 0,
        'paired_timing_repeats': 3, 'maximum_tip_penetration_m': .002, 'same_original_robot_meshes': True})
    (root/'source_registered').mkdir()
    (root/'source_registered/reference.py').write_bytes(original_file.read_bytes())
    (root/'source_registered/current.py').write_bytes(current_file.read_bytes())
    model, rows = ArmModel(), []
    for case in registration['cases']:
        state = read_json(ROOT/registration['source_run']/'trials'/case/'stationing/coarse/state.json')
        original = reference.HandTablePrecheck(model, state, .002)
        updated = HandTablePrecheck(model, state, .002)
        proposals = [p for p in read_json(source/'searches'/case/'map/search_result.json')['events']
                     if p.get('source') == 'inverse_reachability']
        before, after = [original.check(p) for p in proposals], [updated.check(p) for p in proposals]
        for i, (a, b) in enumerate(zip(before, after)):
            if a.keys() != b.keys():
                raise ValueError('Geometry receipt schema differs')
            for key in a:
                equal = abs(a[key]-b[key]) < 2e-12 if key == 'depth_m' else a[key] == b[key]
                if not equal:
                    raise ValueError(f'Geometry output differs: {case}/{i}/{key}')
        timings = {'reference': [], 'updated': []}
        for repeat in range(3):
            order = ['reference', 'updated'] if repeat % 2 == 0 else ['updated', 'reference']
            for name in order:
                checker = original if name == 'reference' else updated
                started = perf_counter()
                for proposal in proposals:
                    checker.check(proposal)
                timings[name].append(perf_counter()-started)
        a, b = [float(np.median(timings[name])) for name in ['reference', 'updated']]
        row = {'case': case, 'proposals': len(proposals), 'equivalent': True,
               'reference_median_seconds': a, 'updated_median_seconds': b, 'speedup': a/b,
               'timing_repeats': timings}
        rows.append(row)
        print(row, flush=True)
    result = {'at': now(), 'functional_equivalence_passed': True,
              'performance_improved': all(row['speedup'] > 1. for row in rows),
              'physical_grasps': 0, 'new_IK_calls': 0, 'paired_cases': rows,
              'original_V2_tip_rules_retained': True}
    write_json(root/'acceptance.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(ensure_run_path(args.root), ensure_run_path(args.source))
