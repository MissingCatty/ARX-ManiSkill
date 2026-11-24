"""Confirm exact-mesh behavior for deterministic best archived witnesses."""
from pathlib import Path
import time

import numpy as np

from audit import HERE, ROOT as AUDIT, ASSET, now, read, save, sha
from geometry import Geometry
from narrow_phase import ConvexFingerPolicy

ROOT = HERE / 'runs/mesh_witnesses_20261123_02'
LEVELS = [0., .0005, .001, .002]


def run():
    assert read(AUDIT / 'completion_verification.json')['passed']
    assert not ROOT.exists(), 'Preserve previous witness runs'
    original = read(AUDIT / 'registration.json')
    schedule = []
    hashes = {str(AUDIT / 'completion_verification.json'): sha(AUDIT / 'completion_verification.json')}
    for row in original['schedule']:
        source = AUDIT / 'cases' / row['id']
        result = read(source / 'result.json')
        with np.load(source / 'clearance.npz') as data:
            keep = data['other_world_clear'] & (data['nonfinger'] <= 0)
            chosen = []
            if keep.any(): chosen.append(int(np.flatnonzero(keep)[np.argmin(data['finger'][keep])]))
            for t in result['thresholds']:
                if t['first_distal_solution'] is not None: chosen.append(t['first_distal_solution'])
        for name in ['best_table_witness', 'best_local_witness']:
            if result[name] is not None: chosen.append(result[name]['solution_index'])
        schedule.append({**row, 'solution_indices': sorted(set(chosen))})
        for n in ['result.json', 'clearance.npz']: hashes[str(source / n)] = sha(source / n)
    save(ROOT / 'registration.json', {'at': now(), 'stage': 'targeted_exact_mesh_crosscheck',
        'schedule': schedule, 'positions': 67, 'witnesses': sum(len(s['solution_indices']) for s in schedule),
        'selection': 'Best whole-finger clearance, first distal solution at every tolerance, and both archived minimum-depth witnesses; deduplicated',
        'candidate_cap_for_this_witness_crosscheck': 'Deterministic best witnesses only, not exhaustive exact-mesh IK enumeration',
        'thresholds_m': LEVELS, 'physical_attempts': 0, 'input_sha256': hashes,
        'source_sha256': {str(p): sha(p) for p in [Path(__file__), HERE/'narrow_phase.py', HERE/'geometry.py']}})
    rows = []; begin = time.perf_counter()
    for row in schedule:
        assert not (ROOT / 'STOP').exists()
        source = Path(row['source_case']); state = read(source / 'state.json')
        geometry = Geometry(source / 'plan/arx.yml', state)
        policy = ConvexFingerPolicy(geometry, state['scene_cuboids']['table'], 0., ASSET)
        with np.load(source / 'endpoint_ik.npz') as saved:
            qs, indices = saved['solved_q'], saved['solved_indices']
        data = np.load(AUDIT / 'cases' / row['id'] / 'clearance.npz')
        receipts = []
        for index in row['solution_indices']:
            levels = []
            for tolerance in LEVELS:
                policy.tolerance_m = tolerance
                result = policy.evaluate_one(qs[index], stage='final_approach', other_world_clear=bool(data['other_world_clear'][index]))
                levels.append(result)
            receipts.append({'solution_index': index, 'candidate_index': int(indices[index, 0]),
                'branch_index': int(indices[index, 1]), 'q': qs[index], 'thresholds': levels})
        result = {**row, 'witnesses': receipts,
                  'thresholds': [{'tolerance_m': t, 'valid_witnesses': sum(w['thresholds'][i]['valid'] for w in receipts),
                       'valid_candidates': len({w['candidate_index'] for w in receipts if w['thresholds'][i]['valid']}),
                       'requires_actual_contact': any(w['thresholds'][i]['valid'] and w['thresholds'][i]['actual_tip_contact'] for w in receipts)}
                      for i, t in enumerate(LEVELS)], 'physical_attempts': 0}
        save(ROOT / 'cases' / row['id'] / 'result.json', result); rows.append(result)
        save(ROOT / 'progress.json', {'phase': 'running', 'registered': 67, 'completed': len(rows), 'last': row['id']})
        print('MESH', len(rows), row['id'], [(x['tolerance_m'], x['valid_candidates']) for x in result['thresholds']], flush=True)
    cohorts = {}
    for group in ['original53', 'unresolved_near14']:
        part = [r for r in rows if r['cohort'] == group]
        cohorts[group] = [{'tolerance_m': t, 'confirmed_endpoint_recovered_positions': sum(r['thresholds'][i]['valid_candidates'] > 0 for r in part),
                           'confirmed_real_contact_positions': sum(r['thresholds'][i]['requires_actual_contact'] for r in part)}
                          for i, t in enumerate(LEVELS)]
    for p, h in hashes.items(): assert sha(p) == h
    save(ROOT / 'summary.json', {'completed': 67, 'positions': 67, 'witnesses': sum(len(r['witnesses']) for r in rows),
         'cohorts': cohorts, 'physical_attempts': 0, 'wall_s': time.perf_counter()-begin,
         'scope': 'Confirmed lower bound from deterministic witnesses; exact meshes not enumerated for every branch'})
    save(ROOT / 'completion_verification.json', {'passed': True, 'completed': 67, 'inputs_unchanged': True, 'physical_attempts': 0})
    save(ROOT / 'progress.json', {'phase': 'complete', 'completed': 67, 'registered': 67})


if __name__ == '__main__':
    try: run()
    except BaseException:
        import traceback
        if ROOT.exists(): (ROOT / 'STOP').write_text(traceback.format_exc())
        raise
