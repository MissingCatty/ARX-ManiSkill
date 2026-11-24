"""Offline exact-hand target sweeps; explicitly not full-arm motion certificates."""
import copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scripts.curobo_arx.data_factory.common import read, write, sha, digest
from .advisor36_export import load_lines
from .advisor37_surfaces import SurfaceChecker


def straight_target_sweep(snapshot, canonical_object, geometry):
    hand = ('right_link26', 'right_link27', 'right_link28')
    if not all(geometry['robot'].get(k) for k in hand):
        raise ValueError('Missing original ARX hand pieces')
    data = {k: v for k, v in geometry.items() if k != 'geometry_sha256'}
    data['robot'] = {k: data['robot'][k] for k in hand}
    checker = SurfaceChecker({**data, 'geometry_sha256': digest(data)})
    current = np.asarray(snapshot['link_world']['right_eef_link'])
    target = np.asarray(snapshot['objects']['test_object']['world']) @ canonical_object
    rotation = Rotation.from_matrix([current[:3, :3], target[:3, :3]])
    angle = float((rotation[0].inv() * rotation[1]).magnitude())
    distance = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    count = max(1, int(np.ceil(distance / .002)), int(np.ceil(angle / np.deg2rad(1.))))
    if count > 128:
        raise ValueError('Straight-sweep diagnostic exceeded registered near-field extent')
    interpolator, inverse = Slerp([0., 1.], rotation), np.linalg.inv(current)
    checked, first = 0, None
    for t in np.linspace(0, 1, count + 1):
        frame = np.eye(4)
        frame[:3, :3] = interpolator(t).as_matrix()
        frame[:3, 3] = current[:3, 3] * (1 - t) + target[:3, 3] * t
        delta = frame @ inverse
        state = copy.deepcopy(snapshot)
        for link in hand:
            state['link_world'][link] = (delta @ snapshot['link_world'][link]).tolist()
        check = checker.check(state)
        checked += 1
        if not check['pass']:
            first = {'fraction': float(t), 'result': check}
            break
    return {'target_sweep_clear': first is None, 'sampled_poses': checked,
            'first_intersection': first, 'target_geometry_sha256': geometry['geometry_sha256'],
            'endpoint_labels_changed': False, 'full_arm_IK_or_scene_path_certified': False,
            'physical_trajectory_executed': False,
            'scope': 'Original palm/finger cooked pieces at fixed measured opening; linear EEF translation and SLERP rotation against native target pieces; other links and environmental path feasibility remain separate'}


def report(campaign):
    root = Path(campaign).resolve()
    out = root / 'far_straight_target_sweep.json'
    if out.exists():
        raise FileExistsError('Do not overwrite diagnostic evidence')
    admission = root / 'dataset/admission.jsonl'
    records, cache = [], {}
    for row in load_lines(admission):
        if not row['admitted'] or row['category'] != 'funnel':
            continue
        dp = np.asarray(row['actual_translation_m'])
        if dp[2] <= .035 or np.linalg.norm(dp[:2]) <= .010:
            continue
        folder = Path(row['observation']).parent
        path = read(folder / 'surface_request.json')['geometry']
        if path not in cache:
            cache[path] = read(path)
        snapshot = read(row['observation'])['snapshot']
        result = straight_target_sweep(snapshot, np.array(row['candidate']['canonical_object']), cache[path])
        records.append({'sample_id': row['sample_id'], 'split': row['split'], **result})
    clear = sum(r['target_sweep_clear'] for r in records)
    result = {'complete': True, 'source_admission_sha256': sha(admission), 'observations': len(records),
              'target_sweep_clear': clear, 'target_sweep_clear_fraction': clear / len(records) if records else None,
              'selection': 'All independently admitted Z>35mm and lateral norm>10mm observations',
              'labels_still_point_to_Gcanonical': True, 'full_arm_path_feasibility_claim': False,
              'physical_trials': 0, 'training_started': False, 'records': records}
    write(out, result)
    return {k: v for k, v in result.items() if k != 'records'}
