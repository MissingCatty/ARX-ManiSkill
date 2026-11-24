"""Finite masher handle reselection with unchanged local surface normals."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, write, sha
from scripts.curobo_arx.data_factory.visual_geometry import TargetGeometry
from .advisor36_geometry import pose
from .advisor36_contract import boundary_poses


def clearance_bound(check):
    value = check.get('camera_surface_distance_m')
    if value is None:
        value = check.get('camera_surface_distance_lower_bound_m')
    if value is None or not np.isfinite(value):
        raise ValueError('Missing measured/geometric camera clearance evidence')
    return float(value)


def masher_candidates(item, original, policy):
    episode = Path(item['seed']['episode'])
    with (episode / 'samples.jsonl').open() as stream:
        row = json.loads(next(stream))
    mount = np.linalg.inv(row['snapshot']['link_world']['right_eef_link']) @ row['camera_world_optical']
    from PIL import Image
    image = Image.open(episode / row['rgb_relative'])
    shape = [image.height, image.width]
    geometry = TargetGeometry.from_asset(item['asset'])
    strict = {**policy, 'minimum_surface_distance_m': .010}
    base_candidates = sorted([r for r in original['candidates'] if r['pass_local_geometry']],
                             key=lambda r: r['cost'])[:64]
    old = np.array(item['historical_G_object'])
    target_world = np.array(item['source_object_world'])
    rows = []
    for source in base_candidates:
        g = np.array(source['Gcanonical_object'])
        for roll in (0., 5., -5., 10., -10., 15., -15., 20., -20., 30., -30., 45., -45., 60., -60.):
            oriented = g.copy()
            oriented[:3, :3] = g[:3, :3] @ Rotation.from_rotvec([0., np.deg2rad(roll), 0.]).as_matrix()
            goal = pose(np.array(source['pad_center_object']), g[:3, 1], oriented)
            camera = target_world @ goal @ mount
            check = geometry.inspect(target_world, camera, row['intrinsics'], shape,
                                     row['camera_clipping_range_m'], strict)
            result = copy.deepcopy(source)
            result.update(index=370000 + len(rows), parent_index=source['index'],
                Gcanonical_object=goal.tolist(), closing_axis_roll_degrees=roll,
                method='advisor37_actual_handle_contact_center_with_closing_axis_roll',
                center_camera_check=check, all_seven_camera_checks=None,
                wrist_clearance_pass=False, physical_preflight_pass=False)
            distance = float(np.linalg.norm(old[:3, 3] - goal[:3, 3]))
            angle = float(np.rad2deg(Rotation.from_matrix(old[:3, :3].T @ goal[:3, :3]).magnitude()))
            result.update(cost=distance * 1000 + angle * .3,
                          goal_position_change_m=distance, goal_rotation_change_degrees=angle)
            rows.append(result)
    # Near historical wrist pose first; seven camera queries only for the
    # first32 center-clear candidates, with a fixed finite CPU search budget.
    clear = sorted([r for r in rows if r['center_camera_check']['pass']], key=lambda r: r['cost'])[:32]
    for candidate in clear:
        checks = []
        for boundary in boundary_poses(target_world @ np.array(candidate['Gcanonical_object']), 'coupled_xyz'):
            check = geometry.inspect(target_world, np.array(boundary['goal_world']) @ mount,
                                     row['intrinsics'], shape, row['camera_clipping_range_m'], strict)
            checks.append({'case': boundary['case'], **check})
        candidate['all_seven_camera_checks'] = checks
        candidate['wrist_clearance_pass'] = all(c['pass'] and clearance_bound(c) >= .010 for c in checks)
    good = [r for r in clear if r['wrist_clearance_pass']]
    return {'object_id': item['object_id'], 'candidates': rows,
            'selected': min(good, key=lambda r: r['cost']) if good else None,
            'pass_local_geometry': bool(good), 'normals_and_contact_centers_unchanged': True,
            'physical_trials': 0, 'source_sha256': sha(__file__),
            'wrist_minimum_distance_m': .010, 'seven_camera_checks_are_not_physical_7of7': True,
            'input_observation': str(episode / 'samples.jsonl'),
            'input_observation_sha256': sha(episode / 'samples.jsonl')}
