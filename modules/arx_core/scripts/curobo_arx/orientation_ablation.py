"""H023 fixed-input orientation intervention; original ARX calibration is read-only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from grasp_geometry_candidates import construct
from sphere_refinement import PAD_CENTER_EEF


def proposal(reference, arm):
    if arm not in ('leveled', 'neural_direction'):
        raise ValueError('Unregistered orientation arm')
    reference = Path(reference)
    source = json.loads((reference / 'selected_grasp.json').read_text())
    camera = np.asarray(json.loads((reference / 'camera_world_optical.json').read_text()))
    rows = np.load(reference / 'zerograsp/object_1.grasp.npy')
    row = np.asarray(source['network_row'])
    assert np.array_equal(row, rows[source['index']]), 'Archived row/index mismatch'
    rg = camera[:3, :3] @ row[4:13].reshape(3, 3)
    axes = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    contact = np.asarray(source['target_contact_world'], dtype=float)
    change = source.get('research_change', {})
    shift = float(change.get('world_z_pose_search_offset_m', 0.))
    assert shift in (0., .003, .006, .009, .012)
    trials = []
    for index, flip in enumerate((np.eye(3), np.diag([1, -1, -1]))):
        raw = np.eye(4)
        raw[:3, :3] = rg @ flip @ axes
        raw[:3, 3] = contact - .040 * rg[:, 0]
        horizontal, pre, lift, metadata = construct(raw, contact, {'accepted': False}, shift)
        expected = horizontal if change.get('hypothesis') == 'H011' else raw
        error = float(np.linalg.norm(expected[:3, :3] - np.asarray(source['grasp_world'])[:3, :3]))
        trials.append((error, index, raw, horizontal, pre, lift, metadata))
    error, flip_index, raw, leveled, pre, lift, metadata = min(trials, key=lambda x: x[0])
    assert error < 1e-5, 'Archived orientation cannot be reconstructed under frozen conventions'
    pad_target = np.asarray(metadata['pad_target_world'])
    if arm == 'leveled':
        grasp = leveled
    else:
        grasp = np.eye(4)
        grasp[:3, :3] = raw[:3, :3]
        grasp[:3, 3] = pad_target - grasp[:3, :3] @ PAD_CENTER_EEF
        pre = grasp.copy(); pre[:3, 3] -= .08 * grasp[:3, 2]
        lift = grasp.copy(); lift[2, 3] += .12
    assert np.linalg.norm(grasp[:3, 3] + grasp[:3, :3] @ PAD_CENTER_EEF - pad_target) < 1e-10
    assert np.linalg.norm(grasp[:3, 2] - raw[:3, 2]) < 1e-6
    metadata.update(hypothesis='H023', arm=arm,
                    original_hypothesis=change.get('hypothesis', 'original'),
                    applied=arm == 'neural_direction', sampled_tracking_delta_rad=.003,
                    proposal_reference=str(reference.resolve()), fixed_network_index=int(source['index']),
                    frozen_row_sha256=hashlib.sha256(row.tobytes()).hexdigest(),
                    flip_index=flip_index, archived_rotation_reconstruction_error=error,
                    source_pose_search_height_m=shift,
                    operation='Keep frozen pad target and approach direction; change closing-axis roll only',
                    nominal_goal_angular_difference_rad=float(Rotation.from_matrix(leveled[:3, :3] @ raw[:3, :3].T).magnitude()))
    selected = {k: source[k] for k in ('index', 'score', 'width_m', 'depth_m', 'network_row',
                                      'table_clearance_raise_m', 'target_contact_world')}
    selected.update(research_change=metadata, pregrasp_world=pre.tolist(),
                    grasp_world=grasp.tolist(), lift_world=lift.tolist())
    return selected
