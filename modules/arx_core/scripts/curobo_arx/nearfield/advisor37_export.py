"""Independent paired admission and per-stratum quota checks; no training."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read, write, sha
from .advisor36_export import export as original_export, load_lines
from .advisor37_samples import quota_status


def independence(records):
    rows = [r for r in records if r['admitted']]
    poses = np.asarray([r['actual_eef_world'] for r in rows])
    if not rows:
        return {'pass': True, 'admitted': 0, 'cross_split_near_duplicates': 0}
    conflicts = []
    for i, j in cKDTree(poses[:, :3, 3]).query_pairs(.001):
        if rows[i]['split'] == rows[j]['split']:
            continue
        angle = Rotation.from_matrix(poses[i, :3, :3].T @ poses[j, :3, :3]).magnitude()
        if angle <= np.deg2rad(.6):
            conflicts.append([rows[i]['sample_id'], rows[j]['sample_id']])
    return {'pass': not conflicts, 'admitted': len(rows),
            'cross_split_near_duplicates': len(conflicts), 'conflicts': conflicts,
            'actual_pose_near_duplicate_translation_m': .001,
            'actual_pose_near_duplicate_rotation_degrees': .6}


def export(campaign, output, group_ids=None):
    root, out = Path(campaign), Path(output)
    groups = (sorted((root / 'groups').iterdir()) if group_ids is None
              else [root / 'groups' / name for name in group_ids])
    geometry_files = {}
    for group in groups:
        for path in group.glob('objects/*/static_collision_surfaces.json'):
            geometry = read(path)
            if (geometry['schema'] != 'arx_static_target_cooked_convex_v2'
                or not geometry['target_all_pieces_native']
                or geometry['whole_target_hull_fallback']
                or not all(p['native_convex'] for p in geometry['target'])):
                raise ValueError('Independent admission requires original native target convexes')
            geometry_files[str(path)] = sha(path)
    if not geometry_files:
        raise ValueError('No native ARX target geometry evidence')
    summary = original_export(root, out, group_ids)
    records = load_lines(out / 'admission.jsonl')
    splits = independence(records)
    write(out / 'independent_split_check.json', splits)
    if not splits['pass']:
        raise ValueError('Actual admitted ARX poses leak across dataset splits')
    quota = quota_status(records)
    write(out / 'quota_status.json', {**quota, 'geometry_files': geometry_files,
          'training_started': False, 'training_requires_user_confirmation': True})
    return {**summary, 'quota_minimum_met': quota['minimum_met']}
