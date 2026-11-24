"""Reject inconsistent frozen inference replay; never aligns or changes a scene."""
import json,os
from pathlib import Path
import numpy as np


def check(live, reference, target):
    live, reference = Path(live), Path(reference)
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(live, 'capture_snapshot.json'), read(reference, 'capture_snapshot.json')
    def pose(x, y):
        x, y = np.asarray(x), np.asarray(y)
        distance = float(np.linalg.norm(x[:3, 3] - y[:3, 3]))
        angle = float(np.arccos(np.clip((np.trace(x[:3, :3] @ y[:3, :3].T) - 1) / 2, -1, 1)))
        return {'position_error_m': distance, 'orientation_error_rad': angle,
                'pass': bool(np.isfinite([distance, angle]).all() and distance <= .001 and angle <= .005)}
    camera = pose(read(live, 'camera_world_optical.json'), read(reference, 'camera_world_optical.json'))
    obj = pose(a['objects'][target]['world'], b['objects'][target]['world'])
    base = pose(a['link_world']['body4'], b['link_world']['body4'])
    intrinsics = read(live, 'camera/params.json')['intrinsics'] == read(reference, 'camera/params.json')['intrinsics']
    result = {'pass': camera['pass'] and obj['pass'] and base['pass'] and intrinsics,
              'reference': str(reference), 'camera': camera, 'target': obj, 'base': base,
              'intrinsics_equal': intrinsics, 'thresholds': {'translation_m': .001, 'rotation_rad': .005},
              'scope': 'Frozen prediction input-consistency gate only; no scene adjustment, pose compensation or evaluator change'}
    (live / 'replay_observation_check.json').write_text(json.dumps(result, indent=2))
    if not result['pass']:
        raise RuntimeError('Frozen prediction observation consistency rejected; no robot motion allowed')
    return result


if __name__ == '__main__':
    reference = os.environ.get('GRASP_REPLAY_REFERENCE')
    if reference:
        check(Path(__file__).resolve().parents[2] / 'runs/arx_grasp', reference, os.environ.get('GRASP_TARGET', 'red_cube'))
