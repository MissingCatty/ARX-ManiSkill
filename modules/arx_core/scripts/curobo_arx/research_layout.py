"""Check matched initial ARX scene states; tolerate only small settling noise."""
import numpy as np
from scipy.spatial.transform import Rotation


def compare_layout(expected, actual):
    failures = []; objects = {}
    if expected['joint_names'] != actual['joint_names']:
        failures.append('joint_names')
    else:
        errors = np.abs(np.asarray(expected['joint_positions']) - actual['joint_positions'])
        # Frozen tolerances account for numerical/settling noise, not scene changes.
        tolerance = np.array([.001 if n in ('joint4', 'right_joint27', 'right_joint28',
                                          'left_joint17', 'left_joint18') else .005
                              for n in expected['joint_names']])
        if (errors > tolerance).any():
            failures.append('initial_joint_positions')
    if set(expected['objects']) != set(actual['objects']):
        failures.append('object_names')
    for name in sorted(set(expected['objects']) & set(actual['objects'])):
        left, right = expected['objects'][name], actual['objects'][name]
        a, b = np.asarray(left['world']), np.asarray(right['world'])
        position = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        angle = float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
        ok = left['path'] == right['path'] and position <= .001 and angle <= .02
        objects[name] = {'position_error_m': position, 'rotation_error_rad': angle, 'pass': ok}
        if not ok:
            failures.append('object:' + name)
    base_error = float(np.abs(np.asarray(expected['robot_base_world']) - actual['robot_base_world']).max())
    if base_error > .0005:
        failures.append('robot_base')
    return {'pass': not failures, 'failures': failures, 'objects': objects,
            'initial_joint_max_error': float(errors.max()) if expected['joint_names'] == actual['joint_names'] else None,
            'robot_base_matrix_max_error': base_error,
            'limits': {'object_translation_m': .001, 'object_rotation_rad': .02,
                       'revolute_rad': .005, 'prismatic_m': .001, 'base_matrix': .0005}}
