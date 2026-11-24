"""Canonical active ARX right-arm collision model; retired spheres cannot load."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT / 'configs/collision/arx_refined_871.json'
COUNTS = {f'right_link{i}': n for i, n in zip(range(21, 29), [26, 115, 97, 36, 24, 89, 258, 226])}
PARAMETER_SHA256 = '66eced4df51a49edd97ccece7c9276065bbf33a75974a8b29aed127d604ca085'


def sphere_digest(spheres):
    return hashlib.sha256(json.dumps(spheres, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_profile():
    profile = json.loads(PROFILE_PATH.read_text())
    spheres = profile['collision_spheres']
    if profile.get('profile_id') != 'arx_refined_871_v1' or {n: len(v) for n, v in spheres.items()} != COUNTS:
        raise ValueError('The active ARX collision profile must contain the approved 871 spheres')
    if sphere_digest(spheres) != PARAMETER_SHA256:
        raise ValueError('Active ARX sphere centers/radii differ from the approved 871-sphere model')
    for values in spheres.values():
        for sphere in values:
            if len(sphere['center']) != 3 or not all(math.isfinite(v) for v in [*sphere['center'], sphere['radius']]) or sphere['radius'] <= 0:
                raise ValueError('Invalid collision sphere')
    return profile


def active_spheres():
    return deepcopy(load_profile()['collision_spheres'])


def require_current_profile(config):
    kin = config.get('robot_cfg', config).get('kinematics', config)
    if sphere_digest(kin.get('collision_spheres', {})) != PARAMETER_SHA256:
        raise ValueError('Retired ARX collision-sphere configuration rejected; current model requires 871 approved spheres')
    if set(kin.get('collision_link_names', [])) != set(COUNTS) or kin.get('collision_sphere_buffer', 0.) != 0.:
        raise ValueError('Current ARX collision links or sphere buffer changed')
    return {'profile_id': 'arx_refined_871_v1', 'total_spheres': 871,
            'sphere_parameter_sha256': PARAMETER_SHA256, 'counts': COUNTS,
            'profile_path': str(PROFILE_PATH)}
