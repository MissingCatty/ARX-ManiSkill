"""Frozen sources and explicit reach-map definitions."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT as PROJECT_ROOT, canonical_path, read_json

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / 'tmp/reachability/experiments'
PROJECTS = ROOT.parent
MS = PROJECTS / 'maniskill'
REFERENCE = PROJECTS / 'claro/docs/06_做配对验证/resources/S5_固定挡物配对01/试次/I0069/01_抓挡'
MODEL = PROJECTS / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.yml'
URDF = PROJECTS / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
TCP = PROJECTS / 'arx_core/configs/runtime/action_tcp.json'
JOINTS = ['right_joint' + str(n) for n in range(21, 27)]
YAW_BINS, COS_BINS, ROLL_BINS = 12, 8, 12
ORIENTATIONS = YAW_BINS * COS_BINS * ROLL_BINS
DOWN_ORIENTATIONS = YAW_BINS * 2 * ROLL_BINS
FIELDS = ['q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'tcp_x', 'tcp_y', 'tcp_z',
          'qw', 'qx', 'qy', 'qz', 'joint_margin_rad', 'joint_margin_normalized', 'jacobian_condition_index']


def read(p):return read_json(p)


def save(p, value):
    p = canonical_path(p);p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n');temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with canonical_path(p).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):h.update(chunk)
    return h.hexdigest()


def now():return datetime.now().astimezone().isoformat()


def orientation_ids(rotations):
    z = rotations[:, :, 2]
    phi = np.arctan2(z[:, 1], z[:, 0])
    cos = np.clip(z[:, 2], -1., 1.)
    sin = np.sqrt(np.maximum(0., 1.-cos*cos))
    x0 = np.column_stack([cos*np.cos(phi), cos*np.sin(phi), -sin])
    y0 = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(len(phi))])
    x = rotations[:, :, 0]
    roll = np.arctan2(np.sum(x*y0, axis=1), np.sum(x*x0, axis=1))
    a = np.floor((phi+np.pi)/(2*np.pi)*YAW_BINS).astype(np.int64) % YAW_BINS
    b = np.minimum(COS_BINS-1, np.floor((cos+1.)/2*COS_BINS).astype(np.int64))
    c = np.floor((roll+np.pi)/(2*np.pi)*ROLL_BINS).astype(np.int64) % ROLL_BINS
    return (a*COS_BINS+b)*ROLL_BINS+c, b < 2


def voxel_ids(positions, resolution):
    cells = np.floor(np.asarray(positions, dtype=np.float64)/resolution).astype(np.int64)
    shifted = cells+2048
    if not ((shifted >= 0) & (shifted < 4096)).all():raise ValueError('Position outside map encoding domain; check units')
    return (shifted[:, 0]*4096+shifted[:, 1])*4096+shifted[:, 2], cells


def unpack_voxels(ids):
    z = ids % 4096-2048;y = (ids//4096) % 4096-2048;x = ids//4096//4096-2048
    return np.column_stack([x, y, z]).astype(np.int16)


def static_scene(state):
    inverse = np.linalg.inv(np.asarray(state['links']['body4']))
    scene = {'cuboid': {}}
    for name, box in state['scene_cuboids'].items():
        transform = np.eye(4);transform[:3, 3] = box['center'];transform = inverse @ transform
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        scene['cuboid'][name] = {'pose': [*transform[:3, 3].tolist(), *quaternion.tolist()], 'dims': box['dims']}
    return scene
