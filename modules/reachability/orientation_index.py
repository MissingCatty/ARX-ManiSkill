"""Persistent orientation index over the original, immutable FK witnesses.

Base yaw leaves R.T @ world_up invariant. Index that two-dimensional quantity
first, then check the complete rotation. Missing entries remain unknown.
"""
from itertools import product
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ensure_run_path, now, read_json, sha256, write_json


def gravity_vectors(quaternions_wxyz):
    q = np.asarray(quaternions_wxyz, dtype=float)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid map quaternion')
    w, x, y, z = (q / norm).T
    return np.column_stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def cell_keys(points, cell_size, dimension):
    xyz = np.floor((np.clip(points, -1., 1.)+1.)/cell_size).astype(np.int64)
    xyz = np.clip(xyz, 0, dimension-1)
    return (xyz[:, 0]*dimension+xyz[:, 1])*dimension+xyz[:, 2]


class OrientationIndex:
    def __init__(self, map_directory, index_directory):
        started = time.perf_counter()
        self.directory = Path(map_directory).resolve()
        meta_path = self.directory/'地图元数据.json'
        complete = read_json(self.directory/'完成.json')
        if not complete.get('passed') or sha256(meta_path) != complete['file_sha256'][meta_path.name]:
            raise ValueError('Unverified reachability map metadata')
        self.meta = read_json(meta_path)
        if self.meta['frame'] != 'body4' or self.meta['pose_frame'] != 'action_tcp43':
            raise ValueError('Unexpected map coordinate convention')
        if self.meta['fields'][6:13] != ['tcp_x', 'tcp_y', 'tcp_z', 'qw', 'qx', 'qy', 'qz']:
            raise ValueError('Unexpected witness schema')
        witness = self.directory/self.meta['witness_file']
        if witness.resolve().parent != self.directory:
            raise ValueError('Witness must belong to the map directory')
        digest = complete['file_sha256'][witness.name]
        if sha256(witness) != digest:
            raise ValueError('Reachability witness checksum mismatch')
        shape = (self.meta['witnesses'], len(self.meta['fields']))
        if witness.stat().st_size != np.prod(shape)*np.dtype(self.meta['dtype']).itemsize:
            raise ValueError('Witness byte count differs from metadata')
        self.bank = np.memmap(witness, mode='r', dtype=self.meta['dtype'], shape=shape)
        index_directory = Path(index_directory)
        manifest = read_json(index_directory/'completion.json')
        if not manifest.get('passed') or manifest['witness_sha256'] != digest:
            raise ValueError('Orientation index belongs to another map')
        if manifest['metadata_sha256'] != sha256(meta_path):
            raise ValueError('Orientation index metadata differs from the map')
        for name, expected in manifest['file_sha256'].items():
            file = index_directory/name
            if file.resolve().parent != index_directory.resolve() or sha256(file) != expected:
                raise ValueError('Orientation index checksum mismatch')
        self.order = np.load(index_directory/'order.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(index_directory/'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.cell_size, self.dimension = manifest['cell_size'], manifest['dimension']
        if len(self.order) != len(self.bank) or len(self.offsets) != self.dimension**3+1:
            raise ValueError('Orientation index dimensions differ from witnesses')
        self.load_seconds = time.perf_counter()-started
        self.manifest = manifest

    @classmethod
    def build(cls, map_directory, index_directory, cell_size=.03):
        index_directory = ensure_run_path(index_directory)
        if index_directory.exists():
            raise ValueError('Index directory exists; retain the previous build')
        if not .005 <= cell_size <= .1:
            raise ValueError('Invalid orientation index cell size')
        index_directory.mkdir(parents=True)
        started = time.perf_counter()
        map_directory = Path(map_directory)
        meta = read_json(map_directory/'地图元数据.json')
        complete = read_json(map_directory/'完成.json')
        witness = map_directory/meta['witness_file']
        if not complete.get('passed') or sha256(witness) != complete['file_sha256'][witness.name]:
            raise ValueError('Cannot index an unverified witness bank')
        if sha256(map_directory/'地图元数据.json') != complete['file_sha256']['地图元数据.json']:
            raise ValueError('Map metadata changed')
        bank = np.memmap(witness, mode='r', dtype=meta['dtype'],
                         shape=(meta['witnesses'], len(meta['fields'])))
        dimension = int(np.floor(2./cell_size))+1
        keys = np.empty(len(bank), dtype=np.int32)
        for begin in range(0, len(bank), 250_000):
            keys[begin:begin+250_000] = cell_keys(
                gravity_vectors(bank[begin:begin+250_000, 9:13]), cell_size, dimension)
        order = np.argsort(keys, kind='stable').astype(np.uint32)
        counts = np.bincount(keys, minlength=dimension**3)
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        np.save(index_directory/'order.npy', order, allow_pickle=False)
        np.save(index_directory/'offsets.npy', offsets, allow_pickle=False)
        report = {'passed': True, 'at': now(), 'witnesses': len(bank),
                  'cell_size': cell_size, 'dimension': dimension,
                  'witness_sha256': complete['file_sha256'][witness.name],
                  'metadata_sha256': sha256(map_directory/'地图元数据.json'),
                  'build_seconds': time.perf_counter()-started,
                  'file_sha256': {name: sha256(index_directory/name) for name in ['order.npy', 'offsets.npy']}}
        write_json(index_directory/'completion.json', report)
        return report

    def query(self, goal_rotation, tolerance_rad):
        """Return every indexed witness satisfying the yaw-free necessary test."""
        rotation = np.asarray(goal_rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T@rotation, np.eye(3), atol=2e-5) or
                np.linalg.det(rotation) < .9999 or not 0 < tolerance_rad <= .1):
            raise ValueError('Invalid rotation query or tolerance')
        gravity = Rotation.from_matrix(rotation).as_matrix()[2]
        radius = 2*np.sin(tolerance_rad/2)+1e-6
        lower = np.floor((gravity-radius+1.)/self.cell_size).astype(int)
        upper = np.floor((gravity+radius+1.)/self.cell_size).astype(int)
        lower, upper = np.clip(lower, 0, self.dimension-1), np.clip(upper, 0, self.dimension-1)
        parts = []
        for x, y, z in product(*(range(lo, hi+1) for lo, hi in zip(lower, upper))):
            key = (x*self.dimension+y)*self.dimension+z
            a, b = self.offsets[key:key+2]
            if b > a:
                parts.append(self.order[a:b])
        if not parts:
            return np.empty(0, dtype=np.uint32)
        indices = np.concatenate(parts)
        dot = gravity_vectors(self.bank[indices, 9:13])@gravity
        return indices[dot >= np.cos(tolerance_rad+1e-7)]
