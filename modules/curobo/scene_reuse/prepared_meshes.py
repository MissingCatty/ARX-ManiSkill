"""Content-addressed, bounded CPU cache of the original processed mesh arrays."""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import time
from typing import Any

import numpy as np

from arx_project.layout import ensure_run_path, sha256, write_json


@dataclass(frozen=True)
class PreparedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    key: str


class PreparedMeshCache:
    def __init__(self, root: Path, maximum_memory_bytes: int = 128 * 1024**2) -> None:
        self.root = ensure_run_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if maximum_memory_bytes < 0:
            raise ValueError('The CPU cache memory limit must be nonnegative')
        self.limit = maximum_memory_bytes
        self.memory: OrderedDict[str, PreparedMesh] = OrderedDict()
        self.bytes = 0
        self.lock = RLock()
        self.counts: Counter = Counter()
        self.timings: Counter = Counter()
        self.events: list[dict] = []
        self.versions = {'numpy': version('numpy'), 'trimesh': version('trimesh')}

    def identity(self, mesh: Any) -> tuple[dict, str]:
        descriptor = {'schema': 1, 'source_sha256': sha256(mesh.file_path),
                      'scale': None if mesh.scale is None else np.asarray(mesh.scale).tolist(),
                      'process': True, 'process_color': True, 'versions': self.versions}
        text = json.dumps(descriptor, sort_keys=True, allow_nan=False, separators=(',', ':'))
        return descriptor, hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def validate(vertices: np.ndarray, faces: np.ndarray) -> None:
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or vertices.dtype.kind != 'f' or faces.dtype.kind not in 'iu'
                or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid prepared collision geometry')

    def remember(self, item: PreparedMesh) -> None:
        size = item.vertices.nbytes + item.faces.nbytes
        if size > self.limit:
            return
        while self.memory and self.bytes + size > self.limit:
            _, old = self.memory.popitem(last=False)
            self.bytes -= old.vertices.nbytes + old.faces.nbytes
            self.counts['memory_evictions'] += 1
        self.memory[item.key] = item
        self.bytes += size

    def read_disk(self, directory: Path, descriptor: dict, key: str) -> PreparedMesh:
        meta = json.loads((directory / 'metadata.json').read_text())
        if meta['identity'] != descriptor:
            raise ValueError('Prepared mesh identity mismatch')
        arrays = []
        for name in ['vertices', 'faces']:
            file = directory / (name + '.npy')
            if sha256(file) != meta['files'][file.name]:
                raise ValueError('Corrupt prepared mesh array: ' + str(file))
            arrays.append(np.load(file, allow_pickle=False))
        self.validate(*arrays)
        for array in arrays:
            array.setflags(write=False)
        return PreparedMesh(*arrays, key)

    def build(self, mesh: Any, directory: Path, descriptor: dict, key: str) -> PreparedMesh:
        began = time.perf_counter()
        vertices, faces = mesh.get_mesh_data(process=True)
        self.timings['original_mesh_processing_seconds'] += time.perf_counter() - began
        self.counts['original_mesh_processing_calls'] += 1
        if sha256(mesh.file_path) != descriptor['source_sha256']:
            raise ValueError('Collision source changed during preprocessing')
        vertices = np.array(vertices, copy=True, order='C')
        faces = np.array(faces, copy=True, order='C')
        self.validate(vertices, faces)
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=self.root) as work:
            temporary = Path(work)
            np.save(temporary / 'vertices.npy', vertices, allow_pickle=False)
            np.save(temporary / 'faces.npy', faces, allow_pickle=False)
            write_json(temporary / 'metadata.json', {'identity': descriptor,
                'files': {name: sha256(temporary / name) for name in ['vertices.npy', 'faces.npy']},
                'vertices': len(vertices), 'triangles': len(faces)})
            # The manifest and arrays become visible together to all consumers.
            os.replace(temporary, directory)
        self.timings['disk_write_seconds'] += time.perf_counter() - began
        for array in [vertices, faces]:
            array.setflags(write=False)
        return PreparedMesh(vertices, faces, key)

    def get(self, mesh: Any) -> PreparedMesh:
        began = time.perf_counter()
        with self.lock:
            stamp = time.perf_counter()
            descriptor, key = self.identity(mesh)
            self.timings['source_identity_seconds'] += time.perf_counter() - stamp
            if key in self.memory:
                item = self.memory.pop(key)
                self.memory[key] = item
                origin = 'memory'
            else:
                directory = self.root / key
                # Locks are per content key, including when different processes share a cache.
                with (self.root / (key + '.lock')).open('a+b') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    if directory.exists():
                        stamp = time.perf_counter()
                        item = self.read_disk(directory, descriptor, key)
                        self.timings['disk_read_seconds'] += time.perf_counter() - stamp
                        origin = 'disk'
                    else:
                        item = self.build(mesh, directory, descriptor, key)
                        origin = 'processed'
                self.remember(item)
            self.counts[origin] += 1
            elapsed = time.perf_counter() - began
            self.timings['lookup_total_seconds'] += elapsed
            self.events.append({'key': key, 'source_sha256': descriptor['source_sha256'],
                                'origin': origin, 'seconds': elapsed,
                                'vertices': len(item.vertices), 'triangles': len(item.faces)})
            return item

    def report(self) -> dict:
        return {'counts': dict(self.counts), 'timings': dict(self.timings), 'events': self.events,
                'memory_bytes': self.bytes, 'maximum_memory_bytes': self.limit,
                'directory': str(self.root), 'original_processing_options_retained': True}
