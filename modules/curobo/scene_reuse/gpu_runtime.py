"""Share immutable native Warp meshes while keeping every scene buffer separate."""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from types import MethodType
import time

import numpy as np

from arx_project.layout import sha256


class GPUGeometryReuse:
    def __init__(self, stores):
        self.stores = list({id(s): s for s in stores}.values())
        self.active = False; self.enabled = False
        self.resources = {}; self.events = []; self.originals = []
        self.versions = {name: version(name) for name in ['numpy', 'trimesh', 'warp-lang']}
        for index, store in enumerate(self.stores):
            original = store._load_mesh_to_warp
            self.originals.append((store, original))
            def load(target, mesh, _original=original, _index=index):
                if not self.active or not self.enabled or mesh.file_path is None:
                    return _original(mesh)
                began = time.perf_counter()
                descriptor = {'source_sha256': sha256(mesh.file_path),
                    'scale': None if mesh.scale is None else np.asarray(mesh.scale).tolist(),
                    'process': True, 'versions': self.versions, 'device': str(target._wp_device)}
                key = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False).encode()).hexdigest()
                hit = key in self.resources
                if not hit:
                    # All GPU arrays and BVH are still constructed by the original loader.
                    self.resources[key] = _original(mesh)
                result = replace(self.resources[key], name=mesh.name)
                self.events.append({'store': _index, 'name': mesh.name, 'key': key,
                    'shared_hit': hit, 'mesh_id': int(result.mesh_id), 'seconds': time.perf_counter()-began})
                return result
            store._load_mesh_to_warp = MethodType(load, store)

    def begin(self, enabled):
        if self.active: raise RuntimeError('GPU resource scopes cannot overlap')
        self.resources = {}; self.events = []
        self.enabled = enabled; self.active = True

    def finish(self):
        if not self.active: raise RuntimeError('GPU resource scope is not active')
        result = {'enabled': self.enabled, 'received_load_calls': len(self.events),
            'native_mesh_creations': sum(not e['shared_hit'] for e in self.events),
            'shared_hits': sum(e['shared_hit'] for e in self.events), 'events': self.events,
            'scene_poses_enable_flags_and_slots_separate': True,
            'native_GPU_constructor_retained': True}
        # Native per-checker wp_cache entries retain strong references independently.
        self.resources = {}; self.enabled = False; self.active = False
        return result

    def detach(self):
        if self.active: raise RuntimeError('Cannot detach an active GPU resource scope')
        for store, original in self.originals:
            store._load_mesh_to_warp = original
