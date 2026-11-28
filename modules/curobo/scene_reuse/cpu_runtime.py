"""Feed cached CPU arrays into each instance's unchanged native Warp loader."""
from types import MethodType
import time
from typing import Any

from .prepared_meshes import PreparedMeshCache


class MeasuredMesh:
    def __init__(self, source: Any, owner: 'CPUGeometryReuse') -> None:
        self.source, self.owner = source, owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def get_mesh_data(self, process: bool = True) -> tuple:
        began = time.perf_counter()
        try:
            if self.owner.cache is not None and self.source.file_path is not None and process:
                item = self.owner.cache.get(self.source)
                return item.vertices, item.faces
            return self.source.get_mesh_data(process=process)
        finally:
            self.owner.cpu_seconds += time.perf_counter() - began


class CPUGeometryReuse:
    def __init__(self, stores: list) -> None:
        self.stores = list({id(store): store for store in stores}.values())
        self.cache = None
        self.active = False
        self.events: list[dict] = []
        self.cpu_seconds = 0.
        self.originals = []
        self.store_indices = {id(store): i for i, store in enumerate(self.stores)}
        for store in self.stores:
            original = store._load_mesh_to_warp
            self.originals.append((store, original))
            def load(target: Any, mesh: Any, _original=original) -> Any:
                if not self.active:
                    return _original(mesh)
                began = time.perf_counter()
                before = self.cpu_seconds
                passed = False
                try:
                    result = _original(MeasuredMesh(mesh, self))
                    passed = True
                    return result
                finally:
                    self.events.append({'name': mesh.name, 'store': self.store_indices[id(target)],
                        'seconds': time.perf_counter() - began,
                        'cpu_geometry_seconds': self.cpu_seconds - before, 'passed': passed})
            store._load_mesh_to_warp = MethodType(load, store)

    def begin(self, cache: PreparedMeshCache | None) -> None:
        if self.active:
            raise RuntimeError('CPU geometry scopes cannot overlap')
        self.cache = cache
        self.events, self.cpu_seconds = [], 0.
        self.active = True

    def finish(self) -> dict:
        if not self.active:
            raise RuntimeError('CPU geometry scope is not active')
        self.active = False
        total = sum(e['seconds'] for e in self.events)
        result = {'enabled': self.cache is not None, 'native_load_calls': len(self.events),
                  'cpu_geometry_seconds': self.cpu_seconds, 'native_load_total_seconds': total,
                  'native_load_other_host_seconds': total - self.cpu_seconds,
                  'native_GPU_upload_and_BVH_implementation_unchanged': True,
                  'events': self.events, 'cache': self.cache.report() if self.cache is not None else None}
        self.cache = None
        return result

    def detach(self) -> None:
        if self.active:
            raise RuntimeError('Cannot detach during an active request')
        for store, original in self.originals:
            store._load_mesh_to_warp = original
