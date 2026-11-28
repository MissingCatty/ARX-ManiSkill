"""Independent consumers, aliases, invalidation and failed GPU construction."""
from dataclasses import dataclass
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref

from arx_project.layout import ROOT
from .gpu_runtime import GPUGeometryReuse


@dataclass(frozen=True)
class Resource:
    name: str
    mesh_id: int
    mesh: object


class SharingTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tmp/curobo/tests'; parent.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(dir=parent, prefix='gpu-resource-test-')
        self.file = Path(self.work.name)/'mesh'; self.file.write_text('a')
        self.calls = []
        def create(mesh):
            self.calls.append(mesh.name)
            return Resource(mesh.name, len(self.calls), SimpleNamespace())
        self.stores = [SimpleNamespace(_load_mesh_to_warp=create, _wp_device='cuda:0') for _ in range(2)]
        self.manager = GPUGeometryReuse(self.stores)
        self.mesh = SimpleNamespace(name='one', file_path=str(self.file), scale=None)

    def tearDown(self): self.work.cleanup()

    def test_one_native_creation_two_independent_consumers(self):
        self.manager.begin(True)
        a = self.stores[0]._load_mesh_to_warp(self.mesh)
        b = self.stores[1]._load_mesh_to_warp(self.mesh)
        self.assertEqual(len(self.calls), 1); self.assertIs(a.mesh, b.mesh)
        self.assertIsNot(a, b); self.assertEqual(self.manager.finish()['shared_hits'], 1)

    def test_alias_name_changes_without_changing_shared_mesh(self):
        self.manager.begin(True); a = self.stores[0]._load_mesh_to_warp(self.mesh)
        other = SimpleNamespace(**vars(self.mesh)); other.name = 'two'
        b = self.stores[1]._load_mesh_to_warp(other)
        self.assertEqual(b.name, 'two'); self.assertEqual(a.name, 'one'); self.assertIs(a.mesh, b.mesh)
        self.manager.finish()

    def test_changed_source_and_scale_get_new_geometry(self):
        self.manager.begin(True); a = self.stores[0]._load_mesh_to_warp(self.mesh)
        self.file.write_text('b'); b = self.stores[1]._load_mesh_to_warp(self.mesh)
        self.mesh.scale = [2, 1, 1]; c = self.stores[0]._load_mesh_to_warp(self.mesh)
        self.assertEqual(len({a.mesh_id, b.mesh_id, c.mesh_id}), 3); self.manager.finish()

    def test_other_device_does_not_share_handle(self):
        self.stores[1]._wp_device = 'cuda:1'; self.manager.begin(True)
        a = self.stores[0]._load_mesh_to_warp(self.mesh); b = self.stores[1]._load_mesh_to_warp(self.mesh)
        self.assertNotEqual(a.mesh_id, b.mesh_id); self.manager.finish()

    def test_consumer_keeps_resource_after_scope_release(self):
        self.manager.begin(True)
        a = self.stores[0]._load_mesh_to_warp(self.mesh); b = self.stores[1]._load_mesh_to_warp(self.mesh)
        ref = weakref.ref(a); del a; self.manager.finish(); gc.collect()
        self.assertIsNone(ref()); self.assertIsNotNone(b.mesh)

    def test_control_does_not_share_or_hash(self):
        self.manager.begin(False); self.file.unlink()
        self.stores[0]._load_mesh_to_warp(self.mesh); self.stores[1]._load_mesh_to_warp(self.mesh)
        self.assertEqual(len(self.calls), 2); self.assertEqual(self.manager.finish()['events'], [])

    def test_failed_constructor_does_not_publish_resource(self):
        self.manager.detach()
        def fail(mesh): raise RuntimeError('failed allocation')
        self.stores[0]._load_mesh_to_warp = fail
        self.manager = GPUGeometryReuse(self.stores); self.manager.begin(True)
        with self.assertRaisesRegex(RuntimeError, 'failed allocation'):
            self.stores[0]._load_mesh_to_warp(self.mesh)
        self.assertEqual(self.manager.resources, {}); self.manager.finish()


if __name__ == '__main__': unittest.main()
