"""Cache correctness, invalidation, ownership and failure propagation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from arx_project.layout import ROOT
from .prepared_meshes import PreparedMeshCache
from .cpu_runtime import CPUGeometryReuse


class FixtureMesh:
    def __init__(self, path, scale=None):
        self.file_path, self.scale = str(path), scale
        self.name = 'fixture'; self.calls = 0

    def get_mesh_data(self, process=True):
        self.calls += 1
        value = float(Path(self.file_path).read_text())
        vertices = np.asarray([[0., 0., 0.], [value, 0., 0.], [0., value, 0.]])
        if self.scale is not None:
            vertices *= np.asarray(self.scale)
        return vertices, np.asarray([[0, 1, 2]])


class CacheTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tmp/curobo/tests'; parent.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(dir=parent, prefix='mesh-cache-test-')
        self.root = Path(self.work.name)
        self.source = self.root/'source.mesh'; self.source.write_text('1')
        self.cache = PreparedMeshCache(self.root/'cache')
        self.mesh = FixtureMesh(self.source)

    def tearDown(self):
        self.work.cleanup()

    def test_original_arrays_preserved_and_shared_read_only(self):
        expected = self.mesh.get_mesh_data()
        a, b = self.cache.get(self.mesh), self.cache.get(self.mesh)
        for old, value in zip(expected, [a.vertices, a.faces]):
            self.assertTrue(np.array_equal(old, value))
        self.assertIs(a, b); self.assertEqual(self.mesh.calls, 2)
        with self.assertRaises(ValueError): a.vertices[0, 0] = 1

    def test_different_paths_same_content_share_entry(self):
        other = self.root/'other.mesh'; other.write_bytes(self.source.read_bytes())
        a = self.cache.get(self.mesh); b = self.cache.get(FixtureMesh(other))
        self.assertEqual(a.key, b.key); self.assertIs(a, b)

    def test_same_path_changed_content_invalidates(self):
        a = self.cache.get(self.mesh); self.source.write_text('2'); b = self.cache.get(self.mesh)
        self.assertNotEqual(a.key, b.key); self.assertEqual(b.vertices[1, 0], 2.)

    def test_scale_changes_identity_and_values(self):
        a = self.cache.get(self.mesh); b = self.cache.get(FixtureMesh(self.source, [2., 1., 1.]))
        self.assertNotEqual(a.key, b.key); self.assertEqual(b.vertices[1, 0], 2.)

    def test_version_changes_identity(self):
        a = self.cache.get(self.mesh); self.cache.versions['trimesh'] = 'different'
        self.assertNotEqual(a.key, self.cache.get(self.mesh).key)

    def test_new_instance_reads_disk_without_reprocessing(self):
        a = self.cache.get(self.mesh); mesh = FixtureMesh(self.source)
        cache = PreparedMeshCache(self.root/'cache'); b = cache.get(mesh)
        self.assertEqual(mesh.calls, 0); self.assertTrue(np.array_equal(a.faces, b.faces))
        self.assertEqual(cache.counts['disk'], 1)

    def test_memory_budget_evicts_but_keeps_disk(self):
        cache = PreparedMeshCache(self.root/'cache', 100)
        a = cache.get(self.mesh); self.source.write_text('2'); cache.get(self.mesh)
        self.source.write_text('1'); cache.get(self.mesh)
        self.assertLessEqual(cache.bytes, 100); self.assertEqual(cache.counts['disk'], 1)
        self.assertTrue((cache.root/a.key/'metadata.json').exists())

    def test_corrupt_disk_array_stops_instead_of_recomputing(self):
        a = self.cache.get(self.mesh); (self.cache.root/a.key/'vertices.npy').write_bytes(b'broken')
        other = PreparedMeshCache(self.cache.root)
        with self.assertRaisesRegex(ValueError, 'Corrupt'): other.get(self.mesh)

    def test_missing_manifest_stops(self):
        a = self.cache.get(self.mesh); (self.cache.root/a.key/'metadata.json').unlink()
        with self.assertRaises(FileNotFoundError): PreparedMeshCache(self.cache.root).get(self.mesh)

    def test_multiple_threads_process_once(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: self.cache.get(self.mesh), range(8)))
        self.assertEqual(self.mesh.calls, 1); self.assertEqual(len({r.key for r in results}), 1)

    def test_processing_failure_is_not_cached(self):
        self.source.write_text('invalid')
        with self.assertRaises(ValueError): self.cache.get(self.mesh)
        self.assertEqual(list(self.cache.root.glob('*/metadata.json')), [])

    def test_native_loader_retains_identity_and_arrays(self):
        outputs = []
        def load(mesh):
            outputs.append((mesh.name, mesh.get_mesh_data()))
            return mesh.name
        store = SimpleNamespace(_load_mesh_to_warp=load)
        manager = CPUGeometryReuse([store]); manager.begin(self.cache)
        for _ in range(2): self.assertEqual(store._load_mesh_to_warp(self.mesh), 'fixture')
        report = manager.finish(); self.assertEqual(self.mesh.calls, 1)
        self.assertEqual(report['native_load_calls'], 2)
        self.assertTrue(np.array_equal(outputs[0][1][0], outputs[1][1][0]))
        manager.detach(); self.assertIs(store._load_mesh_to_warp, load)

    def test_control_and_disabled_scope_use_original_loader(self):
        store = SimpleNamespace(_load_mesh_to_warp=lambda mesh: mesh.get_mesh_data())
        manager = CPUGeometryReuse([store]); store._load_mesh_to_warp(self.mesh)
        manager.begin(None); store._load_mesh_to_warp(self.mesh); manager.finish()
        self.assertEqual(self.mesh.calls, 2); self.assertEqual(self.cache.counts, {})


if __name__ == '__main__': unittest.main()
