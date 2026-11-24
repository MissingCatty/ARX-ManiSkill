"""Exercise external SDK asset paths, integrity checks and catalog refresh."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from arx_project import assets
from arx_project.layout import ROOT
from tools.validate_library_assets import validate_library_assets


class LibraryStorageTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        self.assets = self.root / 'simulation/assets'
        self.bundle = self.assets / 'models/sdk'
        self.bundle.mkdir(parents=True)
        self.mesh = self.bundle / 'mesh.obj'
        self.mesh.write_bytes(b'original mesh geometry')
        data = self.mesh.read_bytes()
        inventory = self.bundle / 'inventory.json'
        inventory.write_text(json.dumps({'files': {'mesh.obj': {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}))
        self.catalog = self.assets / 'catalog.json'
        self.record = {'directory': 'models/sdk', 'inventory': 'models/sdk/inventory.json',
                       'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest()}
        self.save_catalog()

    def save_catalog(self):
        self.catalog.write_text(json.dumps({'kind': 'shared_3d_asset_catalog', 'models': {},
                                           'resources': {'sdk': self.record}}))

    def tearDown(self):
        self.work.cleanup()

    def test_catalog_path_refreshes_after_configuration_update(self):
        with patch.object(assets, 'CATALOG', self.catalog):
            self.assertEqual(assets.resource_path('sdk'), self.bundle)
            before = self.catalog.stat().st_mtime_ns
            self.record['directory'] = 'models/another_sdk'
            self.save_catalog()
            os.utime(self.catalog, ns=(before + 10000, before + 10000))
            self.assertEqual(assets.resource_path('sdk'), self.assets / 'models/another_sdk')

    def test_asset_corruption_and_missing_files_are_detected(self):
        self.assertTrue(validate_library_assets(self.root)['passed'])
        self.mesh.write_bytes(b'changed mesh geometry!')
        result = validate_library_assets(self.root)
        self.assertFalse(result['passed'])
        self.assertIn('asset checksum mismatch: mesh.obj', result['checks'][0]['errors'])
        self.mesh.unlink()
        self.assertFalse(validate_library_assets(self.root)['passed'])

    def test_inventory_tampering_and_external_links_are_detected(self):
        self.mesh.unlink()
        self.mesh.symlink_to('/etc/passwd')
        self.assertFalse(validate_library_assets(self.root)['passed'])
        (self.bundle / 'inventory.json').write_text('{}')
        self.assertFalse(validate_library_assets(self.root)['passed'])

    def test_curobo_versions_keep_separate_configs_and_shared_asset_roots(self):
        for owner, resource in [('curobo', 'curobo_assets'), ('curobo_adapter', 'curobo_v090_assets')]:
            path = ROOT / 'modules' / owner / 'source/curobo/content/__init__.py'
            namespace = runpy.run_path(str(path))
            self.assertEqual(namespace['get_assets_path'](), assets.resource_path(resource))
            self.assertEqual(namespace['get_configs_path'](), path.parent / 'configs')
            self.assertTrue(namespace['get_assets_path']().is_relative_to(ROOT / 'simulation/assets/models'))


if __name__ == '__main__':
    unittest.main()
