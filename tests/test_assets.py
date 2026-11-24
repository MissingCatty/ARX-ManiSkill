"""Check shared asset lookup and frozen scene bindings without loading a GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from arx_project.assets import AssetStore, load_object_catalog, recipe_sha256


class SharedAssetsTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        (self.root / 'models/cup').mkdir(parents=True)
        self.record = {'model_id': 'cup', 'source_definition': 'models/cup/model.xml',
                       'visuals': [{'file': 'models/cup/visual.obj'}],
                       'collisions': [{'file': 'models/cup/collision.obj', 'density': 100}],
                       'source_hashes': {'models/cup/visual.obj': 'saved-source-hash'}}
        self.catalog = self.root / 'catalog.json'
        self.save(self.catalog, {'kind': 'shared_3d_asset_catalog', 'models': {'cup': self.record},
                                'resources': {'robot': {'directory': 'models/cup'}}})

    def tearDown(self):
        self.work.cleanup()

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def reference(self, **extra):
        ref = self.root / 'scene_assets.json'
        self.save(ref, {'kind': 'shared_asset_catalog_reference', 'catalog_ref': str(self.catalog),
                        'asset_ids': ['cup'], 'recipe_sha256': {'cup': recipe_sha256(self.record)}, **extra})
        return ref

    def test_two_scenes_share_one_model_and_preserve_physics(self):
        first = load_object_catalog(self.reference())['models']['cup']
        second = load_object_catalog(self.reference())['models']['cup']
        self.assertEqual(first['visuals'][0]['file'], str(self.root / 'models/cup/visual.obj'))
        self.assertEqual(first['collisions'][0]['density'], 100)
        first['collisions'][0]['density'] = 0
        self.assertEqual(second['collisions'][0]['density'], 100)
        self.assertEqual(AssetStore(self.catalog).resource('robot'), self.root / 'models/cup')

    def test_reject_changed_or_missing_scene_assets(self):
        with self.assertRaisesRegex(ValueError, 'recipe changed'):
            load_object_catalog(self.reference(recipe_sha256={'cup': 'wrong'}))
        with self.assertRaises(KeyError):
            load_object_catalog(self.reference(asset_ids=['missing']))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_object_catalog(self.reference(asset_ids=['cup', 'cup']))

    def test_asset_paths_cannot_escape_via_relative_or_absolute_paths(self):
        store = AssetStore(self.catalog)
        for value in ['../secret.obj', '/etc/passwd', 'models/../../secret.obj']:
            with self.assertRaises(ValueError): store.path_for(value)
        (self.root / 'models/link').symlink_to(Path('/etc'), target_is_directory=True)
        with self.assertRaises(ValueError): store.path_for('models/link/passwd')

    def test_archived_catalog_receipts_remain_readable(self):
        path = self.root / 'archived_catalog.json'
        self.save(path, {'models': {'cup': self.record}})
        self.assertEqual(load_object_catalog(path)['models']['cup'], self.record)


if __name__ == '__main__': unittest.main()
