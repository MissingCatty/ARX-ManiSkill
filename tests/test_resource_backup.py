"""Exercise split archive recovery and corruption detection on small resources."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from tools.drive_backup import archive_group, reuse_groups, groups
from tools.restore_resources import restore


class ResourceBackupTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name) / 'source'
        self.payload = Path(self.work.name) / 'backup'
        self.destination = Path(self.work.name) / 'restored'
        (self.root / 'weights/demo').mkdir(parents=True)
        (self.root / 'simulation/assets/models/cup').mkdir(parents=True)
        self.data = random.Random(42).randbytes(2048)
        (self.root / 'weights/demo/model.bin').write_bytes(self.data)
        (self.root / 'simulation/assets/models/cup/mesh.bin').write_bytes(b'original geometry')
        (self.root / 'simulation/assets/models/cup/shared.bin').symlink_to('mesh.bin')
        self.group = {'id': 'fixture', 'members': ['weights', 'simulation']}

    def tearDown(self):
        self.work.cleanup()

    def pack(self):
        result = archive_group(self.root, self.payload, self.group, chunk_bytes=128, threads=1)
        (self.payload / 'manifest.json').write_text(json.dumps({
            'state': 'ready', 'source_commit': 'fixture', 'groups': [result]}))
        return result

    def test_split_round_trip_preserves_bytes_layout_and_relative_symlink(self):
        result = self.pack()
        self.assertGreater(len(result['parts']), 2)
        restore(self.payload, self.destination)
        self.assertEqual((self.destination / 'weights/demo/model.bin').read_bytes(), self.data)
        shared = self.destination / 'simulation/assets/models/cup/shared.bin'
        self.assertTrue(shared.is_symlink())
        self.assertEqual(shared.read_bytes(), b'original geometry')
        self.assertEqual(archive_group(self.root, self.payload, self.group, 128, 1), result)

    def test_corrupt_part_is_rejected_before_writing_destination(self):
        result = self.pack()
        part = self.payload / result['parts'][1]['path']
        part.write_bytes(b'x' * part.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            restore(self.payload, self.destination)
        self.assertFalse(self.destination.exists())

    def test_resume_rejects_changed_source_weights(self):
        self.pack()
        (self.root / 'weights/demo/model.bin').write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'create a new snapshot'):
            archive_group(self.root, self.payload, self.group, 128, 1)


    def test_new_snapshot_reuses_unchanged_parts_and_restores_them(self):
        result = self.pack()
        previous = Path(self.work.name)/'previous'
        previous.mkdir()
        self.payload.rename(previous/'payload')
        new = Path(self.work.name)/'new_payload'
        ids = reuse_groups(self.root, previous, new, [self.group])
        self.assertEqual(ids, ['fixture'])
        source = previous/'payload'/result['parts'][0]['path']
        reused = new/result['parts'][0]['path']
        self.assertEqual(source.stat().st_ino, reused.stat().st_ino)
        (new/'manifest.json').write_text(json.dumps({'state':'ready','groups':[result]}))
        restore(new, self.destination)
        self.assertEqual((self.destination/'weights/demo/model.bin').read_bytes(), self.data)

    def test_new_snapshot_does_not_reuse_changed_weights(self):
        self.pack()
        previous = Path(self.work.name)/'previous';previous.mkdir()
        self.payload.rename(previous/'payload')
        (self.root/'weights/demo/model.bin').write_bytes(b'different model')
        new = Path(self.work.name)/'new_payload'
        self.assertEqual(reuse_groups(self.root, previous, new, [self.group]), [])
        self.assertFalse((new/'packages/fixture').exists())

    def test_archived_runs_keep_their_tmp_path_in_the_backup(self):
        (self.root / 'configs').mkdir()
        (self.root / 'tmp/reachability/experiments').mkdir(parents=True)
        policy = self.root / 'configs/cloud_backup.json'
        policy.write_text(json.dumps({'archive_runs': ['tmp/reachability/experiments']}))
        result = groups(self.root)
        archive = next(r for r in result if r['id'] == 'archive_reachability_experiments')
        self.assertEqual(archive['members'], ['tmp/reachability/experiments'])
        for path in ['modules/reachability/experiments', '/tmp/reachability/experiments',
                     'tmp/../escape', 'tmp/reachability/missing']:
            policy.write_text(json.dumps({'archive_runs': [path]}))
            with self.assertRaises(ValueError):
                groups(self.root)


if __name__ == '__main__':
    unittest.main()
