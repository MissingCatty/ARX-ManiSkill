"""Offline storage mapping and failure checks; no robot or GPU service."""
from pathlib import Path
import tempfile
import unittest
from .storage import register_bulk_storage,container_storage_mounts
from .common import read,sha


class StorageTests(unittest.TestCase):
    def test_legacy_campaign_needs_no_extra_mount(self):
        self.assertEqual(container_storage_mounts('/unused',{}),[])

    def test_registered_overflow_roundtrip_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as local,tempfile.TemporaryDirectory(dir='/coolas/data/arx_teacher_zhaoyechong') as remote:
            campaign=Path(local);root=Path(remote)/'bulk'
            receipt=register_bulk_storage(campaign,root)
            self.assertEqual(container_storage_mounts(campaign,{'bulk_storage':receipt}),['-v',str(root)+':'+str(root)+':rw'])
            before=sha(campaign/'bulk_storage.json')
            with self.assertRaises(FileExistsError):register_bulk_storage(campaign,root)
            self.assertEqual(before,sha(campaign/'bulk_storage.json'))
            self.assertEqual(read(campaign/'bulk_storage.json'),receipt)
            (campaign/'workers').unlink();(campaign/'workers').mkdir()
            with self.assertRaisesRegex(ValueError,'symlink differs'):container_storage_mounts(campaign,{'bulk_storage':receipt})

    def test_non_data_or_missing_storage_is_refused(self):
        with tempfile.TemporaryDirectory() as local:
            for root in ('relative','/tmp/arx-undeclared-overflow'):
                with self.assertRaises(ValueError):register_bulk_storage(local,root)
            with self.assertRaises(ValueError):container_storage_mounts(local,{'bulk_storage':{'root':'/coolas/data/nonexistent-storage-check','directories':['groups','workers']}})


if __name__=='__main__':unittest.main()
