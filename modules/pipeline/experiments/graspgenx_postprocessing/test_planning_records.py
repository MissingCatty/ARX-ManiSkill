"""Record faults must stop acceptance even when a planning result remains."""
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import write_json,sha256
from .planning_review import check_receipt


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.claim={'state':'complete'}
        for key,name in [('metrics_sha256','metrics.json'),('plan_sha256','plan_result.json'),('order_sha256','candidate_order.json')]:
            write_json(self.root/name,{'fixture':True});self.claim[key]=sha256(self.root/name)
        write_json(self.root/'candidate_snapshot_commit.json',{'persisted':True})

    def tearDown(self):self.temp.cleanup()

    def test_complete_receipts_are_accepted(self):check_receipt(self.root,self.claim)

    def test_result_without_order_manifest_is_rejected(self):
        (self.root/'candidate_order.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):check_receipt(self.root,self.claim)

    def test_changed_result_is_rejected(self):
        write_json(self.root/'plan_result.json',{'fixture':False})
        with self.assertRaises(ValueError):check_receipt(self.root,self.claim)

    def test_uncommitted_candidate_snapshot_is_rejected(self):
        write_json(self.root/'candidate_snapshot_commit.json',{'persisted':False})
        with self.assertRaises(ValueError):check_receipt(self.root,self.claim)

    def test_claimed_request_cannot_be_counted_complete(self):
        self.claim['state']='claimed'
        with self.assertRaises(ValueError):check_receipt(self.root,self.claim)


if __name__=='__main__':unittest.main()
