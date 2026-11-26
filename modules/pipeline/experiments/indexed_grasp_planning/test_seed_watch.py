"""Ensure the resource watcher cannot replay a claimed or completed E trial."""
from pathlib import Path
import tempfile
import unittest

from arx_project.layout import write_json,read_json
from .seed_watch import validate_predecessor


class PredecessorGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        write_json(self.root/'registration.json',{'experiment':'index_seed_probe',
            'IK_calls':3132,'fixtures':[{}]*261})
        write_json(self.root/'stop_summary.json',{'phase':'stopped_before_model_loading',
            'completed_IK_calls':0,'record_integrity':True})
        write_json(self.root/'STOP',{'reason':'resource gate'})

    def tearDown(self):self.temp.cleanup()

    def test_wholly_unexecuted_registration_may_be_copied(self):
        before=(self.root/'STOP').read_bytes()
        self.assertEqual(validate_predecessor(self.root)['IK_calls'],3132)
        self.assertEqual(before,(self.root/'STOP').read_bytes())

    def test_started_registration_is_not_silently_replayed(self):
        write_json(self.root/'started.json',{})
        with self.assertRaises(ValueError):validate_predecessor(self.root)

    def test_partial_IK_execution_is_not_silently_replayed(self):
        path=self.root/'stop_summary.json';value=read_json(path)
        value['completed_IK_calls']=1;write_json(path,value)
        with self.assertRaises(ValueError):validate_predecessor(self.root)

    def test_completed_results_are_not_replayed(self):
        write_json(self.root/'acceptance.json',{'passed':True})
        with self.assertRaises(ValueError):validate_predecessor(self.root)

    def test_missing_STOP_is_not_bypassed(self):
        (self.root/'STOP').unlink()
        with self.assertRaises(ValueError):validate_predecessor(self.root)


if __name__=='__main__':unittest.main()
