"""Exercise the real GraspGenX-to-ARX converter with both recording modes."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from arx_project import artifacts
from arx_project.layout import ROOT, write_json
from .loader import load_file

STAGED = ROOT/'tmp/candidate_io/refactor_20261125_01/staged_source'


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROOT/'modules/graspgenx'), str(ROOT/'modules/maniskill'),
                       str(ROOT/'modules/pipeline/experiments/rayst3r_graspgenx')]
        cls.module = load_file('candidate_conversion_under_test',STAGED/
            'modules/pipeline/experiments/repaired_grasp_benchmark/repair_contract.py')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='conversion_',dir=ROOT/'tmp/candidate_io')
        self.root = Path(self.temp.name)

    def tearDown(self):
        artifacts.close_owned_transports(); self.temp.cleanup()

    def inputs(self, name):
        directory=self.root/name;raw=directory/'graspgenx_raw';raw.mkdir(parents=True)
        poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,3]=[[.1,0,.2],[.2,.1,.1],[.3,.2,.2]]
        np.save(raw/'grasps_generator_world.npy',poses)
        np.save(raw/'scores.npy',np.array([.5,.9,.5]))
        write_json(raw/'inference.json',{'repair_contract':self.module.VERSION,
            'T_eef_generator':self.module.T_EEF_GENERATOR,'branch_tags':['diff','obb','obb']})
        write_json(directory/'state.json',{'camera_world_optical':np.eye(4)})
        return directory

    def test_file_consumer_default_remains_synchronous(self):
        directory=self.inputs('file_consumer')
        self.assertEqual(self.module.make_candidates(directory),3)
        self.assertTrue((directory/'candidates/candidate_pool.json').exists())
        self.assertIsNone(artifacts.candidate_transport(directory))

    def test_memory_handoff_matches_the_original_converter_before_save(self):
        old=self.inputs('old');new=self.inputs('new')
        self.module.make_candidates(old)
        expected=json.loads((old/'candidates/candidate_pool.json').read_text())
        entered,release=threading.Event(),threading.Event()
        original=artifacts.atomic_json

        def blocked(path,value):
            entered.set()
            if not release.wait(5):raise RuntimeError('test writer release timeout')
            return original(path,value)

        try:
            with patch('arx_project.artifacts.atomic_json',side_effect=blocked):
                self.assertEqual(self.module.make_candidates(new,background=True),3)
                self.assertTrue(entered.wait(1))
                self.assertFalse((new/'candidates/candidate_pool.json').exists())
                descriptor=artifacts.candidate_transport(new)
                self.assertEqual(artifacts.read_candidate_transport(new,descriptor),expected)
                release.set();artifacts.finish_candidate_transport(new)
        finally:release.set()
        self.assertEqual(json.loads((new/'candidates/candidate_pool.json').read_text()),expected)
        np.testing.assert_array_equal(np.load(old/'raw_grasps.npy'),np.load(new/'raw_grasps.npy'))


if __name__=='__main__':unittest.main()
