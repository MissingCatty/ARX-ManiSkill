import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from modules.curobo.dual_arm.accelerated_adapter import opening_samples
from .adapter import narrowing_canonical,optimized_pool
from modules.graspgenx.native_postprocessing.width_planning import FINGERS


class SweepTests(unittest.TestCase):
    def state(self):return {'joint_names':['arm',*FINGERS],'qpos':[.3,.044,.044]}

    def test_identical_narrowing_sample_values(self):
        for width in [.06,.04,.08283961046871056]:
            canonical=narrowing_canonical(self.state(),width);fractions,target,_=opening_samples(canonical)
            expected=[float(.044+(width/2-.044)*x) for x in np.linspace(0,1,max(2,int(np.ceil(abs(.044-width/2)/.0005))+1))]
            self.assertEqual([float(.044+x*(target-.044)) for x in fractions],expected)
            self.assertEqual(canonical['qpos'][0],.3)

    def test_batch_failure_keeps_first_invalid_sample(self):
        pool=object.__new__(optimized_pool(object));pool.batch_sweep_enabled=True
        from modules.curobo.indexed_planning.stage_timing import StageTiming
        pool.stage_timing=StageTiming()
        with patch('modules.curobo.gripper_sweep.adapter.batch_sweep',return_value={'passed':False,'first_invalid_sample':3}):
            result=pool.validate_narrowing(Path('/tmp/unused'),self.state(),.06)
        self.assertFalse(result['passed']);self.assertEqual(len(result['samples']),3)

    def test_batch_narrowing_preserves_sample_list(self):
        pool=object.__new__(optimized_pool(object));pool.batch_sweep_enabled=True
        from modules.curobo.indexed_planning.stage_timing import StageTiming
        pool.stage_timing=StageTiming()
        with patch('modules.curobo.gripper_sweep.adapter.batch_sweep',return_value={'passed':True,'first_invalid_sample':None}):
            result=pool.validate_narrowing(Path('/tmp/unused'),self.state(),.06)
        self.assertEqual(result['samples'][0],.044);self.assertEqual(result['samples'][-1],.03)


if __name__=='__main__':unittest.main()
