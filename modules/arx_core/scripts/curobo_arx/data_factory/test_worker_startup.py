"""Offline ARX resident-worker startup fault cases; no robot/model commands."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image

from .launch import wait_for_workers,worker_startup_timeout
from .common import ROOT,read
from .runtime import Group


class WorkerStartupTests(unittest.TestCase):
    def run_startup(self,ready_after,timeout,exit_after=None):
        with tempfile.TemporaryDirectory(prefix='arx_resident_startup_') as temp:
            directories=[Path(temp)/'planner',Path(temp)/'graspness']
            for d in directories:d.mkdir()
            (directories[0]/'ready.json').write_text('{}')
            now=[0.];worker=Mock()
            worker.poll.side_effect=lambda: 1 if exit_after is not None and now[0]>=exit_after else None
            def sleep(seconds):
                now[0]+=seconds
                if ready_after is not None and now[0]>=ready_after:
                    (directories[1]/'ready.json').write_text('{}')
            with patch('scripts.curobo_arx.data_factory.launch.time.monotonic',side_effect=lambda:now[0]),\
                    patch('scripts.curobo_arx.data_factory.launch.time.sleep',side_effect=sleep):
                return wait_for_workers([worker],directories,timeout)

    def test_recorded_cold_start_183_seconds_is_admitted_only_by_new_registration(self):
        with self.assertRaisesRegex(RuntimeError,'initialization timeout'):
            self.run_startup(183,worker_startup_timeout({}))
        elapsed=self.run_startup(183,worker_startup_timeout({'worker_startup_timeout_seconds':600}))
        self.assertGreaterEqual(elapsed,183)
        self.assertLess(elapsed,184)

    def test_missing_ready_still_times_out_under_extended_limit(self):
        with self.assertRaisesRegex(RuntimeError,'initialization timeout'):
            self.run_startup(None,600)

    def test_crashed_worker_fails_promptly_despite_extended_limit(self):
        with self.assertRaisesRegex(RuntimeError,'process exited'):
            self.run_startup(183,600,exit_after=1)

    def test_unchanged_registration_keeps_original_180_second_limit(self):
        self.assertEqual(worker_startup_timeout({}),180)

    def test_invalid_startup_budget_is_rejected(self):
        for seconds in (0,-1,float('nan'),float('inf'),1801):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                worker_startup_timeout({'worker_startup_timeout_seconds':seconds})


FIXTURE=ROOT/'runs/graspness_startup_fix_20261104_01/fixtures'


@unittest.skipUnless((FIXTURE/'index.json').exists(),'Recorded ARX cold-preparation evidence unavailable')
class RecordedPerceptionBudgetTests(unittest.TestCase):
    """Exercise the original ARX block with its recorded frames and mock clocks.

    No World, model or motion starts; stop before actual planning. Only timeout
    scope is checked, not grasp quality, contacts or physical execution.
    """
    def run_recorded_block(self,reply_after):
        timing=read(FIXTURE/'preparation_timing.json');request=read(FIXTURE/'request.json')
        response=read(FIXTURE/'response.json');observed=read(FIXTURE/'head_observation.json')
        rgb=np.asarray(Image.open(FIXTURE/'head.png').convert('RGB'))
        depth=np.load(FIXTURE/'head_depth.npy');now=[0.];steps=[0];submitted=[]
        class PlanningReached(Exception):pass
        with tempfile.TemporaryDirectory(prefix='arx_recorded_perception_budget_') as temp:
            group=Group.__new__(Group);group.out=Path(temp)/'group';group.out.mkdir()
            group.campaign=Path(temp)/'campaign';group.campaign.mkdir();group.stop=False
            group.renderer='RaytracedLighting';group.registration={'reference_files':read(FIXTURE/'reference_files.json')}
            group.envs=[]
            for i in range(2):
                bridge=Mock();bridge.active=False
                bridge.observation.return_value=(observed,rgb,depth)
                bridge.export_scene.return_value={'scope':'offline timing only; never consumed by a planner'}
                bridge.snapshot.return_value=observed['snapshot']
                env=SimpleNamespace(task=None,bridge=bridge,block_count=0,state='idle',id='ARX/env_'+str(i),
                    reset=Mock(),visual_context=lambda:request['visual_context'])
                if i==0:env.reset.side_effect=lambda _asset:now.__setitem__(0,now[0]+timing['reset_seconds'])
                group.envs.append(env)
            def step(_record):
                now[0]+=timing['settle_seconds']/15 if steps[0]<15 else 1.
                steps[0]+=1
            group.step=step
            checks=iter([timing['environment_0_check_seconds'],timing['environment_1_check_seconds']])
            def fingerprint(*_):now[0]+=next(checks);return {'pass':True}
            group.neural=Mock()
            def submit(*_,**__):submitted.append(now[0]);return request['id']
            group.neural.submit.side_effect=submit
            group.neural.poll.side_effect=lambda _:response if now[0]-submitted[0]>=reply_after else None
            group.planner=Mock();group.planner.submit.side_effect=PlanningReached
            with patch('scripts.curobo_arx.data_factory.runtime.time.monotonic',side_effect=lambda:now[0]),\
                    patch('scripts.curobo_arx.data_factory.frozen.fingerprint',side_effect=fingerprint):
                try:group.run_block(request['asset'],0)
                except PlanningReached:return now[0]-submitted[0],submitted[0]
            self.fail('Expected the recorded response to reach qualification, without actual planning')

    def test_recorded_750_second_preparation_does_not_consume_perception_budget(self):
        seconds=read(FIXTURE/'response.json')['seconds']
        elapsed,preparation=self.run_recorded_block(seconds)
        self.assertGreater(preparation,700)
        self.assertGreater(preparation+elapsed,900)
        self.assertLess(elapsed,900)

    def test_actual_perception_wait_over_900_seconds_is_still_rejected(self):
        with self.assertRaisesRegex(TimeoutError,'Resident perception budget exceeded'):
            self.run_recorded_block(901)


if __name__=='__main__':unittest.main()
