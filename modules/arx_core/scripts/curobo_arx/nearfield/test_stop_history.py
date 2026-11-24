"""Offline ARX runtime regression: retain measured history, never relax settling."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
import numpy as np
from .runtime_advisor33 import Environment
from .advisor32_rules import settle
from .advisor33_audit import bound_stop


class StopHistory(unittest.TestCase):
    def environment(self):
        env=Environment.__new__(Environment)
        env.id='arx/env_0';env.task={'id':'offline-arx'};env.asset={'object_id':'milk'}
        env.group=types.SimpleNamespace(registration={})
        env.recorder=Mock();env.stage='close';env.last_command={'label':'close'}
        env.stop_samples=[]
        env.bridge=Mock(active=False,fault=None)
        env.bridge.context=types.SimpleNamespace(cameras={'right_wrist':types.SimpleNamespace(capture_arrays={})})
        return env

    def capture(self,env,seconds,position=0.,joint=0.):
        ee=np.eye(4);ee[0,3]=position
        row={'snapshot':{'sim_time':seconds,'joint_names':['right_joint21'],
             'right_arm':['right_joint21'],'joint_positions':[joint],'joint_velocities':[.9],
             'joint_limits':[[-2.094,3.1416]],'link_world':{'right_eef_link':ee.tolist()}}}
        env.bridge.observation.return_value=(row,np.zeros((2,2,3)),np.ones((2,2)))
        env.record();return row

    def test_continuous_hold_survives_completion_and_is_raw_bound(self):
        env=self.environment();raw=[self.capture(env,i*.2) for i in range(11)]
        env.bridge.active=True
        self.assertFalse(settle(env.stop_samples,active=True,fault=None,elapsed_wall_s=0.)['pass'])
        env.bridge.active=False;env.state='executing';env.started=0.;env.group.registration['attempt_seconds']=1000
        env.episode=Path('/not-used');env.stages=[];env.desired=np.eye(4);env.visual_issues=[]
        env.bridge.result.return_value={'completed':True}
        with patch('scripts.curobo_arx.nearfield.runtime_advisor33.write'), \
             patch('scripts.curobo_arx.nearfield.runtime_advisor33.time.monotonic',return_value=5.), \
             patch('scripts.curobo_arx.nearfield.runtime_advisor33.assess_settling',return_value={}), \
             patch('scripts.curobo_arx.nearfield.runtime_advisor33.forbidden_contacts',return_value=False):
            env.tick()
        self.assertEqual(env.state,'waiting_idle');self.assertEqual(len(env.stop_samples),11)
        result=settle(env.stop_samples,active=False,fault=None,elapsed_wall_s=.1)
        self.assertTrue(result['pass']);self.assertEqual(result['origin'],'settled_by_position_window')
        proof={**result,'files':{},'samples':[asdict(s) for s in env.stop_samples],'wait_elapsed_wall_s':.1}
        self.assertTrue(bound_stop(proof,raw)['pass'])

    def test_clear_history_reproduces_ten_wall_second_starvation(self):
        env=self.environment()
        for i in range(7):self.capture(env,i*.2)
        result=settle(env.stop_samples,active=False,fault=None,elapsed_wall_s=9.9)
        self.assertFalse(result['pass']);self.assertFalse(result['window_applicable'])
        self.assertTrue(settle(env.stop_samples,active=False,fault=None,elapsed_wall_s=10.)['timeout'])

    def test_motion_in_tail_still_rejected(self):
        for kind in ('position','joint'):
            env=self.environment()
            for i in range(11):self.capture(env,i*.2,**{kind:i*(.0001 if kind=='position' else .004)})
            self.assertFalse(settle(env.stop_samples,active=False,fault=None,elapsed_wall_s=.1)['pass'])

    def test_old_stable_tail_cannot_hide_latest_motion(self):
        env=self.environment()
        for i in range(11):self.capture(env,i*.2)
        self.capture(env,2.2,position=.001)
        self.assertFalse(settle(env.stop_samples,active=False,fault=None,elapsed_wall_s=.1)['pass'])

    def test_duplicate_time_not_counted_and_tail_bounded(self):
        env=self.environment()
        for i in range(40):self.capture(env,i*.2)
        self.capture(env,7.8)
        self.assertEqual(len(env.stop_samples),30)
        self.assertAlmostEqual(env.stop_samples[0].sim_time,2.)

    def test_reset_cannot_reuse_previous_trial(self):
        env=self.environment();self.capture(env,2.)
        with patch('scripts.curobo_arx.nearfield.runtime_advisor33.ActionEnvironment.reset',side_effect=RuntimeError('stop before real reset')):
            with self.assertRaisesRegex(RuntimeError,'stop before real reset'):env.reset({})
        self.assertEqual(env.stop_samples,[])


if __name__=='__main__':unittest.main()
