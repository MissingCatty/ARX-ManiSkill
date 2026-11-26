"""Exercise global ranking, resumable windows and failures without a GPU."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from arx_project.layout import ROOT,read_json,write_json
from modules.arx_core.arms import get_arm
from modules.curobo.candidate_io.store import (bind_candidate_view,candidate_pool,
    candidate_scope,candidate_window,read_snapshot)
from .adapter import choose_plan
from .interleaved_adapter import InterleavedDualArmPlanner


class FixturePool:
    def __init__(self, winners, travel=None):
        self.winners = winners
        self.travel = travel or {'right':1.,'left':2.}
        self.checks = {'right':[],'left':[]}
        self.fail_side = None

    def request(self,directory,**flags):
        side = Path(directory).name
        if side == self.fail_side:
            raise RuntimeError('injected planner resource failure')
        candidates = candidate_pool(directory)['candidates']
        events = []
        for local,candidate in enumerate(candidates):
            index = candidate['metadata']['original_index']
            self.checks[side].append(index)
            events.append({'candidate_index':local})
            if index == self.winners[side]:
                return {'success':True,'selected':{
                    'index':local,'score':candidate['score'],
                    'contact_pose_world':candidate['pose'],
                    'q_pre':[self.travel[side]]*6},
                    'candidate_events':events,'rejected':{'grasp_ik':len(events)-1}}
        return {'success':False,'reason':'no_feasible_native_post_or_recalled_candidate',
            'candidate_events':events,'rejected':{'grasp_ik':len(events)}}


class FixturePlanner(InterleavedDualArmPlanner):
    def prepare_arm(self,directory,state,arm,filename,*,planned_open=False):
        work = directory/'arms'/arm
        write_json(work/filename,state)
        bind_candidate_view(work,directory)
        return work

    def opening_sweep(self,work,canonical):
        passed = work.name not in getattr(self,'blocked',set())
        return {'passed':passed,'reason':'native_clear' if passed else 'gripper_opening_sweep_in_collision'}


class InterleavingTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tmp/dual_arm_interleaving'
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='offline_',dir=parent)
        self.root = Path(self.temporary.name)
        self.state = {'joint_names':[n for side in ['right','left'] for n in get_arm(side).joints],
                      'qpos':[0.]*12}

    def tearDown(self):
        self.temporary.cleanup()

    def setup_planner(self,count,winners,batch=16,travel=None):
        payload = {'candidates':[{'pose':(np.eye(4)+np.array(
            [[0.,0.,0.,i*.001],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])).tolist(),
            'score':float(count-i),'metadata':{'original_index':i}} for i in range(count)]}
        write_json(self.root/'candidates/candidate_pool.json',payload)
        write_json(self.root/'state.json',self.state)
        pool = FixturePool(winners,travel)
        planner = FixturePlanner(pool,None,self.root)
        planner.candidate_batch_size = batch
        return planner,pool,payload

    def test_left_first_candidate_avoids_exhausting_3320_right_candidates(self):
        planner,pool,payload = self.setup_planner(3320,{'right':None,'left':0})
        result = planner.request(self.root)
        self.assertEqual((result['selected']['arm'],result['selected']['index']),('left',0))
        self.assertEqual(len(pool.checks['right']),16)
        self.assertEqual(pool.checks['left'],[0])
        self.assertEqual(read_snapshot(self.root/'arms/left')[0],payload)

    def test_equal_rank_still_checks_both_arms_and_uses_joint_motion(self):
        planner,pool,_ = self.setup_planner(50,{'right':0,'left':0},travel={'right':2.,'left':1.})
        result = planner.request(self.root)
        self.assertEqual(result['selected']['arm'],'left')
        self.assertEqual(pool.checks,{'right':[0],'left':[0]})

    def test_resume_keeps_original_indices_without_duplicate_checks(self):
        planner,pool,payload = self.setup_planner(80,{'right':47,'left':32})
        result = planner.request(self.root)
        self.assertEqual((result['selected']['arm'],result['selected']['index']),('left',32))
        self.assertEqual(result['selected']['contact_pose_world'],payload['candidates'][32]['pose'])
        self.assertEqual(pool.checks['right'],list(range(48)))
        self.assertEqual(pool.checks['left'],list(range(33)))
        self.assertEqual(read_json(self.root/'arms/left/plan_result.json')['selected']['index'],32)

    def test_small_exhaustive_oracle_agrees_with_original_selection(self):
        for batch,right,left in product([1,3],[None,0,1,2,3,4],[None,0,1,2,3,4]):
            with self.subTest(batch=batch,right=right,left=left):
                planner,pool,_ = self.setup_planner(5,{'right':right,'left':left},batch)
                originals = {side:({'success':False} if index is None else
                    {'success':True,'selected':{'index':index,'q_pre':[pool.travel[side]]*6}})
                    for side,index in [('right',right),('left',left)]}
                expected = choose_plan(originals,self.state)
                result = planner.request(self.root)
                self.assertEqual(result['success'],expected is not None)
                if expected:
                    self.assertEqual(result['selected']['arm'],expected)
                    self.assertEqual(result['selected']['index'],pool.winners[expected])
                for side,checks in pool.checks.items():
                    self.assertEqual(checks,list(range(len(checks))))

    def test_all_failures_exhaust_both_arms_and_preserve_counts(self):
        planner,pool,_ = self.setup_planner(35,{'right':None,'left':None})
        result = planner.request(self.root)
        self.assertFalse(result['success'])
        for side in ['right','left']:
            self.assertEqual(pool.checks[side],list(range(35)))
            saved = read_json(self.root/'arms'/side/'plan_result.json')
            self.assertTrue(saved['full_pool_exhausted'])
            self.assertEqual(saved['rejected']['grasp_ik'],35)

    def test_later_windows_expand_and_keep_the_original_winning_rank(self):
        planner,pool,_ = self.setup_planner(100,{'right':56,'left':56})
        result = planner.request(self.root)
        self.assertEqual((result['selected']['arm'],result['selected']['index']),('right',56))
        schedule = read_json(self.root/'interleaved_search.json')['schedule']
        self.assertEqual(len(schedule),4)
        self.assertEqual([(r['start_index'],r['stop_index_exclusive']) for r in schedule],
                         [(0,16),(0,16),(16,80),(16,57)])
        self.assertEqual(pool.checks['right'],list(range(57)))
        self.assertEqual(pool.checks['left'],list(range(57)))

    def test_opening_collision_prevents_that_arm_search(self):
        planner,pool,_ = self.setup_planner(35,{'right':0,'left':20})
        planner.blocked = {'right'}
        result = planner.request(self.root)
        self.assertEqual(result['selected']['arm'],'left')
        self.assertEqual(pool.checks['right'],[])

    def test_only_remaining_arm_searches_its_prefix_in_one_window(self):
        planner,pool,_ = self.setup_planner(100,{'right':0,'left':80})
        planner.blocked = {'right'}
        result = planner.request(self.root)
        self.assertEqual(result['selected']['index'],80)
        self.assertEqual(len(read_json(self.root/'interleaved_search.json')['schedule']),1)
        self.assertEqual(pool.checks['left'],list(range(81)))

    def test_resource_failure_propagates_and_window_restores_source(self):
        planner,pool,payload = self.setup_planner(35,{'right':None,'left':0})
        pool.fail_side = 'left'
        with self.assertRaisesRegex(RuntimeError,'resource failure'):
            planner.request(self.root)
        self.assertEqual(read_snapshot(self.root/'arms/left')[0],payload)
        with candidate_scope(self.root):
            self.assertEqual(candidate_pool(self.root),payload)

    def test_window_keeps_shared_candidate_objects_and_restores_on_error(self):
        _,_,payload = self.setup_planner(35,{'right':None,'left':0})
        with candidate_scope(self.root):
            arm = self.root/'arms/right'
            bind_candidate_view(arm,self.root)
            original = candidate_pool(arm)
            with self.assertRaisesRegex(RuntimeError,'fault'):
                with candidate_window(arm,16,32):
                    self.assertIs(candidate_pool(arm)['candidates'][0],original['candidates'][16])
                    self.assertEqual(candidate_pool(self.root),payload)
                    raise RuntimeError('fault')
            self.assertIs(candidate_pool(arm),original)

    def test_forced_arm_retains_full_native_request(self):
        planner,pool,_ = self.setup_planner(35,{'right':None,'left':20})
        result = planner.request(self.root,arm='left')
        self.assertEqual(result['selected']['arm'],'left')
        self.assertEqual(pool.checks['right'],[])
        self.assertEqual(pool.checks['left'],list(range(21)))


if __name__ == '__main__':
    unittest.main()
