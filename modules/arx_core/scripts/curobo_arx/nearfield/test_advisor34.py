"""Offline ARX geometry, dispatch budgets and fault-clock regressions."""
import copy
from dataclasses import replace
from pathlib import Path
import types
import unittest
from unittest.mock import Mock,patch
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read
from .advisor34_rules import settle
from .advisor34_preflight import sample,PREVIOUS
from .advisor34_geometry import compensate,first_plane,column_intersections,observe
from .advisor34_policy import decision
from .runtime_advisor34 import Environment
from training.nearfield_arx import native_mask


class ARXRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg=read(PREVIOUS/'registration.json')
        ep=next(PREVIOUS.glob('groups/*/trials/a33m_spam_canon_normal_g0/episodes/*'))
        cls.req=read(read(ep/'policy_requests.json')[-1]['request'])
        cls.row=native_mask.request_row(cls.req)
        cls.gate,_,cls.points=observe(native_mask.projector_for(cls.reg),cls.req['asset'],cls.row,np.load(cls.req['depth'],allow_pickle=False))
        cls.current=np.array(cls.row['snapshot']['link_world']['right_eef_link'])
        cls.box=cls.gate['box'];cls.scale=cls.reg['learned_policy']['visual_contract']['depth_scale']
        # Fault injections remain in the frozen ARX hand frame; no surrogate robot.
        cls.inhand=np.array([[-.006,0.,.0335],[-.005,0.001,.0335]])
        cls.shallow=cls.inhand+np.array([0.,0.,.025])
        cls.regular={**cls.gate,'pass':True,'in_hand_points':2,'target_direction_EE_xy':[1.,0.]}

    def request(self,mode='decision',index=1):
        return {**self.req,'method':'D','vision_arm':'fresh','mode':mode,'motions_completed':index,
                'inputs':{'eef_world':self.current.tolist()},'axial_corrections_completed':0}

    def test_real_spam_visible_top_passes_column_without_fake_visible_points(self):
        self.assertEqual(self.gate['target_points'],124240)
        self.assertEqual(self.gate['in_hand_points'],0)
        self.assertTrue(self.gate['pass']);self.assertEqual(self.gate['gate_origin'],'gravity_column')
        self.assertEqual(decision(self.request('final_gate',2),'GRASP',self.gate,self.points,self.row,self.scale)['kind'],'close')

    def test_empty_negative_and_too_high_or_axially_far_columns(self):
        g=self.gate['gravity_down_EE']
        for pts in (np.empty((0,3)),np.array([[-1.,0.,.0335]]),np.array([[-.006,1.,.0335]]),np.array([[-.006,0.,1.]])):
            keep,_=column_intersections(pts,g,self.box,include_z=True)
            self.assertFalse(keep.any())

    def test_gravity_rotation_covariance(self):
        from scipy.spatial.transform import Rotation
        R=Rotation.from_rotvec([.3,.2,-.4]).as_matrix()
        down=self.current[:3,:3].T@np.array([0.,0.,-1.])
        # Turning the whole scene changes world gravity expressed in the same
        # hand. It must not silently become hard-coded local +X.
        wrong=(R@self.current[:3,:3]).T@np.array([0.,0.,-1.])
        self.assertGreater(np.linalg.norm(down-wrong),.1)
        actual,_=column_intersections(self.points,down,self.box,include_z=True)
        self.assertEqual(int(actual.sum()),self.gate['occlusion_envelope']['source_points'])

    def test_first_plane_335_and_axial_cap(self):
        first=first_plane(self.current,'REFINE 0 0 0 0 0 0',self.shallow,self.box)
        self.assertTrue(first['pass']);self.assertAlmostEqual(first['near_plane_z_EE_m'],.0335)
        self.assertAlmostEqual(first['used_translation_m'][2],.025)
        capped=first_plane(self.current,'REFINE 0 0 0 0 0 0',self.shallow+[0,0,.2],self.box)
        self.assertTrue(capped['axial_cap_active']);self.assertAlmostEqual(capped['used_translation_m'][2],.065)

    def test_second_preserves_XY_rotation_and_not_clamped_to_hide_rejection(self):
        r=np.array([0.,0.,.01]);p=np.array([.001,0.,.002])
        a=compensate(self.current,p,r,self.shallow,self.box)
        self.assertTrue(a['pass']);self.assertTrue(a['applied'])
        np.testing.assert_allclose(a['used_translation_m'][:2],p[:2],atol=0,rtol=0)
        np.testing.assert_allclose(a['rotation_rad'],r,atol=0,rtol=0)
        bad=compensate(self.current,p,r,self.shallow+[0,0,.1],self.box)
        self.assertFalse(bad['pass']);self.assertEqual(bad['reason'],'axial_alignment_exceeds_local_bound')

    def test_every_closure_route_checks_axial_insertion_once(self):
        cases=[(self.request(),'GRASP'),(self.request('final_gate',2),'GRASP'),
               (self.request(),'REFINE 0 0 0 0 0 0'),
               ({**self.request(),'method':'C','contact_closure':True},'STEP 0 0 0 0 0 0')]
        for req,token in cases:
            with self.subTest(req=req,token=token):
                a=decision(req,token,self.regular,self.shallow,self.row,self.scale)
                self.assertEqual(a['kind'],'move');self.assertTrue(a['preclosure_axial_correction'])
                b=decision({**req,'axial_corrections_completed':1},token,self.regular,self.shallow,self.row,self.scale)
                self.assertEqual(b['kind'],'reject')
                c=decision({**req,'mode':'final_gate','axial_corrections_completed':1},'GRASP',self.regular,self.inhand,self.row,self.scale)
                self.assertEqual(c['kind'],'close')

    def test_G0_cannot_cross_D_boundary(self):
        req=self.request();req['inputs']['G0_world']=np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError,'Coarse prior'):
            decision(req,'GRASP',self.gate,self.points,self.row,self.scale)

    def test_runtime_controller_motion_does_not_consume_or_restart_neural_refines(self):
        env=Environment.__new__(Environment);env.group=types.SimpleNamespace(campaign=Path('/absent'))
        env.pending_policy={'path':Path('/absent/request.json'),'request':self.request('final_gate',2),
                            'snapshot':self.row['snapshot']}
        req=env.pending_policy['request'];req.update(request_id='offline',raw_index=0)
        req['inputs']['decision_history_by_method']={'D':{}}
        env.local_index=2;env.axial_corrections=0;env.closure_pending=False
        env.session=types.SimpleNamespace(refines=2);env.calls=2;env.neural_calls=2;env.traces=[];env.episode=Path('/absent')
        env.proof=Mock(return_value={'pass':True});env.log_event=Mock();env.submit=Mock()
        action=decision(req,'GRASP',self.regular,self.shallow,self.row,self.scale)
        reply={'prediction':'GRASP','model_inference':False,'action':action,'finger_box':self.regular}
        mod='scripts.curobo_arx.nearfield.runtime_advisor34.'
        with patch(mod+'read_bound_reply',return_value=(req,reply)),patch(mod+'read',return_value={'frame_time':1.}), \
             patch(mod+'write'),patch('scripts.curobo_arx.nearfield.advisor34_policy.stationary_dispatch',return_value={'pass':True}), \
             patch('pathlib.Path.exists',side_effect=[False,True]):
            env.consume_prediction(self.row['snapshot'])
        self.assertEqual(env.local_index,2);self.assertEqual(env.session.refines,2);self.assertEqual(env.neural_calls,2)
        self.assertEqual(env.axial_corrections,1);self.assertTrue(env.closure_pending)
        self.assertEqual(env.submit.call_args.args[0],'axial_align_001')


class StopClock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        path=Path(read(PREVIOUS/'timeout_window_timing_context.json')['raw'])
        last=sample(json.loads(path.read_text().splitlines()[-1]));cls.last=last

    def tail(self,n):
        return [replace(self.last,sim_time=.2*i,velocities=tuple(.9 for _ in self.last.velocities)) for i in range(n)]

    def check(self,n,wall=0.,sim=0.,active=False):
        return settle(self.tail(n),active=active,fault=None,elapsed_wall_s=wall,elapsed_sim_s=sim)

    def test_ten_points_are_not_two_seconds(self):
        self.assertFalse(self.check(10)['pass']);self.assertTrue(self.check(11)['pass'])

    def test_slow_sim_does_not_time_out_at_ten_wall_seconds(self):
        self.assertFalse(self.check(6,wall=10.,sim=1.)['timeout'])
        self.assertTrue(self.check(11,wall=20.,sim=2.)['pass'])

    def test_sim_limit_and_wall_safety_are_independent(self):
        self.assertEqual(self.check(6,wall=30.,sim=10.)['timeout_reason'],'stationary_wait_10sim_s_timeout')
        self.assertEqual(self.check(11,wall=120.,sim=2.)['timeout_reason'],'stationary_wall_safety_120s_timeout')
        self.assertTrue(self.check(11,wall=100.,sim=10.)['pass'])

    def test_active_controller_and_fault_never_pass(self):
        self.assertFalse(self.check(11,active=True)['pass'])
        with self.assertRaises(RuntimeError):settle(self.tail(11),active=False,fault='offline fault',elapsed_wall_s=0.,elapsed_sim_s=0.)

    def test_missing_nonmonotonic_moving_samples_cannot_pass(self):
        samples=self.tail(11)
        for variant in (samples[:-1]+[replace(samples[-1],position=(1.,1.,1.))],samples[:5]+samples[6:]):
            self.assertFalse(settle(variant,active=False,fault=None,elapsed_wall_s=0.,elapsed_sim_s=0.)['pass'])
        for variant in (samples+[samples[-1]],samples[::-1]):
            with self.assertRaises(ValueError):settle(variant,active=False,fault=None,elapsed_wall_s=0.,elapsed_sim_s=0.)


if __name__=='__main__':unittest.main()
