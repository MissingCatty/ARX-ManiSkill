"""Fault tests for the new actual-frame/stop/zero-prior integration boundaries."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha
from .advisor32_rules import StopSample,settle,continuation_after_slot
from .advisor33_policy import decision,stationary_dispatch
from .advisor33_geometry import first_plane,PAD
from .advisor33_audit import bound_stop,image_binding,audit_partial
from training.nearfield_arx.history import before
from training.nearfield_arx.lateral_recovery import inference_example


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.req={'method':'D','vision_arm':'fresh','motions_completed':0,'mode':'decision','inputs':{'eef_world':np.eye(4).tolist()}}
        self.box={'x_EE_m':[-.011,-.001],'y_EE_m':[-.045,.045],'z_EE_m':[0,.045],'pad_center_EE_m':PAD.tolist()}
        self.gate={'pass':True,'box':self.box,'target_points':10,'in_hand_points':10,'target_direction_EE_xy':[0,0]}
    def test_axial_cap_preserves_model_lateral(self):
        r=first_plane(np.eye(4),'REFINE +10.000 +0.000 +130.000 +0.0000 +0.0000 +0.0000',np.array([[.004,0,.180]]),self.box)
        self.assertTrue(r['pass']);self.assertTrue(r['axial_cap_active']);self.assertEqual(r['used_translation_m'][0],.01)
        self.assertAlmostEqual(np.linalg.norm(r['used_translation_m']),.065)
    def test_empty_corridor_does_not_move(self):
        r=first_plane(np.eye(4),'REFINE +0.000 +0.000 +30.000 +0.0000 +0.0000 +0.0000',np.array([[1.,1.,.08]]),self.box)
        self.assertFalse(r['pass'])
    def test_G0_cannot_cross_D_boundary(self):
        self.req['inputs']['G0_world']=np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError,'Coarse prior'):decision(self.req,'GRASP',self.gate,[],{}, {})
    def test_gate_never_means_model_grasp(self):
        self.req['mode']='final_gate';self.req['motions_completed']=2
        result=decision(self.req,'REFINE +1.000 +0.000 +0.000 +0.0000 +0.0000 +0.0000',self.gate,[],{}, {})
        self.assertEqual(result['closure_origin'],'visual_gate_after_motion_budget')
    def test_GRASP_outside_box_rejected(self):
        self.gate['pass']=False
        self.assertEqual(decision(self.req,'GRASP',self.gate,[],{}, {})['kind'],'reject')
    def test_invalid_mode_rejected(self):
        self.req['mode']='guess'
        with self.assertRaises(ValueError):decision(self.req,'GRASP',self.gate,[],{}, {})
    def test_prompt_interfaces_unchanged(self):
        inputs={'eef_world':np.eye(4).tolist(),'G0_world':np.eye(4).tolist(),'stage':'local_observation',
            'decision_history_by_method':{m:before(m,0,0,None,1.) for m in ('C','D')}}
        contract={'depth_scale':{'near_m':.035,'far_m':.327}}
        c=inference_example(inputs,'C',['rgb','depth'],contract);d=inference_example(inputs,'D',['rgb','depth'],contract)
        self.assertIn('G0_in_current_TCP',c['instruction']);self.assertNotIn('G0',d['instruction'])
    def test_window_handles_reported_velocity_without_changing_drift_bounds(self):
        snap={'joint_names':['q'],'joint_positions':[0.],'right_arm':['q'],'joint_velocities':[1.],
            'objects':{'test_object':{'world':np.eye(4).tolist()}}}
        self.assertTrue(stationary_dispatch(snap,snap,{'pass':True})['pass'])
        changed=copy.deepcopy(snap);changed['joint_positions']=[.001]
        self.assertFalse(stationary_dispatch(snap,changed,{'pass':True})['pass'])
        self.assertFalse(stationary_dispatch(snap,snap,{'pass':False})['pass'])
    def test_stop_samples_bound_to_raw(self):
        samples=[StopSample(i*.2,(0,0,0),(0.,),(1.,)) for i in range(11)]
        proof={**settle(samples,active=False,fault=None,elapsed_wall_s=3.),'samples':[asdict(s) for s in samples],
               'wait_elapsed_wall_s':3.,'files':{}}
        raw=[{'snapshot':{'sim_time':s.sim_time,'link_world':{'right_eef_link':np.eye(4).tolist()},
            'joint_names':['q'],'right_arm':['q'],'joint_positions':[0.],'joint_velocities':[1.]}} for s in samples]
        self.assertTrue(bound_stop(proof,raw)['pass'])
        raw[3]['snapshot']['joint_positions']=[.1]
        with self.assertRaisesRegex(RuntimeError,'Altered'):bound_stop(proof,raw)
    def test_missing_stop_sample_rejected(self):
        proof={'files':{},'samples':[asdict(StopSample(1.,(0,0,0),(0.,),(0.,)))]}
        with self.assertRaisesRegex(RuntimeError,'missing'):bound_stop(proof,[])
    def test_unconfirmed_world_cannot_continue(self):
        self.assertFalse(continuation_after_slot(record_intact=True,configuration_intact=True,services_healthy=True,
            old_world_absent=False,new_world_reset_verified=True)['continue_next_slot'])
    def test_orphan_execution_at_timeout_stops_scheduling(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);ep=p/'episode';ep.mkdir();write(ep/'requests.json',[{'label':'close'}]);write(ep/'close_execution.json',{})
            write(p/'ISOLATED_TIMEOUT.json',{'files':{},'episode':str(ep),'known_partial_record':True,'outcome':'interrupted_unknown'})
            with self.assertRaisesRegex(RuntimeError,'command'):audit_partial(p)
    def test_full_stale_second_view_cannot_fall_back(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);req={'observation':str(p/'id_observation.json'),'request_id':'id','vision_arm':'full_stale',
                'object_id':'object','motions_completed':1,'method':'D'}
            reg={'full_stale_donors':{'object':{'images_by_plane':{'1':['oldrgb','olddepth']}}},'learned_policy':{'checkpoints':{'D':'fixed'}}}
            rep={'model_inference':True,'checkpoint':'fixed','visual_control':{'images':['currentrgb','currentdepth']}}
            with self.assertRaisesRegex(RuntimeError,'images'):image_binding(reg,req,rep)
    def test_container_mount_uses_registered_host_project(self):
        from unittest.mock import patch
        from .advisor33_launch import container_command
        reg={'project_root':'/host/project','source_snapshot':'/host/project/run/source','runtime_module':'test'}
        with patch('scripts.curobo_arx.nearfield.advisor33_launch.seed_kit_cache',return_value=Path('/host/cache')):
            cmd=container_command(Path('/host/run'),reg,'UUID',Path('/host/launch'),'container')
        self.assertIn('/host/project/asset:/workspace/asset:ro',cmd)
        self.assertIn('/host:/host:rw',cmd)
    def test_terminal_timeout_requires_new_world_marker(self):
        from .advisor33_launch import terminal_isolation
        rows=[{'trial_id':'slot','outcome':'stationary_wait_10s_timeout'}]
        with self.assertRaises(RuntimeError):terminal_isolation(rows,['slot','next'],None)
        marker={'trial_id':'slot','terminal_records_complete':True,'configuration_intact':True,'retry':False,'next_slot_requires_new_World':True}
        self.assertTrue(terminal_isolation(rows,['slot','next'],marker))
        marker['next_slot_requires_new_World']=False
        with self.assertRaises(RuntimeError):terminal_isolation(rows,['slot','next'],marker)
    def test_ordinary_failure_cannot_forge_timeout_isolation(self):
        from .advisor33_launch import terminal_isolation
        marker={'trial_id':'slot','terminal_records_complete':True,'configuration_intact':True,'retry':False,'next_slot_requires_new_World':True}
        with self.assertRaises(RuntimeError):terminal_isolation([{'trial_id':'slot','outcome':'empty_grasp'}],['slot','next'],marker)


if __name__=='__main__':unittest.main()
