"""Offline checks use the admitted ARX sources, never another robot."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read
from training.nearfield_arx.contract import parse_action, pose_error, encode
from .physical_spec import physical_trials, bounded_teacher_goal, MAX_TEACHER_STEPS
from .prepare_physical import source_inventory
from .runtime_physical import PhysicalEnvironment


class PhysicalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.seeds, cls.families, cls.geometries = source_inventory()
        cls.trials = physical_trials(cls.sources)
        seed = next(iter(cls.seeds.values()))
        cls.snapshot = read(Path(seed['episode'])/'actual_entry.json')['snapshot']

    def test_exact_budget_and_disjoint_original_splits(self):
        self.assertEqual(len(self.trials),36)
        self.assertEqual(sum(t['source_split']=='train' for t in self.trials),24)
        for field in ('source_family','geometry_group'):
            a={t[field] for t in self.trials if t['source_split']=='train'}
            b={t[field] for t in self.trials if t['source_split']=='validation'}
            self.assertFalse(a & b)

    def test_prior_is_bound_to_real_entry_and_resets_are_retained(self):
        for t in self.trials:
            H,G=np.asarray(t['entry_world']),np.asarray(t['G0_world'])
            relative=np.linalg.inv(H)@G
            expected=np.eye(4);expected[2,3]=t['perturbation']['backoff_m']
            np.testing.assert_allclose(relative,expected,atol=1e-12)
            self.assertEqual(t['placement'],self.seeds[t['seed_id']]['placement'])
            self.assertEqual(t['renderer'],self.seeds[t['seed_id']]['renderer'])
            self.assertEqual(t['perturbation']['virtual_prior_variants'],0)
            parse_action('D',encode('D',H,t['source_Gstar_world'],G))

    def test_registered_physical_variance_covers_angles_axes_and_signs(self):
        offsets=[t['perturbation']['rotation_Gstar_axes_degrees'] for t in self.trials]
        self.assertEqual({round(np.linalg.norm(x)) for x in offsets},{5,10,15})
        self.assertEqual({int(np.argmax(np.abs(x))) for x in offsets},{0,1,2})
        self.assertTrue(any(sum(x)>0 for x in offsets) and any(sum(x)<0 for x in offsets))
        self.assertEqual({t['perturbation']['backoff_m'] for t in self.trials if t['physical_entry_kind']=='near_stopped_recovery'}, {.005,.010,.020})

    def test_teacher_path_respects_unchanged_C_bounds_for_actual_ARX_goals(self):
        for t in self.trials:
            actual=np.asarray(t['entry_world']);star=t['source_Gstar_world']
            for step in range(MAX_TEACHER_STEPS):
                token,goal,final=bounded_teacher_goal(actual,star)
                dp,dr=parse_action('C',token)
                self.assertLessEqual(np.linalg.norm(dp),.020001)
                self.assertLessEqual(np.linalg.norm(dr),np.deg2rad(4.5001))
                actual=goal
                if final:break
            self.assertTrue(final)
            self.assertLess(pose_error(actual,star)['translation_m'],1e-6)
            self.assertLess(pose_error(actual,star)['rotation_rad'],2e-6)

    def test_rotation_limited_teacher_provenance_is_json_serializable(self):
        from scripts.curobo_arx.data_factory.common import canonical
        for t in self.trials:
            token,goal,final=bounded_teacher_goal(t['entry_world'],t['source_Gstar_world'])
            self.assertIs(type(final),bool)
            canonical({'token':token,'next_goal_world':goal.tolist(),'final_goal':final})
        actual=ROOT/'runs/nearfield_physical_20261108_03/groups/nearfield_raytracedlighting/trials/02__household_hot3d_potato_masher__physical_2/episodes/261e3e0e8ab24173a020758c15916a3b/physical_entry_observation.json'
        row=read(actual);token,goal,final=bounded_teacher_goal(row['snapshot']['link_world']['right_eef_link'],self.trials[2]['source_Gstar_world'])
        canonical({'token':token,'next_goal_world':goal.tolist(),'final_goal':final})
        self.assertFalse(final)

    def env(self):
        e=PhysicalEnvironment.__new__(PhysicalEnvironment)
        snap=copy.deepcopy(self.snapshot);trial=self.trials[0]
        snap['link_world']['right_eef_link']=trial['entry_world']
        e.bridge=SimpleNamespace(active=False,fault=False,snapshot=lambda:snap)
        e.physical_entry_ready=False;e.teacher_final_issued=False;e.local_index=0
        e.entry_stopped_sim_time=snap['sim_time']
        e.last_observation={'frame_time':snap['sim_time'],'snapshot':snap}
        e.group=SimpleNamespace(active_trial=trial)
        e.apply_action=lambda m,t:('apply',m,t);e.finish=lambda reason:('finish',reason)
        e.log_event=lambda event:None
        return e,snap

    def test_stale_entry_image_does_not_dispatch(self):
        e,_=self.env();self.assertIsNone(e.decide());self.assertFalse(e.physical_entry_ready)

    def test_busy_or_faulted_ARX_stops_before_entry(self):
        for field,value in [('active',{'command':'in_flight'}),('fault','unknown')]:
            e,_=self.env();setattr(e.bridge,field,value)
            with self.assertRaises(RuntimeError):e.decide()

    def test_moving_entry_is_not_positive_supervision(self):
        e,s=self.env();index=s['joint_names'].index(s['right_arm'][0]);s['joint_velocities'][index]=.02
        e.last_observation['frame_time']+=.2
        self.assertIsNone(e.decide());self.assertFalse(e.physical_entry_ready)

    def test_recorded_ARX_residual_velocity_waits_then_stopped_frame_can_proceed(self):
        e,s=self.env();e.last_observation['frame_time']+=.2
        for name in s['right_arm']:s['joint_velocities'][s['joint_names'].index(name)]=.04
        self.assertIsNone(e.decide())
        self.assertFalse(e.physical_entry_ready)
        for name in s['right_arm']:s['joint_velocities'][s['joint_names'].index(name)]=0.
        e.last_observation['frame_time']+=.2
        with tempfile.TemporaryDirectory() as tmp:
            e.episode=Path(tmp);self.assertEqual(e.decide()[:2],('apply','C'))

    def test_fresh_live_state_cannot_replace_a_moving_rendered_state(self):
        e,s=self.env();e.last_observation['frame_time']+=.2
        e.last_observation['snapshot']=copy.deepcopy(s)
        for name in s['right_arm']:s['joint_velocities'][s['joint_names'].index(name)]=0.
        other=e.last_observation['snapshot'];other['joint_velocities'][other['joint_names'].index(other['right_arm'][0])]=.04
        self.assertIsNone(e.decide());self.assertFalse(e.physical_entry_ready)

    def test_fresh_entry_persists_only_after_real_motion(self):
        e,s=self.env();e.last_observation['frame_time']+=.2
        for name in s['right_arm']:s['joint_velocities'][s['joint_names'].index(name)]=0.
        with tempfile.TemporaryDirectory() as tmp:
            e.episode=Path(tmp);reply=e.decide()
            self.assertTrue(e.physical_entry_ready)
            self.assertEqual(reply[:2],('apply','C'))
            self.assertTrue((e.episode/'physical_entry_observation.json').exists())
            self.assertNotEqual(reply[-1],'GRASP')

    def test_terminal_decision_uses_original_GRASP_gate_and_budget_is_failure(self):
        e,s=self.env();e.physical_entry_ready=True;e.teacher_final_issued=True
        e.previous_execution={'final':{'sim_time':s['sim_time']}}
        e.last_observation['frame_time']+=.2
        for name in s['right_arm']:s['joint_velocities'][s['joint_names'].index(name)]=0.
        self.assertEqual(e.decide(),('apply','C','GRASP'))
        e.teacher_final_issued=False;e.local_index=MAX_TEACHER_STEPS
        self.assertEqual(e.decide(),('finish','teacher_step_budget_exhausted'))

    def test_postapproach_GRASP_waits_for_a_fresh_stationary_sample(self):
        e,s=self.env();e.physical_entry_ready=True;e.teacher_final_issued=True
        e.previous_execution={'final':{'sim_time':s['sim_time']}}
        self.assertIsNone(e.decide())
        e.last_observation['frame_time']+=.2
        s['joint_velocities'][s['joint_names'].index(s['right_arm'][0])]=.04
        self.assertIsNone(e.decide())

    def test_only_accepted_state_generation_changes_stage_to_entry(self):
        e,_=self.env();e.state='executing';e.stage='state_generation'
        e.last_command={'label':'state_generation'};e.group.world=SimpleNamespace(current_time=12.)
        def accepted(env):env.state='local_decision'
        with patch('scripts.curobo_arx.nearfield.runtime_v2.ActionEnvironment.tick',accepted):e.tick()
        self.assertEqual(e.stage,'entry');self.assertEqual(e.entry_stopped_sim_time,12.)
        e.state='executing';e.stage='state_generation'
        def failed(env):env.state='idle'
        with patch('scripts.curobo_arx.nearfield.runtime_v2.ActionEnvironment.tick',failed):e.tick()
        self.assertEqual(e.stage,'state_generation')


if __name__=='__main__':unittest.main()
