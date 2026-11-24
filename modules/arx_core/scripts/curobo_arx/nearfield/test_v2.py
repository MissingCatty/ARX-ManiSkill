import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,digest
from .closure import bind,validate_reply
from .runtime_v2 import ActionEnvironment
from training.nearfield_arx.lookahead import select
from training.nearfield_arx.history import before,unavailable,for_method
from training.nearfield_arx.contract import DSession,encode


class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        ee=np.eye(4).tolist()
        self.command={'id':'accepted','label':'c_step_001','joint_names':['arm'],
            'position':[[0.],[0.]],'position_tolerance_rad':.005}
        self.snapshot={'sim_time':1.2,'joint_names':['arm'],'joint_positions':[0.],
            'right_arm':['arm'],'joint_limits':[[-1.,1.]],'link_world':{'right_eef_link':ee}}
        self.obs={'attempt_id':'trial','frame_time':1.2,'snapshot':copy.deepcopy(self.snapshot),'visual_geometry':{'pass':True}}
        self.execution={'settled':True,'records':[{'time':t/10,'position':[0.],'target':[0.],'ee_world':ee} for t in range(11)],
            'final':{'sim_time':1.}}
        write(self.root/'c_step_001_command.json',self.command)
        write(self.root/'c_step_001_execution.json',self.execution)
        write(self.root/'c_step_001_settling.json',{'accepted':True})
        write(self.root/'requests.json',[self.command]);self.ee=ee

    def tearDown(self):self.tmp.cleanup()

    def binding(self,**kw):
        return bind(self.root,self.snapshot,self.obs,self.command,self.ee,
            **{'active':False,'fault':False,'request_id':'grasp','configuration_identity':'frozen',**kw})

    def test_binding_holds_real_last_command_and_requires_matching_certificate(self):
        b=self.binding();self.assertTrue(b['pass']);self.assertEqual(b['hold_q'],[0.])
        cert={'pass':True,'binding_sha256':digest(b),'hold_q':[0.]}
        self.assertTrue(validate_reply(b,cert,self.command))
        cert['binding_sha256']='other'
        with self.assertRaises(RuntimeError):validate_reply(b,cert,self.command)

    def test_execution_without_command_stops(self):
        (self.root/'c_step_001_command.json').unlink()
        with self.assertRaises(RuntimeError):self.binding()

    def test_missing_receipt_stops(self):
        (self.root/'c_step_001_execution.json').unlink()
        with self.assertRaises(RuntimeError):self.binding()

    def test_busy_fault_and_stale_observation_stop(self):
        for kw in ({'active':True},{'fault':'unknown'}):
            with self.assertRaises(RuntimeError):self.binding(**kw)
        self.snapshot['sim_time']=2.
        with self.assertRaises(RuntimeError):self.binding()

    def test_real_render_clock_rounding_requires_same_physics_capture_evidence(self):
        from .capture import VERSION
        self.snapshot['sim_time']=1.200003;self.obs['snapshot']['sim_time']=1.200003
        with self.assertRaises(RuntimeError):self.binding()  # Legacy path is unchanged.
        self.obs['capture']={'version':VERSION,
            'reference_time':{'referenceTimeNumerator':12,'referenceTimeDenominator':10},
            'physics_time_before_render':1.200003,'physics_time_after_render':1.200003,
            'after_render':{'sim_time':1.200003},'pose_sync_state':{'sim_time':1.200003}}
        self.assertTrue(self.binding()['pass'])
        self.obs['capture']['after_render']['sim_time']=1.
        with self.assertRaises(RuntimeError):self.binding()

    def test_old_frame_cannot_pass_the_render_clock_rounding_path(self):
        from .closure import observation_timing
        from .capture import VERSION
        self.obs['capture']={'version':VERSION,
            'reference_time':{'referenceTimeNumerator':1,'referenceTimeDenominator':1},
            'physics_time_before_render':1.2,'physics_time_after_render':1.2,
            'after_render':{'sim_time':1.2},'pose_sync_state':{'sim_time':1.2}}
        self.obs['frame_time']=1.
        with self.assertRaises(RuntimeError):observation_timing(self.obs)

    def test_training_event_join_uses_the_verified_physics_clock(self):
        from training.nearfield_arx.export import observed_history
        from .capture import VERSION
        self.obs['frame_time']=1.19997
        self.obs['capture']={'version':VERSION,
            'reference_time':{'referenceTimeNumerator':119997,'referenceTimeDenominator':100000},
            'physics_time_before_render':1.2,'physics_time_after_render':1.2,
            'after_render':{'sim_time':1.2},'pose_sync_state':{'sim_time':1.2}}
        command={**self.command,'label':'approach','requested_sim_time':.2}
        self.obs['teacher_command']=command
        write(self.root/'selected_grasp.json',{'grasp_world':self.ee})
        history=observed_history(self.root,self.obs,[command],1.2,1.2)
        self.assertEqual(history['stage'],'post_approach_stopped')
        self.assertEqual(history['elapsed_since_command_s'],1.)

    def test_changed_target_and_duplicate_close_stop(self):
        self.command['position'][-1]=[.1]
        with self.assertRaises(RuntimeError):self.binding()
        self.command['position'][-1]=[0.]
        write(self.root/'requests.json',[self.command,{'id':'close','label':'close'}])
        with self.assertRaises(RuntimeError):self.binding()

    def test_video_keeps_onset_frame_across_verified_render_clock_rounding(self):
        from .report import execution_frames
        from .capture import VERSION
        self.obs['frame_time']=1.19997
        self.obs['capture']={'version':VERSION,
            'reference_time':{'referenceTimeNumerator':119997,'referenceTimeDenominator':100000},
            'physics_time_before_render':1.2,'physics_time_after_render':1.2,
            'after_render':{'sim_time':1.2},'pose_sync_state':{'sim_time':1.2}}
        write(self.root/'requests.json',[{'label':'approach','requested_sim_time':1.4}])
        write(self.root/'approach_execution.json',{'sim_duration_s':.4})
        self.assertEqual(execution_frames(self.root,[self.obs]),[self.obs])
        self.obs['capture']['after_render']['sim_time']=1.
        with self.assertRaises(RuntimeError):execution_frames(self.root,[self.obs])

    def test_rejected_model_never_accepts_closure(self):
        self.assertFalse(validate_reply(self.binding(),{'pass':False,'rejected':'collision'},self.command))

    def env(self):
        e=ActionEnvironment.__new__(ActionEnvironment);e.bridge=SimpleNamespace(active=False,fault=False,snapshot=lambda:self.snapshot)
        e.grasp_seen=False;e.local_index=99999;e.trace_before=lambda m,t:{'decision_id':'d'}
        e.log_event=lambda event:None;e.submit=lambda label,goal,phase:(label,phase)
        e.request_closure=lambda ident:('closure_gate',ident)
        return e

    def test_step_never_becomes_approach_or_closure_based_on_teacher_length(self):
        e=self.env();target=np.eye(4);target[0,3]=.01
        result=e.apply_action('C',encode('C',np.eye(4),target))
        self.assertEqual(result,('c_step_100000','local'));self.assertFalse(e.grasp_seen)

    def test_explicit_grasp_ignores_teacher_length_and_cannot_repeat(self):
        e=self.env();e.local_index=0
        self.assertEqual(e.apply_action('C','GRASP'),('closure_gate','d'))
        with self.assertRaises(RuntimeError):e.apply_action('C','GRASP')

    def test_lookahead_is_contiguous_and_bounded(self):
        waypoints=[]
        for x in (.005,.019,.021,.010):
            pose=np.eye(4);pose[0,3]=x;waypoints.append(pose)
        self.assertEqual(select(np.eye(4),waypoints),1)
        with self.assertRaises(ValueError):select(np.eye(4),waypoints,2)

    def test_budget_history_is_before_output_and_exhaustion_is_not_grasp(self):
        session=DSession(np.eye(4));snapshots=[]
        for _ in range(3):
            snapshots.append(before('D',session.calls,session.refines,None,1.))
            event=session.request(np.eye(4),'REFINE 0.000 0.000 0.000 0.0000 0.0000 0.0000')
        self.assertEqual([h['calls_completed'] for h in snapshots],[0,1,2])
        self.assertEqual([h['remaining_refines'] for h in snapshots],[2,1,0])
        self.assertEqual(event['kind'],'budget_exhausted')

    def test_unknown_and_inapplicable_histories_are_distinct(self):
        self.assertIsNone(unavailable('D')['calls_completed'])
        self.assertEqual(unavailable('C')['refine_budget_availability'],'not_applicable')
        real=before('C',1,0,{'sim_time':1.,'outcome':'accepted','dispatched':True},1.2)
        self.assertEqual(for_method({'decision_history_by_method':{'C':real}},'D')['availability'],'unrecorded')

    def test_original_ARX_evaluator_receives_chronological_real_actions(self):
        from scripts.curobo_arx.data_factory.common import ROOT
        from scripts.curobo_arx.data_factory.recording import physical
        source=next((ROOT/'runs/nearfield_stage1_20261105_01/execution_03').glob('groups/*/trials/00*/episodes/*'))
        fixture=self.root/'evaluation';fixture.mkdir()
        for p in source.iterdir():
            if p.is_file() and p.name not in ('physical_receipt.json','execution_roles.json'):
                (fixture/p.name).symlink_to(p)
        requests=read(source/'requests.json')
        sequence=[(r['label'] if r['label'] in ('pregrasp','approach','close','lift','hold') else 'approach',r['label']) for r in requests]
        write(fixture/'execution_roles.json',{'sequence':sequence})
        result=physical(fixture)
        self.assertTrue(result['pass']);self.assertEqual(result['executed_stages'],[r['label'] for r in requests])
        sequence[1]=('hold',sequence[1][1]);write(fixture/'execution_roles.json',{'sequence':sequence})
        with self.assertRaises(ValueError):physical(fixture)

    def test_actual_seven_step_visual_chain_is_required_for_admission(self):
        from scripts.curobo_arx.data_factory.common import ROOT
        from scripts.curobo_arx.data_factory.ledger import visual_evidence_files
        source=next((ROOT/'runs/nearfield_stage1_20261105_02/replay_02').glob('groups/*/trials/00*/episodes/*'))
        quality=read(source/'quality_receipt.json');names=visual_evidence_files(source,quality)
        self.assertEqual(sum(n.startswith('c_step_') for n in names),7)
        self.assertNotIn('approach_visual_validation.json',names)
        for name in names:self.assertIn(name,quality['files']);self.assertTrue(read(source/name)['pass'])
        fixture=self.root/'admission';fixture.mkdir()
        for n in ('execution_roles.json','requests.json'):write(fixture/n,read(source/n))
        roles=read(fixture/'execution_roles.json');roles['sequence'].pop(1);write(fixture/'execution_roles.json',roles)
        with self.assertRaises(ValueError):visual_evidence_files(fixture,quality)
        no_role=copy.deepcopy(quality);no_role['files'].pop('execution_roles.json')
        with self.assertRaises(ValueError):visual_evidence_files(source,no_role)


if __name__=='__main__':unittest.main()
