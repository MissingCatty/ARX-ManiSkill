"""Offline ARX evidence/dispatch faults for the48-slot add-on; no GPU/World."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha
from .main48_spec import make_trials,validate_trials,validate_schedule,point_cloud,select_candidate,GEOMETRY
from .main48_audit import audit_baseline,same_selection
from .runtime import audit_actions
from .runtime_main48 import MainEnvironment


class Main48Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=[];cls.seeds={}
        for split,folder in [('train','nearfield_stage1_20261105_04/recovery_24'),('validation','nearfield_validation_20261106_03')]:
            reg=read(ROOT/'runs'/folder/'registration.json');cls.seeds.update(reg['seeds'])
            for sid,seed in reg['seeds'].items():
                t=next(t for t in reg['seed_trials'] if t['seed_id']==sid)
                asset=next(a for a in reg['catalog']['objects'] if a['object_id']==seed['object_id'])
                item={k:t[k] for k in ('seed_id','object_id','renderer','placement','source_Gstar_world')}
                item.update(source_family=seed.get('source_family',sid),source_split=split,geometry_group=asset['geometry_group'])
                cls.sources.append(item)
        cls.trials=make_trials(cls.sources,cls.seeds)
        reg=read(ROOT/'runs/nearfield_stage_recovery_20261109_02/training/physical/summary.json')
        cls.episode=Path(reg['trials'][0]['attempts'][0]['episode'])
        cls.row=read(cls.episode/'physical_entry_observation.json')
        cls.reference=np.asarray(cls.trials[0]['G0_world'])

    def test_original48_paired_conditions_and_balanced_order(self):
        self.assertEqual(len(self.trials),48)
        for index,m in enumerate('ABCD'):
            self.assertEqual(sum(t['method']==m for t in self.trials),12)
            self.assertEqual(sum(self.trials[i]['method']==m for i in range(0,48,4)),3)
        for obj in {t['object_id'] for t in self.trials}:
            self.assertEqual(len([t for t in self.trials if t['object_id']==obj]),12)

    def test_joint_condition_moves_H_with_G0_but_entry_only_preserves_G0(self):
        for t in self.trials:
            base=np.asarray(t['Gbase_world']);g0=np.asarray(t['G0_world']);h=np.asarray(t['entry_world'])
            back=np.eye(4);back[2,3]=-.08
            if t['condition']=='joint_prior_entry_offset':np.testing.assert_allclose(h,g0@back,atol=1e-12)
            else:np.testing.assert_array_equal(g0,base)
            if t['condition']=='entry_offset':self.assertGreater(np.linalg.norm(h[:3,3]-(base@back)[:3,3]),.0199)

    def test_Gstar_never_changes_the_generated_inputs(self):
        sources=copy.deepcopy(self.sources)
        for s in sources:s['source_Gstar_world']=np.eye(4).tolist()
        changed=make_trials(sources,self.seeds)
        for old,new in zip(self.trials,changed):
            for key in ('G0_world','entry_world','Gbase_world'):self.assertEqual(old[key],new[key])

    def test_duplicate_or_unpaired_trials_rejected(self):
        for key in ('trial_id','G0_world','placement'):
            rows=copy.deepcopy(self.trials)
            if key=='trial_id':rows[1][key]=rows[0][key]
            else:rows[1][key]=['tampered']
            with self.assertRaises(ValueError):validate_trials(rows)

    def grasp(self,goal=None,score=.5):
        goal=self.reference if goal is None else goal
        axes=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        rotation=np.asarray(goal)[:3,:3]@axes.T;row=np.zeros(17);row[:4]=[score,.05,.02,.02]
        row[4:13]=rotation.ravel();row[13:16]=goal[:3,3]+(.04-row[3])*rotation[:,0]
        return row

    def target(self,row):return np.array([row[13:16]+row[3]*row[4:13].reshape(3,3)[:,0]])

    def test_ARX40mm_TCP_calibration_and_symmetric_branch(self):
        row=self.grasp();target=self.target(row)
        selected=select_candidate([row],np.eye(4),self.reference,target)['selected']
        np.testing.assert_allclose(selected['goal_world'],self.reference,atol=1e-12)
        flip=self.reference.copy();flip[:3,:3]=flip[:3,:3]@np.diag([-1,-1,1]);row=self.grasp(flip)
        selected=select_candidate([row],np.eye(4),self.reference,self.target(row))['selected']
        self.assertTrue(selected['symmetry_flip']);np.testing.assert_allclose(selected['goal_world'],self.reference,atol=1e-12)

    def test_high_score_outside_neighborhood_is_not_selected(self):
        distant=self.reference.copy();distant[0,3]+=.041
        rows=np.stack([self.grasp(distant,.99),self.grasp(score=.2)])
        result=select_candidate(rows,np.eye(4),self.reference,np.concatenate([self.target(r) for r in rows]))
        self.assertEqual(result['selected']['index'],1)

    def test_unsupported_surface_and_invalid_rotation_are_rejected(self):
        row=self.grasp()
        self.assertIsNone(select_candidate([row],np.eye(4),self.reference,self.target(row)+1)['selected'])
        row[4:13]=0
        self.assertIsNone(select_candidate([row],np.eye(4),self.reference,self.target(row))['selected'])

    def test_cloud_uses_depth_and_visible_mask_not_object_pose(self):
        depth=np.full((8,8),.1);m={r:np.zeros_like(depth,dtype=bool) for r in ('target','right_link27','right_link28')}
        m['target'][:6,:6]=True;m['right_link27'][7,:]=True
        depth[6,0]=np.nan;depth[6,1]=0;depth[6,2]=5
        points,target=point_cloud(depth,[[100,0,4],[0,100,4],[0,0,1]],m)
        self.assertEqual(len(target),36);self.assertEqual(len(points),53)
        m['right_link27'][0,0]=True
        with self.assertRaises(ValueError):point_cloud(depth,np.eye(3),m)

    def fixture(self,path):
        path=Path(path);trial=next(t for t in self.trials if t['method']=='B');g0=np.asarray(trial['G0_world'])
        row=self.grasp(g0);target=self.target(row);result=select_candidate([row],np.eye(4),g0,target)
        folder=path/'geometry';e=folder/'query_evidence';e.mkdir(parents=True)
        observation=folder/'query_observation.json';write(observation,{'camera_world_optical':np.eye(4).tolist()})
        np.save(e/'filtered_predictions.npy',[row]);np.save(e/'target_points.npy',target);write(e/'candidate_selection.json',result)
        request={'method':'B','request_id':'query','trial_id':trial['trial_id'],'G0_world':trial['G0_world'],
            'observation':str(observation),'registration_sha256':'registered','raw_index':0,'files':{str(observation):sha(observation)}}
        req=folder/'query_request.json';write(req,request)
        reply={**result,'rejected':None,'projection':{'target_visibility_pass':True},'total_seconds':1.,
            'ok':True,'method':'B','request_id':'query','registration_sha256':'registered','request_sha256':sha(req),
            'files':{str(p):sha(p) for p in e.iterdir()}}
        write(folder/'query_reply.json',reply);write(folder/'query_dispatch_check.json',{'pass':True})
        write(path/'geometry_requests.json',[{'request_id':'query'}]);write(path/'requests.json',[{'label':'b_refine_001'}])
        write(path/'interpreter_events.json',[{'kind':'target_request','baseline_method':'B','geometry_request_id':'query',
            'command_label':'b_refine_001','goal_world':result['selected']['goal_world']}])
        return trial

    def test_geometry_request_reply_ledger_loss_stops(self):
        for missing in ('geometry/query_request.json','geometry/query_reply.json','geometry_requests.json','geometry/query_dispatch_check.json'):
            with tempfile.TemporaryDirectory() as f:
                root=Path(f);trial=self.fixture(root);self.assertTrue(audit_baseline(root,trial)['pass'])
                (root/missing).unlink()
                with self.assertRaises((RuntimeError,FileNotFoundError)):audit_baseline(root,trial)

    def test_executed_goal_must_match_ranked_geometry(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);trial=self.fixture(root);ev=read(root/'interpreter_events.json');ev[0]['goal_world'][0][3]+=.01
            write(root/'interpreter_events.json',ev)
            with self.assertRaises(AssertionError):audit_baseline(root,trial)

    def test_A_cannot_substitute_measured_historical_Gstar(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);trial=next(t for t in self.trials if t['method']=='A');changed=np.asarray(trial['G0_world']);changed[0,3]+=.01
            write(root/'interpreter_events.json',[{'kind':'target_request','baseline_method':'A','goal_world':changed.tolist()}])
            with self.assertRaises(AssertionError):audit_baseline(root,trial)

    def test_orphan_execution_without_command_stops(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);write(root/'b_refine_001_execution.json',{})
            with self.assertRaises(RuntimeError):audit_actions(root,[{'label':'b_refine_001'}])

    def test_successive_B_queries_not_same_observation_or_automatic_retry(self):
        env=object.__new__(MainEnvironment);trial=next(t for t in self.trials if t['method']=='B')
        env.group=SimpleNamespace(active_trial=trial);env.bridge=SimpleNamespace(active=False,fault=None,snapshot=lambda:self.row['snapshot'])
        env.pending_geometry=None;env.last_observation=self.row;env.physical_entry_ready=True
        env.previous_execution={'final':{'sim_time':self.row['frame_time']}};env.baseline_motions=1
        env.submit=Mock();env.finish=Mock()
        env.decide();env.submit.assert_not_called();env.finish.assert_not_called()

    def test_C_D_keep_inherited_decision_method(self):
        for method in ('C','D'):
            env=object.__new__(MainEnvironment);env.group=SimpleNamespace(active_trial={'method':method})
            with patch('scripts.curobo_arx.nearfield.runtime_stage_recovery.StageEnvironment.decide',return_value='inherited') as called:
                self.assertEqual(env.decide(),'inherited');called.assert_called_once()

    def test_cross_numpy_diagnostic_angle_difference_is_not_decision_change(self):
        old={'selected':None,'accepted':[],'rejected':[{'index':24,'reason':'outside_G0_neighborhood',
             'error':{'rotation_rad':.9260579967612226,'translation_m':.05}}]}
        new=copy.deepcopy(old);new['rejected'][0]['error']['rotation_rad']=.9260579908508849
        self.assertTrue(same_selection(old,new))
        for field,value in [('index',25),('reason','no_visible_target_support')]:
            changed=copy.deepcopy(new);changed['rejected'][0][field]=value;self.assertFalse(same_selection(old,changed))
        changed=copy.deepcopy(new);changed['rejected'][0]['error']['rotation_rad']+=1e-5
        self.assertFalse(same_selection(old,changed))
        changed=copy.deepcopy(new);changed['selected']={'index':24};self.assertFalse(same_selection(old,changed))

    def test_real_B_no_candidate_chain_survives_environment_numeric_recheck(self):
        root=ROOT/'runs/nearfield_main48_20261109_01'
        reg=read(root/'registration.json');trial=reg['seed_trials'][1]
        episode=next((root/'groups/nearfield_raytracedlighting/trials'/trial['trial_id']/'episodes').iterdir())
        result=audit_baseline(episode,trial)
        self.assertEqual(result['geometry_calls'],1)
        self.assertIsNone(result['geometry_queries'][0]['selected'])
        self.assertFalse(result['geometry_queries'][0]['dispatched_event'])

    def test_continuation_cannot_replay_or_change_a_consumed_case(self):
        carried=[{'trial_id':t['trial_id'],'method':t['method'],'record_integrity_pass':True} for t in self.trials[:2]]
        reg={'registered_trials':self.trials,'seed_trials':self.trials[2:],'carried_details':carried}
        validate_schedule(reg)
        for mutation in ('repeat','reorder','change','unverified'):
            changed=copy.deepcopy(reg)
            if mutation=='repeat':changed['seed_trials'].insert(0,self.trials[1])
            if mutation=='reorder':changed['seed_trials'].reverse()
            if mutation=='change':changed['seed_trials'][0]['G0_world'][0][3]+=.01
            if mutation=='unverified':changed['carried_details'][1]['record_integrity_pass']=False
            with self.assertRaises(ValueError):validate_schedule(changed)

    def test_continuation_blocks_an_unexpected_claim_in_remaining46(self):
        from .main48_continue import partition,ledger_rows
        root=ROOT/'runs/nearfield_main48_20261109_01'
        correct=partition(root);self.assertEqual(len(correct['remaining']),46)
        def changed(path):
            rows=ledger_rows(path)
            return [{'physical':0,'state':'claimed'}] if path.stem==correct['remaining'][0]['trial_id'] else rows
        with patch('scripts.curobo_arx.nearfield.main48_continue.ledger_rows',side_effect=changed):
            with self.assertRaises(ValueError):partition(root)

    def test_supplement_is_failure_without_inventing_native_terminal_records(self):
        from .main48_continue import partition,supplement
        root=ROOT/'runs/nearfield_main48_20261109_01';part=partition(root)
        detail,pins=supplement(root,part)
        self.assertFalse(detail['physical_success']);self.assertFalse(detail['native_runtime_completed'])
        self.assertTrue(detail['supplemental_accounting_verified']);self.assertIsNone(detail['wall_seconds'])
        self.assertTrue(detail['event_applicability']['geometry_query'])
        self.assertFalse(detail['event_applicability']['close'])
        self.assertEqual(set(detail['physical']['missing_stages']),{'close','lift','hold'})
        self.assertFalse((Path(detail['source_episode'])/'physical_receipt.json').exists())
        self.assertEqual(len(partition(root)['remaining']),46)


if __name__=='__main__':unittest.main()
