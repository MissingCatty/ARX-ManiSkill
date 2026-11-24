"""Offline ARX-specific fault tests. No CUDA, simulator, policy or RPC access."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from .common import ROOT,VERSION,digest,read,sha,write
from .geometry import Roles,handoff,grasp_pool,pose_equal,translated
from .ledger import Ledger
from .recording import Recording,Writer,physical,quality
from .launch import allocate


QUOTAS={'max_total_dispatches':100,'max_physical_attempts':20,'goals_per_object':2,
        'distances_mm':[10,40,60,80],'successes_per_bin':2,'physical_per_bin':4,'planning_rejections_per_bin':3}


class IsolationTests(unittest.TestCase):
    def test_boundary_noise_is_checked_before_existing_planning_margin(self):
        from .teacher import bounded_planning_start, Rejection
        # A real initial ARX elbow is slightly below zero after settling.
        raw=np.array([-.00000557,.5]);bounds=np.array([[0.,3.24],[-2.,2.]])
        projected=bounded_planning_start(raw,bounds)
        self.assertAlmostEqual(projected[0],.0001)
        self.assertEqual(projected[1],raw[1])
        with self.assertRaisesRegex(Rejection,'outside_frozen_limits'):
            bounded_planning_start([-.001,.5],bounds)

    def test_render_step_measures_world_advance_instead_of_assuming_physics_dt(self):
        from .runtime import advance_render_tick
        class World:
            current_time=3.016666824
            def step(self,render):self.current_time+=self.delta
        world=World();world.delta=.2
        advance_render_tick(world,.2)
        world.delta=.1916666766628623  # Actual native startup alignment, 11 receipt.
        advance_render_tick(world,.2)
        world.delta=1/120
        with self.assertRaisesRegex(RuntimeError,'World render step advanced'):
            advance_render_tick(world,.2)

    def test_recorder_rejects_subframe_or_skipped_frame_cadence_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=Writer(4)
            try:
                rec=Recording(Path(tmp)/'episode',{'env':'env0','id':'a'},writer)
                row={'env_id':'env0','attempt_id':'a','frame_time':0.,'capture_sim_time':0.,
                    'snapshot':{'link_world':{'right_eef_link':np.eye(4).tolist()}},'stage':'entry'}
                rec.add(row,np.zeros((2,2,3)),np.zeros((2,2)))
                for t in (1/120,.4):
                    with self.assertRaisesRegex(ValueError,'0.2-second'):
                        rec.add({**row,'frame_time':t,'capture_sim_time':t},np.zeros((2,2,3)),np.zeros((2,2)))
                rec.add({**row,'frame_time':.2,'capture_sim_time':.2},np.zeros((2,2,3)),np.zeros((2,2)))
            finally:writer.close()

    def test_truncated_group_pass_covers_top_and_bottom_surface_regions(self):
        from .geometry import balanced_group_order
        groups={str(((axis,sign),(direction,side),roll)):[{'score':1.-axis*.1}]
            for axis in range(3) for sign in (-1,1) for direction in range(3)
            for side in (-1,1) for roll in range(4)}
        import ast
        first=[ast.literal_eval(key) for key in balanced_group_order(groups)[:64]]
        self.assertEqual({key[0] for key in first},{(a,s) for a in range(3) for s in (-1,1)})
        self.assertEqual({key[1] for key in first},{(a,s) for a in range(3) for s in (-1,1)})

    def test_ik_failure_diagnostics_do_not_mislabel_collision_as_unreachable(self):
        from types import SimpleNamespace
        from .teacher import ik_failure_details, Rejection
        details=ik_failure_details(SimpleNamespace(position_error=np.array([.0001,np.nan]),
            rotation_error=np.array([.003,.005]),feasible=np.array([False,False])))
        self.assertEqual(details['feasible_seeds'],0)
        self.assertEqual(details['reported_seeds'],2)
        self.assertEqual(details['minimum_position_error'],.0001)
        failure=Rejection('no_ik_solution',details)
        self.assertEqual(str(failure),'no_ik_solution')
        self.assertEqual(failure.details,details)

    def test_recording_rejects_wrong_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=Writer(4)
            try:
                rec=Recording(Path(tmp)/'episode',{'env':'gpu3/env_0','id':'attempt'},writer)
                with self.assertRaisesRegex(ValueError,'Cross-environment'):
                    rec.add({'env_id':'gpu3/env_1','attempt_id':'attempt','frame_time':0.,'capture_sim_time':0.},np.zeros((2,2,3)),np.zeros((2,2)))
            finally:writer.close()

    def test_cross_environment_contact_is_fault_not_filtered(self):
        roles=Roles('/World/Envs/env_0',[0,0,0])
        with self.assertRaisesRegex(ValueError,'Cross-environment'):
            roles.event({'a':'/World/Envs/env_0/Robot/right_link27','b':'/World/Envs/env_1/Task/assets/asset_egad_A1'})

    def test_path_mapping_retains_raw_identity(self):
        root='/World/Envs/env_1';roles=Roles(root,[100,0,0])
        event=roles.event({'a':root+'/Robot/link','b':root+'/Task/object','num_contacts':1})
        self.assertEqual(event['a'],'/World/Robot/link')
        self.assertEqual(event['raw_a'],root+'/Robot/link')
        g=np.eye(4);g[:3,3]=[4.048,2.08,.7]
        np.testing.assert_allclose(translated(translated(g,[100,0,0]),[100,0,0],True),g,atol=1e-12,rtol=0)

    def test_g_local_perturbation_and_undisturbed_first(self):
        from scipy.spatial.transform import Rotation
        g=np.eye(4);g[:3,:3]=Rotation.from_euler('y',1.).as_matrix()
        a=handoff(g,40,0,20261029)
        expected=g.copy();expected[:3,3]-=.04*g[:3,2]
        np.testing.assert_allclose(a['world'],expected)
        b=handoff(g,80,12,20261029)
        self.assertEqual(b,handoff(g,80,12,20261029))
        local=np.linalg.inv(g)@np.array(b['world'])
        self.assertLessEqual(np.linalg.norm(local[:2,3]),.02+1e-10)
        self.assertLessEqual(abs(local[2,3]+.08),.02+1e-10)
        self.assertLessEqual(Rotation.from_matrix(local[:3,:3]).magnitude(),np.deg2rad(15)+1e-10)

    def test_parallel_jaw_flip_not_new_grasp(self):
        g=np.eye(4);h=np.eye(4);h[:3,:3]=np.diag([-1,-1,1])
        self.assertTrue(pose_equal(g,h,symmetric=True))
        self.assertFalse(pose_equal(g,h,symmetric=False))

    def test_sideways_candidate_not_rejected_by_legacy_downward_heuristic(self):
        row=np.zeros(17);row[0]=.9;row[1]=.04;row[3]=.02;row[4:13]=np.eye(3).reshape(-1)
        pool=grasp_pool([row],np.eye(4),np.eye(4),[.1,.1,.1])
        self.assertEqual(len(pool['candidates']),1)


class LedgerTests(unittest.TestCase):
    def test_first_object_priority_does_not_reorder_assets_or_reset_indices(self):
        other=Ledger(self.root/'priority.db')
        catalog={'objects':[{'object_id':'A','placement':{'position':[10.,10.,.4]}},
                            {'object_id':'B','placement':{'position':[10.4,10.,.4]}}]}
        try:
            other.initialize(catalog,{**QUOTAS,'first_object':'B'})
            self.assertEqual(other.available_object()['object_id'],'B')
            actual=[json.loads(r[0]) for r in other.db.execute('SELECT record FROM objects ORDER BY rowid')]
            self.assertEqual(actual,catalog['objects'])
            other.acquire_object('owner',object_id='B')
            self.assertEqual(other.available_object()['object_id'],'A')
        finally:other.close()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.ledger=Ledger(self.root/'ledger.db')
        catalog={'objects':[{'object_id':'egad_A1'}]}
        self.ledger.initialize(catalog,QUOTAS)
        self.goal={'goal_id':'g0','grasp_world':np.eye(4).tolist(),'group':'side'}
        self.ledger.add_goals('egad_A1',{'candidates':[self.goal]})

    def tearDown(self):
        self.ledger.close();self.tmp.cleanup()

    def claim(self,env='gpu3/env0'):
        return self.ledger.claim('egad_A1','owner',env)

    def receipts(self,task,*,physical_pass=True,write_complete=True,content='unique',actual=None):
        path=self.root/task['id'];path.mkdir()
        write(path/'physical_receipt.json',{'pass':physical_pass})
        write(path/'episode.json',{'version':VERSION,'attempt_id':task['id'],'write_complete':write_complete,
            'actual_entry':{'snapshot':{'link_world':{'right_eef_link':(np.eye(4) if actual is None else actual).tolist()}}}})
        write(path/'quality_receipt.json',{'pass':True,'files':{},'content_id':content,'physical_receipt_sha256':sha(path/'physical_receipt.json')})
        return path

    def writing(self,task):
        self.ledger.transition(task['id'],'owner','executing',physical=True)
        self.ledger.transition(task['id'],'owner','writing')

    def test_claims_are_exclusive_across_connections(self):
        task=self.claim()
        second=Ledger(self.root/'ledger.db')
        try:
            with self.assertRaisesRegex(RuntimeError,'already claimed'):
                second.claim('egad_A1','second','gpu3/env0')
            other=second.claim('egad_A1','second','gpu3/env1')
            self.assertNotEqual((task['distance'],task['sample']),(other['distance'],other['sample']))
        finally:second.close()

    def test_fresh_pool_deduplicates_global_goal_and_keeps_actual_G(self):
        new=copy.deepcopy(self.goal);new['goal_id']='new_network_row';new['grasp_world'][0][3]=.002
        pool=self.ledger.add_goals('egad_A1',{'candidates':[new]})
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM goals').fetchone()[0],1)
        task=self.ledger.claim('egad_A1','owner','env',active_pool=pool['candidates'])
        self.assertEqual(task['goal_id'],'g0')
        self.assertEqual(task['goal']['source_goal_id'],'new_network_row')
        self.assertEqual(task['goal']['grasp_world'][0][3],.002)

    def test_unavailable_cached_goal_is_not_dispatched(self):
        self.assertIsNone(self.ledger.claim('egad_A1','owner','env',active_pool=[]))

    def test_object_reservation_is_exclusive_and_not_reclaimed_by_heartbeat(self):
        self.assertEqual(self.ledger.acquire_object('one',object_id='egad_A1')['object_id'],'egad_A1')
        self.assertIsNone(self.ledger.acquire_object('two',object_id='egad_A1'))
        self.assertEqual(self.ledger.acquire_object('one')['object_id'],'egad_A1')
        self.ledger.release_object('one')
        self.assertEqual(self.ledger.acquire_object('two')['object_id'],'egad_A1')

    def test_unresolved_attempt_blocks_object_release(self):
        self.ledger.acquire_object('owner',object_id='egad_A1')
        self.claim()
        with self.assertRaisesRegex(RuntimeError,'unresolved'):
            self.ledger.release_object('owner')

    def test_physical_failure_never_commits(self):
        task=self.claim();self.writing(task)
        with self.assertRaisesRegex(ValueError,'Physical failure'):
            self.ledger.commit(task['id'],'owner',self.receipts(task,physical_pass=False))
        self.assertFalse(any(r['state']=='committed' for r in self.ledger.rows()))

    def test_partial_write_never_commits(self):
        task=self.claim();self.writing(task)
        with self.assertRaisesRegex(ValueError,'incomplete write'):
            self.ledger.commit(task['id'],'owner',self.receipts(task,write_complete=False))

    def test_missing_durable_file_never_commits(self):
        task=self.claim();self.writing(task);p=self.receipts(task)
        receipt=read(p/'quality_receipt.json');receipt['files']={'raw/000000.png':'missing'};write(p/'quality_receipt.json',receipt)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            self.ledger.commit(task['id'],'owner',p)

    def test_crash_after_commit_cannot_count_twice(self):
        task=self.claim();self.writing(task);p=self.receipts(task)
        self.assertEqual(self.ledger.commit(task['id'],'owner',p),'committed')
        with self.assertRaisesRegex(RuntimeError,'cannot count twice'):
            self.ledger.commit(task['id'],'owner',p)
        self.assertEqual(sum(r['state']=='committed' for r in self.ledger.rows()),1)

    def test_new_seed_same_actual_entry_not_counted(self):
        a=self.claim();self.writing(a);self.ledger.commit(a['id'],'owner',self.receipts(a))
        b=self.claim();self.writing(b)
        self.assertEqual(self.ledger.commit(b['id'],'owner',self.receipts(b,content='second')),'duplicate_excluded')

    def test_copied_recording_new_entry_not_counted(self):
        a=self.claim();self.writing(a);self.ledger.commit(a['id'],'owner',self.receipts(a))
        b=self.claim();self.writing(b);pose=np.eye(4);pose[0,3]=.02
        self.assertEqual(self.ledger.commit(b['id'],'owner',self.receipts(b,actual=pose)),'duplicate_excluded')

    def test_interruption_requires_exit_and_idle_evidence(self):
        task=self.claim()
        with self.assertRaisesRegex(ValueError,'process-exit'):
            self.ledger.recover('owner',{'old_process_exited':True,'environments_idle':False,'evidence':['receipt']})
        self.ledger.recover('owner',{'old_process_exited':True,'environments_idle':True,'evidence':['exit_and_idle_receipt']})
        self.assertEqual(self.ledger.rows()[0]['state'],'recovered_interrupted')
        replacement=self.claim()
        self.assertNotEqual(replacement['id'],task['id'])
        self.assertEqual(len(self.ledger.rows()),2)


class EvidenceTests(unittest.TestCase):
    def test_shared_gpu_intervals_are_not_double_counted(self):
        from .report import union_seconds
        self.assertEqual(union_seconds([(0,10),(2,7),(9,12),(20,23)]),15)
    def test_installed_curobo_scene_refresh_constructor(self):
        from curobo._src.geom.types import SceneCfg
        self.assertTrue(callable(SceneCfg.create))
        self.assertEqual(len(SceneCfg.create({'mesh':{}})),0)
    def test_clone_canonicalization_keeps_physics_scene_and_mass_changes(self):
        import gzip
        from .frozen import canonical_physics
        old=json.loads(gzip.decompress((ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/setup.json.gz').read_bytes()))['fingerprint']['invariant']
        canonical=canonical_physics(old)
        changed=copy.deepcopy(old)
        changed['usd']['/physicsScene']['attributes']['physics:gravityMagnitude']=1.
        self.assertNotEqual(canonical_physics(changed),canonical)
        changed=copy.deepcopy(old)
        changed['runtime_robot']['get_link_masses'][0][0]+=1.
        self.assertNotEqual(canonical_physics(changed),canonical)

    def test_closed_goal_gate_checks_actual_open_goal_configuration(self):
        from .teacher import Teacher,Rejection
        class Update:
            names=['right_joint27','right_joint28']
            def apply(self,models,values):self.applied=values
        class Checker:
            def validate(self,q):self.last=q.copy();return np.array([False])
        t=Teacher.__new__(Teacher);u=Update();c=Checker()
        t.context={'closed':(None,c,u,[]),'open':(None,None,None,None)}
        t.ik=lambda *args: (_ for _ in ()).throw(AssertionError('Must use the actual already-selected approach joint configuration'))
        actual=np.array([.1,.2,.3,.4,.5,.6])
        with self.assertRaisesRegex(Rejection,'nominal_closed_grasp_collision'):
            t.nominal_closed(np.eye(4),np.zeros(6),np.eye(4),q_grasp_override=actual)
        np.testing.assert_array_equal(c.last.reshape(-1),actual)
        self.assertEqual(u.applied,{'right_joint27':.022,'right_joint28':.022})

    def test_clone_filters_preserve_original_arx_group_membership(self):
        from pxr import Usd,UsdGeom,UsdPhysics,Sdf
        from .isolation import isolate
        stage=Usd.Stage.CreateInMemory();roots=['/World/Envs/env_0','/World/Envs/env_1']
        robot=ROOT/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd'
        originals={}
        for root in roots:
            UsdGeom.Xform.Define(stage,root)
            stage.DefinePrim(root+'/Robot','Xform').GetReferences().AddReference(str(robot))
            colliders=[p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
            self.assertTrue(colliders)
            original=UsdPhysics.CollisionGroup.Define(stage,root+'/OriginalARXFixtureGroup')
            original.GetCollidersCollectionAPI().CreateExpansionRuleAttr().Set('explicitOnly')
            original.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([colliders[0]])
            original.CreateFilteredGroupsRel().SetTargets([original.GetPath()])
            originals[root]=(original,colliders)
        result=isolate(stage,roots)
        for root,(original,colliders) in originals.items():
            queries=[UsdPhysics.CollisionGroup(stage.GetPrimAtPath(p)).GetCollidersCollectionAPI().ComputeMembershipQuery()
                for p in result[root]['original_groups']+[result[root]['default_group']]]
            for collider in colliders:self.assertEqual(sum(q.IsPathIncluded(collider) for q in queries),1)
            self.assertIn(str(original.GetPath()),[str(p) for p in original.GetFilteredGroupsRel().GetTargets()])
            self.assertEqual(result[root]['original_filters'][str(original.GetPath())],[str(original.GetPath())])

    def test_established_roles_are_preserved(self):
        from .catalog import KNOWN_ROLES
        old=read(ROOT/'configs/lilo_vla/teacher_pilot20.json')['roles']
        for name,role in old.items():self.assertEqual(KNOWN_ROLES[name],role)

    def test_declared_historical_placements_are_preserved(self):
        from .catalog import build
        catalog={a['object_id']:a for a in build()['objects']}
        for case in read(ROOT/'configs/lilo_vla/teacher_pilot20.json')['cases']:
            self.assertEqual(catalog[case['object_name']]['placement'],case['placement'])
        old=read(ROOT/'scripts/curobo_arx/static_pick/reference/historical_reference/registration.json')['scenario']
        self.assertEqual(catalog['bbq_sauce_bottle']['placement']['position'],old['placement']['position'])

    def test_new_protocol_rejects_cross_environment_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'episode.json',{'version':VERSION,'write_complete':True,'samples':9,'env_id':'env0',
                'attempt_id':'attempt','object_id':'egad_A1','environment_root':'/World/Envs/env_0'})
            row={'env_id':'env0','attempt_id':'attempt','object_id':'egad_A1',
                 'camera_owner_root':'/World/Envs/env_1','camera_source_path':'/World/Envs/env_1/Robot/camera'}
            (p/'samples.jsonl').write_text('\n'.join(json.dumps(row) for _ in range(9))+'\n')
            write(p/'requests.json',[])
            with self.assertRaisesRegex(ValueError,'Camera source belongs'):
                quality(p)

    def test_execution_without_command_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'requests.json',[{'label':'close','id':'c'}]);write(p/'close_execution.json',{'settled':True})
            with self.assertRaisesRegex(ValueError,'missing evidence: close_command'):
                physical(p)

    def test_orphan_execution_without_request_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'requests.json',[]);write(p/'close_execution.json',{})
            with self.assertRaisesRegex(ValueError,'without registered command'):
                physical(p)

    def test_legacy_success_still_passes_original_evaluator(self):
        # Read actual historical ARX receipts, never create a fake physical success.
        source=ROOT/'runs/bottle_normal_run_20261026_144441'
        if not source.exists():self.skipTest('Historical ARX trace not installed')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);requests=[]
            for label in ('pregrasp','approach','close','lift','hold'):
                command=read(source/(label+'_command.json'));requests.append({'label':label,'id':label})
                for suffix in ('_command.json','_execution.json','_settling.json','_optimization.json'):
                    src=source/(label+suffix)
                    if src.exists():(p/src.name).symlink_to(src)
            for name in ('capture_snapshot.json','grasp_report.json'):(p/name).symlink_to(source/name)
            write(p/'requests.json',requests)
            result=physical(p)
            self.assertTrue(result['pass'],result)
            self.assertGreater(result['hold_min_rise_m'],.06)

    def test_real_arx_recording_has_valid_new_protocol_action_onsets(self):
        # Temporary, explicitly historical offline fixture. Never admitted to a
        # production ledger or counted as a newly collected demonstration.
        source=ROOT/'runs/lilo_P2_teacher_probe_20261028_01/bottle_grasp_frame_r1'
        if not source.exists():self.skipTest('Historical ARX wrist recording unavailable')
        import sys
        sys.path.insert(0,str(ROOT/'scripts/curobo_arx/lilo_vla'))
        from observations import encode_state
        rows=[json.loads(line) for line in (source/'teacher_raw/samples.jsonl').read_text().splitlines()]
        goal=read(source/'selected_grasp.json')['grasp_world']
        root='/World/Envs/env_0'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'raw').mkdir();requests={}
            for i,row in enumerate(rows):
                row.update(env_id='offline_env',attempt_id='historical_fixture',object_id='bbq_sauce_bottle',
                    camera_owner_root=root,camera_source_path=root+'/Robot/calibrated_right_wrist',
                    rgb_relative=f'raw/{i:06d}.png',depth_relative=f'raw/{i:06d}.npy')
                (p/row['rgb_relative']).symlink_to(row['rgb_path'])
                (p/row['depth_relative']).symlink_to(row['depth_path'])
                row['state']=encode_state(row['snapshot'],'grasp_goal_world_error_v1',goal).tolist()
                cmd=row.get('teacher_command')
                if cmd:requests[cmd['id']]=cmd
            (p/'samples.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            for name in ('selected_grasp.json','capture_snapshot.json','grasp_report.json'):(p/name).symlink_to(source/name)
            for path in source.glob('*_*.json'):
                if path.name.endswith(('_command.json','_execution.json','_settling.json','_optimization.json')):
                    (p/path.name).symlink_to(path)
            write(p/'requests.json',list(requests.values()))
            self.assertTrue(physical(p)['pass'])
            entry=read(source/'handoff_actual_observation/observation.json')
            entry_row=min(rows,key=lambda row:abs(row['frame_time']-entry['frame_time']))
            write(p/'episode.json',{'version':VERSION,'write_complete':True,'samples':len(rows),'env_id':'offline_env',
                'attempt_id':'historical_fixture','object_id':'bbq_sauce_bottle','environment_root':root,
                'goal':{'grasp_world':goal},'actual_entry':entry_row,'historical_fixture_only':True})
            receipt=quality(p)
            self.assertTrue(receipt['pass'])
            self.assertGreater(receipt['chunks'],0)
            self.assertTrue(all(n>=1 for n in receipt['onsets'].values()))

    def test_resource_floor_and_reduced_env_count(self):
        inventory=[{'index':0,'uuid':'compute','free_mb':7000,'arch':'8.9'},
                   {'index':1,'uuid':'sim','free_mb':9000,'arch':'8.9'}]
        compute,sim=allocate(inventory,4,compute='0',sim=['1'])
        self.assertEqual(sim[0]['env_count'],2)
        inventory[0]['free_mb']=5000
        with self.assertRaisesRegex(RuntimeError,'6500'):
            allocate(inventory,4,compute='0',sim=['1'])

    def test_single_available_gpu_can_share_compute_without_exceeding_budget(self):
        inventory=[{'index':3,'uuid':'shared','free_mb':15200,'arch':'8.9'}]
        compute,sim=allocate(inventory,4)
        self.assertEqual(compute['uuid'],sim[0]['uuid'])
        self.assertEqual(sim[0]['env_count'],2)
        inventory[0]['free_mb']=13999
        with self.assertRaisesRegex(RuntimeError,'startup budget'):
            allocate(inventory,4)


if __name__=='__main__':unittest.main()
