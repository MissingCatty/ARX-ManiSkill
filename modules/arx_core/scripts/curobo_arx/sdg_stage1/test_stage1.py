"""Offline ARX frame/queue/guard checks; not new physical demonstrations."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,digest,read
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.runtime import Group
from scripts.curobo_arx.data_factory.geometry import equivalent_grasp,grasp_comparison_frame,grasp_equal
from .sampler import candidate_pool,composed_mesh,frozen_pad_center,SDGSampler
from .export import entry_record
from .asset_contract import source_contract,check_and_separate
from .scene_transport import geometry_pose,original_vertices,within_original_bound


REFERENCE=ROOT/'runs/data_factory_20261030_15'


def successful_fixture():
    db=sqlite3.connect('file:'+str(REFERENCE/'ledger.sqlite3')+'?mode=ro',uri=True)
    try:return Path(db.execute("SELECT episode FROM attempts WHERE state='committed' LIMIT 1").fetchone()[0])
    finally:db.close()


class Stage1Tests(unittest.TestCase):
    def precision_fixture(self):
        p=ROOT/'runs/sdg_stage1_20261031_06'
        snapshot=read(p/'groups/group_6397bc8545f1/episodes/c61aa1512c9a4e9190ad901d3c4946d8/capture_snapshot.json')
        asset=next(a for a in read(p/'registration.json')['catalog']['objects'] if a['object_id']=='egad_B4')
        mesh,_=composed_mesh(asset['usd_path'])
        return mesh.vertices,np.asarray(snapshot['objects']['test_object']['world']),np.linalg.inv(snapshot['link_world']['body4'])

    def test_quaternion_roundtrip_precision_stress_preserves_original_ARX_geometry(self):
        # Numeric frame-representation stress only: frozen USD/physics are not edited.
        from scipy.spatial.transform import Rotation
        points,obj,inverse=self.precision_fixture();inverse=inverse.astype(np.float32).astype(float)
        direct=original_vertices(points,obj,world_to_base=inverse);details={}
        local,pose,vertices,error=geometry_pose(points,obj,world_to_base=inverse,origin=[1000,0,0],evidence=details)
        self.assertTrue(within_original_bound(direct))
        self.assertTrue(details['identity_pose_fallback'])
        self.assertGreater(details['initial_quaternion_error_m'],1e-6)
        self.assertLess(error,1e-8)
        np.testing.assert_array_equal(vertices,direct)
        decoded=local@Rotation.from_quat([*pose[4:],pose[3]]).as_matrix().T+pose[:3]
        np.testing.assert_array_equal(decoded,direct)

    def test_original_ARX_export_bound_and_invalid_transform_checks_remain_strict(self):
        points,obj,inverse=self.precision_fixture()
        self.assertTrue(within_original_bound(original_vertices(points,obj,world_to_base=inverse)))
        far=obj.copy();far[0,3]+=1000.
        self.assertFalse(within_original_bound(original_vertices(points,far,world_to_base=inverse)))
        bad=inverse.copy();bad[0,:3]*=2
        with self.assertRaisesRegex(RuntimeError,'not rigid'):
            geometry_pose(points,obj,world_to_base=bad,origin=[1000,0,0])
        corrupt=points.copy();corrupt[0,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            geometry_pose(corrupt,obj,world_to_base=inverse)

    def test_export_deferred_until_every_ARX_group_has_exited(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess
        from .run import final_export
        from scripts.curobo_arx.data_factory.common import write,sha
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write(folder/'state.json',{'status':'fault','unresolved_groups':['group_late']})
            before=sha(folder/'state.json')
            with patch('scripts.curobo_arx.sdg_stage1.run.export') as exporter:
                self.assertIsNone(final_export(folder,{'full_ARX_successful_demonstrations':37}))
                exporter.assert_not_called()
            self.assertEqual(sha(folder/'state.json'),before)
            self.assertTrue((folder/'EXPORT_DEFERRED.json').is_file())
            write(folder/'state.json',{'status':'fault','unresolved_groups':[]})
            write(folder/'group_registrations/group_late.process.json',{'container':'arx-data-factory-group_late'})
            with patch('scripts.curobo_arx.sdg_stage1.run.subprocess.run',return_value=CompletedProcess([],0)),patch('scripts.curobo_arx.sdg_stage1.run.export') as exporter:
                self.assertIsNone(final_export(folder,{'full_ARX_successful_demonstrations':37}))
                exporter.assert_not_called()

    def test_cleanup_waits_for_late_ARX_writer_without_dispatching_new_work(self):
        from unittest.mock import patch
        from scripts.curobo_arx.data_factory.launch import wait_for_groups
        clock=[0.];process=types.SimpleNamespace(poll=lambda:None if clock[0]<187 else 0)
        active={'gpu':{'id':'group_late','process':process}}
        def advance(duration):clock[0]+=duration
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.curobo_arx.data_factory.launch.time.monotonic',side_effect=lambda:clock[0]),patch('scripts.curobo_arx.data_factory.launch.time.sleep',side_effect=advance),patch('scripts.curobo_arx.data_factory.launch.write') as checkpoint:
            self.assertEqual(wait_for_groups(active,tmp,960),[])
            self.assertGreaterEqual(clock[0],187)
            self.assertFalse(checkpoint.call_args.args[1]['new_dispatch_allowed'])
            clock[0]=0.
            self.assertEqual(wait_for_groups(active,tmp,180),['group_late'])

    def test_reach_keeps_actual_ARX_target_geometry_while_grasp_gate_allows_contact(self):
        from scripts.curobo_arx.data_factory.teacher import Teacher
        fixture=next((ROOT/'runs/sdg_stage1_20261031_04/groups').glob('*/episodes/*/reach_scene.json')).parent
        scene=read(fixture/'reach_scene.json')
        snapshot=read(fixture/'capture_snapshot.json');target=snapshot['objects']['test_object']['path']
        source=copy.deepcopy(scene)
        qualifies=Teacher.scene_meshes(scene,target)
        reaches=Teacher.scene_meshes(scene,target,include_target=True)
        is_target=lambda m:m['path']==target or m['path'].startswith(target+'/')
        self.assertGreater(sum(map(is_target,reaches)),0)
        self.assertFalse(any(map(is_target,qualifies)))
        self.assertEqual([m for m in reaches if not is_target(m)],qualifies)
        self.assertEqual([m for m in reaches if is_target(m)],[m for m in scene['meshes'] if is_target(m)])
        self.assertEqual(scene,source)
        self.assertFalse(any(m['path'].startswith('/World/Robot/') for m in reaches))

    def test_source_fix_preserves_budget_before_preparing_new_ARX_registration(self):
        from unittest.mock import patch
        import time
        from .prepare import prepare
        from scripts.curobo_arx.data_factory.common import write
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'old';parent.mkdir()
            write(parent/'registration.json',{'stage':'sdg_stage1','deadline_unix':time.time()+3600,
                'wall_hours':24,'quotas':{'max_physical_attempts':2048}})
            write(parent/'coordinator_process.json',{'pid':999999999})
            write(parent/'launch.json',{'stop_unix':time.time()+3600})
            db=sqlite3.connect(parent/'ledger.sqlite3')
            db.execute('CREATE TABLE attempts(object_id TEXT,state TEXT,physical INTEGER)')
            db.execute('CREATE TABLE events(event TEXT,value TEXT)')
            db.executemany('INSERT INTO attempts VALUES (?,?,?)',[('rubiks_cube','committed',1)]*8+[('rubiks_cube','failed',1)]*2)
            db.commit();db.close()
            proof={'old_process_exited':True,'environments_idle':True,'containers_absent':True,'evidence':{'offline_control_flow_fixture':True}}
            with patch('scripts.curobo_arx.data_factory.resume.exit_proof',return_value=proof),patch('scripts.curobo_arx.sdg_stage1.prepare.base_prepare',side_effect=RuntimeError('captured before prepare')) as base:
                with self.assertRaisesRegex(RuntimeError,'captured before prepare'):
                    prepare(Path(tmp)/'new',source_fix_parent=parent,object_policy='all',placement_profile='historical')
                self.assertEqual(base.call_args.kwargs['max_attempts'],2038)
                self.assertEqual(base.call_args.kwargs['wall_hours'],24)
            db=sqlite3.connect(parent/'ledger.sqlite3');db.execute("UPDATE attempts SET state='executing' WHERE rowid=1");db.commit();db.close()
            with patch('scripts.curobo_arx.data_factory.resume.exit_proof',return_value=proof),patch('scripts.curobo_arx.sdg_stage1.prepare.base_prepare') as base:
                with self.assertRaisesRegex(ValueError,'unresolved execution'):prepare(Path(tmp)/'new',source_fix_parent=parent,object_policy='all',placement_profile='historical')
                base.assert_not_called()

    def test_source_fix_continuation_prioritizes_unvisited_ARX_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'ledger.sqlite3')
            q={'max_physical_attempts':24,'max_total_dispatches':32,'goals_per_object':4,
                'distances_mm':[10,40,60,80],'successes_per_bin':8,'first_object':'rubiks_cube',
                'prior_dispatch_counts':{'rubiks_cube':8}}
            ledger.initialize({'objects':[{'object_id':'rubiks_cube'},{'object_id':'egad_A1'}]},q)
            self.assertEqual(ledger.available_object()['object_id'],'egad_A1')
            self.assertEqual(len(ledger.rows()),0)
            ledger.close()

    def test_previously_screened_empty_pool_does_not_preempt_new_ARX_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'ledger.sqlite3')
            q={'max_physical_attempts':24,'max_total_dispatches':32,'goals_per_object':4,
                'distances_mm':[10,40,60,80],'successes_per_bin':8,
                'prior_candidate_screened_objects':['lemon_01']}
            ledger.initialize({'objects':[{'object_id':'lemon_01'},{'object_id':'egad_B6'}]},q)
            self.assertEqual(ledger.available_object()['object_id'],'egad_B6')
            self.assertEqual(ledger.rows(),[])
            ledger.close()

    def test_parallel_jaw_flip_preserves_actual_ARX_contact_with_off_axis_pad(self):
        source=copy.deepcopy(read(successful_fixture()/'episode.json')['goal'])
        source['pad_center_eef_m']=frozen_pad_center().tolist()
        center=grasp_comparison_frame(source)[:3,3].copy()
        flipped={**source,'grasp_world':equivalent_grasp(source,True).tolist()}
        np.testing.assert_allclose(grasp_comparison_frame(flipped)[:3,3],center,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(np.asarray(flipped['grasp_world'])[:3,3]-np.asarray(source['grasp_world'])[:3,3]),.012)
        self.assertTrue(grasp_equal(source,flipped))
        legacy={k:v for k,v in source.items() if k!='pad_center_eef_m'}
        np.testing.assert_array_equal(equivalent_grasp(legacy,True)[:3,3],np.asarray(legacy['grasp_world'])[:3,3])
        wrong={**source,'grasp_world':equivalent_grasp(legacy,True).tolist()}
        self.assertFalse(grasp_equal(source,wrong))

    def test_sampler_deduplicates_contact_centered_ARX_parallel_jaw_poses(self):
        g=np.asarray(read(successful_fixture()/'episode.json')['goal']['grasp_world'])
        contact=g.copy();contact[:3,3]+=g[:3,:3]@frozen_pad_center()
        flipped=contact.copy();flipped[:3,:3]=contact[:3,:3]@np.diag([-1.,-1.,1.])
        separate=contact.copy();separate[:3,3]+=np.array([.02,0,0])
        pool=candidate_pool([contact,flipped,separate],np.eye(4),[.2,.2,.2],8,frozen_pad_center(),{})
        self.assertEqual(len(pool['candidates']),2)
        self.assertEqual(len(pool['rejected']),1)

    def test_ledger_deduplicates_contact_symmetry_across_ARX_observations(self):
        source=copy.deepcopy(read(successful_fixture()/'episode.json')['goal'])
        source.update(goal_id='first',pad_center_eef_m=frozen_pad_center().tolist())
        flipped={**source,'goal_id':'equivalent','grasp_world':equivalent_grasp(source,True).tolist()}
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'ledger.sqlite3')
            ledger.initialize({'objects':[{'object_id':'arx_fixture'}]},{'candidate_reserve':8})
            ledger.add_goals('arx_fixture',{'candidates':[source]})
            pool=ledger.add_goals('arx_fixture',{'candidates':[flipped]})
            self.assertEqual(pool['candidates'][0]['goal_id'],'first')
            self.assertEqual(pool['candidates'][0]['goal_equivalence'],'contact_center_5mm_10deg_parallel_jaw_v2')
            self.assertEqual(ledger.db.execute('select count(*) from goals').fetchone()[0],1)
            ledger.close()

    def test_original_and_additional_assets_reuse_translation_without_altering_other_ops(self):
        from pxr import Usd,UsdGeom
        from scripts.curobo_arx.data_factory.vector_env import set_initial_asset_translation
        assets=read(REFERENCE/'catalog.json')['objects']+read(ROOT/'runs/sdg_stage1_development_20261031_01/additional_assets_v2/catalog.json')
        for asset in assets:
            stage=Usd.Stage.CreateInMemory();p=stage.DefinePrim('/Object','Xform')
            p.GetReferences().AddReference(asset.get('usd_path',asset.get('path')))
            xf=UsdGeom.Xformable(p)
            old={o.GetOpName():str(o.Get()) for o in xf.GetOrderedXformOps() if o.GetOpName()!='xformOp:translate'}
            set_initial_asset_translation(p,[10.,10.,.4]);set_initial_asset_translation(p,[11.,10.,.4])
            self.assertEqual(sum(o.GetOpName()=='xformOp:translate' for o in xf.GetOrderedXformOps()),1)
            self.assertEqual({o.GetOpName():str(o.Get()) for o in xf.GetOrderedXformOps() if o.GetOpName()!='xformOp:translate'},old)
    def test_region_grouping_is_invariant_to_object_pivot_translation(self):
        meta=read(successful_fixture()/'episode.json');g=np.asarray(meta['goal']['grasp_world'])
        contact=g.copy();contact[:3,3]+=g[:3,:3]@frozen_pad_center()
        first=candidate_pool([contact],np.eye(4),[.2,.2,.2],8,frozen_pad_center(),{},[0,0,0])
        shift=np.array([3.,0.,2.]);local=contact.copy();local[:3,3]+=shift
        world=np.eye(4);world[:3,3]=-shift
        second=candidate_pool([local],world,[.2,.2,.2],8,frozen_pad_center(),{},shift)
        self.assertEqual(first['candidates'][0]['group'],second['candidates'][0]['group'])
        np.testing.assert_allclose(first['candidates'][0]['grasp_world'],second['candidates'][0]['grasp_world'],atol=1e-12)

    def test_new_object_contract_rejects_kinematic_or_gravity_disabled_source(self):
        from pxr import Usd,UsdGeom,UsdPhysics,Sdf
        asset=read(ROOT/'runs/sdg_stage1_development_20261031_01/additional_assets/catalog.json')[0]
        with tempfile.TemporaryDirectory() as tmp:
            for i,field in enumerate(['kinematic','gravity']):
                path=Path(tmp)/(str(i)+'.usda');stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1.)
                root=stage.DefinePrim('/Object','Xform');root.GetReferences().AddReference(asset['path']);stage.SetDefaultPrim(root)
                if field=='kinematic':UsdPhysics.RigidBodyAPI(root).CreateKinematicEnabledAttr(True)
                else:root.CreateAttribute('physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(True)
                stage.GetRootLayer().Save()
                with self.assertRaisesRegex(ValueError,'dynamic.*gravity'):source_contract(path)

    def test_scene_transport_preserves_actual_ARX_asset_vertices_static_and_live(self):
        meta=read(successful_fixture()/'episode.json')
        asset=next(a for a in read(REFERENCE/'catalog.json')['objects'] if a['object_id']==meta['object_id'])
        mesh,_=composed_mesh(asset['usd_path'])
        base=np.asarray(meta['actual_entry']['snapshot']['link_world']['body4'])
        actual=np.asarray(meta['actual_entry']['snapshot']['objects']['test_object']['world'])
        for authored,live in [(None,None),(actual,actual)]:
            _,pose,vertices,error=geometry_pose(mesh.vertices,actual,authored,live,np.linalg.inv(base))
            direct=(np.c_[mesh.vertices,np.ones(len(mesh.vertices))]@(np.linalg.inv(base)@actual).T)[:,:3]
            np.testing.assert_allclose(vertices,direct,atol=1e-10)
            self.assertLess(error,1e-10)
        translated=actual.copy();translated[0,3]+=100
        shifted_base=base.copy();shifted_base[0,3]+=100
        local,_,vertices,error=geometry_pose(mesh.vertices,translated,None,None,np.linalg.inv(shifted_base),[100,0,0])
        self.assertLess(abs(local).max(),10.)
        np.testing.assert_allclose(vertices,direct,atol=1e-10)
    def test_contact_to_tcp_uses_unchanged_ARX_pad_calibration(self):
        meta=read(successful_fixture()/'episode.json');g=np.asarray(meta['goal']['grasp_world'])
        contact=g.copy();contact[:3,3]+=contact[:3,:3]@frozen_pad_center()
        pool=candidate_pool([contact],np.eye(4),[.2,.2,.2],8,frozen_pad_center(),{'name':'offline_ARX_frame_fixture'})
        np.testing.assert_allclose(pool['candidates'][0]['grasp_world'],g,atol=1e-12)
        np.testing.assert_array_equal(frozen_pad_center(),[-.006,0,.0375])

    def test_official_source_generates_candidates_for_actual_ARX_asset(self):
        reg=read(ROOT/'runs/sdg_stage1_development_20261031_01/bank67_reference/registration.json')
        config=copy.deepcopy(reg);config['sdg'].pop('candidate_bank',None)
        config['sdg']['sampler_parameters'].update(num_candidates=64,num_orientations=8)
        backend=SDGSampler(config);asset=next(a for a in reg['catalog']['objects'] if a['object_id']=='mustard_bottle')
        poses,evidence=backend.local_candidates(asset)
        self.assertGreater(len(poses),0);self.assertFalse(evidence['physics_evaluation'])
        for pose in poses:
            self.assertTrue(np.isfinite(pose).all());self.assertAlmostEqual(np.linalg.det(pose[:3,:3]),1.,places=6)

    def test_per_G_seed_preserves_nominal_and_changes_relative_perturbations(self):
        meta=read(successful_fixture()/'episode.json')
        group=object.__new__(Group);group.registration={'seed':20261029,'entry_sampler':'per_object_G_seed_v1'}
        task={'object_id':meta['object_id'],'goal_id':'gA','goal':meta['goal'],'distance':40,'sample':0}
        nominal=group.sample_handoff(task)
        self.assertEqual(nominal['local_rotation_vector_rad'],[0.,0.,0.])
        first=group.sample_handoff({**task,'sample':1})
        other=group.sample_handoff({**task,'sample':1,'goal_id':'gB'})
        self.assertNotEqual(first['seed'],other['seed'])
        self.assertFalse(np.allclose(first['local_rotation_vector_rad'],other['local_rotation_vector_rad']))

    def test_paired_queue_reserves_nominal_and_perturbed_four_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            q={'candidate_reserve':128,'goals_per_object':32,'distances_mm':[10,40,60,80],
                'successes_per_bin':8,'physical_per_bin':32,'planning_rejections_per_bin':64,
                'max_total_dispatches':100,'max_physical_attempts':32,'sampling_policy':'paired_nominal_perturbed_rounds_v1'}
            ledger=Ledger(Path(tmp)/'ledger.sqlite3')
            ledger.initialize({'objects':[{'object_id':'egad_A3'}]},q)
            g=np.asarray(read(successful_fixture()/'episode.json')['goal']['grasp_world'])
            ledger.add_goals('egad_A3',{'candidates':[{'goal_id':'g','grasp_world':g.tolist(),'group':'offline_ARX_queue_fixture'}]})
            tasks=[ledger.claim('egad_A3','owner','env'+str(i)) for i in range(8)]
            self.assertEqual([(t['distance'],t['sample']) for t in tasks],[(d,i) for d in [10,40,60,80] for i in [0,1]])
            self.assertEqual(sum(r['physical'] for r in ledger.rows()),0)
            ledger.close()

    def test_entry_index_reconstructs_requested_and_measured_ARX_entry(self):
        row=entry_record(successful_fixture());g=np.asarray(row['G_world'])
        np.testing.assert_allclose(g@row['D_requested'],row['H_requested_world'],atol=1e-9)
        np.testing.assert_allclose(g@row['D_actual'],row['H_actual_world'],atol=1e-9)
        self.assertIsNone(row['VLA_success_label']);self.assertEqual(row['camera_role'],'right_wrist')

    def test_entry_export_rejects_shifted_SDK_contact_metadata(self):
        from unittest.mock import patch
        from . import export as module
        fixture=successful_fixture();original=module.read
        def damaged(path):
            value=original(path)
            if Path(path).name=='episode.json':
                value=copy.deepcopy(value);g=value['goal']
                g['pad_center_eef_m']=frozen_pad_center().tolist()
                center=grasp_comparison_frame(g)[:3,3]
                g['target_contact_world']=(center+[.012,0,0]).tolist()
            return value
        with patch.object(module,'read',side_effect=damaged):
            with self.assertRaisesRegex(ValueError,'contact-center metadata'):entry_record(fixture)

    def actual_extra_contract_fixture(self):
        extra=read(ROOT/'runs/sdg_stage1_development_20261031_01/additional_assets/catalog.json')[0]
        contract=source_contract(extra['path']);name=extra['name'];prefix='/World/Task/assets/asset_'+name
        usd={}
        for rel,value in contract['authored'].items():
            value=copy.deepcopy(value)
            value['relationships']={k:[prefix+t[len(contract['source_root']):] if t==contract['source_root'] or t.startswith(contract['source_root']+'/') else t for t in targets] for k,targets in value['relationships'].items()}
            usd[prefix+rel]=value
        usd['/World/Robot/unchanged']={'attributes':{'drive:stiffness':123.}}
        current={'usd':usd,'runtime_objects':{prefix:{'get_masses':[[contract['expected_mass_kg']]],'get_inertias':np.eye(3).reshape(1,1,9).tolist()}}}
        return name,contract,current

    def test_extra_guard_retains_historical_subset_and_checks_source_mass(self):
        name,contract,current=self.actual_extra_contract_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            old,review=check_and_separate(copy.deepcopy(current),{name:contract},Path(tmp)/'baseline.json')
            self.assertTrue(review['pass']);self.assertIn('/World/Robot/unchanged',old['usd'])
            self.assertEqual(len(old['usd']),1)
            prefix='/World/Task/assets/asset_'+name
            bad=copy.deepcopy(current);bad['runtime_objects'][prefix]['get_masses']=[[contract['expected_mass_kg']*2]]
            _,review=check_and_separate(bad,{name:contract},Path(tmp)/'baseline.json')
            self.assertFalse(review['pass'])

    def test_extra_guard_stops_on_later_inertia_change(self):
        name,contract,current=self.actual_extra_contract_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'baseline.json';_,review=check_and_separate(copy.deepcopy(current),{name:contract},path)
            self.assertTrue(review['pass'])
            bad=copy.deepcopy(current);bad['runtime_objects']['/World/Task/assets/asset_'+name]['get_inertias']=(np.eye(3)*2).reshape(1,1,9).tolist()
            _,review=check_and_separate(bad,{name:contract},path)
            self.assertFalse(review['pass'])

    def test_extra_guard_does_not_hide_unregistered_extra_prim(self):
        name,contract,current=self.actual_extra_contract_fixture()
        current['usd']['/World/Task/assets/asset_unregistered']={'attributes':{'physics:mass':10}}
        with tempfile.TemporaryDirectory() as tmp:
            old,_=check_and_separate(current,{name:contract},Path(tmp)/'baseline.json')
            self.assertIn('/World/Task/assets/asset_unregistered',old['usd'])


if __name__=='__main__':unittest.main()
