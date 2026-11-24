"""Offline ARX collection guards; no simulator, CUDA or physical trials."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .common import ROOT,read
from .placements import placement_bank,active_asset,tag_pool,validation_subset
from .visual_geometry import TargetGeometry,near_intersection,camera_path,inspect_row,registered_clipping
from .ledger import Ledger


class VisualTests(unittest.TestCase):
    def setUp(self):
        self.geometry=TargetGeometry(trimesh.creation.box(extents=[.04,.04,.04]))
        self.k=np.array([[100.,0,32],[0,100.,32],[0,0,1]])

    def check(self,z):
        target=np.eye(4);target[2,3]=z
        return self.geometry.inspect(target,np.eye(4),self.k,(64,64),[.05,10])

    def test_safe_target_is_not_rejected_for_its_rgb_color(self):
        self.assertTrue(self.check(.15)['pass'])

    def test_near_clipping_without_camera_inside_stops_admission(self):
        result=self.check(.06)
        self.assertFalse(result['pass'])
        self.assertFalse(result['camera_inside_target'])
        self.assertIn('target_intersects_camera_near_slab',result['reasons'])
        self.assertGreater(result['camera_surface_distance_m'],.01)

    def test_inside_and_missing_optics_are_not_silently_admitted(self):
        self.assertTrue(self.check(0)['camera_inside_target'])
        for clip in (None,[0,10],[.1,.01]):
            with self.assertRaises(ValueError):
                self.geometry.inspect(np.eye(4),np.eye(4),self.k,(64,64),clip)

    def test_triangle_crossing_view_is_found_even_when_all_vertices_are_outside(self):
        vertices=np.array([[-1,-1,.04],[1,-1,.04],[0,1,.04]])
        self.assertEqual(near_intersection(vertices,[[0,1,2]],np.eye(4),self.k,(64,64),.05),0)
        vertices[:,0]+=10
        self.assertIsNone(near_intersection(vertices,[[0,1,2]],np.eye(4),self.k,(64,64),.05))

    def test_path_checks_intermediate_optical_poses_with_fixed_mount(self):
        first=np.eye(4);first[2,3]=-.1
        second=first.copy();second[2,3]=.1
        mount=np.eye(4);mount[0,3]=.071
        poses=list(camera_path([first,second],mount))
        self.assertGreaterEqual(len(poses),101)
        np.testing.assert_allclose(poses[0],first@mount)
        np.testing.assert_allclose(poses[-1],second@mount)
        self.assertTrue(any(abs(p[2,3])<1e-9 for p in poses))

    def test_actual_ARX_milk_black_frame_is_rejected_and_sources_are_preserved(self):
        campaign=ROOT/'runs/sdg_household_20261101_01'
        episode=campaign/'groups/group_004139233b28/episodes/343e50388801488d90cf6f2f0b8dd3f0'
        row=json.loads((episode/'samples.jsonl').read_text().splitlines()[16])
        asset=next(a for a in read(campaign/'registration.json')['catalog']['objects'] if a['object_id']==row['object_id'])
        original=copy.deepcopy(row);clip,evidence=registered_clipping(campaign)
        self.assertEqual(clip,[.05,10])
        self.assertFalse(evidence['historical_live_clipping_verified'])
        result=inspect_row(TargetGeometry.from_asset(asset),row,(480,640),clip)
        self.assertFalse(result['pass'])
        self.assertIn('target_intersects_camera_near_slab',result['reasons'])
        self.assertLess(result['camera_surface_distance_m'],.003)
        self.assertEqual(original,row)

    def test_old_unfiltered_training_dataset_cannot_be_started(self):
        from training.show_harness_arx.train import validate_dataset
        with self.assertRaisesRegex(ValueError,'clipping admission'):
            validate_dataset(ROOT/'runs/show_harness_train_20261102_01/dataset_pilot_v3')


class PlacementTests(unittest.TestCase):
    def test_preferred_first_bin_changes_queue_only_and_preserves_full_coverage(self):
        asset,profile,support,vertices,bank=self.fixture();preferred=bank['placements'][-1]
        profile['first_placements']={'mustard_bottle':{k:preferred[k] for k in ('offset_xy_m','yaw_delta_degrees')}}
        ordered=placement_bank(asset,vertices,profile,support,42)
        self.assertEqual(ordered['placements'][0],preferred)
        self.assertEqual({p['placement_id'] for p in ordered['placements']},{p['placement_id'] for p in bank['placements']})
        profile['first_placements']['mustard_bottle']['offset_xy_m']=[99,99]
        with self.assertRaisesRegex(ValueError,'not a supported'):placement_bank(asset,vertices,profile,support,42)

    def test_bounded_validation_keeps_scene_assets_and_cannot_reenable_excluded_objects(self):
        objects=[{'object_id':'mustard_bottle','available':True,'placement':{'position':[4,2,.7]}},
                 {'object_id':'bbq_sauce_bottle','available':True},
                 {'object_id':'egad_A1','available':True,'collection_enabled':False}]
        catalog={'objects':objects};original=copy.deepcopy(catalog)
        subset=validation_subset(catalog,['mustard_bottle'])
        self.assertEqual(catalog,original);self.assertEqual(len(subset['objects']),3)
        self.assertEqual(subset['objects'][0],objects[0])
        self.assertFalse(subset['objects'][1]['collection_enabled'])
        for names in ([],['egad_A1'],['missing']):
            with self.assertRaises(ValueError):validation_subset(catalog,names)

    def fixture(self):
        asset={'object_id':'mustard_bottle','available':True,'renderer':'RaytracedLighting',
            'placement':{'position':[4.,2.,.75],'orientation':[1.,0,0,0]},'mass_kg':.3,'split':'train','issues':[]}
        profile=read(ROOT/'configs/lilo_vla/household_placement_diversity.json')
        support={'polygon_xy':[[3.7,1.7],[4.3,1.7],[4.3,2.3],[3.7,2.3]]}
        vertices=trimesh.creation.box(extents=[.04,.06,.1]).vertices
        bank=placement_bank(asset,vertices,profile,support,42)
        asset['collection_placements']=bank['placements']
        return asset,profile,support,vertices,bank

    def ledger(self,folder,asset):
        ledger=Ledger(Path(folder)/'ledger.db')
        ledger.initialize({'objects':[asset]}, {'max_total_dispatches':100,'max_physical_attempts':20,
            'goals_per_object':1,'candidate_reserve':1,'distances_mm':[10,40,60,80],
            'successes_per_bin':1,'physical_per_bin':4,'planning_rejections_per_bin':3})
        return ledger

    def test_full_position_yaw_cross_product_is_deterministic_and_keeps_original_properties(self):
        asset,profile,support,vertices,bank=self.fixture();original=copy.deepcopy(asset)
        again=placement_bank(asset,vertices,profile,support,42)
        self.assertEqual(bank,again);self.assertEqual(len(bank['placements']),72)
        self.assertEqual(len({p['placement_id'] for p in bank['placements']}),72)
        self.assertEqual({p['xy_bin'] for p in bank['placements']},set(range(9)))
        self.assertEqual({p['yaw_bin'] for p in bank['placements']},set(range(8)))
        for p in bank['placements']:
            active=active_asset(asset,p)
            self.assertEqual(active['placement']['position'][2],.75)
            self.assertEqual(active['mass_kg'],.3)
            self.assertEqual(active['reference_placement'],asset['placement'])
        self.assertEqual(asset,original)

    def test_roll_and_pitch_resting_orientation_are_not_randomized(self):
        asset,profile,support,vertices,_=self.fixture()
        original=Rotation.from_euler('x',.2);q=original.as_quat()
        asset['placement']['orientation']=np.r_[q[3],q[:3]].tolist()
        for p in placement_bank(asset,vertices,profile,support,42)['placements']:
            q=np.asarray(p['placement']['orientation']);actual=Rotation.from_quat(np.r_[q[1:],q[0]])
            expected=Rotation.from_euler('z',p['yaw_delta_degrees'],degrees=True)*original
            np.testing.assert_allclose(actual.as_matrix(),expected.as_matrix(),atol=1e-12)

    def test_unsupported_footprint_bins_are_explicit_rejections(self):
        asset,profile,_,vertices,_=self.fixture()
        support={'polygon_xy':[[3.95,1.95],[4.05,1.95],[4.05,2.05],[3.95,2.05]]}
        bank=placement_bank(asset,vertices,profile,support,42)
        self.assertGreater(len(bank['rejected']),0)
        self.assertLess(len(bank['placements']),72)
        self.assertEqual(len(bank['placements'])+len(bank['rejected']),72)

    def test_schedule_rotates_placements_and_held_lease_keeps_its_pose(self):
        asset,*_=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(tmp,asset)
            try:
                poses=[]
                for _ in range(3):
                    active=ledger.acquire_object('group')
                    self.assertEqual(active,ledger.acquire_object('group'))
                    poses.append(active['collection_placement']['placement_id'])
                    ledger.release_object('group')
                self.assertEqual(len(set(poses)),3)
            finally:ledger.close()

    def test_empty_candidate_pool_exhausts_only_that_placement(self):
        asset,*_=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(tmp,asset)
            try:
                first=ledger.acquire_object('group');ledger.release_object('group',exhausted_reason='candidate_exhausted')
                second=ledger.acquire_object('group')
                self.assertNotEqual(first['collection_placement']['placement_id'],second['collection_placement']['placement_id'])
                self.assertEqual(ledger.db.execute('SELECT COUNT(*) FROM exhausted_objects').fetchone()[0],0)
            finally:ledger.close()

    def test_identical_world_goals_at_distinct_placements_have_separate_reserves_and_claims(self):
        asset,*_=self.fixture()
        raw={'candidates':[{'goal_id':'same_G','grasp_world':np.eye(4).tolist(),'group':'side'}]}
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(tmp,asset)
            try:
                identifiers=[]
                for _ in range(2):
                    active=ledger.acquire_object('group');pool=ledger.add_goals(asset['object_id'],tag_pool(raw,active))
                    self.assertEqual(len(pool['candidates']),1)
                    task=ledger.claim(asset['object_id'],'group','env',active_pool=pool['candidates'])
                    identifiers.append(task['goal_id'])
                    ledger.transition(task['id'],'group','rejected',outcome='offline_test_no_motion')
                    ledger.release_object('group')
                self.assertNotEqual(*identifiers)
                self.assertEqual(ledger.db.execute('SELECT COUNT(*) FROM goals').fetchone()[0],2)
                with self.assertRaisesRegex(ValueError,'fresh qualified active pool'):
                    ledger.claim(asset['object_id'],'group','env')
            finally:ledger.close()

    def test_stale_placement_pool_cannot_be_dispatched_after_pose_switch(self):
        asset,*_=self.fixture()
        raw={'candidates':[{'goal_id':'G','grasp_world':np.eye(4).tolist(),'group':'side'}]}
        with tempfile.TemporaryDirectory() as tmp:
            ledger=self.ledger(tmp,asset)
            try:
                old=ledger.acquire_object('group');pool=ledger.add_goals(asset['object_id'],tag_pool(raw,old))
                ledger.release_object('group');ledger.acquire_object('group')
                with self.assertRaisesRegex(ValueError,'differs from the current'):
                    ledger.claim(asset['object_id'],'group','env',active_pool=pool['candidates'])
                self.assertEqual(ledger.rows(),[])
            finally:ledger.close()

    def test_report_distinguishes_registered_attempted_and_successful_pose_coverage(self):
        from .common import write
        from .report import report
        asset,*_=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);ledger=self.ledger(tmp,asset)
            try:
                quotas=ledger.quotas;quotas['target_episodes']=72*4
                write(folder/'registration.json',{'catalog':{'objects':[asset]},'quotas':quotas})
                write(folder/'state.json',{'status':'prepared'})
                # The report opens the registered standard ledger filename.
                ledger.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');ledger.close()
                (folder/'ledger.db').rename(folder/'ledger.sqlite3')
                result=report(folder)
                self.assertEqual(result['registered_object_placements'],72)
                self.assertEqual(result['physically_attempted_object_placements'],0)
                self.assertEqual(result['successful_object_placements'],0)
                self.assertFalse(result['quota_complete'])
            finally:
                try:ledger.close()
                except Exception:pass


if __name__=='__main__':unittest.main()
