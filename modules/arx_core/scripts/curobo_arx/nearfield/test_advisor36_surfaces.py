"""CPU fault checks against the actual frozen ARX/milk fixture; no World."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,write,digest
from .advisor36_surfaces import export_body,SurfaceChecker
from .advisor36_static_launch import prior_observations
from .advisor36_export import load_lines


FIXTURE=ROOT/'runs/nearfield_advisor36_20261112_01/static_objects/household_hope_milk_carton_05'


class ActualARXSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pxr import Usd,UsdPhysics
        sys.path[:0]=[str(ROOT/'arx-sim'),str(ROOT/'shared')]
        from arx_sim.inprocess_app import _apply_gripper_contact_overrides
        reg=read(FIXTURE/'registration.json')
        entry=read(FIXTURE/'canonical_registry.json')['objects']['household_hope_milk_carton']
        stage=Usd.Stage.Open(reg['learned_policy']['robot_usd']);stage.SetEditTarget(stage.GetSessionLayer())
        _apply_gripper_contact_overrides(stage,reg['render_configs']['RaytracedLighting'])
        robot={p.GetName():export_body(stage,p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        asset=Usd.Stage.Open(entry['source']['asset']['usd_path'])
        data={'schema':'arx_static_target_surfaces_v1','robot':robot,
            'target':export_body(asset,asset.GetDefaultPrim().GetPath()),
            'scope':'Actual frozen ARX/milk CPU fixture; no physical execution'}
        cls.geometry={**data,'geometry_sha256':digest(data)}
        cls.checker=SurfaceChecker(cls.geometry)
        cls.reference=read(entry['center_closure_request'])['measured_hold_snapshot']

    def test_actual_successful_open_center_has_clear_target(self):
        result=self.checker.check(self.reference)
        self.assertTrue(result['pass'],result['checks'])
        self.assertGreater(result['minimum_distance_m'],.004)

    def test_near_observation_not_false_sphere_collision(self):
        request=read(FIXTURE/'groups/pilot/observations/household_hope_milk_carton_0384_p/IK_request.json')
        state=copy.deepcopy(self.reference)
        desired=np.asarray(request['snapshot']['objects']['test_object']['world'])@request['desired_eef_object']
        change=desired@np.linalg.inv(self.reference['link_world']['right_eef_link'])
        for name in ('right_link26','right_link27','right_link28'):
            state['link_world'][name]=(change@self.reference['link_world'][name]).tolist()
        state['objects']['test_object']=request['snapshot']['objects']['test_object']
        self.assertTrue(self.checker.check(state)['pass'])

    def test_shifted_arx_finger_intersection_is_rejected(self):
        # A corrupted static pose fixture, not a new robot or physics trial.
        state=copy.deepcopy(self.reference)
        axis=np.asarray(state['link_world']['right_eef_link'])[:3,1]
        for name in ('right_link26','right_link27','right_link28'):
            t=np.asarray(state['link_world'][name]);t[:3,3]+=axis*.03
            state['link_world'][name]=t.tolist()
        self.assertFalse(self.checker.check(state)['pass'])

    def test_changed_surface_payload_is_record_fault(self):
        changed=copy.deepcopy(self.geometry)
        changed['target'][0]['points_body'][0][0]+=.001
        with self.assertRaises(ValueError):SurfaceChecker(changed)


class ContinuationFaultTests(unittest.TestCase):
    def fixture(self,root):
        rows=load_lines(FIXTURE/'candidates.jsonl');candidate=next(r for r in rows if r['sample_id'].endswith('0384_p'))
        group=root/'groups/pilot';folder=group/'observations'/candidate['sample_id']
        row={'sample_id':candidate['sample_id'],'candidate':candidate,'record_integrity_pass':True,
            'physical_trajectory_executed':False,'pass_':False}
        for name,value in [('STARTED.json',{'sample_id':candidate['sample_id']}),
                           ('IK_request.json',{'sample_id':candidate['sample_id']}),
                           ('static_IK_receipt.json',{'pass':False}),('capture_receipt.json',row)]:write(folder/name,value)
        write(group/'completion.json',{'pass':True,'processed':1,'registered':1})
        write(group/'shutdown_receipt.json',{'all_trajectories_idle':True})
        write(group/'results.json',{'observations':[row]})
        write(root/'registration.json',{'static_observations':{}})
        return rows,folder

    def test_terminal_rejection_is_carried_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,folder=self.fixture(Path(tmp));r=prior_observations(Path(tmp),rows)
            self.assertEqual(r['observed'],[folder.name]);self.assertFalse(r['started_without_terminal'])

    def test_execution_receipt_without_request_stops_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,folder=self.fixture(Path(tmp));(folder/'IK_request.json').unlink()
            with self.assertRaises(FileNotFoundError):prior_observations(Path(tmp),rows)

    def test_unknown_started_member_excludes_the_whole_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,folder=self.fixture(Path(tmp));other=rows[0]
            write(Path(tmp)/'groups/fault/observations'/other['sample_id']/'STARTED.json',{'sample_id':other['sample_id']})
            r=prior_observations(Path(tmp),rows)
            self.assertEqual(set(r['excluded']),{s['sample_id'] for s in rows if s['pair_id']==other['pair_id']})

    def test_mismatched_terminal_receipt_stops_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,folder=self.fixture(Path(tmp));r=read(folder/'capture_receipt.json');r['pass_']=True
            write(folder/'capture_receipt.json',r)
            with self.assertRaises(RuntimeError):prior_observations(Path(tmp),rows)


if __name__=='__main__':unittest.main()
