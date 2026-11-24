"""Offline checks of the actual ARX camera definition and retained mesh gates."""
import ast
from pathlib import Path
from typing import Any
import unittest
import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom
from .camera_render import correct_wrist_near_clip
from .common import ROOT, read
from .visual_geometry import TargetGeometry


class CameraRenderTests(unittest.TestCase):
    def camera(self):
        # Execute only the original camera-definition helpers; never import or
        # launch Isaac Sim. Use the real, unchanged ARX calibration YAML.
        tree=ast.parse((ROOT/'arx-sim/arx_sim/inprocess_app.py').read_text())
        helpers=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name in ('_set_xform_matrix','_apply_camera_definition')],type_ignores=[])
        scope={'Any':Any};exec(compile(helpers,'arx_camera_helpers','exec'),scope)
        stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
        path='/World/Robot/ARX/right_link26/right_color_camera'
        UsdGeom.Xform.Define(stage,str(Path(path).parent))
        spec=yaml.safe_load((ROOT/'arx-sim/config/controllers.yaml').read_text())['controllers']['camera']['definitions']['right_wrist']
        scope['_apply_camera_definition'](stage,path,spec)
        return stage,path

    def test_only_renderer_near_clip_changes_on_real_ARX_camera_definition(self):
        stage,path=self.camera();camera=UsdGeom.Camera(stage.GetPrimAtPath(path))
        before={a.GetName():str(a.Get()) for a in camera.GetPrim().GetAttributes()}
        result=correct_wrist_near_clip(stage,path,[.001,10.])
        after={a.GetName():str(a.Get()) for a in camera.GetPrim().GetAttributes()}
        self.assertEqual([k for k in before if before[k]!=after[k]],['clippingRange'])
        self.assertTrue(result['all_other_camera_attributes_unchanged'])
        np.testing.assert_allclose(result['before_m'],[.05,10])
        np.testing.assert_allclose(result['after_m'],[.001,10])

    def test_invalid_or_broader_optics_changes_are_refused_without_mutation(self):
        for values in ([.05,10],[.1,10],[0,10],[.001,9],[float('nan'),10],[.001]):
            stage,path=self.camera();before=stage.GetRootLayer().ExportToString()
            with self.assertRaises(ValueError):correct_wrist_near_clip(stage,path,values)
            self.assertEqual(before,stage.GetRootLayer().ExportToString())

    def test_missing_camera_and_nonmetre_stage_are_refused(self):
        stage,path=self.camera()
        with self.assertRaises(ValueError):correct_wrist_near_clip(stage,'/missing',[.001,10])
        UsdGeom.SetStageMetersPerUnit(stage,.01)
        with self.assertRaises(ValueError):correct_wrist_near_clip(stage,path,[.001,10])

    def test_saved_ARX_corn_goal_loses_near_slab_rejection_only(self):
        folder=ROOT/'runs/graspness_teacher_validation_20261102_02'
        requests=[read(p) for p in (folder/'workers/planner/requests').glob('*.done')]
        request=next(r for r in requests if r.get('pool',{}).get('candidates')
            and r['pool']['candidates'][0]['goal_id']=='b2c75f20f583688ca9cd')
        ctx=request['visual_context'];geometry=TargetGeometry.from_asset(ctx['asset'])
        goal=np.asarray(request['pool']['candidates'][0]['grasp_world'])
        args=(request['snapshot']['objects']['test_object']['world'],
            goal@np.asarray(ctx['eef_from_camera']),ctx['intrinsics'],ctx['shape'])
        old=geometry.inspect(*args,[.05,10]);new=geometry.inspect(*args,[.001,10])
        self.assertEqual(old['reasons'],['target_intersects_camera_near_slab'])
        self.assertTrue(new['pass'])
        self.assertEqual(old['policy'],new['policy'])
        self.assertGreater(old['camera_surface_distance_m'],.01)

    def test_real_ARX_milk_camera_too_close_stays_rejected_with_short_near_clip(self):
        folder=ROOT/'runs/sdg_household_20261101_01'
        episode=folder/'groups/group_004139233b28/episodes/343e50388801488d90cf6f2f0b8dd3f0'
        import json
        row=json.loads((episode/'samples.jsonl').read_text().splitlines()[16])
        asset=next(a for a in read(folder/'registration.json')['catalog']['objects'] if a['object_id']==row['object_id'])
        result=TargetGeometry.from_asset(asset).inspect(row['snapshot']['objects']['test_object']['world'],
            row['camera_world_optical'],row['intrinsics'],(480,640),[.001,10])
        self.assertFalse(result['pass'])
        self.assertIn('camera_surface_clearance',result['reasons'])


if __name__=='__main__':unittest.main()
