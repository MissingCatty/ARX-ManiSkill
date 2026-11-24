from __future__ import annotations

import unittest

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = Usd = UsdGeom = None

from arx_sim.runtime.scene_cameras import (
    create_scene_cameras,
    demote_calibration_cameras,
    disable_imported_cameras,
)


@unittest.skipIf(Usd is None, "OpenUSD is not installed")
class SceneCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World")
        UsdGeom.Xform.Define(self.stage, "/World/Environment")
        UsdGeom.Xform.Define(self.stage, "/World/Robot")

    def test_imported_cameras_are_inactive(self) -> None:
        paths = (
            "/World/Environment/viewer_camera",
            "/World/Environment/external_sensor0",
            "/World/Robot/left_eef_link/Camera",
        )
        for path in paths:
            UsdGeom.Camera.Define(self.stage, path)
        disable_imported_cameras(self.stage, ("/World/Environment", "/World/Robot"))
        self.assertTrue(all(not self.stage.GetPrimAtPath(path).IsActive() for path in paths))

    def test_calibration_pose_remains_without_extra_camera(self) -> None:
        path = "/World/Robot/head_color_camera"
        camera = UsdGeom.Camera.Define(self.stage, path)
        camera.CreateFocalLengthAttr().Set(24.0)
        camera.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
        demote_calibration_cameras(self.stage, (path,))
        prim = self.stage.GetPrimAtPath(path)
        self.assertEqual(prim.GetTypeName(), "Xform")
        self.assertEqual(prim.GetAttribute("focalLength").Get(), 24.0)
        self.assertEqual(
            UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation(),
            Gf.Vec3d(0.1, 0.2, 0.3),
        )

    def test_scene_cameras_look_toward_robot(self) -> None:
        root = UsdGeom.Xform.Define(self.stage, "/World/Robot/articulation")
        root.AddTranslateOp().Set(Gf.Vec3d(4.3, 2.85, 0.01))
        root.AddOrientOp().Set(Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, -0.70710678)))
        spec = {
            "target_local_xyz": [0.35, 0.0, 0.65],
            "views": {
                "left_oblique": [-1.45, 1.15, 2.15],
                "right_oblique": [-1.45, -1.15, 2.15],
                "front_oblique": [1.60, 0.95, 2.30],
            },
        }
        create_scene_cameras(self.stage, str(root.GetPath()), spec)
        robot_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim())
        target = robot_to_world.Transform(Gf.Vec3d(*spec["target_local_xyz"]))
        for name in spec["views"]:
            prim = self.stage.GetPrimAtPath(f"/World/SceneCameras/{name}")
            self.assertTrue(prim.IsA(UsdGeom.Camera))
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            eye = matrix.ExtractTranslation()
            forward = matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
            self.assertGreater(Gf.Dot(forward, (target - eye).GetNormalized()), 0.9999)


if __name__ == "__main__":
    unittest.main()
