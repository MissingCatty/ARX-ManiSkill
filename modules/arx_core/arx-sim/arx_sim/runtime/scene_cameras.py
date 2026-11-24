from __future__ import annotations

from typing import Any


def disable_imported_cameras(stage: Any, roots: tuple[str, ...]) -> None:
    from pxr import Usd, UsdGeom

    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            raise RuntimeError(f"camera asset root is missing: {root_path}")
        cameras = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Camera)]
        for prim in cameras:
            prim.SetActive(False)


def demote_calibration_cameras(stage: Any, paths: tuple[str, ...]) -> None:
    from pxr import UsdGeom

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"calibration camera is missing: {path}")
        prim.SetTypeName("Xform")


def create_scene_cameras(stage: Any, articulation_path: str, spec: dict[str, Any]) -> None:
    from pxr import Gf, UsdGeom

    articulation = stage.GetPrimAtPath(articulation_path)
    if not articulation:
        raise RuntimeError(f"articulation root is missing: {articulation_path}")
    robot_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(articulation)
    target = robot_to_world.Transform(Gf.Vec3d(*spec["target_local_xyz"]))
    UsdGeom.Xform.Define(stage, "/World/SceneCameras")
    for name, local_eye in spec["views"].items():
        eye = robot_to_world.Transform(Gf.Vec3d(*local_eye))
        camera = UsdGeom.Camera.Define(stage, f"/World/SceneCameras/{name}")
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 50.0))
        camera.CreateFocalLengthAttr().Set(18.0)
        camera.CreateHorizontalApertureAttr().Set(24.0)
        camera.CreateVerticalApertureAttr().Set(13.5)
        view = Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
        camera.MakeMatrixXform().Set(view.GetInverse())


def activate_scene_camera(spec: dict[str, Any]) -> bool:
    """Select the configured robot view for the UI and WebRTC viewport."""
    name = spec.get("default_view")
    if not name:
        return False
    if name not in spec["views"]:
        raise ValueError(f"default scene camera {name!r} is absent from views")
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        return False
    viewport.camera_path = f"/World/SceneCameras/{name}"
    return True
