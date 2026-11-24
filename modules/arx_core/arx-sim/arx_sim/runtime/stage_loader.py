from __future__ import annotations

from typing import Any


def _wait_for_stage(simulation_app: Any) -> None:
    from isaacsim.core.experimental.utils.stage import is_stage_loading

    simulation_app.update()
    while is_stage_loading() and simulation_app.is_running():
        simulation_app.update()
    if not simulation_app.is_running():
        raise RuntimeError("SimulationApp stopped while loading the USD stage")


def load_composed_stage(simulation_app: Any, config: dict[str, Any]) -> Any:
    """Compose environment, robot and task beneath a clean runtime root layer."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    simulation = config["simulation"]
    context = omni.usd.get_context()
    environment = str(simulation["environment"])
    context.new_stage()
    _wait_for_stage(simulation_app)
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world_prim = stage.DefinePrim(Sdf.Path("/World"), "Xform")
    stage.SetDefaultPrim(world_prim)

    # Referencing only the environment's default prim deliberately excludes
    # source-layer session objects such as /Render and viewport settings. Those
    # objects otherwise prevent Replicator annotators from producing data.
    environment_path = str(simulation.get("environment_prim", "/World/Environment"))
    environment_prim = stage.DefinePrim(Sdf.Path(environment_path), "Xform")
    if not environment_prim.GetReferences().AddReference(environment):
        raise RuntimeError(f"could not reference environment USD at {environment_path}")

    robot_path = str(simulation.get("robot_prim", "/World/Robot"))
    if stage.GetPrimAtPath(robot_path):
        raise RuntimeError(f"environment already contains the reserved robot prim: {robot_path}")
    robot_prim = stage.DefinePrim(Sdf.Path(robot_path), "Xform")
    if not robot_prim.GetReferences().AddReference(str(simulation["robot"])):
        raise RuntimeError(f"could not reference robot USD at {robot_path}")
    # A task pose describes the articulation root in world coordinates. The
    # referenced robot already has a capture pose, so compensate for it here.
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    pose = simulation.get("robot_pose")
    if pose:
        position = [float(v) for v in pose.get("position", [])]
        orientation = [float(v) for v in pose.get("orientation_wxyz", [])]
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("simulation.robot_pose must contain position[3] and orientation_wxyz[4]")
        _wait_for_stage(simulation_app)
        articulation_prim = stage.GetPrimAtPath(str(simulation["articulation_root"]))
        if not articulation_prim:
            raise RuntimeError("articulation root is missing while applying task pose")
        source = UsdGeom.XformCache().GetLocalToWorldTransform(articulation_prim)
        desired = Gf.Matrix4d(1.0)
        desired.SetRotateOnly(Gf.Rotation(Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))))
        desired.SetTranslateOnly(Gf.Vec3d(*position))
        xform.AddTransformOp().Set(source.GetInverse() * desired)

    overlay_path = str(simulation.get("task_overlay", "")).strip()
    if overlay_path:
        overlay_prim = stage.DefinePrim(Sdf.Path("/World/Task"), "Xform")
        if not overlay_prim.GetReferences().AddReference(overlay_path):
            raise RuntimeError(f"could not reference task overlay: {overlay_path}")
    _wait_for_stage(simulation_app)
    return stage


def require_prim(stage: Any, path: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"required USD prim is missing: {path}")
    return prim
