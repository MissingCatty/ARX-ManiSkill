from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from arx_sim.config import available_task_layouts, load_sim_config, write_resolved_config
from arx_sim.preflight import validate_install_and_assets


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Isaac Sim and its arx-ops-compatible client")
    parser.add_argument(
        "--webrtc-port",
        type=int,
        help="WebRTC signaling port (default: ARX_SIM_WEBRTC_PORT or 49100)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--disable-ros2", action="store_true", help="diagnostic only; DualMap topics will be absent")
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--task",
        choices=available_task_layouts(),
        help="optional startup task layout; base scene is unchanged when omitted",
    )
    return parser.parse_args()


def _full_path(root: str, relative: str) -> str:
    return str(Path(root) / relative.strip("/"))


def _set_xform_matrix(xform: Any, rows: list[list[float]]) -> None:
    from pxr import Gf

    matrix = Gf.Matrix4d(*[float(value) for row in rows for value in row])
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:transform":
            op.Set(matrix)
            return
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)


def _apply_camera_definition(stage: Any, path: str, spec: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        parent_path = Sdf.Path(path).GetParentPath()
        parent = stage.GetPrimAtPath(parent_path)
        if not parent:
            raise RuntimeError(
                f"camera parent prim is missing: {parent_path}; refusing to create "
                "a detached transform hierarchy"
            )
        prim = UsdGeom.Camera.Define(stage, path).GetPrim()
    camera = UsdGeom.Camera(prim)
    intrinsics = spec.get("color_intrinsics", {})
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    width = float(intrinsics.get("width", 640.0))
    height = float(intrinsics.get("height", 480.0))
    focal_mm = 24.0
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 10.0))
    camera.CreateFocalLengthAttr().Set(focal_mm)
    if fx > 0.0 and fy > 0.0:
        camera.CreateHorizontalApertureAttr().Set(focal_mm * width / fx)
        camera.CreateVerticalApertureAttr().Set(focal_mm * height / fy)
    # Match PR8: the camera prim must remain renderable for RGB-D output.
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    if spec.get("xform_rows"):
        _set_xform_matrix(UsdGeom.Xformable(prim), spec["xform_rows"])


def _apply_task_initial_joint_positions(robot: Any, config: dict[str, Any]) -> None:
    """Apply optional task-only articulation state after the first world reset."""
    positions = config.get("task_layout", {}).get("initial_joint_positions", {})
    if not positions:
        return
    if not isinstance(positions, dict):
        raise ValueError("task_layout.initial_joint_positions must be a mapping")
    dof_index = {name: index for index, name in enumerate(robot.dof_names)}
    missing = sorted(set(positions) - set(dof_index))
    if missing:
        raise RuntimeError(f"task initial joints absent from articulation: {missing}")
    indices = [dof_index[name] for name in positions]
    values = [float(positions[name]) for name in positions]
    robot.set_dof_positions(values, dof_indices=indices)
    robot.set_dof_position_targets(values, dof_indices=indices)


def _apply_gripper_contact_overrides(stage: Any, config: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    contact = config["controllers"].get("gripper_contact", {})
    if not contact:
        return

    material_cfg = contact.get("material", {})
    material_root = Sdf.Path("/World/PhysicsMaterials")
    if not stage.GetPrimAtPath(material_root).IsValid():
        stage.DefinePrim(material_root, "Scope")
    material = UsdShade.Material.Define(stage, material_root.AppendChild("gripper_rubber"))
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(material_cfg.get("static_friction", 3.0)))
    physics_material.CreateDynamicFrictionAttr().Set(float(material_cfg.get("dynamic_friction", 2.5)))
    physics_material.CreateRestitutionAttr().Set(float(material_cfg.get("restitution", 0.0)))

    replacement_cfg = contact.get("replacement_colliders", {})
    replacement_enabled = bool(replacement_cfg.get("enabled", False))
    disable_link_names = set(
        replacement_cfg.get("disable_existing_collision_links", [])
        if replacement_enabled else []
    )
    split_links = set(replacement_cfg.get("split_existing_collision_links", []) if replacement_enabled else [])
    split_axis_names = [
        str(axis).lower()
        for axis in replacement_cfg.get(
            "split_axes",
            [replacement_cfg.get("split_axis", "y")],
        )
    ]
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    split_axes = [(axis_name, axis_lookup[axis_name]) for axis_name in split_axis_names if axis_name in axis_lookup]
    if not split_axes:
        split_axes = [("y", 1)]
    segment_cfg = replacement_cfg.get("split_segments", 6)
    if isinstance(segment_cfg, dict):
        split_segments_by_axis = {
            axis_name: max(1, int(segment_cfg.get(axis_name, 1)))
            for axis_name, _axis in split_axes
        }
    else:
        split_segments_by_axis = {
            axis_name: max(1, int(segment_cfg))
            for axis_name, _axis in split_axes
        }
    split_overlap_m = max(0.0, float(replacement_cfg.get("split_overlap_m", 0.001)))

    link_names = set(contact.get("bind_existing_collision_links", []))
    pad_specs = contact.get("rubber_pads", {})
    pad_links = pad_specs.get("links", {}) if bool(pad_specs.get("enabled", True)) else {}
    wanted_links = link_names | set(pad_links.keys()) | disable_link_names | split_links

    found_links: dict[str, Any] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        name = prim.GetName()
        if name in wanted_links:
            found_links[name] = prim

    bound_colliders = 0
    disabled_colliders = 0
    created_replacement_colliders = 0

    def _copy_split_collision_mesh(link_name: str, source_prim: Any) -> int:
        source_mesh = UsdGeom.Mesh(source_prim)
        points = list(source_mesh.GetPointsAttr().Get() or [])
        face_counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        face_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not points or not face_counts or not face_indices:
            print(f"[runtime] gripper replacement skipped empty collision mesh: {source_prim.GetPath()}", flush=True)
            return 0

        axis_ranges: list[tuple[str, int, int, float, float, float]] = []
        for axis_name, axis_index in split_axes:
            values = [float(point[axis_index]) for point in points]
            min_coord = min(values)
            max_coord = max(values)
            span = max_coord - min_coord
            if span <= 1e-6:
                continue
            segments = split_segments_by_axis.get(axis_name, 1)
            axis_ranges.append((axis_name, axis_index, segments, min_coord, max_coord, span / segments))

        if not axis_ranges:
            return 0

        faces: list[tuple[tuple[float, ...], list[int]]] = []
        cursor = 0
        for count in face_counts:
            indices = [int(index) for index in face_indices[cursor : cursor + int(count)]]
            cursor += int(count)
            if not indices:
                continue
            centroid = tuple(
                sum(float(points[index][axis_index]) for index in indices) / len(indices)
                for _axis_name, axis_index, _segments, _min_coord, _max_coord, _step in axis_ranges
            )
            faces.append((centroid, indices))

        if not faces:
            return 0

        piece_count = 0
        parent_path = source_prim.GetPath().GetParentPath()
        source_name = source_prim.GetName()
        from itertools import product

        segment_ranges = [range(segments) for _axis_name, _axis_index, segments, _min_coord, _max_coord, _step in axis_ranges]
        for segment_indices in product(*segment_ranges):
            bounds: list[tuple[float, float]] = []
            suffix_parts: list[str] = []
            for segment, (axis_name, _axis_index, segments, min_coord, max_coord, step) in zip(segment_indices, axis_ranges):
                lower = min_coord + segment * step - split_overlap_m
                upper = min_coord + (segment + 1) * step + split_overlap_m
                if segment == 0:
                    lower = min_coord - split_overlap_m
                if segment == segments - 1:
                    upper = max_coord + split_overlap_m
                bounds.append((lower, upper))
                suffix_parts.append(f"{axis_name}{segment:02d}")

            selected_faces = []
            for centroid, indices in faces:
                if all(lower <= value <= upper for value, (lower, upper) in zip(centroid, bounds)):
                    selected_faces.append(indices)
            if not selected_faces:
                continue

            old_to_new: dict[int, int] = {}
            piece_points: list[Any] = []
            piece_counts: list[int] = []
            piece_indices: list[int] = []
            for indices in selected_faces:
                piece_counts.append(len(indices))
                for old_index in indices:
                    if old_index not in old_to_new:
                        old_to_new[old_index] = len(piece_points)
                        piece_points.append(points[old_index])
                    piece_indices.append(old_to_new[old_index])

            if len(piece_points) < 4:
                continue

            piece_path = parent_path.AppendChild(f"replacement_{source_name}_piece_{'_'.join(suffix_parts)}")
            piece_mesh = UsdGeom.Mesh.Define(stage, piece_path)
            piece_prim = piece_mesh.GetPrim()
            piece_mesh.CreatePointsAttr().Set(piece_points)
            piece_mesh.CreateFaceVertexCountsAttr().Set(piece_counts)
            piece_mesh.CreateFaceVertexIndicesAttr().Set(piece_indices)
            piece_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            UsdGeom.Imageable(piece_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            UsdPhysics.CollisionAPI.Apply(piece_prim).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(piece_prim).CreateApproximationAttr().Set("convexHull")
            UsdShade.MaterialBindingAPI.Apply(piece_prim).Bind(material)
            piece_count += 1

        print(
            f"[runtime] split gripper collider {source_prim.GetPath()} into "
            f"{piece_count} convex pieces along {','.join(axis_name for axis_name, _axis in split_axes)}",
            flush=True,
        )
        return piece_count

    for link_name in sorted(link_names):
        link = found_links.get(link_name)
        if link is None:
            print(f"[runtime] gripper contact link not found for material binding: {link_name}", flush=True)
            continue
        for prim in Usd.PrimRange(link):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if prim.GetName().startswith("replacement_"):
                    continue
                if link_name in split_links and prim.IsA(UsdGeom.Mesh):
                    created_replacement_colliders += _copy_split_collision_mesh(link_name, prim)
                if link_name in disable_link_names:
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                    disabled_colliders += 1
                else:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                    bound_colliders += 1

    created_pads = 0
    size_x, size_y, size_z = [float(value) for value in pad_specs.get("size_m", [0.060, 0.004, 0.050])]
    pad_x = float(pad_specs.get("x_m", 0.030))
    pad_z = float(pad_specs.get("z_m", -0.020))
    for link_name, spec in sorted(pad_links.items()):
        link = found_links.get(link_name)
        if link is None:
            print(f"[runtime] gripper contact link not found for rubber pad: {link_name}", flush=True)
            continue
        pad_path = link.GetPath().AppendChild("rubber_pad_inner")
        cube = UsdGeom.Cube.Define(stage, pad_path)
        prim = cube.GetPrim()
        cube.CreateSizeAttr().Set(1.0)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(pad_x, float(spec["y_m"]), pad_z))
        xform.AddScaleOp().Set(Gf.Vec3f(size_x, size_y, size_z))
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        created_pads += 1

    print(
        "[runtime] applied gripper rubber contact material "
        f"to {bound_colliders} existing colliders; disabled {disabled_colliders} "
        f"existing colliders; created {created_replacement_colliders} replacement "
        f"convex pieces and {created_pads} rubber pads",
        flush=True,
    )


def _prepare_ros_process(config: dict[str, Any], disabled: bool) -> None:
    """Re-exec before Kit starts when using Isaac's bundled ROS libraries."""
    ros = config.get("ros2", {})
    if disabled or not bool(ros.get("enabled", True)):
        return
    distro = str(ros.get("distro", "jazzy"))
    os.environ["ROS_DISTRO"] = distro
    os.environ["ROS_DOMAIN_ID"] = str(ros.get("domain_id", 0))
    os.environ["RMW_IMPLEMENTATION"] = str(ros.get("rmw_implementation", "rmw_fastrtps_cpp"))
    internal_lib = Path(config["simulation"]["isaac_root"]) / "exts" / "isaacsim.ros2.core" / distro / "lib"
    if not internal_lib.is_dir():
        return  # A sourced system ROS distribution may provide the libraries.
    paths = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    if str(internal_lib) in paths:
        return
    if os.environ.get("COOLAS_ISAAC_ROS_REEXEC") == "1":
        raise RuntimeError(f"Isaac bundled ROS library path was not retained: {internal_lib}")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(internal_lib), *paths])
    environment["COOLAS_ISAAC_ROS_REEXEC"] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "arx_sim.inprocess_app", *sys.argv[1:]],
        environment,
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _enable_debug_ui_extensions(app_utils: Any, simulation_app: Any) -> None:
    debug_extensions = [
        "omni.kit.window.extensions",
        "omni.kit.window.script_editor",
        "omni.physx.ui",
        "omni.physx.bundle",
        "omni.debugdraw",
    ]
    print("[runtime] ARX_SIM_DEBUG_UI=1; enabling Isaac debug UI extensions", flush=True)
    for extension in debug_extensions:
        try:
            app_utils.enable_extension(extension)
            print(f"[runtime] enabled debug extension: {extension}", flush=True)
        except Exception as exc:
            print(
                f"[runtime] failed to enable debug extension {extension}: {exc}",
                flush=True,
            )
    simulation_app.update()


def main() -> int:
    args = _arguments()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    config = load_sim_config(
        task=args.task,
        run_dir=run_dir,
        webrtc_port=args.webrtc_port,
    )
    _prepare_ros_process(config, args.disable_ros2 or args.validate_only)
    resolved_path = write_resolved_config(config)
    errors = validate_install_and_assets(config, check_port=not args.validate_only)
    if errors:
        for error in errors:
            print(f"[preflight] ERROR: {error}", file=sys.stderr)
        return 2
    print(f"[preflight] Isaac Sim adapter config={resolved_path}", flush=True)
    if config.get("task_layout"):
        print(f"[preflight] task layout={config['task_layout']['name']}", flush=True)
    if args.validate_only:
        print("[preflight] Isaac Sim installation and locked assets are valid", flush=True)
        return 0

    ros = config.get("ros2", {})
    os.environ["ROS_DISTRO"] = str(ros.get("distro", "jazzy"))
    os.environ["ROS_DOMAIN_ID"] = str(ros.get("domain_id", 0))
    os.environ["RMW_IMPLEMENTATION"] = str(ros.get("rmw_implementation", "rmw_fastrtps_cpp"))

    simulation_app: Any = None
    context: Any = None
    api_server: Any = None
    lifecycle: Any = None
    exit_code = 0
    try:
        # This is intentionally the first Isaac/Omniverse import.
        from isaacsim import SimulationApp

        sim = config["simulation"]
        vray_mdl_path = Path(sim["isaac_root"]) / "kit" / "mdl" / "core" / "VRay"
        simulation_app = SimulationApp({
            "headless": bool(sim.get("headless", True)),
            "hide_ui": False,
            "renderer": str(sim.get("renderer", "RaytracedLighting")),
            "anti_aliasing": int(sim.get("anti_aliasing", 3)),
            "denoiser": bool(sim.get("denoiser", True)),
            "samples_per_pixel_per_frame": int(sim.get("samples_per_pixel_per_frame", 64)),
            "multi_gpu": False,
            "active_gpu": int(sim.get("active_gpu", 0)),
            "physics_gpu": int(sim.get("physics_gpu", sim.get("active_gpu", 0))),
            "fast_shutdown": True,
            # The environment's OmniGibson material depends on Isaac's VRay
            # compatibility modules, which Isaac 6.0 does not put on its
            # default MDL system path in the Python experience.
            "extra_args": [
                f'--/app/mdl/additionalSystemPaths=["{vray_mdl_path}"]'
            ],
        })

        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
        from pxr import PhysxSchema, Usd, UsdPhysics

        from arx_sim.api import create_app
        from arx_sim.api_server import BackgroundApiServer
        from arx_sim.backend import IsaacBackend
        from arx_sim.backend.command_queue import CommandQueue
        from arx_sim.backend.context import IsaacContext
        from arx_sim.lifecycle import JointLifecycle
        from arx_sim.robot.controller import IsaacRobotController
        from arx_sim.runtime.determinism import configure_determinism
        from arx_sim.runtime.scene_cameras import (
            activate_scene_camera,
            create_scene_cameras,
            demote_calibration_cameras,
            disable_imported_cameras,
        )
        from arx_sim.runtime.simulation_loop import run_simulation_loop
        from arx_sim.runtime.stage_loader import load_composed_stage, require_prim
        from arx_sim.sensors.camera import IsaacRgbdCamera

        webrtc_port = int(config["webrtc"]["port"])
        simulation_app.set_setting("/app/window/drawMouse", True)
        simulation_app.set_setting(
            "/exts/omni.kit.livestream.app/primaryStream/signalPort", webrtc_port
        )
        simulation_app.set_setting(
            "/exts/omni.kit.livestream.app/primaryStream/streamPort", webrtc_port + 1
        )
        app_utils.enable_extension("omni.kit.livestream.app")
        simulation_app.update()
        if _env_flag("ARX_SIM_DEBUG_UI"):
            _enable_debug_ui_extensions(app_utils, simulation_app)

        configure_determinism(int(sim["seed"]))
        stage = load_composed_stage(simulation_app, config)
        world = World(
            physics_dt=float(sim["physics_dt"]),
            rendering_dt=float(sim["rendering_dt"]),
            stage_units_in_meters=1.0,
        )
        articulation_path = str(sim["articulation_root"])
        articulation_prim = require_prim(stage, articulation_path)
        articulation_physics = PhysxSchema.PhysxArticulationAPI.Apply(
            articulation_prim
        )
        articulation_physics.CreateEnabledSelfCollisionsAttr().Set(
            bool(sim.get("enable_self_collisions", False))
        )
        articulation_physics.CreateSolverPositionIterationCountAttr().Set(
            int(sim.get("solver_position_iteration_count", 16))
        )
        articulation_physics.CreateSolverVelocityIterationCountAttr().Set(
            int(sim.get("solver_velocity_iteration_count", 4))
        )
        for prim in Usd.PrimRange(articulation_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(
                    prim
                ).CreateDisableGravityAttr().Set(False)
        _apply_gripper_contact_overrides(stage, config)
        robot = Articulation(articulation_path)
        world.reset()
        _apply_task_initial_joint_positions(robot, config)
        world.step(render=False)
        disable_imported_cameras(
            stage,
            (str(sim.get("environment_prim", "/World/Environment")), str(sim["robot_prim"])),
        )
        create_scene_cameras(stage, articulation_path, sim["scene_cameras"])
        activate_scene_camera(sim["scene_cameras"])

        controller_cfg = config["controllers"]
        frames = controller_cfg["frames"]
        base_path = _full_path(articulation_path, frames["base_link"])
        lift_path = _full_path(articulation_path, frames["lift_link"])
        arm_base_path = _full_path(articulation_path, frames["arm_base_link"])
        ee_path = _full_path(articulation_path, frames["ee_link"])
        left_ee_path = _full_path(articulation_path, controller_cfg["left_arm"]["ee_link"])
        tcp_path = _full_path(articulation_path, frames["gripper_tcp"])
        camera_cfg = controller_cfg["camera"]
        camera_paths = {
            role: _full_path(articulation_path, relative)
            for role, relative in camera_cfg["roles"].items()
        }
        for role, path in camera_paths.items():
            spec = dict(camera_cfg.get("definitions", {}).get(role, {}))
            if spec:
                _apply_camera_definition(stage, path, spec)
        demote_calibration_cameras(stage, tuple(camera_paths.values()))
        for path in [base_path, lift_path, arm_base_path, ee_path, left_ee_path, tcp_path, *camera_paths.values()]:
            require_prim(stage, path)
        base_prim = RigidPrim(base_path)
        lift_prim = RigidPrim(lift_path)
        arm_base_prim = lift_prim if arm_base_path == lift_path else RigidPrim(arm_base_path)
        ee_prim = RigidPrim(ee_path)
        left_ee_prim = RigidPrim(left_ee_path)
        tcp_prim = ee_prim if tcp_path == ee_path else RigidPrim(tcp_path)
        camera_prims = {role: XformPrim(path) for role, path in camera_paths.items()}
        resolution = tuple(int(v) for v in camera_cfg["resolution"])
        publish_hz = float(sim.get("camera_tick_hz", config["ros2"].get("publish_hz", 30.0)))
        enabled_camera_roles = set(sim.get("camera_roles", camera_paths))
        if not enabled_camera_roles or not enabled_camera_roles.issubset(camera_paths):
            raise ValueError("simulation.camera_roles must select existing RGB-D cameras")

        def create_cameras() -> dict[str, Any]:
            return {
                role: IsaacRgbdCamera(
                    f"/World/COOLASSensors/{role}", resolution, publish_hz,
                    int(sim.get("active_gpu", 0)), camera_prims[role],
                    camera_cfg.get("definitions", {}).get(role, {}).get("color_intrinsics"),
                )
                for role in camera_paths if role in enabled_camera_roles
            }

        from arx_sim.runtime.rendering import configure_camera_antialiasing

        configure_camera_antialiasing(simulation_app, int(sim.get("anti_aliasing", 3)))
        cameras = create_cameras()
        queue = CommandQueue(capacity=int(sim["queue_capacity"]))
        context = IsaacContext(
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            robot=robot,
            base_prim=base_prim,
            lift_prim=lift_prim,
            arm_base_prim=arm_base_prim,
            ee_prim=ee_prim,
            left_ee_prim=left_ee_prim,
            tcp_prim=tcp_prim,
            camera_prims=camera_prims,
            cameras=cameras,
            config=config,
            command_queue=queue,
            camera_factory=create_cameras,
            contact_materials_applier=lambda: _apply_gripper_contact_overrides(stage, config),
        )
        controller = IsaacRobotController(context)
        context.controller = controller
        # One tick-synchronous controller owns the four actuator groups; expose
        # each role explicitly in IsaacContext without adding forwarding layers.
        context.arm_controller = controller
        context.gripper_controller = controller
        context.chassis_controller = controller
        context.lift_controller = controller

        if bool(ros.get("enabled", True)) and not args.disable_ros2:
            app_utils.enable_extension("isaacsim.ros2.bridge")
            simulation_app.update()
            from arx_sim.sensors.ros2_publishers import Ros2Publishers

            context.ros2_publishers = Ros2Publishers(
                config, camera_role=str(ros.get("camera_role", "right_arm"))
            )

        for _ in range(int(sim.get("camera_warmup_steps", 8))):
            world.step(render=True)
        # Define EE Home from the settled startup pose. This keeps the initial
        # arm status at zero instead of exposing gravity-settling offsets.
        controller.capture_ee_home_frames()
        queue.set_ready()
        controller.publish_snapshot(world.current_time)

        backend = IsaacBackend(context)
        api = create_app(backend)
        api_server = BackgroundApiServer(
            api,
            host=str(config["services"]["arx_ops_host"]),
            port=int(config["services"]["arx_ops_port"]),
        )
        api_server.start()
        lifecycle = JointLifecycle()
        lifecycle.install_signal_handlers()
        print(
            f"[runtime] Isaac Sim ready; compatible API="
            f"{config['services']['arx_ops_host']}:{config['services']['arx_ops_port']}",
            flush=True,
        )
        print(
            f"[runtime] WebRTC signaling=tcp:{webrtc_port} media=udp:{webrtc_port + 1}",
            flush=True,
        )
        run_simulation_loop(
            context,
            lifecycle.stop_event,
            max_steps=max(0, int(args.max_steps)),
            render=True,
            api_server=api_server,
        )
    except BaseException as exc:
        exit_code = 1
        if context is not None:
            context.command_queue.fail(str(exc))
            context.publish(lifecycle="FAILED", error=str(exc))
        traceback.print_exc()
    finally:
        if lifecycle is not None:
            try:
                lifecycle.shutdown(context, api_server)
            finally:
                lifecycle.restore_signal_handlers()
        elif api_server is not None:
            api_server.stop()
        if simulation_app is not None:
            simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
