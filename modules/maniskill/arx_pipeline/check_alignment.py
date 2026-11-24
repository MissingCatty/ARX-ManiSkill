"""Check recorded ARX reset and calibrated head camera without grasp execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from environment import (make, ALIGNMENT, PERCEPTION_CAMERA, INITIAL_JOINTS,
                         INITIAL_SOURCE, CONTROLLERS_SOURCE, RUNTIME_SOURCE,
                         CURRENT_CONTROLLERS, HEAD, META, BASE_POSE, TABLE_HEIGHT, SCENE_CUBOIDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"alignment.json").exists():
        raise FileExistsError("Retain previous validation; choose a new output")
    env = make()
    try:
        obs, _ = env.reset(seed=20261116)
        robot = env.agent.robot
        names = [j.name for j in robot.active_joints]
        actual = robot.qpos.cpu().numpy()[0]
        limits = robot.qlimits.cpu().numpy()[0]
        expected = np.clip([INITIAL_JOINTS[n] for n in names], limits[:, 0], limits[:, 1])
        params = obs["sensor_param"][PERCEPTION_CAMERA]
        k = params["intrinsic_cv"].cpu().numpy()[0]
        color = HEAD["color_intrinsics"]
        source_k = np.array([[color["fx"], 0, color["cx"]], [0, color["fy"], color["cy"]], [0, 0, 1]])
        gl = params["cam2world_gl"].cpu().numpy()[0]
        mount = CURRENT_CONTROLLERS["camera"]["roles"]["head"].split("/", 1)[0]
        mount_world = robot.links_map[mount].pose.to_transformation_matrix().cpu().numpy()[0]
        expected_gl = mount_world @ np.asarray(HEAD["xform_rows"]).T
        camera = obs["sensor_data"][PERCEPTION_CAMERA]
        rgb = camera["rgb"].cpu().numpy()[0]
        depth = camera["depth"].cpu().numpy()[0, :, :, 0]
        seg = camera["segmentation"].cpu().numpy()[0, :, :, 0]
        mask = seg == env.obj._objs[0].per_scene_id
        for name, value in [("head.png", rgb), ("depth.png", depth.astype(np.uint16)), ("mask.png", mask.astype(np.uint8)*255),
                            ("overview.png", env.render().cpu().numpy()[0])]:
            imageio.imwrite(out/name, value)
        checks = {"initial_q_max_error": float(np.max(np.abs(actual-expected))),
                  "intrinsics_max_error_px": float(np.max(np.abs(k-source_k))),
                  "camera_world_transform_max_error": float(np.max(np.abs(gl-expected_gl))),
                  "head_shape": list(rgb.shape), "drive_parameters_match": True,
                  "joint_velocity_limits_match": True}
        drives = {}
        for joint in robot.active_joints:
            native = joint._objs[0]
            source = META["joints"][joint.name]
            cfg = CURRENT_CONTROLLERS[source["section"]]
            actual_drive = {"stiffness": native.stiffness, "damping": native.damping,
                            "force_limit": native.force_limit, "drive_mode": native.drive_mode,
                            "max_velocity": native.max_velocity}
            passed = (np.isclose(native.stiffness, cfg["drive_stiffness"]) and
                      np.isclose(native.damping, cfg["drive_damping"]) and
                      np.isclose(native.force_limit, cfg["max_effort"]) and
                      native.drive_mode == source["drive_mode"])
            checks["drive_parameters_match"] &= bool(passed)
            checks["joint_velocity_limits_match"] &= bool(np.isclose(native.max_velocity, source["limits"]["velocity"]))
            drives[joint.name] = actual_drive
        passed = bool(checks["initial_q_max_error"] < 1e-6 and checks["intrinsics_max_error_px"] < 1e-4
                      and checks["camera_world_transform_max_error"] < 1e-6
                      and checks["drive_parameters_match"] and checks["joint_velocity_limits_match"])
        report = {"alignment_passed": passed, "scope": "Initialization and render checks only; zero grasp trials and zero commanded physics steps",
                  "configuration": ALIGNMENT, "checks": checks, "joints": dict(zip(names, map(float, actual))),
                  "joint_drives": drives, "intrinsics": k.tolist(), "camera_world_gl": gl.tolist(),
                  "camera_world_optical": (gl@np.diag([1,-1,-1,1])).tolist(), "base_pose": BASE_POSE,
                  "target_world": env.obj.pose.raw_pose.cpu().numpy()[0].tolist(),
                  "target_mask_pixels": int(mask.sum()), "target_visible_for_inference": bool(mask.sum()>=50),
                  "tabletop_z_m": 0., "table_height_m": TABLE_HEIGHT, "scene_cuboids": SCENE_CUBOIDS,
                  "camera_z_m": float(gl[2,3]),
                  "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [INITIAL_SOURCE, CONTROLLERS_SOURCE, RUNTIME_SOURCE]}}
        (out/"alignment.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ["alignment_passed", "checks", "target_mask_pixels", "camera_z_m"]}, indent=2))
        if not passed:
            raise AssertionError("Alignment failed; inspect alignment.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
