"""仅核对冻结 ARX 头部相机的静态 RGB-D，不调用规划或执行抓取。"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent / "maniskill/arx_pipeline"))
from environment import CURRENT_CONTROLLERS
import environment
import gymnasium as gym
from head_camera_config import head_configuration
from camera_imaging import camera_profiles, intrinsic_matrix, sensor_and_model_frames, write_profiles


def main():
    out = PROJECT / "experiments/pointcloud_benchmark/runs/five_methods_50_20261122_01/preparation/头部朝下核查"
    out.mkdir(parents=True, exist_ok=True)
    override, camera_record = head_configuration(CURRENT_CONTROLLERS)
    env = gym.make("ARXPickYCB-v1", model_id="005_tomato_soup_can", perception_camera="head",
                   num_envs=1, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   reward_mode="none", render_mode="rgb_array", sim_backend="cpu",
                   render_backend="sapien_cuda", sensor_configs=override).unwrapped
    try:
        obs, _ = env.reset(seed=20261122)
        data = obs["sensor_data"]["head"]
        rgb = data["rgb"].cpu().numpy()[0]
        depth = data["depth"].cpu().numpy()[0, :, :, 0]
        seg = data["segmentation"].cpu().numpy()[0, :, :, 0]
        mask = seg == env.obj._objs[0].per_scene_id
        Image.fromarray(rgb).save(out / "head_render.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / "target_mask_gt.png")
        np.save(out / "depth_mm.npy", depth)
        params = obs["sensor_param"]["head"]
        profile = camera_profiles()["head"]
        frames = sensor_and_model_frames(rgb, depth.astype(np.float32)*.001, profile)
        Image.fromarray(frames["raw_rgb"]).save(out / "d405_raw_rgb.png")
        Image.fromarray(frames["raw_depth"]).save(out / "d405_depth_raw.png")
        Image.fromarray(frames["rgb"]).save(out / "model_rgb.png")
        np.save(out / "model_depth_m.npy", frames["depth_m"])
        checks = {}
        for role in ["head", "right_wrist"]:
            got = obs["sensor_param"][role]["intrinsic_cv"].cpu().numpy()[0]
            expected = intrinsic_matrix(camera_profiles()[role])
            error = float(np.max(np.abs(got-expected)))
            assert error < 1e-4, (role, error)
            checks[role] = {"maximum_K_error": error, "calibration_matched": True}
        state = {"model_id": env.model_id, "target_mask_pixels": int(mask.sum()),
                 "intrinsics": params["intrinsic_cv"].cpu().numpy()[0].tolist(),
                 "camera_world_optical": (params["cam2world_gl"].cpu().numpy()[0] @ np.diag([1, -1, -1, 1])).tolist(),
                 "qpos": env.agent.robot.qpos.cpu().numpy()[0].tolist(),
                 "object_world": env.obj.pose.to_transformation_matrix().cpu().numpy()[0].tolist(),
                 "camera_alignment": {**env.camera_alignment, "camera_definition": camera_record["camera_definition"], "experiment_override": camera_record},
                 "real_camera_profile": profile, "renderer_intrinsic_checks": checks, "physical_grasp_trials": 0}
        write_profiles()
        (PROJECT / "configs/head_camera_mount.json").write_text(json.dumps(camera_record, ensure_ascii=False, indent=2) + "\n")
        (out / "preflight.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(out), "target_mask_pixels": int(mask.sum())}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
