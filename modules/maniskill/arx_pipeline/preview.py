import json
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from environment import make, ROOT

env = make()
try:
    obs, _ = env.reset(seed=20261116)
    action = torch.tensor(env.initial_q[None], device=env.device)
    for _ in range(80):
        obs, *_ = env.step(action)
    out = ROOT / "outputs/arx_preview"
    out.mkdir(exist_ok=True)
    imageio.imwrite(out / "overview.png", env.render().cpu().numpy()[0])
    imageio.imwrite(out / "head.png", obs["sensor_data"]["head"]["rgb"].cpu().numpy()[0])
    state = {"joint_names": [j.name for j in env.agent.robot.active_joints],
             "qpos": env.agent.robot.qpos.cpu().numpy()[0].tolist(),
             "links": {n: env.agent.robot.links_map[n].pose.to_transformation_matrix().cpu().numpy()[0].tolist()
                       for n in ["base_link", "body4", "right_eef_link", "head_camera_link"]},
             "object_pose": env.obj.pose.raw_pose.cpu().numpy()[0].tolist(),
             "sensor_params": {k: v.cpu().numpy().tolist() for k, v in obs["sensor_param"]["head"].items()},
             "sensor_shapes": {k: list(v.shape) for k,v in obs["sensor_data"]["head"].items()},
             "spawn": env.spawn}
    (out / "state.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state))
finally:
    env.close()
