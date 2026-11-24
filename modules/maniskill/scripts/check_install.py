"""Check the standalone installation; this does not measure grasp success."""
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

root = Path(os.environ["MANISKILL_ROOT"])
report = {
    "versions": {p: version(p) for p in ("mani-skill", "sapien", "torch", "numpy")},
    "cuda_available": torch.cuda.is_available(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "purpose": "installation check only; no grasp success evaluation",
}
assert report["cuda_available"], "CUDA is unavailable"
report["cuda_device"] = torch.cuda.get_device_name(0)
assert torch.equal(torch.arange(8, device="cuda").square().cpu(), torch.arange(8).square())
report["cuda_tensor_check"] = True
env = gym.make(
    "PickCube-v1", num_envs=1, obs_mode="rgb+depth+segmentation",
    control_mode="pd_joint_delta_pos", render_mode="rgb_array",
    sim_backend="gpu", render_backend="sapien_cuda",
)
try:
    obs, info = env.reset(seed=20261115)
    frames = []
    for step in range(5):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render().detach().cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        frames.append(frame)
    report["sensors"] = {
        name: {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
               for key, value in channels.items()}
        for name, channels in obs["sensor_data"].items()
    }
    rgb = frames[-1]
    assert np.isfinite(rgb).all() and rgb.std() > 1, "Rendered image is invalid"
    import imageio.v2 as imageio
    imageio.imwrite(root / "outputs/install_check.png", rgb)
    imageio.mimwrite(root / "outputs/install_check.mp4", frames, fps=10)
    report.update({"reset_and_steps": 5, "render_check": True, "status": "passed"})
finally:
    env.close()
(root / "outputs/install_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
