"""Render one high-quality frame; no policy, training or grasp trial."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ["MANISKILL_ROOT"])
sys.path.insert(0, str(ROOT))
import high_quality
import imageio.v2 as imageio
import numpy as np

start = time.monotonic()
env = high_quality.make("PickCube-v1")
try:
    obs, info = env.reset(seed=20261115)
    frame = env.render().cpu().numpy()[0]
    assert frame.shape == (2160, 3840, 3)
    assert np.isfinite(frame).all() and frame.std() > 1
    imageio.imwrite(ROOT / "outputs/high_quality_4k.png", frame)
    sensor = obs["sensor_data"]["base_camera"]
    imageio.imwrite(ROOT / "outputs/high_quality_sensor.png", sensor["rgb"][0].cpu().numpy())
    for key in ("depth", "segmentation"):
        np.save(ROOT / f"outputs/high_quality_{key}.npy", sensor[key].cpu().numpy())
    configs = env.unwrapped._sensor_configs
    shader = configs["base_camera"].shader_config.shader_pack_config
    assert shader["ray_tracing_samples_per_pixel"] == 256
    report = dict(status="passed", rendering=shader, observation_resolution=[1920, 1080],
                  recording_resolution=[3840, 2160], elapsed_seconds=time.monotonic()-start,
                  gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  physics="official task defaults; no overrides", grasp_trials=0,
                  entrypoint="high_quality.make")
finally:
    env.close()
(ROOT / "outputs/high_quality_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
