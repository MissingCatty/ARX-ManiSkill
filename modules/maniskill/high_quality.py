"""High-fidelity rendering configuration for new standalone ManiSkill runs."""
import copy
import gymnasium as gym
import mani_skill.envs
from mani_skill.render.shaders import PREBUILT_SHADER_CONFIGS


def make(env_id, **kwargs):
    shader = copy.deepcopy(PREBUILT_SHADER_CONFIGS["rt"])
    shader.shader_pack_config.update(
        ray_tracing_samples_per_pixel=256,
        ray_tracing_path_depth=16,
        ray_tracing_denoiser="optix",
    )
    config = dict(
        num_envs=1,
        # Official single-environment backend; SAPIEN 3.0 ray tracing does not
        # support ManiSkill's GPU parallel renderer.
        sim_backend="auto",
        render_backend="sapien_cuda",
        obs_mode="rgb+depth+segmentation",
        render_mode="rgb_array",
        enable_shadow=True,
        sensor_configs=dict(shader_config=shader, width=1920, height=1080),
        human_render_camera_configs=dict(shader_config=shader, width=3840, height=2160),
        viewer_camera_configs=dict(shader_config=shader),
    )
    config.update(kwargs)
    return gym.make(env_id, **config)
