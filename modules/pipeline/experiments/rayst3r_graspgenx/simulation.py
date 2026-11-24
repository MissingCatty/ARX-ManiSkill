"""Reuse frozen physical execution; supply head images and predicted masks."""
from copy import deepcopy
import inspect
import os
from pathlib import Path
import textwrap
import types

import numpy as np

from common import MS, paths, read, save


def install(root):
    paths()
    os.environ.setdefault('MS_ASSET_DIR', str(MS / 'assets'))
    import arx_runtime_imports
    arx_runtime_imports.install()
    import environment
    import run
    import baseline
    import short_lift_protocol
    from head_camera_config import head_configuration
    from camera_imaging import camera_profiles, intrinsic_matrix
    import gymnasium as gym
    reg = read(root / 'registration.json')
    environment.INITIAL_JOINTS = {**environment.INITIAL_JOINTS, 'joint4': .452}
    profiles = camera_profiles()
    override, camera_record = head_configuration(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS = deepcopy(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS['camera']['definitions']['head'] = camera_record['camera_definition']
    short_lift_protocol.install_execution_policy(run, baseline)
    keeper = []

    def make(model_id='014_lemon', perception_camera='head', scene_spec=None):
        assert perception_camera == 'head'
        env = gym.make('ARXPickYCB-v1', model_id=model_id, perception_camera='head',
            scene_spec=scene_spec, num_envs=1, obs_mode='rgb+depth+segmentation',
            control_mode='pd_joint_pos', reward_mode='none', render_mode='rgb_array',
            sim_backend='cpu', render_backend='sapien_cuda', sensor_configs=override).unwrapped
        if not keeper:
            import sapien
            keeper.append(sapien.render.SapienRenderer(env._render_device))
        return env
    run.make = make

    # Retain the exact historical arm observation motion and its collision checks.
    # Its wrist image visibility gate is replaced by the actual head observation
    # and SAM3 outcome; no head image is admitted using a wrist-only bbox rule.
    source = getattr(run.trial, 'registered_short_lift_source', None) or inspect.getsource(run.trial)
    source = textwrap.dedent(source)
    old = "if not view['admitted_view']:"
    assert source.count(old) == 1
    source = source.replace(old, "if perception_camera != 'head' and not view['admitted_view']:")
    scope = dict(run.__dict__)
    exec(compile(source, str(Path(__file__)) + ':head_trial', 'exec'), scope)
    generated = scope['trial']
    run.trial = types.FunctionType(generated.__code__, run.__dict__, 'trial', generated.__defaults__)
    original_execution = run.Execution

    class HeadExecution(original_execution):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sensor_writer = run.imageio.get_writer(self.out / 'head.mp4', fps=20,
                codec='libx264', quality=7, macro_block_size=2)
    run.Execution = HeadExecution

    def capture_training():
        out = root / 'preflight/head_warmup'
        out.mkdir(parents=True, exist_ok=False)
        env = make()
        try:
            env.reset(seed=0)
            for _ in range(80): env._step_action(run.torch.tensor(env.initial_q[None], device=env.device))
            obs = env.get_obs()
            state = run.snapshot(env, obs)
            assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
            expected = read(reg['common_initial_conditions'])
            assert np.array_equal(env.initial_q, np.asarray(expected['initial_q'], dtype=np.float32))
            k = np.asarray(state['intrinsics'])
            assert np.max(np.abs(k - intrinsic_matrix(profiles['head']))) < 1e-4
            data = obs['sensor_data']['head']
            rgb, depth = data['rgb'].cpu().numpy()[0], data['depth'].cpu().numpy()[0, :, :, 0]
            out.joinpath('camera').mkdir()
            run.Image.fromarray(rgb).save(out / 'camera/color.png')
            run.Image.fromarray(depth.astype(np.uint16)).save(out / 'camera/depth.png')
            # Warmup SAM3 sees RGB and text only; it creates the mask itself.
            save(out / 'camera/params.json', {'depth_scale': .001, 'intrinsics': {
                'width': 640, 'height': 480, 'fx': k[0, 0], 'fy': k[1, 1], 'ppx': k[0, 2], 'ppy': k[1, 2]}})
            save(out / 'state.json', state)
            save(out / 'entry.json', {'model_id': '014_lemon', 'target_text': 'lemon', 'warmup': True})
            run.Image.fromarray(env.render().cpu().numpy()[0]).save(out / 'overview.png')
            save(root / 'preflight/simulation.json', {'passed': True, 'joint4_m': .452,
                'physical_grasps': 0, 'head_configuration': camera_record,
                'same_initial_q_as_fixed452': True})
        finally:
            env.close()
        return out
    return run, baseline, capture_training
