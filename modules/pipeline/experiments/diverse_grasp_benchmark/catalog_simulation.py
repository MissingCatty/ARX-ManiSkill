"""Bind the verified 80-model catalog to the unchanged short-lift executor."""
from arx_project.assets import load_object_catalog
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def install(root):
    from common import MS, read, save
    sys.path.insert(0, str(PROJECT / 'experiments/diverse_object_scenes'))
    import pipeline as catalog_pipeline
    catalog = load_object_catalog(root / 'object_catalog.json')['models']
    environment, baseline, run, sapien, torch, factory, profile, convert, get_maps = catalog_pipeline.runtime(catalog)
    import short_lift_protocol
    import gymnasium as gym
    from camera_imaging import intrinsic_matrix
    short_lift_protocol.install_execution_policy(run, baseline)
    keeper = []

    def make(model_id='014_lemon', perception_camera='head', scene_spec=None):
        assert perception_camera == 'head'
        if model_id in catalog:
            assert scene_spec is not None
            return factory(model_id, scene_spec)
        # Only the historical development lemon is used for warmup; evaluation
        # always takes the verified catalog branch above.
        assert model_id == '014_lemon' and scene_spec is None
        env = gym.make('ARXPickYCB-v1', model_id=model_id, perception_camera='head',
            num_envs=1, obs_mode='rgb+depth+segmentation', control_mode='pd_joint_pos',
            reward_mode='none', render_mode='rgb_array', sim_backend='cpu', render_backend='sapien_cuda').unwrapped
        if not keeper: keeper.append(sapien.render.SapienRenderer(env._render_device))
        return env
    run.make = make
    original_execution = run.Execution

    class HeadExecution(original_execution):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sensor_writer = run.imageio.get_writer(self.out / 'head.mp4', fps=20,
                codec='libx264', quality=7, macro_block_size=2)
    run.Execution = HeadExecution

    def capture_training():
        out = root / 'preflight/head_warmup'; out.mkdir(parents=True, exist_ok=False)
        entry=read(root/'registration.json')['schedule'][0]
        env = make(entry['model_id'],scene_spec=entry['scene'])
        try:
            env.reset(seed=0)
            for _ in range(80): env._step_action(torch.tensor(env.initial_q[None], device=env.device))
            obs = env.get_obs(); state = run.snapshot(env, obs)
            assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
            expected = read(root / 'common_initial_conditions.json')
            assert np.array_equal(env.initial_q, np.asarray(expected['initial_q'], dtype=np.float32))
            k = np.asarray(state['intrinsics'])
            assert np.max(np.abs(k - intrinsic_matrix(profile))) < 1e-4
            data = obs['sensor_data']['head']; out.joinpath('camera').mkdir()
            run.Image.fromarray(data['rgb'].cpu().numpy()[0]).save(out / 'camera/color.png')
            depth = data['depth'].cpu().numpy()[0, :, :, 0]
            run.Image.fromarray(depth.astype(np.uint16)).save(out / 'camera/depth.png')
            save(out / 'camera/params.json', {'depth_scale': .001, 'intrinsics': {
                'width': 640, 'height': 480, 'fx': k[0, 0], 'fy': k[1, 1], 'ppx': k[0, 2], 'ppy': k[1, 2]}})
            save(out / 'state.json', state)
            # Both pools warm from this fresh measured state, then rebind the
            # measured fingers on each request. No closed-state history is used.
            save(out / 'closed_state.json', state)
            save(out / 'entry.json', {**entry,'warmup':True})
            run.Image.fromarray(env.render().cpu().numpy()[0]).save(out / 'overview.png')
            save(root / 'preflight/simulation.json', {'passed': True, 'physical_grasps': 0,
                'joint4_m': .452, 'camera': 'head', 'K_matched': True,
                'same_initial_command_as_fixed452': True, 'catalog_models': len(catalog),
                'environment': 'ARXPickCatalog-v1', 'wrist_observation_motion': False})
        finally: env.close()
        return out
    return run, baseline, capture_training
