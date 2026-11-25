"""Restore the paired measured handoff, then use the unchanged grasp executor."""
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from arx_project.layout import read_json, sha256, write_json
from .cohort import check_pair


def state_difference(saved, actual):
    if saved['joint_names'] != actual['joint_names']:
        raise RuntimeError('Paired replay joint ordering differs')
    objects = {name: float(np.linalg.norm(np.asarray(value['pose'][:3])-
        np.asarray(actual['objects'][name]['pose'][:3]))) for name, value in saved['objects'].items()}
    rotation = {name: float(2*np.arccos(np.clip(abs(np.dot(value['pose'][3:],
        actual['objects'][name]['pose'][3:])), 0., 1.))) for name, value in saved['objects'].items()}
    return {'joint_max_difference': float(np.abs(np.asarray(saved['qpos'])-actual['qpos']).max()),
        'base_matrix_max_difference': float(np.abs(np.asarray(saved['links']['base_link'])-
            actual['links']['base_link']).max()),
        'camera_matrix_max_difference': float(np.abs(np.asarray(saved['camera_world_optical'])-
            actual['camera_world_optical']).max()),
        'object_translation_difference_m': objects, 'object_rotation_difference_rad': rotation}


def validate_difference(difference):
    if (difference['joint_max_difference'] > .001 or difference['base_matrix_max_difference'] > .001
            or difference['camera_matrix_max_difference'] > .001
            or max(difference['object_translation_difference_m'].values(), default=0.) > .001
            or max(difference['object_rotation_difference_rad'].values(), default=0.) > .02):
        raise RuntimeError('Paired replay state does not match the frozen handoff: '+str(difference))


def restore(env, executor, directory, saved):
    import sapien
    import torch
    robot = env.agent.robot
    robot.set_pose(sapien.Pose(np.asarray(saved['links']['base_link'])))
    q = torch.tensor([saved['qpos']], dtype=torch.float32, device=env.device)
    robot.set_qpos(q)
    robot.set_qvel(torch.zeros_like(q))
    for name, item in saved['objects'].items():
        actor = env.objects[name]
        if env.object_models[name] != item['model_id']:
            raise RuntimeError('Paired replay changed an object model')
        actor.set_pose(sapien.Pose(item['pose'][:3], item['pose'][3:]))
        actor._bodies[0].linear_velocity = np.zeros(3)
        actor._bodies[0].angular_velocity = np.zeros(3)
    idle = []
    for tick in range(60):
        env._step_action(q)
        idle.append({'tick': tick, 'qpos': robot.qpos.cpu().numpy()[0].tolist(),
                     'qvel': robot.qvel.cpu().numpy()[0].tolist()})
    write_json(directory/'paired_idle_trace.json', idle)
    if max(np.abs(row['qvel']).max() for row in idle[-30:]) >= .01:
        raise RuntimeError('Paired replay robot idle state is unconfirmed')
    obs = env.get_obs()
    actual = executor.snapshot(env, obs)
    write_json(directory/'paired_actual_state.json', actual)
    Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(directory/'paired_actual_head.png')
    difference = state_difference(saved, actual)
    write_json(directory/'paired_state_difference.json', difference)
    validate_difference(difference)
    write_json(directory/'paired_restore_verification.json', {'passed': True,
        'idle_samples': 60, 'difference': difference, 'cached_SAM_input_used': True,
        'same_objects_and_robot': True, 'source': 'paired measured post-stationing state'})


class PairedPreparation:
    def __init__(self, pair, services, executor):
        self.pair, self.services, self.executor = pair, services, executor

    def prepare(self, env, out, entry):
        started = time.perf_counter()
        frozen = check_pair(self.pair)
        for name in self.pair['input_hashes']:
            target = out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frozen/name, target)
        state = read_json(out/'state.json')
        if self.pair.get('common_precondition_failure'):
            write_json(out/'paired_common_failure.json', {
                'source_result_sha256': self.pair['baseline_result_sha256'],
                'reason': 'unchanged_high_workspace_stationing_prerequisite',
                'native_grasp_planning_called': False})
            return {'success': False, 'reason': 'unable_to_reach_high_workspace', 'motion_started': False}
        restore(env, self.executor, out, state)
        Image.fromarray(env.render().cpu().numpy()[0]).save(out/'initial_overview.png')
        write_json(out/'initial_conditions.json', {'spawn': env.spawn,
            'base_pose': entry['base_pose'], 'initial_q': env.initial_q.tolist(),
            'object_bbox': env.object_bbox, 'paired_station_world': state['links']['base_link']})
        write_json(out/'capture.json', {'camera': 'head', 'mask_source': 'frozen successful SAM3',
            'paired_source': self.pair['baseline_trial'], 'input_hashes': self.pair['input_hashes']})
        request_started = time.perf_counter()
        response = self.services.request('zerograsp', out, seed=2026112500+self.pair['schedule_index'])
        generation_seconds = time.perf_counter()-request_started
        plan_seconds = 0.
        if response['result']['candidate_count']:
            kwargs = {} if self.pair['workspace']['preferred_arm'] is None else {
                'arm': self.pair['workspace']['preferred_arm']}
            request_started = time.perf_counter()
            response = self.services.request('curobo_dual_arm', out, **kwargs)
            plan_seconds = time.perf_counter()-request_started
            plan = response['result']
        else:
            plan = {'success': False, 'reason': 'zerograsp_no_postprocessed_candidate', 'count': 0}
            write_json(out/'plan_result.json', plan)
        write_json(out/'latency.json', {'generation_rpc_seconds': generation_seconds,
            'initial_planning_rpc_seconds': plan_seconds,
            'inference_and_initial_planning_seconds': generation_seconds+plan_seconds,
            'prepare_with_state_restore_seconds': time.perf_counter()-started})
        if not plan['success']:
            return {'success': False, 'reason': plan['reason'], 'motion_started': False}
        names = ['state.json', 'plan_result.json', 'camera/color.png', 'camera/depth_m.npy',
                 'camera/mask.png', 'candidates/candidate_pool.json']
        return {'success': True, 'prepared_plan': True, 'motion_started': False,
            **self.pair['workspace'], 'prepared_artifacts': {n: sha256(out/n) for n in names},
            'paired_source': self.pair['baseline_trial']}
