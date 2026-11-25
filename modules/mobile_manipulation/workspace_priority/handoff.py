"""Verify an observation/plan handoff immediately before any arm execution."""
from pathlib import Path
import numpy as np

from arx_project.layout import now, read_json, sha256, write_json


def validate_prepared(directory, preparation, actual_state):
    directory = Path(directory)
    if not preparation.get('success') or not preparation.get('prepared_plan'):
        raise RuntimeError('Prepared grasp was not successfully verified')
    hashes = preparation.get('prepared_artifacts', {})
    required = {'state.json', 'plan_result.json', 'camera/color.png', 'camera/depth_m.npy',
                'camera/mask.png', 'candidates/candidate_pool.json'}
    if not required.issubset(hashes):
        raise RuntimeError('Prepared grasp is missing required artifact hashes')
    for relative, expected in hashes.items():
        path = (directory/relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise RuntimeError('Prepared grasp artifact changed: '+relative)
    saved = read_json(directory/'state.json')
    if saved['joint_names'] != actual_state['joint_names'] or np.max(
            np.abs(np.asarray(saved['qpos'])-actual_state['qpos'])) > .001:
        raise RuntimeError('prepared_plan_stale_robot_state')
    for key in ['base_link', 'body4']:
        if np.max(np.abs(np.asarray(saved['links'][key])-actual_state['links'][key])) > .001:
            raise RuntimeError('prepared_plan_stale_base_or_lift')
    body = np.asarray(actual_state['links']['body4'])
    point = body[:3, :3].T@(np.asarray(preparation['prepared_target_world'])-body[:3, 3])
    cell = np.floor(point/preparation['workspace_resolution_m']).astype(int).tolist()
    if cell != preparation['verified_high_cell']:
        raise RuntimeError('prepared_plan_left_verified_high_workspace')
    plan = read_json(directory/'plan_result.json')
    if not plan.get('success'):
        raise RuntimeError('Prepared native grasp planning failed')
    arm = preparation.get('preferred_arm')
    if arm is not None and plan['selected']['arm'] != arm:
        raise RuntimeError('Prepared grasp used an arm outside its high workspace')
    evidence = {'passed': True, 'at': now(), 'selected_arm': plan['selected']['arm'],
                'workspace_scope': preparation.get('workspace_scope'),
                'verified_high_cell': cell, 'artifact_hashes': hashes,
                'robot_joint_names': actual_state['joint_names'],
                'robot_qpos': actual_state['qpos'],
                'body4_world': actual_state['links']['body4'],
                'base_world': actual_state['links']['base_link']}
    write_json(directory/'prepared_plan_execution_gate.json', evidence)
    return evidence
