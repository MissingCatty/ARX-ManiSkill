"""Resident static preparation-pose IK checks; never sends robot commands."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PREFIX = 'WORKSPACE_IK '


def emit(value):
    print(PREFIX + json.dumps(value), flush=True)


def main():
    state_path, directory = map(Path, sys.argv[1:3])
    directory.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    ms = project.parent / 'maniskill'
    sys.path[:0] = [str(ms), str(ms / 'arx_pipeline')]
    from planner import Planner
    import torch
    state = json.loads(state_path.read_text())
    # The admission check concerns work area, not whether clutter can be solved.
    state['scene_meshes'] = {}
    planner = Planner(directory, state)
    start = planner.measured(state)
    emit({'ready': True, 'pid': __import__('os').getpid(),
          'static_ik_only': True, 'physical_grasps': 0})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            return
        center = np.asarray(request['center_world_m'])
        top = float(request['object_top_m'])
        answer = {'passed': False, 'id': request['id'], 'tested_orientations': 0}
        for yaw in [0., 90., 180., 270., 45., 135., 225., 315.]:
            goal = np.eye(4)
            goal[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
            goal[:3, 3] = [center[0], center[1], top + .08]
            pre, lift = goal.copy(), goal.copy()
            pre[2, 3] += .08
            lift[2, 3] += .03
            answer['tested_orientations'] += 1
            q_goal = planner.ik(goal, start)
            if q_goal is None:
                continue
            q_pre = planner.ik(pre, q_goal)
            q_lift = planner.ik(lift, q_goal)
            if q_pre is None or q_lift is None:
                continue
            qs = torch.stack([q_goal, q_pre, q_lift])
            if not bool(((qs >= planner.limits[:, 0]) & (qs <= planner.limits[:, 1])).all()):
                raise RuntimeError('IK returned a joint outside the measured limits')
            # Save actual FK independently of the requested Cartesian positions.
            actual = []
            for q, desired in zip([q_goal, q_pre, q_lift], [goal, pre, lift]):
                fk = planner.planner.compute_kinematics(planner.js(q)).tool_poses
                local = fk.position.detach().cpu().numpy().reshape(-1, 3)[0]
                world = planner.base[:3, :3] @ local + planner.base[:3, 3]
                error = float(np.linalg.norm(world - desired[:3, 3]))
                if error > .0021:
                    raise RuntimeError('Saved IK FK error exceeds original position tolerance')
                actual.append({'position_world_m': world.tolist(), 'position_error_m': error})
            answer.update(passed=True, goal_world=goal.tolist(), preparation_world=pre.tolist(),
                          lift_world=lift.tolist(), joint_names=[f'right_joint{i}' for i in range(21, 27)],
                          q_goal=q_goal.tolist(), q_preparation=q_pre.tolist(), q_lift=q_lift.tolist(),
                          actual_fk=actual, joint_limits_passed=True, table_collision_passed=True,
                          tested_scene='original table and ground; objects excluded',
                          definition='EEF 8 cm above object top, another 8 cm above it, and 3 cm up; no grasp execution')
            break
        emit(answer)


if __name__ == '__main__':
    main()
