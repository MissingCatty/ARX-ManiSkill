"""Static positional work-area certificates with measured tool orientations."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def emit(value):
    print('WORKSPACE_IK ' + json.dumps(value), flush=True)


def main():
    state_path, directory = map(Path, sys.argv[1:3])
    directory.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    ms = project.parent / 'maniskill'
    sys.path[:0] = [str(ms), str(ms / 'arx_pipeline')]
    from planner import Planner
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    import torch
    state = json.loads(state_path.read_text())
    state['scene_meshes'] = {}
    planner = Planner(directory, state)
    planner.planner.ik_solver.update_tool_pose_criteria({'right_eef_link': ToolPoseCriteria.track_position()})
    start = planner.measured(state)
    emit({'ready': True, 'pid': __import__('os').getpid(), 'static_ik_only': True,
          'orientation': 'selected by position IK; saved from FK', 'physical_grasps': 0})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            return
        center = np.asarray(request['center_world_m'])
        answer = {'passed': False, 'id': request['id'], 'tested_orientations': 0}
        # Different initial orientation guesses help expose distinct joint solutions;
        # the IK criterion itself tracks position, not the guessed orientation.
        for yaw in [0., 90., 180., 270.]:
            goal = np.eye(4)
            goal[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
            goal[:3, 3] = [center[0], center[1], request['object_top_m'] + .08]
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
            assert bool(((qs >= planner.limits[:, 0]) & (qs <= planner.limits[:, 1])).all())
            actual = []
            for q, desired in zip([q_goal, q_pre, q_lift], [goal, pre, lift]):
                result = planner.planner.compute_kinematics(planner.js(q)).tool_poses
                local = result.position.detach().cpu().numpy().reshape(-1, 3)[0]
                quat = result.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
                position = planner.base[:3, :3] @ local + planner.base[:3, 3]
                rotation = planner.base[:3, :3] @ Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                error = float(np.linalg.norm(position - desired[:3, 3]))
                if error > .0021:
                    raise RuntimeError('Position IK FK error exceeds original position tolerance')
                desired[:3, :3] = rotation
                actual.append({'position_world_m': position.tolist(), 'position_error_m': error})
            answer.update(passed=True, goal_world=goal.tolist(), preparation_world=pre.tolist(),
                          lift_world=lift.tolist(), joint_names=[f'right_joint{i}' for i in range(21, 27)],
                          q_goal=q_goal.tolist(), q_preparation=q_pre.tolist(), q_lift=q_lift.tolist(),
                          actual_fk=actual, joint_limits_passed=True, table_collision_passed=True,
                          tested_scene='original table and ground; objects excluded',
                          definition='Position IK, EEF 8 cm above object top, another 8 cm above it, and 3 cm up; actual FK orientation saved')
            break
        emit(answer)


if __name__ == '__main__':
    main()
