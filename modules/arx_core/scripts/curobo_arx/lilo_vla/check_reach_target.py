"""Additional reach-only veto against the unchanged exported target mesh.

The interaction planners intentionally omit the grasp target. This independent
query checks the original open-gripper robot against the original target mesh
before any reaching command. It does not modify either planner or its scene,
robot geometry, collision settings, or the actual-entry contact gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TargetReachCheck:
    def __init__(self, trial):
        from gpu_policy import COMPUTE_GPU,compute_env
        os.environ.update(compute_env(os.environ,os.environ.get('CUDA_VISIBLE_DEVICES',COMPUTE_GPU)))
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        trial = Path(trial).resolve()
        snapshot = read(trial / 'capture_snapshot.json')
        target = snapshot['objects']['test_object']['path']
        sources = read(trial / 'scene_sources.json')
        exported = yaml.safe_load((trial / 'scene.yml').read_text())
        meshes = {key: value for key, value in exported['mesh'].items()
                  if sources[key] == target or sources[key].startswith(target + '/')}
        if not meshes:
            raise ValueError('Original exported target mesh is missing')
        robot = trial / 'model/arx.yml'
        config = RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(robot), scene_model={'mesh': meshes}, n_meshes=len(meshes) + 1,
            collision_activation_distance=0.)
        self.checker = RobotCollisionChecker(config)
        # Match the existing executor's self-collision setting; this extra query
        # only vetoes contacts with the target, never substitutes other checks.
        self.checker.self_collision_cost = None
        if self.checker.kinematics.joint_names != snapshot['right_arm']:
            raise ValueError('Target query joint order differs from the actual ARX arm')
        self.evidence = {'target_path': target, 'open_robot_sha256': sha(robot),
                         'robot_urdf_sha256': sha(trial / 'model/arx.urdf'),
                         'target_mesh_sha256': {key: sha(value['file_path']) for key, value in meshes.items()},
                         'collision_activation_distance': 0., 'original_planners_modified': False,
                         'scope': 'Additional reach-only planning veto; actual measured contacts remain authoritative'}

    def validate(self, positions, subdivisions=10):
        q = torch.as_tensor(positions, device='cuda', dtype=torch.float32)
        if q.ndim != 2 or q.shape[1] != len(self.checker.kinematics.joint_names) or not torch.isfinite(q).all():
            raise ValueError('Expected a finite actual-ARX joint path')
        if len(q) > 1 and subdivisions > 1:
            alpha = torch.arange(subdivisions, device=q.device) / subdivisions
            q = torch.cat([(q[:-1, None] * (1 - alpha[None, :, None]) +
                            q[1:, None] * alpha[None, :, None]).reshape(-1, q.shape[1]), q[-1:]])
        mask = torch.cat([self.checker.validate(chunk[:, None]).flatten() for chunk in q.split(64)])
        invalid = (~mask).nonzero().flatten().cpu().tolist()
        return {'pass': not invalid, 'samples': len(q), 'invalid_samples': len(invalid),
                'invalid_indices': invalid, 'first_sample_valid': bool(mask[0]),
                'last_sample_valid': bool(mask[-1]), 'subdivisions': subdivisions,
                'source': self.evidence}

    def reach(self, path, selected, start, goal, planner, checker, names, limits, base_world, output):
        """Keep the final entry; try one 80mm same-axis waypoint if needed.

        Both legs retain the original PRM/IK/scene checks. The supplementary
        target query can only reject. The final optimized command is checked
        separately by the caller, since optimization may leave its PRM seed.
        """
        import numpy as np
        from .grasp_handoff import corridor, local_callbacks, NEAR_RULES
        output = Path(output)
        selected.pop('reach_via', None)
        report = {'algorithm': 'target_veto_via_080mm_stop_v2', 'accepted': None,
                  'via_backoff_m': .08, 'final_entry_changed': False}

        def finish(route):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            return route

        direct = path(start, goal)
        if direct is not None:
            report['direct'] = self.validate(direct)
            if report['direct']['pass']:
                report['accepted'] = 'direct'
                return finish(direct)
        else:
            report['direct'] = {'pass': False, 'reason': 'Original PRM rejected route'}
        grasp = np.asarray(selected['grasp_world'])
        via = grasp.copy()
        via[:3, 3] -= .08 * grasp[:3, 2]
        # Obtain the entry world pose from frozen FK; never alter the selected G.
        from .tracking import forward
        entry_world = np.asarray(base_world) @ forward(planner.kinematics, goal.detach().cpu().numpy(), names)[0]
        solve, validate = local_callbacks(planner, checker, names, limits, base_world)
        q_via, proof = corridor(entry_world, via, goal, solve, validate, NEAR_RULES['ten_mm'])
        report['reverse_to_via'] = proof
        if q_via is None:
            return finish(None)
        first = path(start, q_via)
        if first is None:
            report['via_route'] = {'pass': False, 'reason': 'Original PRM rejected far waypoint'}
            return finish(None)
        continuation = torch.as_tensor(proof['joint_waypoints'][::-1], device=goal.device, dtype=goal.dtype)
        route = torch.cat([first, continuation[1:]])
        report['via_route'] = self.validate(route)
        if not report['via_route']['pass']:
            return finish(None)
        report['accepted'] = 'via_080mm'
        # Preserve this waypoint as two separately optimized motion commands.
        # One optimizer over the concatenated seed can cut back through target.
        selected['reach_via'] = {'world': via.tolist(), 'joint': q_via.tolist(),
                                 'route_split_index': len(first) - 1,
                                 'execution': 'Two original optimized commands with the original settling gate at the waypoint'}
        report['reach_via'] = selected['reach_via']
        return finish(route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trial = args.trial.resolve()
    check = TargetReachCheck(trial)
    command = read(trial / 'pregrasp_command.json')
    execution = read(trial / 'pregrasp_execution.json')
    report = {'command_path': check.validate(command['position']),
              'measured_samples': check.validate([r['position'] for r in execution['records']], subdivisions=1),
              'measured_sampling_limitation': 'Recorded 5Hz joint samples; no claim of observing the trajectory between records',
              'physical_configuration_changed': False, 'simulator_commands_issued': 0}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: {k: value[k] for k in ('pass', 'samples', 'invalid_samples', 'last_sample_valid')}
                      for key, value in report.items() if isinstance(value, dict)}))


if __name__ == '__main__':
    main()
