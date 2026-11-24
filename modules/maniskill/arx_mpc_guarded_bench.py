"""Settled handoff and bounded Cartesian goals for the resident ARX MPC."""
import argparse
import inspect
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np

import arx_mpc_approach as mpc
import arx_mpc_approach_bench as bench

SELF = Path(__file__).resolve()
MAX_SUBGOAL_M = .004
MAX_FRAME_UPDATE_M = .001
HANDOFF_JOINT_SPEED_RAD_S = .035
MAX_COMMAND_SPEED_RAD_S = .075
MAX_ROLLING_SOLVES = 80


class StableTranslation(mpc.TargetTranslation):
    def update(self, points):
        previous = self.offset.copy()
        result = super().update(points)
        if result['status'] == 'correction_outside_local_region':
            # A new visible surface may fail registration. Retain the last
            # trusted observation and never apply a displacement outside the region.
            result['status'] = 'rejected_surface_registration'
        if result['status'] == 'tracked':
            delta = self.offset - previous
            length = np.linalg.norm(delta)
            if length > MAX_FRAME_UPDATE_M:
                self.offset = previous + delta * (MAX_FRAME_UPDATE_M / length)
            result['translation_m'] = self.offset.tolist()
        return result


class BoundedApproach(mpc.ResidentApproach):
    def __init__(self, context, state):
        from curobo.model_predictive_control import ModelPredictiveControlCfg
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        create = ModelPredictiveControlCfg.create

        def configured(**kwargs):
            kwargs.update(squared_l2_regularization_weight=[.01, 500., 10., 0., 0.])
            return create(**kwargs)

        with patch.object(ModelPredictiveControlCfg, 'create', configured):
            super().__init__(context, state)
        self.solver.update_tool_pose_criteria({mpc.FRAME: ToolPoseCriteria.track_position_and_orientation(
            rpy=[1., 1., 1.], non_terminal_scale=.5)})

    def solve(self, measured):
        torch = self.torch
        result = self.solver.optimize_action_sequence(measured.clone())
        if result.action_sequence is None:
            return {'success': False, 'reason': 'mpc_invalid_action_sequence'}
        proposals = result.action_sequence.position.reshape(-1, 6)[:mpc.STEPS_PER_SOLVE]
        if proposals.shape[0] != mpc.STEPS_PER_SOLVE or not torch.isfinite(proposals).all():
            return {'success': False, 'reason': 'mpc_invalid_action_sequence'}
        start = measured.position
        proposed_speed = float(torch.diff(torch.cat([start, proposals]), dim=0).abs().max()) * mpc.CONTROL_HZ
        scale = min(1., MAX_COMMAND_SPEED_RAD_S / max(proposed_speed, 1e-9))
        alpha = torch.arange(5, device=start.device) / 5
        for _ in range(5):
            # Slow and, when necessary, backtrack the actual next interval.
            # Revalidate the projected command, starting at the actual measured state.
            commands = start + scale * (proposals - start)
            path = torch.cat([start, commands])
            dense = (path[:-1, None] * (1 - alpha[None, :, None])
                     + path[1:, None] * alpha[None, :, None]).reshape(-1, 6)
            dense = torch.cat([dense, commands[-1:]], dim=0)
            if self.context.checker.validate(dense[:, None]).all():
                return {'success': True, 'positions': commands.detach().cpu().tolist(),
                        'solver_seconds': float(result.solve_time), 'collision_checked_samples': len(dense),
                        'native_horizon_feasible': bool(result.success.all()), 'command_scale': scale,
                        'projected_max_joint_speed_rad_s': proposed_speed * scale}
            scale *= .5
        return {'success': False, 'reason': 'mpc_no_safe_command_interval'}

    def step(self, directory, q, qd, goal_world):
        if directory != self.directory:
            raise RuntimeError('MPC session does not own the requested trial')
        goal = np.asarray(goal_world)
        if (goal.shape != (4, 4) or not np.isfinite(goal).all()
                or np.linalg.norm(goal[:3, 3] - self.nominal[:3, 3]) > mpc.MAX_CORRECTION_M + 1e-9
                or not np.array_equal(goal[:3, :3], self.nominal[:3, :3])):
            raise ValueError('MPC goal outside the registered local correction region')
        measured = self.joint_state(q, qd)
        kin = self.solver.compute_kinematics(measured).tool_poses.position
        local = kin.detach().cpu().numpy().reshape(3)
        origin = self.context.base[:3, :3] @ local + self.context.base[:3, 3]
        remaining = goal[:3, 3] - origin
        distance = float(np.linalg.norm(remaining))
        subgoal = goal.copy()
        if distance > MAX_SUBGOAL_M:
            subgoal[:3, 3] = origin + remaining * (MAX_SUBGOAL_M / distance)
        self.set_goal(subgoal)
        response = self.solve(measured)
        return {**response, 'subgoal_displacement_m': min(distance, MAX_SUBGOAL_M)}


def install_variant():
    mpc.ResidentApproach = BoundedApproach
    mpc.TargetTranslation = StableTranslation
    source = inspect.getsource(mpc.execute_approach)
    before = "    state = read(out / 'state.json')"
    assert source.count(before) == 1
    settle = '''    hold = execution.command[indices].copy()
    for brake_ticks in range(CONTROL_HZ):
        joint_state = measured()
        if (np.max(np.abs(joint_state['qd'])) <= HANDOFF_JOINT_SPEED_RAD_S
                and np.max(np.abs(np.asarray(joint_state['q']) - hold)) < .005):
            break
        command(hold)
    else:
        raise RuntimeError('mpc_handoff_not_settled')
    handoff_distance, _ = tool_errors(tool_transform(execution), nominal)
    if handoff_distance > HANDOFF_M + .002:
        raise RuntimeError('mpc_handoff_outside_local_region')

'''
    source = source.replace(before, settle + before)
    source = source.replace("'global_prefix_ticks': k,", "'global_prefix_ticks': k, 'handoff_settle_ticks': brake_ticks,")
    namespace = {**mpc.__dict__, 'HANDOFF_JOINT_SPEED_RAD_S': HANDOFF_JOINT_SPEED_RAD_S,
                 'MAX_SOLVES': MAX_ROLLING_SOLVES}
    exec(compile(source, str(SELF) + ':mpc_execute_approach', 'exec'), namespace)
    mpc.execute_approach = namespace['execute_approach']


def prepare(out, backend_gpu, scene_gpu, cases):
    source = inspect.getsource(bench.prepare)
    old = "slot = canonical + '_MPC01'"
    assert source.count(old) == 1
    namespace = {**bench.__dict__, 'SELF': SELF}
    exec(compile(source.replace(old, "slot = canonical + '_MPC02'"), str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    shutil.copy2(bench.__file__, out / 'source' / Path(bench.__file__).name)
    for file in (out / 'registration.json', out / bench.ARM / 'registration.json'):
        reg = bench.read(file)
        reg['source_hashes'][str(Path(bench.__file__).resolve())] = bench.sha(bench.__file__)
        reg['mpc'].update(handoff_joint_speed_rad_s=HANDOFF_JOINT_SPEED_RAD_S,
                          maximum_cartesian_subgoal_m=MAX_SUBGOAL_M,
                          maximum_observation_update_m=MAX_FRAME_UPDATE_M,
                          maximum_command_speed_rad_s=MAX_COMMAND_SPEED_RAD_S,
                          maximum_rolling_solves=MAX_ROLLING_SOLVES,
                          orientation_axis_weight=1., acceleration_regularization_weight=500.)
        bench.save(file, reg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'serve', 'status'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend-gpu')
    parser.add_argument('--scene-gpu')
    parser.add_argument('--case', action='append')
    args, _ = parser.parse_known_args()
    if args.stage == 'prepare':
        assert args.output and args.backend_gpu and args.scene_gpu
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu, args.case or bench.PILOT)
        return
    install_variant()
    bench.SELF = SELF
    bench.main()


if __name__ == '__main__':
    main()
