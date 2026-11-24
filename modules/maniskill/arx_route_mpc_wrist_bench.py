"""Track the final planned approach with live wrist correction and cuRobo MPC."""
import inspect
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PchipInterpolator

import arx_route_mpc_latency_v2_bench as latency

SELF = Path(__file__).resolve()
smooth = latency.latency.smooth
mpc = smooth.mpc
benchmark = smooth.benchmark
WRIST_REFERENCE_SPEED = .052
WRIST_SPEED_LIMIT = smooth.route.guarded.MAX_COMMAND_SPEED_RAD_S
MIN_VISION_CHANGE_M = .00025
MAX_VISION_JOINT_OFFSET_RAD = .035
BASE_PREPARE = latency.prepare


class WristRouteMPC(smooth.SmoothRouteMPC):
    def begin(self, directory, context, q, qd):
        if self.route_stage != 'approach':
            return {'success': False, 'reason': 'wrist_reference_missing'}
        plan = mpc.read(directory / 'plan_result.json')
        path = plan['paths']['approach']
        handoff = len(self.route.positions) - 1
        positions = np.asarray(path['q'][handoff:])
        stamps = np.asarray(path['times'][handoff:])
        if len(stamps) < 2:
            return {'success': False, 'reason': 'wrist_reference_empty'}
        result = super().begin(directory, context, q, qd)
        if not result['success']:
            return result
        self.wrist_curve = PchipInterpolator(stamps - stamps[0], positions, axis=0)
        self.wrist_span = float(stamps[-1] - stamps[0])
        tangent = self.wrist_curve(np.linspace(0., self.wrist_span, 101), 1)
        self.wrist_duration = max(mpc.SOLVE_DT, 1.5 * np.abs(tangent).max() * self.wrist_span
                                  / WRIST_REFERENCE_SPEED)
        actual = np.asarray(q, dtype=float)
        self.wrist_stream = smooth.StreamState(0., 1., actual - positions[0],
                                               np.asarray(qd), np.zeros(6), actual)
        self.wrist_endpoint = positions[-1]
        self.wrist_delta = np.zeros(6)
        self.last_vision_goal = self.nominal.copy()
        self.set_route_mode(self.joint_state(q, qd))
        return {**result, 'reference_seconds': self.wrist_duration}

    def wrist_reference(self, seconds):
        fraction = float(np.clip(seconds / self.wrist_duration, 0., 1.))
        eased = fraction * fraction * (3. - 2. * fraction)
        return self.wrist_curve(self.wrist_span * eased)

    def step(self, directory, q, qd, goal_world):
        if directory != self.directory:
            raise RuntimeError('MPC session does not own the requested trial')
        goal = np.asarray(goal_world)
        if (goal.shape != (4, 4) or not np.isfinite(goal).all()
                or np.linalg.norm(goal[:3, 3] - self.nominal[:3, 3]) > mpc.MAX_CORRECTION_M + 1e-9
                or not np.array_equal(goal[:3, :3], self.nominal[:3, :3])):
            raise ValueError('MPC goal outside the registered local correction region')
        measured = self.joint_state(q, qd)
        state = self.wrist_stream
        vision_updated = np.linalg.norm(goal[:3, 3] - self.last_vision_goal[:3, 3]) >= MIN_VISION_CHANGE_M
        if vision_updated:
            self.last_vision_goal = goal.copy()
            seed = self.torch.as_tensor(self.wrist_endpoint, dtype=measured.position.dtype,
                                        device=measured.position.device)
            solution = self.context.ik(goal, seed)
            if solution is not None:
                delta = solution.reshape(6).cpu().numpy() - self.wrist_endpoint
                if np.isfinite(delta).all() and np.abs(delta).max() <= MAX_VISION_JOINT_OFFSET_RAD:
                    self.wrist_delta = delta

        # Bring in the visual endpoint adjustment gradually along the nominal path.
        blend = min(1., (state.time + mpc.SOLVE_DT) / max(.4, self.wrist_duration * .5))
        visual = self.wrist_delta * blend
        target = self.wrist_reference(state.time + smooth.route.LOOKAHEAD_S) + visual
        self.solver.update_goal_state(self.joint_state(target, np.zeros(6)))
        prediction = self.solver.optimize_action_sequence(measured.clone())
        if prediction.action_sequence is None:
            return {'success': False, 'reason': 'wrist_mpc_invalid_action_sequence'}
        commands = prediction.action_sequence.position.detach().cpu().numpy().reshape(-1, 6)
        if len(commands) < mpc.STEPS_PER_SOLVE or not np.isfinite(commands).all():
            return {'success': False, 'reason': 'wrist_mpc_invalid_action_sequence'}
        error = np.abs(np.asarray(q) - state.last_command).max()
        next_reference = self.wrist_reference(state.time + mpc.SOLVE_DT) + visual
        correction = np.clip(commands[mpc.STEPS_PER_SOLVE - 1] - next_reference, -.004, .004)
        correction *= np.clip(error / .015, 0., 1.)
        for pace in (1., .7, .4):
            applied, after = smooth.advance_commands(self.wrist_reference, np.inf, state,
                                                      visual + correction, pace)
            speed = np.abs(np.diff(np.vstack([state.last_command, applied]), axis=0)).max() / smooth.DT
            if speed > WRIST_SPEED_LIMIT:
                continue
            points = np.vstack([q, applied])
            alpha = (np.arange(5) / 5)[None, :, None]
            dense = (points[:-1, None] * (1. - alpha) + points[1:, None] * alpha).reshape(-1, 6)
            dense = np.vstack([dense, applied[-1]])
            tensor = self.torch.as_tensor(dense, dtype=measured.position.dtype, device=measured.position.device)
            if self.context.checker.validate(tensor[:, None]).all():
                self.wrist_stream = after
                return {'success': True, 'positions': applied.tolist(),
                        'solver_seconds': float(prediction.solve_time), 'collision_checked_samples': len(dense),
                        'projected_max_joint_speed_rad_s': float(speed), 'reference_seconds': self.wrist_duration,
                        'reference_progress_s': float(after.time), 'visual_goal_updated': bool(vision_updated),
                        'visual_joint_offset_rad': self.wrist_delta.tolist()}
        return {'success': False, 'reason': 'wrist_mpc_no_safe_interval'}


def prepare(out, backend_gpu, scene_gpu, cases):
    source = latency.latency.substitute(inspect.getsource(BASE_PREPARE), "'_LATENCY02'", "'_WRIST01'")
    namespace = {**latency.__dict__, 'SELF': SELF}
    exec(compile(source, str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    original = Path(latency.__file__).resolve()
    shutil.copy2(original, out / 'source' / original.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        registration = benchmark.read(path)
        registration['source_hashes'][str(original)] = benchmark.sha(original)
        registration['mpc'].update(wrist_planned_suffix=True, wrist_reference_speed_rad_s=WRIST_REFERENCE_SPEED,
                                   wrist_vision_update_m=MIN_VISION_CHANGE_M,
                                   wrist_vision_joint_limit_rad=MAX_VISION_JOINT_OFFSET_RAD)
        benchmark.save(path, registration)


def main():
    with patch.multiple(latency, SELF=SELF, prepare=prepare), patch.object(smooth, 'SmoothRouteMPC', WristRouteMPC):
        latency.main()


if __name__ == '__main__':
    main()
