"""Use the timed global reference and a continuous MPC correction for ARX."""
import argparse
from dataclasses import dataclass
import inspect
from pathlib import Path
import shutil

import numpy as np
from scipy.interpolate import PchipInterpolator

import arx_route_mpc_terminal_bench as terminal

SELF = Path(__file__).resolve()
benchmark = terminal.benchmark
route = terminal.route
mpc = route.mpc
DT = 1. / mpc.CONTROL_HZ
MAX_CORRECTION_RAD = .012
CORRECTION_VELOCITY_RAD_S = .035
CORRECTION_ACCEL_RAD_S2 = .30
CORRECTION_JERK_RAD_S3 = 4.
PACE_SLEW_PER_S = .6


@dataclass
class StreamState:
    time: float
    pace: float
    offset: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    last_command: np.ndarray


def advance_commands(curve, duration, state, target_offset, target_pace):
    """Integrate the live correction without resetting its derivatives at each solve."""
    offset, velocity = state.offset.copy(), state.velocity.copy()
    acceleration = state.acceleration.copy()
    t, pace = state.time, state.pace
    commands = []
    for _ in range(mpc.STEPS_PER_SOLVE):
        pace += float(np.clip(target_pace - pace, -PACE_SLEW_PER_S * DT, PACE_SLEW_PER_S * DT))
        t = min(duration, t + DT * pace)
        # Taper the correction while the reference comes to rest at its endpoint.
        desired = target_offset * np.clip((duration - t) / .4, 0., 1.)
        wanted_accel = np.clip(25 * (desired - offset) - 10 * velocity,
                               -CORRECTION_ACCEL_RAD_S2, CORRECTION_ACCEL_RAD_S2)
        change = np.clip(wanted_accel - acceleration, -CORRECTION_JERK_RAD_S3 * DT,
                         CORRECTION_JERK_RAD_S3 * DT)
        acceleration += change
        velocity = np.clip(velocity + DT * acceleration, -CORRECTION_VELOCITY_RAD_S,
                           CORRECTION_VELOCITY_RAD_S)
        offset += DT * velocity
        commands.append(curve(t) + offset)
    return np.asarray(commands), StreamState(t, pace, offset, velocity, acceleration, commands[-1])


class SmoothRouteMPC(terminal.TerminalRouteMPC):
    def begin_route(self, directory, context, phase, path, q, qd):
        result = super().begin_route(directory, context, phase, path, q, qd)
        if not result['success']:
            return result
        self.curve = PchipInterpolator(self.route.times, self.route.positions, axis=0, extrapolate=False)
        actual = np.asarray(q, dtype=float)
        initial_velocity = np.clip(np.asarray(qd) - self.curve(0., 1),
                                   -CORRECTION_VELOCITY_RAD_S, CORRECTION_VELOCITY_RAD_S)
        self.stream = StreamState(0., 1., actual - self.curve(0.), initial_velocity, np.zeros(6), actual)
        return result

    def route_step(self, directory, phase, q, qd):
        if self.directory != directory or self.route_stage != phase:
            raise RuntimeError('MPC session does not own this route')
        measured = self.joint_state(q, qd)
        actual = np.asarray(q, dtype=float)
        duration, state = self.route.duration, self.stream
        endpoint_error = float(np.max(np.abs(actual - self.route.positions[-1])))
        if (duration - state.time <= 1e-8 and endpoint_error < route.SETTLED_JOINT_ERROR_RAD
                and max(map(abs, qd)) < route.SETTLED_SPEED_RAD_S):
            return {'success': True, 'settled': True, 'joint_error_rad': endpoint_error,
                    'reference_progress_s': duration}

        target = self.curve(min(duration, state.time + route.LOOKAHEAD_S))
        self.solver.update_goal_state(self.joint_state(target, np.zeros(6)))
        prediction = self.solver.optimize_action_sequence(measured.clone())
        if prediction.action_sequence is None:
            return {'success': False, 'reason': 'smooth_mpc_invalid_action_sequence'}
        commands = prediction.action_sequence.position.detach().cpu().numpy().reshape(-1, 6)
        if len(commands) < mpc.STEPS_PER_SOLVE or not np.isfinite(commands).all():
            return {'success': False, 'reason': 'smooth_mpc_invalid_action_sequence'}

        error = float(np.max(np.abs(actual - state.last_command)))
        pace = float(np.clip(1. - max(0., error - .02) / .06, .2, 1.))
        nominal_end = self.curve(min(duration, state.time + mpc.SOLVE_DT * pace))
        raw_correction = commands[mpc.STEPS_PER_SOLVE - 1] - nominal_end
        # An already accurate reference requires little feedback. On deviation,
        # admit a bounded correction from the native MPC prediction.
        correction = np.clip(raw_correction, -MAX_CORRECTION_RAD, MAX_CORRECTION_RAD)
        correction *= .65 * np.clip(error / .015, 0., 1.)
        for feedback_scale, pace_scale in ((1., 1.), (.5, .7), (0., .4)):
            applied, after = advance_commands(self.curve, duration, state, correction * feedback_scale,
                                              pace * pace_scale)
            speed = float(np.abs(np.diff(np.vstack([state.last_command, applied]), axis=0)).max()) / DT
            if speed > route.ROUTE_SPEED_RAD_S:
                continue
            points = np.vstack([actual, applied])
            corridor_error = self.route.corridor_error(points)
            if corridor_error > route.ROUTE_CORRIDOR_RAD:
                continue
            alpha = np.arange(5) / 5
            dense = (points[:-1, None] * (1 - alpha[None, :, None])
                     + points[1:, None] * alpha[None, :, None]).reshape(-1, 6)
            dense = np.vstack([dense, applied[-1]])
            tensor = self.torch.as_tensor(dense, dtype=self.torch.float32, device=measured.position.device)
            if not self.context.checker.validate(tensor[:, None]).all():
                continue
            self.stream = after
            self.route.progress = after.time
            return {'success': True, 'settled': False, 'positions': applied.tolist(),
                    'solver_seconds': float(prediction.solve_time), 'collision_checked_samples': len(dense),
                    'joint_error_rad': endpoint_error, 'reference_progress_s': after.time,
                    'reference_pace': after.pace, 'corridor_error_rad': corridor_error,
                    'projected_max_joint_speed_rad_s': speed,
                    'mpc_offset_rad': after.offset.tolist(), 'mpc_offset_velocity_rad_s': after.velocity.tolist(),
                    'feedback_scale': feedback_scale}
        return {'success': False, 'reason': 'smooth_mpc_no_safe_interval',
                'joint_error_rad': endpoint_error, 'reference_progress_s': state.time}


def prepare(out, backend_gpu, scene_gpu, cases):
    source = inspect.getsource(benchmark.prepare)
    old = "canonical + '_ROUTE01'"
    assert source.count(old) == 1
    namespace = {**benchmark.__dict__, 'SELF': SELF}
    exec(compile(source.replace(old, "canonical + '_SMOOTH01'"), str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    dependencies = [Path(m.__file__).resolve() for m in (benchmark, terminal, terminal.reseeded)]
    for path in dependencies:
        shutil.copy2(path, out / 'source' / path.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        reg = benchmark.read(path)
        reg['source_hashes'].update({str(p): benchmark.sha(p) for p in dependencies})
        reg['mpc'].update(timed_reference_feedforward=True, drift_pace_slew_per_s=PACE_SLEW_PER_S,
                          mpc_correction_acceleration_rad_s2=CORRECTION_ACCEL_RAD_S2,
                          mpc_correction_jerk_rad_s3=CORRECTION_JERK_RAD_S3)
        benchmark.save(path, reg)


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
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu, args.case or ('I0117',))
        return
    route.RouteMPC = SmoothRouteMPC
    benchmark.SELF = SELF
    benchmark.main()


if __name__ == '__main__':
    main()
