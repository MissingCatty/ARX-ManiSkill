"""Route MPC with measured reference progress and warm-start recovery."""
import argparse
import inspect
from pathlib import Path
import shutil

import numpy as np

import arx_route_mpc as route
import arx_route_mpc_bench as benchmark

SELF = Path(__file__).resolve()
STALL_SOLVES = 5
STALL_DISPLACEMENT_RAD = .001
REFERENCE_BACKTRACK_S = .4


class MeasuredReference(route.ReferenceRoute):
    def next_target(self, q):
        # Use the nearest portion of the route within the current local window.
        # An arm ahead of or behind the previous clock cannot freeze the clock.
        lo = max(0., self.progress - REFERENCE_BACKTRACK_S)
        hi = min(self.duration, self.progress + route.LOOKAHEAD_S + .1)
        inside = (self.times > lo) & (self.times < hi)
        stamps = np.r_[lo, self.times[inside], hi]
        positions = np.vstack([self.position(lo), self.positions[inside], self.position(hi)])
        distance = np.abs(positions - q).max(axis=1)
        minimum = float(distance.min())
        nearest = float(stamps[np.flatnonzero(distance <= minimum + 1e-8)[-1]])
        if minimum <= route.WAYPOINT_TOLERANCE_RAD:
            self.progress = max(self.progress, nearest)
        target_time = min(self.duration, max(nearest + route.LOOKAHEAD_S,
                                             self.progress + route.mpc.SOLVE_DT))
        return self.position(target_time)


class ReseededRouteMPC(route.RouteMPC):
    def begin_route(self, *args, **kwargs):
        self.seed_position = None
        self.stall_solves = 0
        return super().begin_route(*args, **kwargs)

    def checked_route_commands(self, measured, target):
        actual = measured.position.cpu().numpy().reshape(6)
        at_target = np.max(np.abs(actual - target)) < route.SETTLED_JOINT_ERROR_RAD
        first = self.seed_position is None
        if first or np.max(np.abs(actual - self.seed_position)) >= STALL_DISPLACEMENT_RAD:
            self.seed_position = actual.copy()
            self.stall_solves = 0
        else:
            self.stall_solves += 1
        reseeded = first or (not at_target and self.stall_solves >= STALL_SOLVES)
        if reseeded:
            # Reinitialize the short prediction from measured joints and the
            # next checked reference waypoint. No new global route is sought.
            self.solver.update_current_state(measured.clone())
            self.solver.update_seed_trajectory_from_goal_state(self.joint_state(target, np.zeros(6)))
            self.stall_solves = 0
        return {**super().checked_route_commands(measured, target), 'route_mpc_reseeded': reseeded}


def prepare(out, backend_gpu, scene_gpu, cases):
    source = inspect.getsource(benchmark.prepare)
    old = "canonical + '_ROUTE01'"
    assert source.count(old) == 1
    namespace = {**benchmark.__dict__, 'SELF': SELF}
    exec(compile(source.replace(old, "canonical + '_ROUTE02'"), str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    previous = Path(benchmark.__file__).resolve()
    shutil.copy2(previous, out / 'source' / previous.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        reg = benchmark.read(path)
        reg['source_hashes'][str(previous)] = benchmark.sha(previous)
        reg['mpc'].update(reference_progress_from_measured_joints=True, stall_reseed_solves=STALL_SOLVES,
                          stall_displacement_rad=STALL_DISPLACEMENT_RAD)
        benchmark.save(path, reg)


def install_variant():
    route.ReferenceRoute = MeasuredReference
    route.RouteMPC = ReseededRouteMPC


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
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu, args.case or benchmark.bench.PILOT)
        return
    install_variant()
    benchmark.SELF = SELF
    benchmark.main()


if __name__ == '__main__':
    main()
