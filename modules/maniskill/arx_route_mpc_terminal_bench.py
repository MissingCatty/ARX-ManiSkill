"""Complete route tracking using the measured terminal state."""
import argparse
import inspect
from pathlib import Path
import shutil

import numpy as np

import arx_route_mpc_reseed_bench as reseeded

SELF = Path(__file__).resolve()
benchmark = reseeded.benchmark
route = reseeded.route


def at_terminal(reference, q, qd):
    # B-spline paths can contain identical poses at several terminal samples.
    # Once in the last lookahead window, the actual joint state decides arrival.
    return (reference.duration - reference.progress <= route.LOOKAHEAD_S + 1e-8
            and np.max(np.abs(reference.positions[-1] - q)) < route.SETTLED_JOINT_ERROR_RAD
            and max(map(abs, qd)) < route.SETTLED_SPEED_RAD_S)


class TerminalRouteMPC(reseeded.ReseededRouteMPC):
    def route_step(self, directory, phase, q, qd):
        if (self.directory == directory and self.route_stage == phase
                and at_terminal(self.route, q, qd)):
            self.route.progress = self.route.duration
        return super().route_step(directory, phase, q, qd)


def prepare(out, backend_gpu, scene_gpu, cases):
    source = inspect.getsource(benchmark.prepare)
    old = "canonical + '_ROUTE01'"
    assert source.count(old) == 1
    namespace = {**benchmark.__dict__, 'SELF': SELF}
    exec(compile(source.replace(old, "canonical + '_ROUTE03'"), str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    dependencies = [Path(m.__file__).resolve() for m in (benchmark, reseeded)]
    for path in dependencies:
        shutil.copy2(path, out / 'source' / path.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        reg = benchmark.read(path)
        reg['source_hashes'].update({str(p): benchmark.sha(p) for p in dependencies})
        reg['mpc'].update(reference_progress_from_measured_joints=True,
                          stall_reseed_solves=reseeded.STALL_SOLVES,
                          stall_displacement_rad=reseeded.STALL_DISPLACEMENT_RAD,
                          terminal_uses_measured_state=True)
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
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu, args.case or ('I0031',))
        return
    reseeded.install_variant()
    route.RouteMPC = TerminalRouteMPC
    benchmark.SELF = SELF
    benchmark.main()


if __name__ == '__main__':
    main()
