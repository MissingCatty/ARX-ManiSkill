"""Run the low-latency smooth MPC pilot with the live refinement callbacks."""
import inspect
from pathlib import Path
import shutil
from types import FunctionType
from unittest.mock import patch

import arx_route_mpc_latency_bench as latency

SELF = Path(__file__).resolve()
benchmark = latency.benchmark
base_prepare = latency.prepare
compile_video = latency.video_timing_execute


def video_timing_execute():
    staged = compile_video()
    hooks = {name: staged.__globals__[name] for name in ('VideoQueue', 'record_frames')}
    globals_for_execution = latency.timing.__dict__
    execute = FunctionType(staged.__code__, globals_for_execution, staged.__name__)

    # Keep the original live globals so table_truncation_bench can install its
    # refinement and rescore hooks immediately before invoking timing.execute.
    def execute_video(trial, entry_path):
        with patch.dict(globals_for_execution, hooks):
            return execute(trial, entry_path)

    return execute_video


def prepare(out, backend_gpu, scene_gpu, cases):
    source = latency.substitute(inspect.getsource(base_prepare), "'_LATENCY01'", "'_LATENCY02'")
    namespace = {**latency.__dict__, 'SELF': SELF}
    exec(compile(source, str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    original = Path(latency.__file__).resolve()
    shutil.copy2(original, out / 'source' / original.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        registration = benchmark.read(path)
        registration['source_hashes'][str(original)] = benchmark.sha(original)
        registration['runtime_optimization']['preserve_live_refinement_callbacks'] = True
        benchmark.save(path, registration)


def main():
    with patch.multiple(latency, SELF=SELF, prepare=prepare, video_timing_execute=video_timing_execute):
        latency.main()


if __name__ == '__main__':
    main()
