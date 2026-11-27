"""Route the unchanged selector through lazy endpoint and path batches."""
from pathlib import Path

from arx_project.layout import write_json
from modules.curobo.candidate_io.store import candidate_pool
from .endpoint_windows import EndpointWindow, PreoptimizedPath
from .planning import parallel_pool


def windowed_pool(base):
    class WindowedPool(parallel_pool(base)):
        def __init__(self, *args, **kwargs):
            self.windowed_enabled = False
            self.windowed_path_capacity = 0
            self.windowed_sequence = None
            self.endpoint_engine = None
            self.windowed_statistics = []
            super().__init__(*args, **kwargs)
            for context in self.contexts.values():
                original_ik, original_path, original_optimize = context.ik, context.path, context.optimize
                def ik(world, start, _original=original_ik):
                    seq = self.windowed_sequence
                    return seq.ik(world, start, _original) if seq is not None else _original(world, start)
                def path(start, goal, _context=context, _original=original_path):
                    seq = self.windowed_sequence
                    result = seq.path(_context, start, goal) if seq is not None else None
                    return result if result is not None else _original(start, goal)
                def optimize(label, value, seconds, _original=original_optimize):
                    return value.value if isinstance(value, PreoptimizedPath) else _original(label, value, seconds)
                context.ik, context.path, context.optimize = ik, path, optimize

        def prepare_endpoint_engine(self, context):
            from modules.curobo.indexed_planning.batches import NativeBatchEngine
            if self.endpoint_engine is None:
                self.endpoint_engine = NativeBatchEngine(self, 16)
            return self.endpoint_engine

        def grasp_endpoint(self, context, state, grasp, start, validator):
            sequence = self.windowed_sequence
            if sequence is None:
                return super().grasp_endpoint(context, state, grasp, start, validator)
            self._pending_ik = []
            return sequence.grasp(context, state, grasp, start, validator, super().grasp_endpoint)

        def select_contact(self, context, state, directory, validator):
            if not self.windowed_enabled:
                return super().select_contact(context, state, directory, validator)
            if self.windowed_sequence is not None or self.parallel_capacity:
                raise RuntimeError('Endpoint windows must not enter the old serial collector')
            sequence = EndpointWindow(self, candidate_pool(directory)['candidates'], self.windowed_path_capacity)
            self.windowed_sequence = sequence
            try:
                return super().select_contact(context, state, directory, validator)
            finally:
                self.windowed_sequence = None
                summary = sequence.summary(); self.windowed_statistics.append(summary)
                write_json(Path(directory) / 'windowed_endpoints' / f'{len(self.windowed_statistics):04d}.json', summary)
    return WindowedPool


def create(args):
    from modules.curobo.gripper_sweep.runtime import create as original
    return original(args, pool_extension=windowed_pool)
