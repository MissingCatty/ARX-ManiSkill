"""Capture a real uncached collision evaluation from the first full IK batch."""
from functools import wraps
import inspect

from .measure import measure


class QueryCapture:
    def __init__(self, scene):
        self.scene = scene
        self.enabled = self.sampling = self.sampled = False
        self.snapshots, self.results = [], []
        original = scene.checker.get_sphere_distance
        signature = inspect.signature(original)

        @wraps(original)
        def watched(*args, **kwargs):
            import torch
            if self.sampling and not self.snapshots and not torch.cuda.is_current_stream_capturing():
                values = signature.bind(*args, **kwargs);values.apply_defaults();v = values.arguments
                if v['env_query_idx'] is not None and bool(v['env_query_idx'].any()):
                    raise ValueError('Actual query unexpectedly uses multiple environments')
                self.snapshots.append({'spheres': v['query_sphere'].detach().clone(),
                    'eta': v['activation_distance'].detach().clone(), 'weight': v['weight'].detach().clone(),
                    'env_index': None if v['env_query_idx'] is None else v['env_query_idx'].detach().clone(),
                    'return_loss': v['return_loss'], 'scene_bytes': self.scene_bytes})
            return original(*args, **kwargs)
        scene.checker.get_sphere_distance = watched

    def begin(self, directory, enabled):
        if self.enabled or self.sampling: raise RuntimeError('Query capture requests cannot overlap')
        self.directory, self.enabled = directory, enabled
        self.sampled = False;self.snapshots, self.results = [], []

    def attach(self, engine):
        original = engine._solve

        @wraps(original)
        def measured(solver, worlds, start, returns, stage):
            if not self.enabled or self.sampled or stage != 'full_world_ik':
                return original(solver, worlds, start, returns, stage)
            self.sampled = True;self.sampling = True
            self.scene_bytes = (engine.context.out/'scene.json').read_bytes()
            try:
                result = original(solver, worlds, start, returns, stage)
            finally:
                self.sampling = False
            if not self.snapshots: raise RuntimeError('No uncached actual full-IK collision call was captured')
            self.results.append(measure(self.snapshots[0], self.scene, self.directory/'query_components'))
            self.snapshots.clear()
            return result
        engine._solve = measured

    def finish(self):
        self.enabled = False
        return {'samples': self.results, 'original_solver_called': True,
                'probe_outputs_never_enter_planning': True, 'diagnostic_not_speedup_trial': True}


def capture_pool(base):
    class CapturePool(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.query_capture = QueryCapture(self.contexts['open'].planner.scene_collision_checker)
            for engine in self.batch_engines.values(): self.query_capture.attach(engine)
    return CapturePool
