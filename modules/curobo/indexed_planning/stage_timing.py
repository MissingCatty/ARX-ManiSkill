"""Request-scoped nested wall timings for optional V8 experiments."""
from contextlib import contextmanager
from functools import wraps
import time


class StageTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.active = False
        self.reset()

    def reset(self):
        if getattr(self, 'stack', []):
            raise RuntimeError('Cannot reset timings during a planning stage')
        self.stack, self.rows = [], []

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        frame = {'stage': name, 'start': self.clock(), 'children': 0.}
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock()-frame['start']
            if self.stack.pop() is not frame:
                raise RuntimeError('Planning timing stack was corrupted')
            self.rows.append({'stage': name, 'wall_seconds': elapsed,
                              'exclusive_seconds': max(0., elapsed-frame['children'])})
            if self.stack:
                self.stack[-1]['children'] += elapsed

    def report(self):
        if self.stack:
            raise RuntimeError('Cannot publish an unfinished timing stage')
        summary = {}
        for row in self.rows:
            out = summary.setdefault(row['stage'], {'calls': 0, 'wall_seconds': 0., 'exclusive_seconds': 0.})
            out['calls'] += 1
            for key in ['wall_seconds', 'exclusive_seconds']:
                out[key] += row[key]
        return {'stages': summary, 'events': list(self.rows),
                'timing_scope': 'host_wall_nested_inclusive_and_exclusive',
                'additional_device_synchronizations': 0}


def wrap_method(instance, name, timer, label):
    """Decorate this resident instance only; no shared classes are patched."""
    original = getattr(instance, name)

    @wraps(original)
    def measured(*args, **kwargs):
        stage = label(*args, **kwargs) if callable(label) else label
        with timer.stage(stage):
            return original(*args, **kwargs)
    setattr(instance, name, measured)


def measured_pool(base):
    class MeasuredPool(base):
        def __init__(self, *args, **kwargs):
            self.stage_timing = StageTiming()
            self._timed_validators = set()
            self._following_ik_stages = []
            super().__init__(*args, **kwargs)
            for context in self.contexts.values():
                wrap_method(context, 'ik', self.stage_timing, self._ik_stage)
                wrap_method(context, 'path', self.stage_timing, 'PRM_path')
                wrap_method(context, 'optimize', self.stage_timing, lambda label, *a, **k: 'trajectory_'+label)
            for engine in self.batch_engines.values():
                for name, label in [('bind', 'batch_world_bind'), ('native', 'grasp_batch_full_world_IK'),
                                    ('pure', 'grasp_batch_kinematic_IK')]:
                    wrap_method(engine, name, self.stage_timing, label)

        def _ik_stage(self, *args, **kwargs):
            if any(item['stage'] == 'grasp_endpoint' for item in self.stage_timing.stack):
                return 'grasp_scalar_IK'
            return self._following_ik_stages.pop(0) if self._following_ik_stages else 'other_scalar_IK'

        def bind(self, *args, **kwargs):
            with self.stage_timing.stage('resident_world_bind'):
                return super().bind(*args, **kwargs)

        def contact_view(self, *args, **kwargs):
            with self.stage_timing.stage('contact_world_bind'):
                validator = super().contact_view(*args, **kwargs)
                if id(validator) not in self._timed_validators:
                    wrap_method(validator.geometry, 'frames', self.stage_timing, 'independent_FK')
                    wrap_method(validator, 'validate', self.stage_timing, 'local_collision_validate')
                    self._timed_validators.add(id(validator))
                return validator

        def grasp_endpoint(self, *args, **kwargs):
            self._following_ik_stages = []
            with self.stage_timing.stage('grasp_endpoint'):
                result = super().grasp_endpoint(*args, **kwargs)
            if result[0] is not None:
                self._following_ik_stages = ['pregrasp_scalar_IK', 'lift_scalar_IK']
            return result

        def select_contact(self, *args, **kwargs):
            with self.stage_timing.stage('candidate_search'):
                return super().select_contact(*args, **kwargs)
    return MeasuredPool
