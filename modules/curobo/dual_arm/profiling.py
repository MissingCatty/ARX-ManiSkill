"""Optional synchronized solver-API timing; no solver settings are changed."""
from functools import wraps
import time

import numpy as np


class SolverTrace:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.records, self.wrapped = [], set()

    def wrap(self, owner, name, component, directory=lambda: None):
        key = (id(owner), name)
        if key in self.wrapped:
            return
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            self.synchronize()
            started = time.perf_counter()
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                self.synchronize()
                self.records.append({'component': component, 'seconds': time.perf_counter()-started,
                                     'directory': str(directory()), 'exception': failed})
        setattr(owner, name, measured)
        self.wrapped.add(key)

    def install(self, pool):
        for context in pool.contexts.values():
            where = lambda context=context: context.out
            self.wrap(context.planner.ik_solver, 'solve_pose', 'native_ik', where)
            self.wrap(context.planner.graph_planner, 'find_path', 'graph_path', where)
            self.wrap(context.planner.trajopt_solver, 'solve_cspace', 'trajectory_optimization', where)
        if pool.pure_solver is not None:
            self.wrap(pool.pure_solver, 'solve_pose', 'kinematic_ik_fallback')

    def reset(self):
        self.records.clear()

    def report(self):
        summary = {}
        for key in sorted({r['component'] for r in self.records}):
            values = np.array([r['seconds'] for r in self.records if r['component'] == key])
            summary[key] = {'calls': len(values), 'total_seconds': float(values.sum()),
                            'median_ms': float(np.median(values)*1000),
                            'p95_ms': float(np.quantile(values, .95)*1000)}
        return {'scope': 'CUDA-synchronized solver API wall time, including CPU dispatch; not kernel-only time',
                'components': summary, 'calls': list(self.records)}
