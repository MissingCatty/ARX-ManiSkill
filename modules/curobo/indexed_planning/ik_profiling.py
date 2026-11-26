"""Request-scoped IK diagnostics; leave the resident numerical solvers intact."""
from contextlib import contextmanager
import cProfile
from functools import wraps
import pstats
import time

from arx_project.layout import write_json


def cpu_rows(profile):
    rows = []
    for (filename, line, name), (primitive, total, own, cumulative, _) in pstats.Stats(profile).stats.items():
        rows.append({'function': name, 'file': filename, 'line': line, 'calls': total,
                     'primitive_calls': primitive, 'self_seconds': own, 'cumulative_seconds': cumulative})
    return sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)


def device_rows(events):
    """Count device events once, rather than adding their enclosing CPU events."""
    grouped = {}
    for event in events:
        if (getattr(event.device_type, 'name', str(event.device_type)) != 'CUDA'
                or getattr(event, 'is_user_annotation', False)):
            continue
        row = grouped.setdefault(event.name, {'name': event.name, 'calls': 0, 'device_microseconds': 0.})
        row['calls'] += 1
        row['device_microseconds'] += event.time_range.elapsed_us()
    return sorted(grouped.values(), key=lambda row: row['device_microseconds'], reverse=True)


class IKTrace:
    def __init__(self):
        self.enabled = False
        self.active_sample = False
        self.samples, self.sampled_stages, self.events = [], set(), []

    def begin(self, directory, enabled):
        if self.enabled or self.active_sample:
            raise RuntimeError('IK diagnostic requests cannot overlap')
        self.directory, self.enabled = directory, enabled
        self.samples, self.sampled_stages, self.events = [], set(), []

    @contextmanager
    def stage(self, name):
        if not self.active_sample:
            yield
            return
        import torch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        began = time.perf_counter();start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, time.perf_counter()-began, start, end))

    def wrap(self, instance, name, label):
        original = getattr(instance, name)

        @wraps(original)
        def measured(*args, **kwargs):
            value = label(*args, **kwargs) if callable(label) else label
            with self.stage(value):
                return original(*args, **kwargs)
        setattr(instance, name, measured)

    def attach(self, engine):
        for mode, solver in [('world', engine.full_solver), ('pure', engine.pure_solver)]:
            seed = solver.seed_ik_solver
            for name, label in [('solve_batch', 'LM_total'), ('_generate_seed_configs', 'LM_seed_generation'),
                                ('_optimize', 'LM_iterations'), ('_select_top_solutions', 'LM_select')]:
                self.wrap(seed, name, mode+'_'+label)
            self.wrap(solver, 'solve_pose', lambda *a, _mode=mode, **k:
                      _mode+('_refine_total' if k.get('run_optimizer', True) else '_coarse_metrics'))
            self.wrap(solver.optimizer, 'optimize', mode+'_LBFGS_optimize')
            self.wrap(solver.metrics_rollout, 'compute_metrics_from_action', mode+'_metrics')
        original = engine._solve

        @wraps(original)
        def measured(solver, worlds, start, returns, stage):
            if not self.enabled or stage in self.sampled_stages:
                return original(solver, worlds, start, returns, stage)
            self.sampled_stages.add(stage)
            return self.sample(original, solver, worlds, start, returns, stage)
        engine._solve = measured

    def sample(self, original, solver, worlds, start, returns, stage):
        import torch
        from torch.profiler import profile, ProfilerActivity
        ordinal = len(self.samples);prefix = self.directory/('ik_sample_'+str(ordinal))
        write_json(prefix.with_suffix('.inputs.json'), {'stage': stage, 'worlds': worlds,
            'start': start.detach().cpu().tolist(), 'returns': returns,
            'num_seeds': solver.config.num_seeds,
            'LM_num_seeds': solver.seed_ik_solver.config.num_seeds,
            'optimizer_settings': [{name: getattr(config, name, None) for name in
                ['num_iters', 'inner_iters', 'fixed_iters', 'converged_ratio']}
                for config in solver.config.optimizer_configs]})
        self.events = [];cpu = cProfile.Profile();began = time.perf_counter()
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False, profile_memory=False, with_stack=False) as cuda:
                self.active_sample = True;cpu.enable()
                try:
                    result = original(solver, worlds, start, returns, stage)
                finally:
                    cpu.disable();self.active_sample = False
            # The original _solve already waits for its end event. Profiling
            # is confined to these diagnostic requests, not latency trials.
            kernels = device_rows(cuda.events())
            cuda.export_chrome_trace(str(prefix.with_suffix('.trace.json')))
            stats = cpu_rows(cpu)
            stages = [{'stage': name, 'host_wall_seconds': wall,
                       'stream_elapsed_seconds': start.elapsed_time(end)/1000.}
                      for name, wall, start, end in self.events]
            summary = {'stage': stage, 'diagnostic_wall_seconds': time.perf_counter()-began,
                'host_calls': stats[:80], 'substage_events': stages,
                'CUDA_graph_captures': sum(row['calls'] for row in stats
                    if row['function'] == '_initialize_cuda_graph'),
                'device_event_seconds_sum': sum(row['device_microseconds'] for row in kernels)/1e6,
                'device_events': kernels[:60],
                'stream_elapsed_includes_host_gaps': True,
                'profiling_overhead_excluded_from_speedup_trials': True}
            write_json(prefix.with_suffix('.json'), summary)
            self.samples.append(summary)
            return result
        finally:
            self.active_sample = False

    def finish(self):
        self.enabled = False
        return {'samples': self.samples, 'original_solver_methods_called': True,
                'numerical_configuration_changed': False,
                'diagnostics_are_not_optimization_latency_results': True}


def profiled_pool(base):
    class ProfiledPool(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ik_trace = IKTrace()
            for engine in self.batch_engines.values(): self.ik_trace.attach(engine)
    return ProfiledPool
