"""Profiler-only requests with separate CPU and device event accounting."""
from contextlib import contextmanager
import cProfile
from functools import wraps
import time

from arx_project.layout import write_json
from modules.curobo.indexed_planning.ik_profiling import cpu_rows,device_rows


class FullTrace:
    def __init__(self,planner):
        self.enabled=False
        for obj,names in [(planner,['opening_sweep','prepare_arm']),
                          (planner.pool,['bind','contact_view','grasp_endpoint','validate_narrowing'])]:
            for name in names:self.wrap(obj,name)
        for context in planner.pool.contexts.values():
            for name in ['ik','path','optimize']:self.wrap(context,name)

    def wrap(self,obj,name):
        original=getattr(obj,name)
        @wraps(original)
        def call(*args,**kwargs):
            if not self.enabled:return original(*args,**kwargs)
            from torch.profiler import record_function
            with record_function('ARX_'+name):return original(*args,**kwargs)
        setattr(obj,name,call)

    @contextmanager
    def sample(self,root,enabled):
        if not enabled:
            yield;return
        from torch.profiler import profile,ProfilerActivity
        cpu=cProfile.Profile();began=time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,
                     profile_memory=False,with_stack=False) as trace:
            self.enabled=True;cpu.enable()
            try:yield
            finally:cpu.disable();self.enabled=False
        trace.export_chrome_trace(str(root/'full_planning.trace.json'))
        kernels=device_rows(trace.events())
        write_json(root/'full_planning_profile.json',{'passed':True,'diagnostic_wall_seconds':time.perf_counter()-began,
            'host_calls':cpu_rows(cpu)[:160],'host_self_calls':sorted(cpu_rows(cpu),key=lambda r:r['self_seconds'],reverse=True)[:100],
            'device_events':kernels,'device_event_seconds_sum':sum(r['device_microseconds'] for r in kernels)/1e6,
            'profiling_overhead_excluded_from_latency_comparison':True})
