"""One explicitly registered sphere model per resident child process."""
import argparse
from pathlib import Path
import time

from arx_project.layout import read_json,write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker
from modules.curobo.sphere_study.adapter import create
from modules.curobo.sphere_study.tracing import FullTrace


class Worker(OriginalWorker):
    def __init__(self,planner):
        super().__init__(planner);self.trace=FullTrace(planner)

    def request(self,directory,*,diagnostic_profile=False,micro_source=None,ordinal=0,**flags):
        import torch
        if torch.cuda.mem_get_info()[0]/1024**2<1536:raise RuntimeError('Original CUDA request reserve unavailable')
        directory=Path(directory)
        if micro_source is not None:
            from modules.curobo.sphere_study.micro import run
            return run(self.planner,directory,micro_source,ordinal)
        pool=self.planner.pool;pool.stage_timing.reset();pool._pending_ik=[];pool.stage_timing.active=True
        try:
            with self.trace.sample(directory,diagnostic_profile):result=super().request(directory,**flags)
        finally:
            pool.stage_timing.active=False;timings=pool.stage_timing.report()
            write_json(directory/'stage_timings.json',timings)
        result.update(sphere_count=pool.diagnostic_sphere_count,diagnostic_profile=bool(diagnostic_profile),
            stages=timings['stages'],source_robot_and_871_production_configuration_unchanged=True)
        write_json(directory/'planning_metrics.json',result)
        return result


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','socket','warmup-trial','descriptor','profile','reference-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind',default='curobo_dual_arm');args=parser.parse_args()
    import torch
    from arx_project.service import serve
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]/1024**2<10000:raise RuntimeError('Original startup CUDA reserve unavailable')
    torch.zeros(1,device='cuda');torch.cuda.synchronize();began=time.perf_counter();planner=create(args)
    planner.pool.campaign_root=Path(args.profile).parents[2]
    counts={role:int(context.checker.kinematics.kinematics_config.total_spheres)
            for role,context in planner.pool.contexts.items()}
    expected=planner.pool.diagnostic_sphere_count
    if any(n!=expected for n in counts.values()):raise RuntimeError('Native model count differs from registered diagnostic profile')
    write_json(args.root/'startup.json',{'passed':True,'seconds':time.perf_counter()-began,
        'actual_native_counts':counts,'self_collision_enabled':{role:context.checker.self_collision_cost is not None
            for role,context in planner.pool.contexts.items()},'physical_source_profile_count':871,
        'planning_diagnostic_only':True,'physical_grasps':0})
    serve(args,worker=Worker(planner))


if __name__=='__main__':main()
