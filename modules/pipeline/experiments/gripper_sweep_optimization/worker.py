"""One original solver pool, selectable scalar/batched gripper checks."""
import argparse
from pathlib import Path
import time

from arx_project.layout import now,write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker


class Worker(OriginalWorker):
    def request(self,directory,*,variant,**flags):
        if variant not in ['legacy','batched']:raise ValueError('Unknown sweep mode')
        pool=self.planner.pool;pool.batch_sweep_enabled=variant=='batched'
        pool.stage_timing.reset();pool._pending_ik=[];pool.stage_timing.active=True
        try:result=super().request(directory,**flags)
        finally:
            pool.stage_timing.active=False;timings=pool.stage_timing.report()
            write_json(Path(directory)/'stage_timings.json',timings)
        result.update(variant=variant,stages=timings['stages'],batched_gripper_sweeps=pool.batch_sweep_enabled)
        write_json(Path(directory)/'planning_metrics.json',result)
        return result


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','socket','warmup-trial','descriptor']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind',default='curobo_dual_arm');args=parser.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.gripper_sweep.adapter import create
    from .preflight import run
    torch.set_num_threads(4);free,_=torch.cuda.mem_get_info()
    if free/1024**2<10000:raise RuntimeError('Insufficient original startup CUDA reserve')
    torch.zeros(1,device='cuda');torch.cuda.synchronize();began=time.perf_counter()
    planner=create(args)
    checked=run(planner,args.warmup_trial,args.root/'sweep_preflight')
    write_json(args.root/'startup.json',{'at':now(),'passed':True,'seconds':time.perf_counter()-began,
        'preflight_cases':len(checked),'shared_original_solver':True,'physical_grasps':0})
    serve(args,worker=Worker(planner))


if __name__=='__main__':main()
