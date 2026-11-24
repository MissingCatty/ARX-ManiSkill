"""Keep original collision rejection while separating it from service failure."""
import argparse
from pathlib import Path
import time
import traceback

import resident_backend as service

COLLISION_MESSAGE='Measured state collides in unchanged ARX model'


def checked_pool(base):
    class CheckedPool(base):
        def request(self,directory,observe=False,approach=False,lift=False,selector='graspcl'):
            started=time.perf_counter()
            try:
                return super().request(directory,observe=observe,approach=approach,lift=lift,selector=selector)
            except AssertionError as error:
                frames=traceback.extract_tb(error.__traceback__)
                origin=frames[-1]
                if str(error)!=COLLISION_MESSAGE or Path(origin.filename).name!='resident_planner.py' or origin.name!='bind':
                    raise
                directory=Path(directory)
                name='observation_plan' if observe else 'lift_plan' if lift else 'approach_plan' if approach else 'plan'
                result={'success':False,'reason':'measured_start_state_in_collision',
                    'rejection_type':'planning_constraint','original_collision_gate_preserved':True,
                    'wall_seconds':time.perf_counter()-started}
                service.save(directory / (name+'_result.json'),result)
                service.save(directory / (name+'_constraint_rejection.json'),{
                    'assertion':str(error),'traceback':''.join(traceback.format_exception(error)),
                    'gate_relaxed':False,'motion_commanded_by_request':False})
                return result
    return CheckedPool


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['zerograsp','graspcl','curobo'],required=True)
    parser.add_argument('--socket',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--warmup-trial',type=Path,required=True)
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--policy-registration',type=Path)
    args=parser.parse_args()
    if args.kind=='graspcl':
        from native_resident_services import serve_graspcl
        serve_graspcl(args)
    else:
        if args.kind=='zerograsp':
            from postprocess_backend import NativePostNeural
            service.NeuralWorker=NativePostNeural
        else:
            import resident_planner
            from fast_resident_planner import FastPlannerPool
            from postprocess_planner import PostprocessPlanner
            resident_planner.GraspCLPlanner=PostprocessPlanner
            resident_planner.PlannerPool=checked_pool(FastPlannerPool)
        service.serve(args)
