"""Resident V7 pure-planning worker using the unchanged V8 batch16 backend."""
import argparse
from pathlib import Path
import time

from arx_project.layout import read_json,write_json,now


class MeasuredPlanner:
    def __init__(self,planner):self.planner=planner

    def request(self,directory,**options):
        import torch
        from modules.curobo.scene_cache import mesh_stores,prune_inactive_meshes
        from modules.graspgenx.postprocessing.ordering import selection_source
        directory=Path(directory);free,_=torch.cuda.mem_get_info()
        if free/1024**2<1536:raise RuntimeError('Insufficient original CUDA reserve before V7 request')
        torch.manual_seed(20261126)
        for context in self.planner.pool.contexts.values():context.planner.reset_seed()
        self.planner.pool.pure_solver.reset_seed()
        for engine in self.planner.pool.batch_engines.values():
            engine.reset_seed();engine.calls.clear()
        torch.cuda.synchronize();began=time.perf_counter()
        result=self.planner.request(directory,**options)
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        manifest=read_json(directory/'candidate_order.json')
        if result['success']:result['selected'].update(selection_source(manifest,result['selected']))
        result['candidate_search_policy']='NMS_representatives_then_native_score_deferred' if manifest['variant']=='nms15' else 'original_native_score_order'
        write_json(directory/'plan_result.json',result)
        checks=0;rejected={};batches=[]
        for side in ['right','left']:
            path=directory/'arms'/side/'plan_result.json'
            if path.exists():
                arm=read_json(path);checks+=len(arm.get('candidate_events',[]))
                for key,count in arm.get('rejected',{}).items():rejected[key]=rejected.get(key,0)+count
            for path in sorted((directory/'arms'/side/'search_windows').glob('*.json')):
                batch=read_json(path)['result'].get('batch_ik')
                if batch:batches.append({'arm':side,**{k:v for k,v in batch.items() if k!='solver_calls'}})
        metrics={'planning_request_seconds':seconds,'success':result['success'],
            'selected_arm':result.get('selected',{}).get('arm'),
            'selected_original_source_index':result.get('selected',{}).get('original_source_index'),
            'selected_search_priority_rank':result.get('selected',{}).get('index'),
            'selected_priority_phase':result.get('selected',{}).get('priority_phase'),
            'candidate_checks':checks,'rejected':rejected,'batches':batches,
            'request_free_mib':free/1024**2,'reason':result.get('reason','success')}
        calls=[call for engine in self.planner.pool.batch_engines.values() for call in engine.calls]
        metrics['batched_grasp_IK_solver_calls']={name:sum(c['stage']==name for c in calls)
            for name in ['full_world_ik','pure_kinematic_fallback']}
        metrics['batched_grasp_IK_GPU_seconds']=sum(c['gpu_seconds'] for c in calls)
        write_json(directory/'planning_metrics.json',metrics)
        write_json(directory/'IK_solver_calls.json',
            {str(k):v.calls for k,v in self.planner.pool.batch_engines.items()})
        prune_inactive_meshes(mesh_stores(self.planner),torch.cuda.synchronize)
        return {k:v for k,v in metrics.items() if k!='batches'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','socket','warmup-trial','descriptor']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind',default='curobo_indexed_planning');args=parser.parse_args()
    import numpy as np
    import torch
    from modules.curobo.indexed_planning.service import create
    from modules.curobo.indexed_planning.warmup import warm_solver_branches
    from arx_project.service import serve
    torch.set_num_threads(4);free,_=torch.cuda.mem_get_info()
    if free/1024**2<10000:raise RuntimeError('Insufficient original startup CUDA memory before V7 loading')
    torch.zeros(1,device='cuda');torch.cuda.synchronize();begin=time.perf_counter()
    args.ik_batch_size=16;args.warm_batch_sizes=(16,)
    planner=create(args)
    branches=warm_solver_branches(planner,args.warmup_trial/'state.json',args.root/'solver_prewarm')
    write_json(args.root/'startup.json',{'at':now(),'passed':True,'seconds':time.perf_counter()-begin,
        'shared_model_pool':True,'batch_capacity':16,'model_loads':planner.pool.loads,
        'negative_and_partial_batch_rows_verified':branches['passed'],
        'allocated_mib':torch.cuda.memory_allocated()/1024**2,'reserved_mib':torch.cuda.memory_reserved()/1024**2,
        'simulation_instances':0,'physical_grasps':0})
    serve(args,worker=MeasuredPlanner(planner))


if __name__=='__main__':main()
