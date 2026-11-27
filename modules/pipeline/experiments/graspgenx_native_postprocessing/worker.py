"""One resident, original scalar V3 planner shared by all postprocessing groups."""
import argparse
from pathlib import Path
import time

from arx_project.layout import read_json,write_json


class Worker:
    def __init__(self,planner):self.planner=planner

    def request(self,directory,**flags):
        import torch
        from modules.curobo.scene_cache import mesh_stores,prune_inactive_meshes
        directory=Path(directory);pool=self.planner.pool
        free,_=torch.cuda.mem_get_info()
        if free/1024**2<1536:raise RuntimeError('Insufficient original request CUDA reserve')
        torch.manual_seed(20261126)
        for context in pool.contexts.values():context.planner.reset_seed()
        pool.pure_solver.reset_seed()
        torch.cuda.synchronize();begin=time.perf_counter()
        result=self.planner.request(directory,**flags)
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        if result['success']:
            payload=read_json(directory/'candidates/candidate_pool.json');selected=result['selected']
            source=payload['candidates'][selected['index']];meta=source['metadata']
            if abs(selected['open_finger_m']-float(source['width'])/2)>1e-12:
                raise RuntimeError('Selected opening differs from the delivered candidate width')
            selected.update(original_native_rank=meta.get('original_native_rank',selected['index']),
                source_generator_index=meta['source_generator_index'],
                postprocessing=meta.get('postprocessing','control'))
            write_json(directory/'plan_result.json',result)
        checks=0;rejected={}
        for side in ['right','left']:
            path=directory/'arms'/side/'plan_result.json'
            if not path.exists():continue
            arm=read_json(path);checks+=len(arm.get('candidate_events',[]))
            for name,value in arm.get('rejected',{}).items():rejected[name]=rejected.get(name,0)+value
        metrics={'success':bool(result['success']),'reason':result.get('reason','success'),
            'planning_seconds':elapsed,'selected_original_rank':result.get('selected',{}).get('original_native_rank'),
            'selected_arm':result.get('selected',{}).get('arm'),'candidate_checks':checks,'rejected':rejected,
            'request_free_mib':free/1024**2,'original_V3_scalar_interleaved_planner':True}
        write_json(directory/'planning_metrics.json',metrics)
        prune_inactive_meshes(mesh_stores(self.planner),torch.cuda.synchronize)
        return metrics


def main():
    p=argparse.ArgumentParser()
    for name in ['root','socket','warmup-trial','descriptor']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',default='curobo_dual_arm');args=p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.graspgenx.native_postprocessing.width_planning import create
    torch.set_num_threads(4);free,_=torch.cuda.mem_get_info()
    if free/1024**2<10000:raise RuntimeError('Insufficient original startup CUDA reserve')
    torch.zeros(1,device='cuda');torch.cuda.synchronize()
    planner=create(args)
    from modules.graspgenx.native_postprocessing.width_planning import target_state,binding_proof
    reports=[]
    for side in ['right','left']:
        original=read_json(args.warmup_trial/'state.json')
        for width in [.088,.060,.040]:
            directory=args.root/'width_prewarm'/side/f'{int(width*1000):03d}'
            canonical=planner.model.bind(original,side,directory/'physical_arm')
            canonical=target_state(canonical,width)
            context=planner.pool.bind(directory,canonical,include_target=False)
            validator=planner.pool.contact_view(context,canonical)
            proof=binding_proof(validator,width,context.model)
            if 'native_GPU_finger_locks_m' not in proof:raise RuntimeError('Missing actual GPU width proof')
            reports.append({'arm':side,**proof})
    write_json(args.root/'width_prewarm.json',{'passed':True,'cases':6,'reports':reports,
        'original_EEF_pose_and_joint_limits_verified':True,'physical_grasps':0})
    write_json(args.root/'startup.json',{'passed':True,'shared_planner':True,'batch_IK_enabled':False,
        'opening_and_closed_models_warmed_both_arms':True,'model_loads':planner.pool.loads,
        'simulation_instances':0,'physical_grasps':0})
    serve(args,worker=Worker(planner))


if __name__=='__main__':main()
