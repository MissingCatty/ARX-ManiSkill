"""One warmed planner, three comparison modes, no simulation process."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from arx_project.layout import now,read_json,write_json
from modules.pipeline.experiments.graspgenx_native_postprocessing.worker import Worker as OriginalWorker


def preflight(planner,warmup,root):
    import torch
    from resident_planner import collision_scene_config
    from modules.graspgenx.native_postprocessing.width_planning import target_state
    reports=[]
    for side in ['right','left']:
        state=planner.model.bind(read_json(warmup/'state.json'),side,root/side/'physical_arm')
        state=target_state(state,.088)
        context=planner.pool.bind(root/side/'bound',state,include_target=False)
        validator=planner.pool.contact_view(context,state)
        session=planner.pool.prepare_analytic(context,state)
        # SAPIEN measurements can sit a few nanoradians outside a hard limit.
        # The existing planner uses its recorded 1e-4-rad legal inset; verify
        # analytic recovery against FK of that exact, unchanged start state.
        current=context.measured(state)
        world=validator.geometry.frames(current.cpu().numpy()[None])['right_eef_link'][0]
        targets=(context.invbase@world)[None]
        first=session.host_query(targets,np.array([.088]))
        if not first['valid_ik'].any():raise RuntimeError('Analytic prewarm lost the recorded current pose')
        values=first['q'][0,first['valid_ik'][0]]
        q=torch.as_tensor(values,dtype=torch.float32,device='cuda')
        native=context.checker.validate(q[:,None]).reshape(-1).cpu().numpy()
        if not np.array_equal(native,first['valid_world'][0,first['valid_ik'][0]]):
            raise RuntimeError('Resident analytic collision differs from original locked-width checker')
        closest=int(np.argmin(np.square(first['q'][0]-current.cpu().numpy()).sum(1)))
        if not first['valid_ik'][0,closest]:raise RuntimeError('Prewarm did not recover the measured branch')
        sphere=context.checker.get_kinematics(current[None,None]).robot_spheres[0,0]
        sphere=sphere[sphere[:,3]>0][0].cpu().numpy()
        scene=read_json(context.out/'scene.json');blocked=deepcopy(scene)
        blocked['cuboid']['analytic_preflight_probe']={'pose':[*sphere[:3].tolist(),1.,0.,0.,0.],
            'dims':[2*float(sphere[3])+.01]*3}
        context.checker.clear_scene_cache();context.checker.update_world(collision_scene_config(blocked))
        session.verify_shared_scene(context.checker.scene_model)
        obstructed=session.host_query(targets,np.array([.088]))
        if obstructed['valid_world'][0,closest]:raise RuntimeError('Captured analytic graph ignored the changed world')
        context.checker.clear_scene_cache();context.checker.update_world(collision_scene_config(scene))
        session.verify_shared_scene(context.checker.scene_model)
        restored=session.host_query(targets,np.array([.088]))
        if any(not np.array_equal(first[k],restored[k]) for k in first):
            raise RuntimeError('Native scene restore changed analytic outputs')
        reports.append({'arm':side,'passed':True,'current_pose_recovered':True,
            'native_locked_width_masks_match':True,'scene_change_observed':True,'scene_restore_exact':True,
            'native_parent_scene_shared':True,
            'model_and_graph_rebuilt_on_world_update':False})
    write_json(root/'acceptance.json',{'at':now(),'passed':True,'reports':reports,
        'simulation_instances':0,'physical_grasps':0})


class Worker(OriginalWorker):
    def request(self,directory,*,variant,**flags):
        import torch
        from modules.curobo.scene_cache import prune_inactive_meshes
        if variant not in ['coarse_numeric','fine_numeric','fine_analytic']:raise ValueError('Unknown comparison variant')
        pool=self.planner.pool;pool.analytic_enabled=variant=='fine_analytic'
        pool.analytic_statistics=[];pool.stage_timing.reset();pool._pending_ik=[];pool.stage_timing.active=True
        pool.binding_codec.reset()
        try:result=super().request(directory,**flags)
        finally:
            pool.stage_timing.active=False;timings=pool.stage_timing.report()
            write_json(Path(directory)/'stage_timings.json',timings)
        result.update(mode=variant,stages=timings['stages'],analytic_windows=pool.analytic_statistics,
            original_V3_scalar_interleaved_planner=variant!='fine_analytic')
        stores=[s._collision.checker.scene_model.data.meshes for s in pool.analytic_sessions.values()]
        result['analytic_mesh_cache']=prune_inactive_meshes([s for s in stores if s is not None],torch.cuda.synchronize)
        write_json(Path(directory)/'planning_metrics.json',result)
        return result


def main():
    p=argparse.ArgumentParser()
    for name in ['root','socket','warmup-trial','descriptor']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',default='curobo_dual_arm');args=p.parse_args()
    import torch
    from arx_project.service import serve
    from modules.curobo.analytic_ik.runtime import create
    torch.set_num_threads(4)
    free=torch.cuda.mem_get_info()[0]/1024**2
    if free<10000:raise RuntimeError('Original startup GPU reserve unavailable')
    probe=torch.arange(1024,device='cuda',dtype=torch.float32)
    if float((probe*2).sum())!=1047552.:raise RuntimeError('CUDA arithmetic probe failed')
    began=time.perf_counter();planner=create(args)
    preflight(planner,args.warmup_trial,args.root/'analytic_preflight')
    write_json(args.root/'startup.json',{'at':now(),'passed':True,'seconds':time.perf_counter()-began,
        'GPU_free_mib_before':free,'analytic_sessions':len(planner.pool.analytic_sessions),
        'same_warmed_solver_for_three_modes':True,'physical_grasps':0})
    serve(args,worker=Worker(planner))


if __name__=='__main__':main()
