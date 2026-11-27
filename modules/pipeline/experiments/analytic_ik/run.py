"""Check analytical coverage, then time resident GPU IK and native collision."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.analytic_ik.solver import BatchAnalyticIK
from modules.curobo.analytic_ik.collision import BranchCollisionChecker
from modules.curobo.indexed_planning.geometric.kinematics import ArmChain
from .prepare import guard
from .timing import measure,host_roundtrip


def independent(targets,q,mask,chain):
    ids,branches=np.nonzero(mask)
    values=q[ids,branches].astype(float)
    if not len(values):return {'branches':0,'maximum_position_error_m':0.,'maximum_rotation_error_rad':0.}
    if not ((values>=chain.limits[:,0])&(values<=chain.limits[:,1])).all():
        raise ValueError('Executable branch violates physical joint bounds')
    actual=chain.fk(values)
    position=np.linalg.norm(actual[:,:3,3]-targets[ids,:3,3],axis=-1)
    rotation=Rotation.from_matrix(actual[:,:3,:3]@targets[ids,:3,:3].transpose(0,2,1)).magnitude()
    if (position>.002).any() or (rotation>.03).any():raise ValueError('Independent URDF FK rejected an analytic branch')
    return {'branches':len(values),'maximum_position_error_m':float(position.max()),
        'maximum_rotation_error_rad':float(rotation.max())}


def cpu_validation(root):
    summaries=[]
    for arm in ['right','left']:
        folder=root/'random'/arm;data=np.load(folder/'states.npz',allow_pickle=False)
        chain=ArmChain(arm);solver=BatchAnalyticIK(arm,dtype=torch.float64)
        began=time.perf_counter();result=solver.solve(torch.as_tensor(data['targets']));seconds=time.perf_counter()-began
        q=result['q'].numpy();mask=result['valid'].numpy()
        proof=independent(data['targets'],q,mask,chain)
        original=np.linalg.norm(q-data['q'][:,None],axis=-1)
        recovered=np.where(mask,original,np.inf).min(-1)<1e-6
        if not recovered.all():raise ValueError('A known legal random branch was lost')
        reference=read_json(folder/'EAIK_reference.json');expected=0;matched=0
        for index,row in enumerate(reference['answers']):
            for value,pe,re in zip(row['q'],row['position_errors_m'],row['rotation_errors_rad']):
                if pe>1e-7 or re>1e-7:continue
                expected+=1
                if (np.linalg.norm(q[index][mask[index]]-value,axis=-1)<1e-6).any():matched+=1
        if expected!=matched:raise ValueError('An exact EAIK reference branch was lost')
        summaries.append({'arm':arm,'random_targets':len(data['q']),'known_branches_recovered':int(recovered.sum()),
            'exact_EAIK_reference_branches':expected,'exact_EAIK_branches_matched':matched,
            'CPU_tensor_seconds':seconds,'reference_CPU_query_seconds':reference['wall_seconds'],
            'reference_CPU_targets':reference['targets'],'independent_FK':proof})
    write_json(root/'CPU_validation.json',{'at':now(),'passed':True,'arms':summaries})
    return summaries


def locked_width_equivalence(engine,config,chain):
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    rng=np.random.default_rng(811)
    values=rng.uniform(chain.limits[:,0]+.001,chain.limits[:,1]-.001,(16,6)).astype(np.float32)
    q=torch.as_tensor(values,device='cuda')
    maximum=0.;rows=[]
    for width in [.088,.06,.04]:
        current=deepcopy(config)
        for name in ['right_joint27','right_joint28']:current['robot_cfg']['kinematics']['lock_joints'][name]=width/2
        locked=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=current,
            scene_collision_checker=engine.checker.scene_model,collision_activation_distance=0.))
        q8=torch.cat([q,torch.full((len(q),2),width/2,device='cuda')],-1)[:,None]
        a=engine.checker.get_kinematics(q8);b=locked.get_kinematics(q[:,None])
        error=float((a.robot_spheres-b.robot_spheres).abs().max().item());maximum=max(error,maximum)
        world_a=engine.checker.get_collision_constraint(a).sum(-1)==0
        world_b=locked.get_collision_constraint(b).sum(-1)==0
        self_a=engine.checker.get_self_collision(a.robot_spheres).sum(-1)==0
        self_b=locked.get_self_collision(b.robot_spheres).sum(-1)==0
        world_same=torch.equal(world_a,world_b);self_same=torch.equal(self_a,self_b)
        if error>2e-6 or not world_same or not self_same:
            raise ValueError('Per-candidate gripper FK/collision differs from the locked-width reference')
        rows.append({'width_m':width,'joint_states':len(q),'sphere_coordinate_max_error_m':error,
            'world_masks_equal':world_same,'self_masks_equal':self_same})
        del locked
    return {'passed':True,'states':48,'maximum_sphere_difference_m':maximum,'rows':rows}


def run(root):
    reg=guard(root);cfg=reg['config']
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Do not restart a claimed benchmark')
    write_json(root/'started.json',{'at':now(),'GPU_uuid':reg['gpu']})
    torch.set_num_threads(1)
    cpu_validation(root)
    free=torch.cuda.mem_get_info()[0]/1024**2
    if free<cfg['startup_free_mib']:raise RuntimeError('Original startup memory gate unavailable')
    probe=torch.arange(1024,device='cuda',dtype=torch.float32)
    if float((probe*2).sum().item())!=1047552.:raise RuntimeError('CUDA arithmetic probe failed')
    write_json(root/'resource_probe.json',{'at':now(),'passed':True,'free_mib':free,
        'GPU_uuid':reg['gpu'],'device':torch.cuda.get_device_name(0)})
    pure=[]
    for arm in ['right','left']:
        data=np.load(root/'random'/arm/'states.npz',allow_pickle=False)
        solver=BatchAnalyticIK(arm,device='cuda:0',dtype=torch.float32)
        tensor=torch.as_tensor(data['targets'].astype(np.float32),device='cuda')
        answer=solver.solve(tensor);torch.cuda.synchronize()
        masks=answer['valid'].cpu().numpy();q=answer['q'].cpu().numpy()
        proof=independent(data['targets'],q,masks,ArmChain(arm))
        complete=masks.any(-1);fallback=answer['needs_fallback'].cpu().numpy()
        if (~complete&~fallback).any():raise ValueError('An unresolved target lacks its required fallback flag')
        targets=tensor[:cfg['batch_capacity']].clone()
        timing=measure({'analytic_256x8':lambda:solver.solve(targets)},cfg['GPU_graph_repeats'],cfg['GPU_eager_repeats'])
        pure.append({'arm':arm,'targets':len(tensor),'targets_with_valid_branch':int(complete.sum()),
            'fallback_required':int(fallback.sum()),'no_valid_branch':int((~complete).sum()),
            'independent_FK':proof,'timing':timing})
        write_json(root/'GPU_pure_analytic.json',{'at':now(),'passed':True,'arms':pure})
        del solver,tensor,answer
    rows=[];solver=BatchAnalyticIK(device='cuda:0',dtype=torch.float32);chain=ArmChain()
    for item in reg['cases']:
        guard(root)
        if torch.cuda.mem_get_info()[0]/1024**2<cfg['minimum_request_free_mib']:
            raise RuntimeError('Original per-batch memory gate unavailable')
        directory=ROOT/item['directory'];claim=directory/'claimed.json'
        if claim.exists():raise ValueError('Collision case already claimed; do not retry')
        write_json(claim,{'at':now(),'state':'claimed'})
        write_json(root/'progress.json',{'at':now(),'phase':'GPU_collision','current':item,
            'completed_collision_cases':len(rows),'collision_cases':len(reg['cases'])})
        raw=np.load(directory/'targets.npz',allow_pickle=False)
        targets=torch.as_tensor(raw['targets'].astype(np.float32),device='cuda')
        widths=torch.as_tensor(raw['widths'].astype(np.float32),device='cuda')
        real=torch.as_tensor(raw['input_valid'],device='cuda')
        config=yaml.load((directory/'arx.yml').read_text(),Loader=yaml.CSafeLoader)
        engine=BranchCollisionChecker(config,read_json(directory/'scene.json'))
        equivalent=locked_width_equivalence(engine,config,chain)
        first=solver.solve(targets);mask=first['valid']&real[:,None]
        values=first['q'].detach().clone();frozen_mask=mask.detach().clone()
        collision=engine.check(values,frozen_mask,widths)
        proof=independent(raw['targets'],values.cpu().numpy(),frozen_mask.cpu().numpy(),chain)
        def complete(with_self):
            answer=solver.solve(targets)
            return answer,engine.check(answer['q'],answer['valid']&real[:,None],widths,include_self=with_self)
        functions={'analytic_256x8':lambda:solver.solve(targets),
            'native_FK_world_2048':lambda:engine.check(values,frozen_mask,widths,include_self=False),
            'native_FK_world_self_2048':lambda:engine.check(values,frozen_mask,widths),
            'analytic_and_FK_world':lambda:complete(False),
            'analytic_and_FK_world_self':lambda:complete(True)}
        timing=measure(functions,cfg['GPU_graph_repeats'],cfg['GPU_eager_repeats'])
        transfer=host_roundtrip(solver,engine,raw['targets'].astype(np.float32),
            raw['widths'].astype(np.float32),real,cfg['GPU_eager_repeats'])
        np.savez_compressed(directory/'branches.npz',q=values.cpu().numpy(),valid=frozen_mask.cpu().numpy(),
            world_clear=collision['world_clear'].cpu().numpy(),self_clear=collision['self_clear'].cpu().numpy(),
            valid_world_only=collision['valid_world_only'].cpu().numpy(),
            valid_world_and_self=collision['valid_world_and_self'].cpu().numpy(),
            fallback_required=(first['needs_fallback']&real).cpu().numpy())
        row={**item,'passed':True,'timing':timing,'host_roundtrip':transfer,
            'analytic_targets_with_legal_branch':int((frozen_mask.any(-1)&real).sum().item()),
            'valid_analytic_branches':int(frozen_mask.sum().item()),
            'world_clear_branches':int(collision['valid_world_only'].sum().item()),
            'world_and_self_clear_branches':int(collision['valid_world_and_self'].sum().item()),
            'fallback_targets':int((first['needs_fallback']&real).sum().item()),
            'native_sphere_count':int(engine.checker.kinematics.total_spheres),
            'native_FK_measured_coordinates':8,'planning_arm_DOF':6,
            'width_equivalence':equivalent,'independent_FK':proof,
            'V2_table_narrow_phase_not_applied_to_these_native_masks':True,
            'source_active_self_collision_setting':False,'diagnostic_world_self_mask_also_available':True,
            'branch_receipt_sha256':sha256(directory/'branches.npz')}
        write_json(directory/'metrics.json',row);rows.append(row)
        write_json(claim,{'at':now(),'state':'complete','metrics_sha256':sha256(directory/'metrics.json')})
        print('ANALYTIC_BATCH',len(rows),item['case'],item['arm'],'IK_ms',
            round(timing['analytic_256x8']['graph_device']['median_ms'],4),'combined_world_self_ms',
            round(timing['analytic_and_FK_world_self']['graph_device']['median_ms'],4),flush=True)
        del engine,first,collision,values,frozen_mask,functions
        torch.cuda.synchronize()
    guard(root)
    write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'record_integrity':True,
        'collision_cases':len(rows),'pure_analytic':pure,'cases':rows,
        'planning_requests':0,'physical_grasps':0,'simulation_instances':0,
        'production_defaults_changed':False})
    write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_collision_cases':len(rows),
        'collision_cases':len(reg['cases'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=ensure_run_path(args.root)
    try:run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
