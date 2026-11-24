"""Offline decomposition of saved ARX grasp IK and original table collisions."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from curobo.types import JointState,GoalToolPose

import graspcl_pipeline as common
import resident_planner


def goals(context,poses):
    local=np.asarray([context.invbase @ pose for pose in poses])
    quaternions=Rotation.from_matrix(local[:,:3,:3]).as_quat()[:,[3,0,1,2]]
    return GoalToolPose(tool_frames=['right_eef_link'],
        position=torch.tensor(local[:,:3,3],dtype=torch.float32,device='cuda').reshape(-1,1,1,1,3),
        quaternion=torch.tensor(quaternions,dtype=torch.float32,device='cuda').reshape(-1,1,1,1,4))


def solve(solver,context,poses,start,returns=32):
    qs=[];valid=[];position_error=[];orientation_error=[]
    for offset in range(0,len(poses),64):
        part=poses[offset:offset+64]
        state=JointState.from_position(start[None].repeat(len(part),1),joint_names=resident_planner.frozen.NAMES)
        result=solver.solve_pose(goals(context,part),current_state=state,return_seeds=returns)
        qs.append(result.solution.reshape(len(part),returns,6).detach().clone())
        valid.append(result.success.reshape(len(part),returns).detach().clone())
        position_error.append(result.position_error.reshape(len(part),returns,-1).max(dim=-1).values.detach().clone())
        orientation_error.append(result.rotation_error.reshape(len(part),returns,-1).max(dim=-1).values.detach().clone())
    return torch.cat(qs),torch.cat(valid),torch.cat(position_error),torch.cat(orientation_error)


def box_penetration(spheres,base,geometry,raise_z=0.):
    xyz=spheres[...,:3] @ torch.as_tensor(base[:3,:3].T,dtype=spheres.dtype,device=spheres.device)
    xyz+=torch.as_tensor(base[:3,3],dtype=spheres.dtype,device=spheres.device)
    center=np.asarray(geometry['center']).copy();center[2]+=raise_z
    delta=(xyz-torch.as_tensor(center,dtype=spheres.dtype,device=spheres.device)).abs()-torch.as_tensor(geometry['dims'],dtype=spheres.dtype,device=spheres.device)/2
    sdf=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.max(dim=-1).values.clamp_max(0)
    penetration=spheres[...,3]-sdf
    return torch.where(spheres[...,3]>0,penetration,torch.full_like(penetration,-1e6))


def collision_details(context,state,qs,checker=None,raise_z=0.):
    checker=checker or context.checker
    q=qs.reshape(-1,6)
    kin=checker.get_kinematics(q[:,None])
    spheres=kin.robot_spheres.reshape(len(q),-1,4).detach().clone()
    full=checker.validate(q[:,None]).reshape(-1).detach().clone()
    bound=(checker.get_bound(q[:,None]).sum(dim=-1).reshape(-1)==0)
    table=box_penetration(spheres,context.base,state['scene_cuboids']['table'],raise_z)
    ground=box_penetration(spheres,context.base,state['scene_cuboids']['ground'])
    return {'full_scene_valid':full,'joint_bounds_valid':bound,
        'table_penetration':table.max(dim=-1).values,'ground_penetration':ground.max(dim=-1).values,
        'worst_table_sphere':table.argmax(dim=-1),'spheres':spheres}


def probe(pool,pure,position,root):
    state=common.read(Path(common.read(root / 'registration.json')['warmup_trial']) / 'state.json')
    context=pool.bind(root / 'probe',state)
    start=context.measured(state)
    pose=np.asarray(state['links']['right_eef_link'])
    q,ok,p,r=solve(pure,context,[pose],start)
    assert bool(ok.any()),'Pure full-pose IK cannot recover recorded measured EEF'
    changed=pose.copy();changed[:3,:3]=changed[:3,:3] @ Rotation.from_euler('x',np.pi).as_matrix()
    pq,pok,pp,pr=solve(position,context,[changed],start)
    assert bool(pok.any()),'Position-only diagnostic cannot recover recorded position'
    details=collision_details(context,state,q[ok])
    assert bool(details['full_scene_valid'].any()),'No original-world-valid IK for measured EEF'
    params=context.checker.kinematics.kinematics_config
    names={int(v):k for k,v in params.link_name_to_idx_map.items()}
    sphere_names=[names[int(x)] for x in params.link_sphere_idx_map.cpu().tolist()]
    config=context.planner.ik_solver.config
    common.save(root / 'probe_verification.json',{'passed':True,'physical_attempts':0,
        'pure_recorded_pose_solutions':int(ok.sum()),'position_only_changed_orientation_solutions':int(pok.sum()),
        'original_ik_optimization_seeds':config.num_seeds,'original_position_tolerance':config.position_tolerance,
        'original_orientation_tolerance':config.orientation_tolerance,'sphere_links':dict(Counter(sphere_names))})
    print('PURE/FIXED-POSITION PROBES PASSED',int(ok.sum()),int(pok.sum()),flush=True)


def goals(context,worlds):
    from curobo.types import GoalToolPose
    local=np.stack([context.invbase @ w for w in worlds])
    quaternion=np.stack([np.array(__import__('planner').pose7(t)[3:]) for t in local])
    return GoalToolPose(tool_frames=['right_eef_link'],
        position=torch.as_tensor(local[:,:3,3],device='cuda',dtype=torch.float32).reshape(-1,1,1,1,3),
        quaternion=torch.as_tensor(quaternion,device='cuda',dtype=torch.float32).reshape(-1,1,1,1,4))


def solve(solver,context,worlds,start):
    from curobo.types import JointState
    solutions=[];success=[];position_errors=[];rotation_errors=[]
    for offset in range(0,len(worlds),64):
        chunk=worlds[offset:offset+64];n=len(chunk)
        padded=list(chunk)+[chunk[0]]*(64-n)
        current=JointState.from_position(start[None].repeat(64,1).clone(),joint_names=context.planner.ik_solver.joint_names)
        result=solver.solve_pose(goals(context,padded),current_state=current,return_seeds=32)
        solutions.append(result.solution.reshape(64,32,6)[:n].detach().clone())
        success.append(result.success.reshape(64,32)[:n].detach().clone())
        position_errors.append(result.position_error.reshape(64,32,-1).amax(dim=-1)[:n].detach().clone())
        rotation_errors.append(result.rotation_error.reshape(64,32,-1).amax(dim=-1)[:n].detach().clone())
    return tuple(torch.cat(values,dim=0) for values in [solutions,success,position_errors,rotation_errors])


def original_trace(context,state,candidates):
    start=context.measured(state)
    events=[];rejected=Counter();accepted=None;examples=[]
    params=context.checker.kinematics.kinematics_config
    names={int(v):k for k,v in params.link_name_to_idx_map.items()}
    sphere_names=[names[int(x)] for x in params.link_sphere_idx_map.cpu().tolist()]

    def ik(world,current,index,stage):
        result=context.planner.ik_solver.solve_pose(goals(context,[world]),current_state=context.js(current),return_seeds=4)
        qs=result.solution.reshape(-1,6).detach().clone()
        solved=result.success.reshape(-1).detach().clone()
        valid=qs[solved]
        if len(valid):valid=valid[context.checker.validate(valid[:,None]).reshape(-1)]
        chosen=valid[(valid-current).square().sum(-1).argmin()].clone() if len(valid) else None
        pose_ok=(result.position_error.reshape(4,-1).amax(-1)<.002)&(result.rotation_error.reshape(4,-1).amax(-1)<.03)&torch.isfinite(qs).all(-1)
        event={'candidate':index,'stage':stage,'solver_valid_solutions':int(solved.sum()),
            'postcheck_valid_solutions':len(valid),'pose_converged_solutions':int(pose_ok.sum())}
        if pose_ok.any():
            detail=collision_details(context,state,qs[pose_ok])
            bounded=detail['joint_bounds_valid'];table=(detail['table_penetration']>0)&bounded
            event.update({'pose_and_bounds_valid_solutions':int(bounded.sum()),
                'table_colliding_pose_solutions':int(table.sum()),
                'ground_colliding_pose_solutions':int(((detail['ground_penetration']>0)&bounded).sum()),
                'world_clear_pose_solutions':int((detail['full_scene_valid']&bounded).sum())})
            if table.any() and len(examples)<2:
                masked=torch.where(table,detail['table_penetration'],torch.full_like(detail['table_penetration'],-1e6))
                worst=int(masked.argmax());sphere=int(detail['worst_table_sphere'][worst])
                example={'candidate':index,'stage':stage,'penetration_m':float(masked[worst]),
                    'link':sphere_names[sphere],'joint_positions':qs[pose_ok][worst].cpu().tolist(),
                    'spheres_body4':detail['spheres'][worst].cpu().tolist()}
                examples.append(example)
        events.append(event)
        return chosen

    for index,candidate in enumerate(candidates):
        contact=np.asarray(candidate['pose']);grasp=contact.copy();grasp[:3,3]-=.040*contact[:3,2]
        pre=grasp.copy();pre[:3,3]-=.08*contact[:3,2]
        lift=grasp.copy();lift[2,3]+=.12
        qg=ik(grasp,start,index,'grasp')
        if qg is None:rejected['grasp_ik']+=1;continue
        if not bool(context.closed_checker.validate(qg[None,None]).all()):
            closed=collision_details(context,state,qg[None],context.closed_checker,raise_z=.008)
            events.append({'candidate':index,'stage':'nominal_closed',
                'table_colliding_pose_solutions':int(closed['table_penetration'][0]>0),
                'table_penetration_m':float(closed['table_penetration'][0])})
            rejected['closed_collision']+=1;continue
        qp=ik(pre,qg,index,'pregrasp');ql=ik(lift,qg,index,'lift')
        if qp is None or ql is None:rejected['pre_or_lift_ik']+=1;continue
        reach=context.path(start,qp);approach=context.path(qp,qg)
        if reach is None or approach is None:rejected['path']+=1;continue
        accepted=index;break
    return {'events':events,'rejected':dict(rejected),'first_accepted_candidate':accepted,'table_examples':examples}


def audit_case(pool,pure,position,root,profile,trial):
    import resident_planner
    from curobo._src.geom.types import SceneCfg
    common=__import__('graspcl_pipeline')
    state=common.read(trial / 'state.json');baseline=common.read(trial / 'result.json')
    candidates=common.read(trial / 'graspcl/candidate_pool.json')['candidates']
    destination=root / 'cases' / profile / trial.name
    destination.mkdir(parents=True,exist_ok=False)
    context=pool.bind(destination / 'model',state)
    traced=original_trace(context,state,candidates)
    old=common.read(trial / 'plan_result.json')
    replay_same=all(traced['rejected'].get(k,0)==v for k,v in old.get('rejected',{}).items()) and traced['first_accepted_candidate']==old.get('selected',{}).get('index')
    worlds=[]
    for candidate in candidates:
        t=np.asarray(candidate['pose']).copy();t[:3,3]-=.040*t[:3,2];worlds.append(t)
    start=context.measured(state);pure.reset_seed();position.reset_seed()
    q,ok,pe,re=solve(pure,context,worlds,start)
    pq,pok,ppe,pre=solve(position,context,worlds,start)
    n=len(candidates)
    shape=ok.shape
    open_table_clear=torch.zeros(shape,dtype=torch.bool,device='cuda')
    open_world_clear=open_table_clear.clone();nominal_clear=open_table_clear.clone()
    closed_table_clear=open_table_clear.clone();closed_physical_table_clear=open_table_clear.clone()
    table_mismatch=0;closed_mismatch=0;table_links=Counter();worst_case_example=None
    if ok.any():
        good=q[ok]
        opened=collision_details(context,state,good)
        closed=collision_details(context,state,good,context.closed_checker,raise_z=.008)
        closed_real=box_penetration(closed['spheres'],context.base,state['scene_cuboids']['table']).max(-1).values
        open_table_clear[ok]=opened['joint_bounds_valid']&(opened['table_penetration']<=0)
        open_world_clear[ok]=opened['full_scene_valid']
        nominal_clear[ok]=opened['full_scene_valid']&closed['full_scene_valid']
        closed_table_clear[ok]=closed['joint_bounds_valid']&(closed['table_penetration']<=0)
        closed_physical_table_clear[ok]=closed['joint_bounds_valid']&(closed_real<=0)
        scene=common.read(destination / 'model/scene.json')
        nominal=deepcopy(scene);nominal['cuboid']['table']['pose'][:3]=(np.array(nominal['cuboid']['table']['pose'][:3])+context.invbase[:3,2]*.008).tolist()
        context.checker.clear_scene_cache();context.checker.update_world(SceneCfg.create({'cuboid':{'table':scene['cuboid']['table']}}))
        oracle=context.checker.validate(good[:,None]).reshape(-1)
        table_mismatch=int((oracle!=(opened['joint_bounds_valid']&(opened['table_penetration']<=0))).sum())
        context.closed_checker.clear_scene_cache();context.closed_checker.update_world(SceneCfg.create({'cuboid':{'table':nominal['cuboid']['table']}}))
        oracle_closed=context.closed_checker.validate(good[:,None]).reshape(-1)
        closed_mismatch=int((oracle_closed!=(closed['joint_bounds_valid']&(closed['table_penetration']<=0))).sum())
        context.checker.clear_scene_cache();context.checker.update_world(resident_planner.collision_scene_config(scene))
        context.closed_checker.clear_scene_cache();context.closed_checker.update_world(resident_planner.collision_scene_config(nominal))
        params=context.checker.kinematics.kinematics_config
        names={int(v):k for k,v in params.link_name_to_idx_map.items()}
        sphere_names=[names[int(x)] for x in params.link_sphere_idx_map.cpu().tolist()]
        indices=opened['worst_table_sphere'][opened['table_penetration']>0].cpu().tolist()
        table_links.update({sphere_names[i] for i in indices})
        if indices:
            worst=int(opened['table_penetration'].argmax());sph=int(opened['worst_table_sphere'][worst])
            worst_case_example={'penetration_m':float(opened['table_penetration'][worst]),'link':sphere_names[sph],
                'joint_positions':good[worst].cpu().tolist(),'spheres_body4':opened['spheres'][worst].cpu().tolist()}
    np.savez_compressed(destination / 'solutions.npz',goals=np.asarray(worlds),full_pose_q=q.cpu().numpy(),
        full_pose_success=ok.cpu().numpy(),full_pose_position_error=pe.cpu().numpy(),full_pose_rotation_error=re.cpu().numpy(),
        position_only_q=pq.cpu().numpy(),position_only_success=pok.cpu().numpy(),
        open_table_clear=open_table_clear.cpu().numpy(),open_world_clear=open_world_clear.cpu().numpy(),
        nominal_closed_world_clear=nominal_clear.cpu().numpy(),closed_table_clear=closed_table_clear.cpu().numpy())
    pure_candidates=ok.any(-1);table_candidates=open_table_clear.any(-1)
    summary={'id':trial.name,'profile':profile,'source_trial':str(trial),'candidate_count':n,
        'baseline_reason':baseline['reason'],'baseline_physical_attempt':baseline['physical_attempt'],'baseline_success':baseline['success'],
        'original_replay_matches_baseline':replay_same,'original_replay':traced,
        'full_pose_pure_ik_candidates':int(pure_candidates.sum()),'position_only_candidates':int(pok.any(-1).sum()),
        'open_table_clear_candidates':int(table_candidates.sum()),'open_full_scene_clear_candidates':int(open_world_clear.any(-1).sum()),
        'open_and_nominal_closed_scene_clear_candidates':int(nominal_clear.any(-1).sum()),
        'any_pure_ik':bool(ok.any()),'any_position_only_ik':bool(pok.any()),'any_table_clear_pure_solution':bool(open_table_clear.any()),
        'any_full_scene_clear_pure_solution':bool(open_world_clear.any()),'any_open_closed_clear_pure_solution':bool(nominal_clear.any()),
        'all_found_ik_solutions_table_blocked':bool(ok.any() and not open_table_clear.any()),
        'any_open_table_collision':bool((ok&~open_table_clear).any()),
        'all_found_open_closed_solutions_table_blocked':bool(ok.any() and not (open_table_clear&closed_table_clear).any()),
        'any_nominal_closed_margin_only_collision':bool((closed_physical_table_clear&~closed_table_clear&ok).any()),
        'analytic_vs_original_table_checker_mismatches':table_mismatch,
        'analytic_vs_original_closed_table_checker_mismatches':closed_mismatch,
        'pure_solution_count':int(ok.sum()),'table_collision_links':dict(table_links),'table_example':worst_case_example,
        'contact_center_below_table_candidates':sum(c['pose'][2][3]<0 for c in candidates),
        'worlds_min_xyz':np.asarray(worlds)[:,:3,3].min(0).tolist(),'worlds_max_xyz':np.asarray(worlds)[:,:3,3].max(0).tolist(),
        'interpretation':'Pure IK omits environment collisions only for diagnostic attribution; finite solutions do not prove exhaustive reachability. All-found-table-blocked refers to the solutions found, not all mathematical IK branches.'}
    common.save(destination / 'audit.json',summary)
    return summary


def aggregate(root,rows):
    common=__import__('graspcl_pipeline');output={}
    for profile in ['zerograsp_200','graspcl_200']:
        group=[r for r in rows if r['profile']==profile]
        failed=[r for r in group if not r['baseline_physical_attempt']]
        def count(part):
            return {'cases':len(part),'any_pure_ik':sum(r['any_pure_ik'] for r in part),
                'position_only_ik':sum(r['any_position_only_ik'] for r in part),
                'any_table_collision':sum(r['any_open_table_collision'] for r in part),
                'all_found_solutions_table_blocked':sum(r['all_found_ik_solutions_table_blocked'] for r in part),
                'any_table_clear_solution':sum(r['any_table_clear_pure_solution'] for r in part),
                'any_full_scene_clear_solution':sum(r['any_full_scene_clear_pure_solution'] for r in part),
                'any_open_closed_clear_solution':sum(r['any_open_closed_clear_pure_solution'] for r in part),
                'all_found_open_closed_solutions_table_blocked':sum(r['all_found_open_closed_solutions_table_blocked'] for r in part),
                'original_replay_same':sum(r['original_replay_matches_baseline'] for r in part),
                'table_oracle_mismatches':sum(r['analytic_vs_original_table_checker_mismatches'] for r in part),
                'closed_table_oracle_mismatches':sum(r['analytic_vs_original_closed_table_checker_mismatches'] for r in part)}
        output[profile]={'all_inferred_cases':count(group),'baseline_no_grasp_cases':count(failed),
            'links':dict(sum((Counter(r['table_collision_links']) for r in group),Counter()))}
    common.save(root / 'summary.json',output)
    common.save(root / 'progress.json',{'phase':'running','completed_cases':len(rows),'total_inferred_cases':390,'summary':output})
    return output


def workspace(pool,root,n=131072):
    common=__import__('graspcl_pipeline');reg=common.read(root / 'registration.json')
    state=common.read(Path(reg['warmup_trial']) / 'state.json');context=pool.bind(root / 'workspace',state)
    limits=context.limits.cpu().numpy();rng=np.random.default_rng(20261117)
    q=rng.uniform(limits[:,0],limits[:,1],(n,6)).astype(np.float32);points=[];clear=[]
    for offset in range(0,n,2048):
        chunk=torch.as_tensor(q[offset:offset+2048],device='cuda')
        kin=context.checker.get_kinematics(chunk[:,None]);body=kin.tool_poses.position.reshape(len(chunk),-1,3)[:,0]
        world=body @ torch.as_tensor(context.base[:3,:3].T,device='cuda',dtype=torch.float32)+torch.as_tensor(context.base[:3,3],device='cuda',dtype=torch.float32)
        spheres=kin.robot_spheres.reshape(len(chunk),-1,4)
        hit=box_penetration(spheres,context.base,state['scene_cuboids']['table']).max(-1).values>0
        points.append(world.cpu().numpy());clear.append((~hit).cpu().numpy())
    points=np.concatenate(points);clear=np.concatenate(clear)
    np.savez_compressed(root / 'workspace_samples.npz',q=q,positions=points,table_clear=clear)
    common.save(root / 'workspace.json',{'samples':n,'right_arm_joints':list(context.planner.ik_solver.joint_names),
        'torso_joint4_fixed_at':state['qpos'][state['joint_names'].index('joint4')],
        'base_body4_world':context.base.tolist(),'limits_rad':limits.tolist(),
        'sampled_xyz_min':points.min(0).tolist(),'sampled_xyz_max':points.max(0).tolist(),
        'table_clear_configurations':int(clear.sum()),'table':state['scene_cuboids']['table'],
        'scope':'Sampled FK position envelope for the six-joint right arm at the frozen torso height. Sampling does not prove unreachable gaps. No scene meshes or self-collision exclusions are used to call a position theoretically reachable.'})


def run(pool,pure,position,root):
    common=__import__('graspcl_pipeline');reg=common.read(root / 'registration.json')
    script=Path(__file__).resolve();common.save(root / 'job_registration.json',{'source_sha256':common.sha(script),
        'physical_attempts':0,'candidate_budget':reg['candidate_budget'],'workspace_samples':reg['fk_workspace_samples']})
    rows=[];started=time.perf_counter();source=Path(reg['source_campaign'])
    for profile in ['zerograsp_200','graspcl_200']:
        campaign=source / profile;registration=common.read(campaign / 'registration.json')
        for entry in registration['schedule']:
            trial=campaign / 'trials' / entry['id']
            if not (trial / 'graspcl/candidate_pool.json').exists():continue
            rows.append(audit_case(pool,pure,position,root,profile,trial))
            aggregate(root,rows)
            if len(rows)%5==0:print('AUDIT',len(rows),'/390',round(time.perf_counter()-started,1),'s',flush=True)
    workspace(pool,root,reg['fk_workspace_samples'])
    output=aggregate(root,rows)
    common.save(root / 'progress.json',{'phase':'complete','completed_cases':len(rows),'total_inferred_cases':390,
        'wall_seconds':time.perf_counter()-started,'physical_attempts':0,'summary':output})
    print('AUDIT COMPLETE',time.perf_counter()-started,output,flush=True)
