"""Search shoulder/table separation by moving the table, with the robot fixed."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from 公共 import ROOT, MS, REFERENCE, MODEL, URDF, TCP, JOINTS, read, save, sha, now, static_scene
from 比较升降高度 import shifted

SELF=Path(__file__).resolve()
PREVIOUS=ROOT/'产物/高度细扫_20261120_01'
BATCH=64


def table_state(original, lift, height):
    """The floor and robot base stay fixed. Only the supported table/object group moves."""
    old_lift=original['qpos'][original['joint_names'].index('joint4')]
    state=shifted(original,lift,old_lift)
    table=state['scene_cuboids']['table'];ground=state['scene_cuboids']['ground']
    floor=ground['center'][2]+ground['dims'][2]/2
    old_top=table['center'][2]+table['dims'][2]/2
    delta=floor+height-old_top
    table['dims'][2]=height;table['center'][2]=floor+height/2
    if 'scene_meshes' in state:
        for item in state['scene_meshes'].values():item['world'][2][3]+=delta
    if 'objects' in state:
        for item in state['objects'].values():item['pose'][2]+=delta
    state['object_world'][2][3]+=delta;state['object_center'][2]+=delta
    state['table_height_m']=height
    return state,delta


def prepare(out,gpu):
    selected=read(PREVIOUS/'真实候选核验/高度选择.json');assert selected['passed']
    lift=selected['selected_lift_m'];assert 0<=lift<=.46
    source=read(REFERENCE/'state.json')
    floor=source['scene_cuboids']['ground']['center'][2]+source['scene_cuboids']['ground']['dims'][2]/2
    original_lift=source['qpos'][source['joint_names'].index('joint4')]
    shoulder=source['links']['body4'][2][3]+.02449999935925007+.03999999910593033+lift-original_lift
    heights=sorted({round(v/100,3) for v in range(10,91,2)}|{.50})
    paths=[SELF,Path(__file__).with_name('公共.py'),Path(__file__).with_name('比较升降高度.py'),
           Path(__file__).with_name('机器人几何.py'),MODEL,URDF,TCP,REFERENCE/'state.json',
           PREVIOUS/'固定查询位姿.npz',PREVIOUS/'查询标签.json',PREVIOUS/'真实候选核验/高度选择.json']
    out.mkdir(parents=True,exist_ok=False);(out/'代码快照').mkdir()
    for path in paths:
        if path.suffix=='.py':shutil.copy2(path,out/'代码快照'/path.name)
    reg={'created_at':now(),'source_sha256':{str(p):sha(p) for p in paths},'gpu_uuid':gpu,
         'reference_state':str(REFERENCE/'state.json'),'fixed_legal_joint4_m':lift,
         'floor_world_z_m':floor,'shoulder_world_z_m':shoulder,'shoulder_above_floor_m':shoulder-floor,
         'current_table_above_floor_m':.50,'coarse_table_heights_m':heights,
         'fine_step_m':.002,'fine_half_width_m':.012,'maximum_height_count':54,
         'primary':'Coverage of fixed vertical grasp-pose probes; report 30-degree and equally weighted mixed coverage separately.',
         'selection':'Highest vertical pose coverage, then vertical any-yaw position coverage, then smallest table adjustment from 50cm.',
         'table_interpretation':'Adjustable table/support cuboid: bottom anchored to original floor; XY and tabletop dimensions unchanged; top and all object/goal Z coordinates move together.',
         'frozen':'Robot base mounting, legal joint4, robot model/limits/TCP, ground, object XY/orientation, camera mount/K, IK tolerances and collision policy.',
         'num_seeds':32,'return_seeds':4,'eef_position_tolerance_m':.002,'orientation_tolerance_rad':.03,
         'pose_probes_per_height':9264,'physical_attempts':0,'robot_motion_commands':0,
         'scope':'Independent offline tabletop-height calculation, no simulator or edits to the ongoing 400 physical tests. Pose probes are not contact-qualified grasps.'}
    save(out/'登记.json',reg)
    candidate,delta=table_state(source,lift,.4)
    assert candidate['links']['base_link']==source['links']['base_link']
    assert candidate['scene_cuboids']['ground']==source['scene_cuboids']['ground']
    assert abs(delta+.1)<1e-9
    assert abs(candidate['scene_cuboids']['table']['center'][2]+candidate['scene_cuboids']['table']['dims'][2]/2-(floor+.4))<1e-9
    save(out/'预检.json',{'passed':True,'base_and_floor_unchanged':True,'table_bottom_anchored_to_floor':True,
        'same_fixed_pose_probes':True,'legal_joint4_unchanged':lift,'physical_attempts':0})
    print('PREPARED table 10..90cm, fixed joint4',lift,'shoulder/floor',shoulder-floor,flush=True)


def run(out):
    reg=read(out/'登记.json')
    assert all(sha(p)==digest for p,digest in reg['source_sha256'].items())
    os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid']
    import torch
    from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.types import GoalToolPose,JointState
    from 机器人几何 import RobotGeometry
    profile=np.array(read(TCP)['T_eef_action'])
    reference=read(reg['reference_state'])
    initial,_=table_state(reference,reg['fixed_legal_joint4_m'],.5)
    model=yaml.safe_load(MODEL.read_text());model['robot_cfg']['kinematics']['urdf_path']=str(URDF)
    path=out/'机器人.yml';path.write_text(yaml.safe_dump(model,sort_keys=False))
    cfg=InverseKinematicsCfg.create(robot=str(path),scene_model=static_scene(initial),
        self_collision_check=False,num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,
        orientation_tolerance=.03,optimizer_collision_activation_distance=.003,max_batch_size=BATCH,
        collision_cache={'mesh':1,'primitive':8},use_cuda_graph=True)
    for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    solver=InverseKinematics(cfg)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(path),
        scene_model=static_scene(initial),n_meshes=1,n_cuboids=8,collision_activation_distance=0.))
    checker.self_collision_cost=None
    geometry=RobotGeometry(profile,device='cuda')
    data=np.load(PREVIOUS/'固定查询位姿.npz');positions=data['positions'];rotations=data['rotations'];n=len(positions)
    assert n==reg['pose_probes_per_height']
    current=np.array([initial['qpos'][initial['joint_names'].index(j)] for j in JOINTS])
    rows=[];flags_all=[];height_all=[];maximum_position=0.;maximum_rotation=0.;began=time.perf_counter()
    def assess(height):
        nonlocal maximum_position,maximum_rotation
        state,delta=table_state(reference,reg['fixed_legal_joint4_m'],height)
        base=np.array(state['links']['body4']);inverse=np.linalg.inv(base)
        scene=SceneCfg.create(static_scene(state));solver.update_world(scene);checker.update_world(scene);solver.reset_seed()
        flags=np.zeros(n,bool);margins=np.zeros(n);solutions=np.zeros((n,6));start=time.perf_counter()
        p=positions.copy();p[:,2]+=delta
        for offset in range(0,n,BATCH):
            count=min(BATCH,n-offset);idx=np.minimum(np.arange(offset,offset+BATCH),n-1)
            world=np.repeat(np.eye(4)[None],BATCH,axis=0);world[:,:3,:3]=rotations[idx]
            world[:,:3,3]=p[idx]-.043*rotations[idx,:,2]
            local=inverse@world;quat=Rotation.from_matrix(local[:,:3,:3]).as_quat()[:,[3,0,1,2]]
            goal=GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.as_tensor(np.ascontiguousarray(local[:,:3,3]),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,3),
                quaternion=torch.as_tensor(np.ascontiguousarray(quat),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,4))
            js=JointState.from_position(torch.as_tensor(np.tile(current,(BATCH,1)),device='cuda',dtype=torch.float32),joint_names=JOINTS)
            result=solver.solve_pose(goal,current_state=js,return_seeds=4)
            qs=result.solution.reshape(BATCH,4,6)
            ok=result.success.reshape(BATCH,4)&checker.validate(qs.reshape(-1,6)[:,None]).reshape(BATCH,4)
            q=qs.detach().cpu().numpy();valid=ok.detach().cpu().numpy()
            valid &= ((q>=geometry.lower-1e-6)&(q<=geometry.upper+1e-6)).all(-1)
            margin=np.minimum(q-geometry.lower,geometry.upper-q).min(-1);margin[~valid]=-1
            best=margin.argmax(-1);chosen=q[np.arange(BATCH),best];found=valid.any(-1)
            if found.any():
                xyz,r=geometry.fk(torch.as_tensor(chosen[found],device='cuda'),jacobian=False)
                tcp=xyz.cpu().numpy()@base[:3,:3].T+base[:3,3];eef_r=base[:3,:3]@r.cpu().numpy()@profile[:3,:3].T
                eef=tcp-.043*eef_r[:,:,2]
                ep=np.linalg.norm(eef-world[found,:3,3],axis=1)
                er=Rotation.from_matrix(eef_r@world[found,:3,:3].transpose(0,2,1)).magnitude()
                assert ep.max()<.00205 and er.max()<.0301,(ep.max(),er.max())
                maximum_position=max(maximum_position,float(ep.max()));maximum_rotation=max(maximum_rotation,float(er.max()))
            flags[offset:offset+count]=found[:count];margins[offset:offset+count]=np.maximum(0,margin.max(-1)[:count]);solutions[offset:offset+count]=chosen[:count]
        shaped=flags.reshape(193,3,2,8)
        row={'table_height_m':height,'table_adjustment_m':height-.5,
            'shoulder_above_table_m':reg['shoulder_above_floor_m']-height,
            'vertical_pose_passes':int(shaped[:,:,0].sum()),'vertical_pose_total':193*3*8,
            'vertical_pose_fraction':float(shaped[:,:,0].mean()),
            'vertical_position_fraction':float(shaped[:,:,0].any(-1).mean()),
            'tilted_pose_fraction':float(shaped[:,:,1].mean()),'tilted_position_fraction':float(shaped[:,:,1].any(-1).mean()),
            'mixed_pose_fraction':float(shaped.mean()),'wall_seconds':time.perf_counter()-start}
        rows.append(row);height_all.append(height);flags_all.append(flags)
        np.savez_compressed(out/f'见证_桌高{round(height*1000):03d}.npz',passed=flags,q=solutions,margin=margins)
        save(out/'桌高扫描.json',{'rows':rows,'physical_attempts':0})
        save(out/'进度.json',{'phase':'桌高扫描','heights_completed':len(rows),'latest':row,'physical_attempts':0})
        print('TABLE',height,'difference',round(row['shoulder_above_table_m'],4),'vertical',round(row['vertical_pose_fraction'],4),'mixed',round(row['mixed_pose_fraction'],4),flush=True)
    rank=lambda row:(-row['vertical_pose_fraction'],-row['vertical_position_fraction'],abs(row['table_adjustment_m']))
    for height in reg['coarse_table_heights_m']:assess(height)
    coarse=sorted(rows,key=rank)[0]['table_height_m']
    fine=sorted({round(coarse+mm/1000,3) for mm in range(-12,13,2) if .05<=coarse+mm/1000<=1.0}
        -{r['table_height_m'] for r in rows})
    assert len(rows)+len(fine)<=reg['maximum_height_count']
    save(out/'细化登记.json',{'created_at':now(),'coarse_best_table_height_m':coarse,'fine_table_heights_m':fine})
    for height in fine:assess(height)
    best=sorted(rows,key=rank)[0]
    mixed=sorted(rows,key=lambda r:(-r['mixed_pose_fraction'],abs(r['table_adjustment_m'])))[0]
    tilt=sorted(rows,key=lambda r:(-r['tilted_pose_fraction'],abs(r['table_adjustment_m'])))[0]
    assert all(sha(p)==digest for p,digest in reg['source_sha256'].items())
    result={'passed':True,'best_vertical':best,'best_mixed':mixed,'best_tilted':tilt,
        'fixed_joint4_m':reg['fixed_legal_joint4_m'],'shoulder_above_floor_m':reg['shoulder_above_floor_m'],
        'rows':rows,'heights':len(rows),'pose_queries':len(rows)*n,'physical_attempts':0,
        'maximum_independent_eef_error_m':maximum_position,'maximum_independent_rotation_error_rad':maximum_rotation,
        'source_files_unchanged':True,'wall_seconds':time.perf_counter()-began,
        'scope':reg['scope'],'mounting_choice':'Adjust table height, leave the robot base mounting fixed.'}
    save(out/'相对高度结论.json',result)
    save(out/'进度.json',{'phase':'完成','best_vertical':best,'physical_attempts':0})
    print('BEST_RELATIVE_HEIGHT',best,'MIXED',mixed,'TILTED',tilt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu')
    args=p.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,args.gpu)
    else:
        try:run(out)
        except BaseException:
            save(out/'故障.json',{'time':now(),'error':traceback.format_exc(),'physical_attempts':0});raise
