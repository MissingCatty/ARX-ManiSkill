"""Fixed-pose, static-world height sweep followed by frozen CLARO path checks."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from 公共 import ROOT, PROJECTS, MS, MODEL, URDF, TCP, JOINTS, REFERENCE, read, save, sha, now, static_scene

SELF = Path(__file__).resolve()
HISTORICAL = MS/'outputs/arx_physical_table_parallel_20261118_01/zerograsp/fixed_zerograsp/trials'
CLARO = MS/'outputs/arx_claro_continuous_mpc_20261120_01'
LIFTS = [0.,.04,.08,.12,.16,.20,.22,.24,.28,.32,.36,.40,.44,.46]
BATCH = 64


def prepare(out,gpu):
    reference = read(REFERENCE/'state.json')
    meta = read(ROOT/'产物/首版_01/地图元数据.json')
    robot = ET.parse(MS/'assets/arx/robot.urdf').getroot()
    lift = next(j for j in robot.findall('joint') if j.get('name')=='joint4')
    assert lift.get('type')=='prismatic' and lift.find('axis').get('xyz')=='0.0 0.0 1.0'
    limits = [float(lift.find('limit').get(k)) for k in ['lower','upper']]
    assert min(LIFTS)>=limits[0] and max(LIFTS)<=limits[1]
    files = [SELF,Path(__file__).with_name('公共.py'),Path(__file__).with_name('机器人几何.py'),
        MODEL,URDF,TCP,REFERENCE/'state.json',MS/'assets/arx/robot.urdf',
        ROOT/'产物/首版_01/地图元数据.json',CLARO/'registration.json',CLARO/'summary.json']
    points=[];case_ids=[]
    for directory in sorted(HISTORICAL.iterdir()):
        p=directory/'state.json'
        if not p.exists():continue
        state=read(p);points.append(state['object_center'][:2]);case_ids.append(directory.name);files.append(p)
    assert len(points)==193
    # Exact same desired world poses at every lift. These are kinematics
    # probes, not proposed object contacts or executable grasp candidates.
    positions=[];rotations=[];labels=[]
    shoulder_xy=np.asarray(reference['links']['body4'])[:2,3]+np.array([.059949,-.25])
    for case,xy in zip(case_ids,points):
        forward=np.asarray(xy)-shoulder_xy;forward/=np.linalg.norm(forward)
        for z in [.03,.08,.13]:
            for tilt in [0.,30.]:
                a=np.deg2rad(tilt);axis=np.r_[np.sin(a)*forward,-np.cos(a)]
                x=np.r_[-forward[1],forward[0],0.];y=np.cross(axis,x)
                r=np.column_stack([x,y,axis])
                assert np.allclose(r.T@r,np.eye(3),atol=1e-8) and np.linalg.det(r)>.999
                for yaw in np.arange(8)*45.:
                    positions.append([*xy,z]);rotations.append(r@Rotation.from_euler('z',yaw,degrees=True).as_matrix())
                    labels.append({'case':case,'height_above_table_m':z,'tilt_deg':tilt,'yaw_deg':float(yaw)})
    out.mkdir(parents=True,exist_ok=False);(out/'代码快照').mkdir()
    np.savez_compressed(out/'固定查询位姿.npz',positions=positions,rotations=rotations)
    save(out/'查询标签.json',labels)
    prior = read(CLARO/'registration.json')
    for entry in prior['schedule']:
        d=CLARO/'trials'/entry['case']/'01_抓挡'
        files += [d/'state.json',d/'graspcl/candidate_pool.json']
        files += [Path(g['file_path']) for g in read(d/'state.json')['scene_meshes'].values()]
    for name in ['arx_pipeline/planner.py','resident_planner.py','fast_resident_planner.py',
                 'short_lift_planner.py','table_truncation_model.py','tmp/physical_table_pair_20261118_01.py',
                 'arx_direct_grasp_20261120.py','arx_direct_terminal_20261120.py']:
        files.append(MS/name)
    reference_lift=reference['qpos'][reference['joint_names'].index('joint4')]
    shoulder=reference['links']['body4'][2][3]+.02449999935925007+.03999999910593033
    reg={'created_at':now(),'source_sha256':{str(p.resolve()):sha(p) for p in set(files)},
        'lift_values_m':LIFTS,'legal_lift_limits_m':limits,'reference_lift_m':reference_lift,
        'reference_shoulder_above_table_m':shoulder,'reference_state':str(REFERENCE/'state.json'),
        'gpu_uuid':gpu,'direct_grasp':True,'physical_attempts':0,'robot_motion_commands':0,
        'query_poses':len(positions),'xy_locations':193,'tcp_heights_above_table_m':[.03,.08,.13],
        'tilt_degrees':[0.,30.],'yaw_angles_degrees':list(np.arange(8)*45.),
        'primary_metric':'Fraction of exact vertical pose probes with a solution passing original static table/ground checks. Constant world targets and denominator at all lifts.',
        'secondary_metrics':['At least one of eight yaw angles at a position','IK joint-limit margin','Thirty-degree approach from robot side'],
        'selection':'Highest vertical pose fraction, then vertical position fraction, then smallest change from current lift. Verify current and top two distinct noncurrent lifts on all twelve frozen CLARO cases.',
        'scope':'Offline six-arm-joint geometry and planning only. Static grid omits clutter; CLARO verification includes original clutter. No physical grasp success measured.',
        'models':'IK, static checker and FK load once for the scan; planning pool once before the separate twelve-case path-check phase. No per-height reload.',
        'orientation_changes_in_probes':'The fixed vertical/tilted grid is a kinematic probe, not a contact-valid grasp set. CLARO candidate poses remain unchanged.'}
    save(out/'登记.json',reg)
    for p in files:
        if p.suffix=='.py':
            target=out/'代码快照'/p.name
            if target.exists():assert target.read_bytes()==p.read_bytes()
            else:target.write_bytes(p.read_bytes())
    save(out/'进度.json',{'stage':'已登记','physical_attempts':0,'heights':len(LIFTS)})


def guard(reg):
    for p,h in reg['source_sha256'].items():assert sha(p)==h,'Original source changed: '+p


def shifted(state,lift,reference_lift):
    state=deepcopy(state);delta=lift-reference_lift
    state['qpos'][state['joint_names'].index('joint4')]=lift
    for name in ['body4','right_eef_link']:
        value=np.asarray(state['links'][name]);value[2,3]+=delta;state['links'][name]=value.tolist()
    if 'camera_world_optical' in state:
        value=np.asarray(state['camera_world_optical']);value[2,3]+=delta;state['camera_world_optical']=value.tolist()
    return state


def scan(out):
    reg=read(out/'登记.json');guard(reg);os.environ['CUDA_VISIBLE_DEVICES']=reg['gpu_uuid']
    import torch
    from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.types import GoalToolPose,JointState
    from 机器人几何 import RobotGeometry
    started=time.perf_counter();reference=read(Path(reg['reference_state']));profile=np.asarray(read(TCP)['T_eef_action'])
    model=yaml.safe_load(MODEL.read_text());model['robot_cfg']['kinematics']['urdf_path']=str(URDF)
    model_path=out/'机器人.yml';model_path.write_text(yaml.safe_dump(model,sort_keys=False))
    cfg=InverseKinematicsCfg.create(robot=str(model_path),scene_model=static_scene(reference),
        self_collision_check=False,num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,
        orientation_tolerance=.03,optimizer_collision_activation_distance=.003,max_batch_size=BATCH,
        collision_cache={'mesh':1,'primitive':8},use_cuda_graph=True)
    for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
        for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
    solver=InverseKinematics(cfg)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model_path),
        scene_model=static_scene(reference),n_meshes=1,n_cuboids=8,collision_activation_distance=0.))
    checker.self_collision_cost=None;geometry=RobotGeometry(profile,device='cuda')
    data=np.load(out/'固定查询位姿.npz');p=data['positions'];r=data['rotations'];n=len(p)
    current=np.array([reference['qpos'][reference['joint_names'].index(k)] for k in JOINTS])
    summaries=[];all_flags=[];all_margins=[];all_solutions=[]
    maximum_fk_error=0.;maximum_rotation_error=0.
    for height_index,lift in enumerate(reg['lift_values_m']):
        state=shifted(reference,lift,reg['reference_lift_m']);base=np.asarray(state['links']['body4']);inverse=np.linalg.inv(base)
        scene=SceneCfg.create(static_scene(state));solver.update_world(scene);checker.update_world(scene)
        # Reset the same internal seed stream at each height.
        solver.reset_seed();flags=np.zeros(n,dtype=bool);margins=np.zeros(n);solutions=np.zeros((n,6))
        begun=time.perf_counter()
        for offset in range(0,n,BATCH):
            count=min(BATCH,n-offset);indices=np.minimum(np.arange(offset,offset+BATCH),n-1)
            world=np.repeat(np.eye(4)[None],BATCH,axis=0);world[:,:3,:3]=r[indices]
            world[:,:3,3]=p[indices]-.043*r[indices,:,2]
            local=inverse@world;quat=Rotation.from_matrix(local[:,:3,:3]).as_quat()[:,[3,0,1,2]]
            goal=GoalToolPose(tool_frames=['right_eef_link'],
                position=torch.as_tensor(np.ascontiguousarray(local[:,:3,3]),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,3),
                quaternion=torch.as_tensor(np.ascontiguousarray(quat),device='cuda',dtype=torch.float32).reshape(BATCH,1,1,1,4))
            js=JointState.from_position(torch.as_tensor(np.tile(current,(BATCH,1)),device='cuda',dtype=torch.float32),joint_names=JOINTS)
            solved=solver.solve_pose(goal,current_state=js,return_seeds=4)
            qs=solved.solution.reshape(BATCH,4,6);ok=solved.success.reshape(BATCH,4)
            clear=checker.validate(qs.reshape(-1,6)[:,None]).reshape(BATCH,4);ok=ok&clear
            qcpu=qs.detach().cpu().numpy();valid=ok.detach().cpu().numpy()
            margin=np.minimum(qcpu-geometry.lower,geometry.upper-qcpu).min(-1);margin[~valid]=-1.
            best=margin.argmax(-1);chosen=qcpu[np.arange(BATCH),best];found=valid.any(-1)
            if found.any():
                actual,rotation=geometry.fk(torch.as_tensor(chosen[found],device='cuda'),jacobian=False)
                xyz=actual.cpu().numpy()@base[:3,:3].T+base[:3,3]
                expected_r=r[indices[found]]@profile[:3,:3]
                actual_r=base[:3,:3]@rotation.cpu().numpy()
                ep=np.linalg.norm(xyz-p[indices[found]],axis=1)
                er=Rotation.from_matrix(actual_r@expected_r.transpose(0,2,1)).magnitude()
                maximum_fk_error=max(maximum_fk_error,float(ep.max()));maximum_rotation_error=max(maximum_rotation_error,float(er.max()))
                assert ep.max()<=.00205 and er.max()<=.0301,(ep.max(),er.max())
            flags[offset:offset+count]=found[:count];margins[offset:offset+count]=np.maximum(0,margin.max(-1)[:count])
            solutions[offset:offset+count]=chosen[:count]
        shaped=flags.reshape(193,3,2,8);quality=margins.reshape(193,3,2,8)
        item={'lift_m':lift,'shoulder_above_table_m':reg['reference_shoulder_above_table_m']+lift-reg['reference_lift_m'],
            'vertical_pose_passes':int(shaped[:,:,0].sum()),'vertical_pose_total':193*3*8,
            'vertical_pose_fraction':float(shaped[:,:,0].mean()),
            'vertical_positions_any_yaw':int(shaped[:,:,0].any(-1).sum()),'positions_total':193*3,
            'vertical_position_fraction':float(shaped[:,:,0].any(-1).mean()),
            'tilted_pose_fraction':float(shaped[:,:,1].mean()),'tilted_position_fraction':float(shaped[:,:,1].any(-1).mean()),
            'vertical_margin_over_0p1_fraction':float((quality[:,:,0]>.1).mean()),
            'vertical_pass_by_tcp_height':[float(shaped[:,i,0].mean()) for i in range(3)],
            'wall_seconds':time.perf_counter()-begun}
        summaries.append(item);all_flags.append(flags);all_margins.append(margins);all_solutions.append(solutions)
        save(out/'高度扫描.json',{'rows':summaries,'physical_attempts':0})
        save(out/'进度.json',{'stage':'高度扫描','completed':height_index+1,'total':len(LIFTS),'latest':item,'physical_attempts':0})
        print('HEIGHT',lift,'shoulder',round(item['shoulder_above_table_m'],4),
              'vertical',round(item['vertical_pose_fraction'],4),'positions',round(item['vertical_position_fraction'],4),
              'tilt30',round(item['tilted_pose_fraction'],4),'seconds',round(item['wall_seconds'],1),flush=True)
    np.savez_compressed(out/'逐位姿见证.npz',lifts=LIFTS,passed=all_flags,margin=all_margins,q=all_solutions)
    order=sorted(summaries,key=lambda a:(-a['vertical_pose_fraction'],-a['vertical_position_fraction'],abs(a['lift_m']-.22)))
    checks=[.22]+[a['lift_m'] for a in order if abs(a['lift_m']-.22)>1e-6][:2]
    result={'passed':True,'rows':summaries,'ranked_lifts_m':[a['lift_m'] for a in order],
        'path_check_lifts_m':checks,'maximum_independent_tcp_error_m':maximum_fk_error,
        'maximum_independent_rotation_error_rad':maximum_rotation_error,'wall_seconds':time.perf_counter()-started,
        'model_load_counts':{'ik':1,'checker':1,'fk':1},'physical_attempts':0}
    save(out/'高度扫描.json',result);guard(reg)
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','scan'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve(),args.gpu)
    else:scan(args.output.resolve())
