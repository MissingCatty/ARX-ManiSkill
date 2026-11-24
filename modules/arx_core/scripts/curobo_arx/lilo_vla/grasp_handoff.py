"""Grasp-relative entry goals and planning-only continuation certificates.

The same candidate rule is usable by deployment and teacher collection. Nothing
in this module edits robot, camera, collision or scene configuration.
"""
import math
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


RULE = {
    'version':'grasp_backoff_v1',
    'ordered_backoff_m':[.04,.06,.08],
    'cartesian_step_m':.005,
    'rotation_step_rad':.05,
    'image_margin_pixels':32,
    'reject_pregrasp_target_contact':True,
}
NEAR_RULES={
    'five_mm':{**RULE,'version':'grasp_backoff_005mm_v2','ordered_backoff_m':[.005]},
    'ten_mm':{**RULE,'version':'grasp_backoff_010mm_v2','ordered_backoff_m':[.01]},
}
S02_RULE = {**RULE, 'version':'grasp_backoff_040mm_s02_v1', 'ordered_backoff_m':[.04]}


def validate_rule(rule):
    if rule not in (RULE,*NEAR_RULES.values(),S02_RULE):
        raise ValueError('Unregistered grasp handoff rule; use a separately versioned experiment')


def candidates(grasp_world,rule,entry_perturbation_world=None):
    validate_rule(rule)
    grasp=np.asarray(grasp_world,dtype=float)
    if grasp.shape!=(4,4) or not np.isfinite(grasp).all():raise ValueError('Expected a finite grasp transform')
    if not np.allclose(grasp[3],[0,0,0,1]) or not np.allclose(grasp[:3,:3].T@grasp[:3,:3],np.eye(3),atol=1e-5):
        raise ValueError('Grasp target is not a rigid transform')
    if np.linalg.det(grasp[:3,:3])<0:raise ValueError('Grasp rotation is reflected')
    delta=np.zeros(6) if entry_perturbation_world is None else np.asarray(entry_perturbation_world,dtype=float)
    if delta.shape!=(6,) or not np.isfinite(delta).all() or np.max(np.abs(delta[:3]))>.01 or np.linalg.norm(delta[3:])>.15:
        raise ValueError('Entry perturbation must be finite, at most 1cm per axis and 0.15rad total')
    result=[]
    for distance in rule['ordered_backoff_m']:
        pose=grasp.copy();pose[:3,3]-=distance*grasp[:3,2]
        pose[:3,3]+=delta[:3]
        pose[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()@pose[:3,:3]
        result.append({'name':f'grasp_backoff_{round(distance*1000):03d}mm',
            'backoff_m':distance,'world':pose.tolist()})
    return result


def frustum(ee_world,ee_camera_optical,intrinsics,size,points,margin):
    camera=np.asarray(ee_world)@np.asarray(ee_camera_optical)
    inverse=np.linalg.inv(camera);K=np.asarray(intrinsics);width,height=size
    projected=[]
    for name,point in points.items():
        xyz=(inverse@np.r_[point,1.])[:3]
        pixel=(K@xyz)[:2]/xyz[2] if xyz[2]>.01 else None
        inside=bool(pixel is not None and margin<=pixel[0]<width-margin and margin<=pixel[1]<height-margin)
        projected.append({'name':name,'pixel_xy':pixel.tolist() if pixel is not None else None,
            'camera_depth_m':float(xyz[2]),'inside_margin':inside})
    return {'pass':all(p['inside_margin'] for p in projected),'points':projected,
        'margin_pixels':margin,'camera_world_optical':camera.tolist(),
        'scope':'Geometric image-frustum/crop check only; does not certify absence of occlusion'}


def corridor(start_world,end_world,start_q,solve,validate,rule):
    """Validate a straight Cartesian/SO(3) continuation using frozen callbacks.

    solve(world, seed) -> (joint goal or None, diagnostics)
    validate(previous, goal) -> a certificate with a boolean pass field
    No simulator motion or synthetic policy observations are produced.
    """
    start=np.asarray(start_world,dtype=float);end=np.asarray(end_world,dtype=float)
    rotations=Rotation.from_matrix(np.stack([start[:3,:3],end[:3,:3]]))
    distance=float(np.linalg.norm(end[:3,3]-start[:3,3]))
    angle=float((rotations[1]*rotations[0].inv()).magnitude())
    count=max(1,math.ceil(distance/rule['cartesian_step_m']),math.ceil(angle/rule['rotation_step_rad']))
    samples=np.linspace(0,1,count+1);orientations=Slerp([0,1],rotations)(samples).as_matrix()
    q=start_q;rows=[];joint_rows=[]
    result={'pass':False,'translation_m':distance,'rotation_rad':angle,'waypoints':rows,
        'joint_waypoints':joint_rows,'scope':'Planning certificate only; no physical execution/contact claim'}
    for index,alpha in enumerate(samples):
        world=np.eye(4);world[:3,3]=(1-alpha)*start[:3,3]+alpha*end[:3,3];world[:3,:3]=orientations[index]
        target,ik=solve(world,q)
        row={'index':index,'world':world.tolist(),'ik':ik};rows.append(row)
        if target is None:
            result['rejection']='Frozen local IK rejected Cartesian continuation';return None,result
        segment=validate(q,target);row['segment']=segment
        if not segment['pass']:
            result['rejection']='Frozen local limit/collision checks rejected continuation';return None,result
        q=target;joint_rows.append(q.tolist())
    result['pass']=True
    return q,result


def local_callbacks(planner,checker,names,limits,base_world):
    """Use the unchanged ARX local IK, precision and dense segment gates."""
    import torch
    from curobo.types import JointState,GoalToolPose
    from .observations import smooth_segment
    from .tracking import residual

    inverse=np.linalg.inv(np.asarray(base_world));bounds=limits.detach().cpu().numpy()
    def solve(world,seed):
        goal=inverse@world;quat=Rotation.from_matrix(goal[:3,:3]).as_quat();quat=np.r_[quat[3],quat[:3]]
        pose=GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.tensor(goal[:3,3],device=seed.device,dtype=torch.float32).reshape(1,1,1,1,3),
            quaternion=torch.tensor(quat,device=seed.device,dtype=torch.float32).reshape(1,1,1,1,4))
        result=planner.ik_solver.solve_pose(pose,current_state=JointState.from_position(seed[None],joint_names=names),return_seeds=4)
        diagnostics={'ik_success_count':int(result.success.sum()),'precise_collision_free_count':0}
        if not result.success.any():
            diagnostics['seed_fk_error']=residual(checker.kinematics,seed.detach().cpu().numpy(),goal,names)
            diagnostics['seed_collision_free']=bool(checker.validate(seed[None,None]).all())
            for name in ('position_error','rotation_error'):
                value=getattr(result,name,None)
                if isinstance(value,torch.Tensor):diagnostics['minimum_solver_'+name]=float(value.min())
            return None,diagnostics
        choices=result.solution[result.success].reshape(-1,len(names))
        choices=choices[checker.validate(choices[:,None]).flatten()]
        diagnostics['collision_free_count']=len(choices)
        for choice in choices[(choices-seed).square().sum(-1).argsort()]:
            error=residual(checker.kinematics,choice.detach().cpu().numpy(),goal,names)
            if error['position_error_m']<=.002 and error['orientation_error_rad']<=.03:
                diagnostics.update(precise_collision_free_count=1,selected_ik_error=error)
                return choice.clone(),diagnostics
        return None,diagnostics
    def validate(first,last):
        q,times=smooth_segment(first.detach().cpu().numpy(),last.detach().cpu().numpy())
        if ((q<bounds[:,0]-1e-6)|(q>bounds[:,1]+1e-6)).any():
            return {'pass':False,'rejection':'Frozen position limits'}
        alpha=np.arange(10)/10
        dense=np.concatenate([(q[:-1,None]*(1-alpha[None,:,None])+q[1:,None]*alpha[None,:,None]).reshape(-1,len(names)),q[-1:]])
        tensor=torch.tensor(dense,device=first.device,dtype=torch.float32)
        valid=torch.cat([checker.validate(batch[:,None]).flatten() for batch in tensor.split(64)])
        invalid=(~valid).nonzero().flatten().detach().cpu().tolist()
        return {'pass':not invalid,'dense_samples':len(dense),'invalid_samples':len(invalid),
            'invalid_indices':invalid,'minimum_joint_limit_margin_rad':float(np.minimum(q-bounds[:,0],bounds[:,1]-q).min()),
            'nominal_segment_seconds':float(times[-1]),'velocity_bound':1.5,'acceleration_bound':5.,'jerk_bound':100.}
    return solve,validate


def select(client,out,selected,start,q_grasp,path,planner,checker,names,limits,gripper_update,models,rule,entry_perturbation_world=None,camera_reference_dir=None):
    """Choose the first grasp-relative candidate with a certified continuation."""
    import torch
    from PIL import Image

    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    reference=Path(camera_reference_dir) if camera_reference_dir is not None else out/'handoff_camera_reference'
    observation=client.request('grasp_lilo_observation',directory=str(reference))
    snap=observation['snapshot'];ee=np.asarray(snap['link_world']['right_eef_link'])
    ee_camera=np.linalg.inv(ee)@np.asarray(observation['camera_world_optical'])
    with Image.open(observation['rgb_path']) as image:size=image.size
    points={'grasp_contact':selected['target_contact_world'],
        'object_center':np.asarray(snap['objects']['test_object']['world'])[:3,3].tolist()}
    positions=dict(zip(snap['joint_names'],snap['joint_positions']))
    gripper_update.apply(models,positions)
    solve,validate=local_callbacks(planner,checker,names,limits,snap['link_world']['body4'])
    grasp=np.asarray(selected['grasp_world']);options=candidates(grasp,rule,entry_perturbation_world)
    report={'rule':rule,'grasp_world':grasp.tolist(),'candidates':options,'accepted_name':None,
        'camera_reference':str(reference/'observation.json'),
        'entry_perturbation_world':entry_perturbation_world,
        'scope':'Planning/crop certificates only; actual entry is checked again after motion'}
    route=None
    for option in options:
        world=np.asarray(option['world'])
        option['frustum']=frustum(world,ee_camera,observation['intrinsics'],size,points,rule['image_margin_pixels'])
        if not option['frustum']['pass']:
            option['rejection']='Target/contact outside wrist image crop margin';continue
        # Start from the known grasp branch and validate backwards to entry.
        q_pre,proof=corridor(grasp,world,q_grasp,solve,validate,rule)
        option['reverse_continuation']=proof
        if q_pre is None:
            option['rejection']=proof['rejection'];continue
        candidate_route=path(start,q_pre)
        option['pregrasp_route_valid']=candidate_route is not None
        if candidate_route is None:
            option['rejection']='Frozen open-gripper PRM approach failed';continue
        option['q_pre']=q_pre.tolist();option['accepted']=True
        report['accepted_name']=option['name'];route=candidate_route
        selected.update(pregrasp_world=world.tolist(),q_pre=q_pre.tolist(),handoff_rule=rule['version'],
            handoff_backoff_m=option['backoff_m'],entry_perturbation_world=entry_perturbation_world)
        break
    planner.reset_seed()
    save('handoff_candidates.json',report)
    if route is None:raise RuntimeError('No grasp-relative handoff passed frozen continuation and wrist-frustum checks')
    save('handoff_target.json',{'desired_world':selected['pregrasp_world'],'rule':rule,
        'entry_perturbation_world':entry_perturbation_world,
        'grasp_world':selected['grasp_world'],'selected_candidate':report['accepted_name'],
        'scope':'Motion target selection only; robot/camera/scene configuration unchanged'})

    def actual_entry(observation):
        snap=observation['snapshot'];ee=np.asarray(snap['link_world']['right_eef_link'])
        camera=np.linalg.inv(ee)@np.asarray(observation['camera_world_optical'])
        with Image.open(observation['rgb_path']) as image:size=image.size
        points={'grasp_contact':selected['target_contact_world'],
            'object_center':np.asarray(snap['objects']['test_object']['world'])[:3,3].tolist()}
        view=frustum(ee,camera,observation['intrinsics'],size,points,rule['image_margin_pixels'])
        target=snap['objects']['test_object']['path']
        reach_files=[out/'pregrasp_execution.json']
        if selected.get('reach_via'):reach_files.insert(0,out/'pregrasp_via_execution.json')
        contacts=[e for file in reach_files for e in json.loads(file.read_text())['contacts']
            if e['num_contacts']>0 and target in e['a']+e['b']]
        result={'pass':False,'frustum':view,'pregrasp_target_contact_events':len(contacts),
            'frame_state_delta_seconds':observation['capture_sim_time']-observation['frame_time'],
            'actual_ee_world':ee.tolist(),'desired_ee_world':selected['pregrasp_world'],
            'scope':'Actual measured entry + planning certificate; not a grasp success claim'}
        if selected.get('reach_via'):result['reach_execution_files']=[str(file) for file in reach_files]
        if contacts and rule['reject_pregrasp_target_contact']:
            result['rejection']='Target was touched before VLA entry';return result
        if not view['pass']:
            result['rejection']='Actual target/contact outside wrist crop margin';return result
        positions=dict(zip(snap['joint_names'],snap['joint_positions']))
        gripper_update.apply(models,positions)
        current=torch.tensor([positions[n] for n in names],device=limits.device,dtype=torch.float32)
        solve,validate=local_callbacks(planner,checker,names,limits,snap['link_world']['body4'])
        _,proof=corridor(ee,grasp,current,solve,validate,rule)
        planner.reset_seed()
        result['continuation']=proof;result['pass']=proof['pass']
        if not result['pass']:result['rejection']=proof['rejection']
        return result
    return route,actual_entry
