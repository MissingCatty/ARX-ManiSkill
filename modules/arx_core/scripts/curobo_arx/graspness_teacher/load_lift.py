"""Prospective lift-goal selection under the unchanged ARX planner/models.

The object root is a load-point surrogate, not a measured centre of mass. This
cost omits robot gravity and contact dynamics; only real execution is success.
"""
import copy
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read,write,sha
from scripts.curobo_arx.data_factory.teacher import Rejection,bounded_planning_start


def goals(original, body, distances, minimum_radius):
    original=np.asarray(original,dtype=float);body=np.asarray(body,dtype=float)
    direction=body[:2,3]-original[:2,3];radius=np.linalg.norm(direction)
    result=[(0.,original.copy())]
    if radius<=minimum_radius:return result
    for distance in distances:
        if distance<=0:continue
        if radius-distance<minimum_radius:continue
        target=original.copy();target[:2,3]+=distance*direction/radius
        result.append((distance,target))
    return result


def fk_world(teacher, planner, q, body):
    from curobo.types import JointState
    state=JointState.from_position(q[None],joint_names=teacher.context['names'])
    output=planner.compute_kinematics(state).tool_poses
    position=output.position.detach().cpu().numpy().reshape(-1,3)
    quat=output.quaternion.detach().cpu().numpy().reshape(-1,4)
    values=np.tile(np.eye(4),(len(position),1,1))
    values[:,:3,:3]=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();values[:,:3,3]=position
    return np.asarray(body)@values


def load_score(teacher, planner, q, body, root_in_eef, mass, limits,epsilon=.001):
    if not 0<epsilon<=.001:raise ValueError('Invalid finite-difference diagnostic step')
    q=q.detach().cpu().numpy();jac=np.zeros((3,len(q)));root_jac=np.zeros(len(q))
    for axis in range(len(q)):
        lo,hi=q.copy(),q.copy();lo[axis]=max(limits[axis,0],q[axis]-epsilon);hi[axis]=min(limits[axis,1],q[axis]+epsilon)
        delta=hi[axis]-lo[axis]
        if delta<=0:raise RuntimeError('No finite derivative inside frozen joint limits')
        joint=teacher.torch.as_tensor(np.stack([lo,hi]),device='cuda',dtype=teacher.torch.float32)
        pose=fk_world(teacher,planner,joint,body)
        jac[:,axis]=(pose[1,:3,3]-pose[0,:3,3])/delta
        root=(pose[:,:3,:3]@root_in_eef[:,None]).squeeze(-1)+pose[:,:3,3]
        root_jac[axis]=(root[1,2]-root[0,2])/delta
    torque=-mass*9.81*root_jac
    return {'moment_sensitivity_score':float(np.linalg.norm(jac@torque)),
            'maximum_object_root_gravity_moment_proxy_nm':float(np.max(np.abs(torque))),
            'object_root_moment_proxy_nm':torque.tolist(),
            'scope':'Object-root rigid-FK surrogate; not measured force, true COM, or predicted actual tracking error'}


def retime(command, poses, speed=.2, angular_speed=2.):
    times=np.asarray(command['time_from_start'],dtype=float);dt=np.diff(times)
    if len(times)!=len(poses) or not len(dt) or (dt<=0).any():raise RuntimeError('Invalid lift time/pose sequence')
    velocity=float((np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/dt).max())
    delta=poses[1:,:3,:3]@poses[:-1,:3,:3].transpose(0,2,1)
    omega=float((Rotation.from_matrix(delta).magnitude()/dt).max())
    scale=max(1.,velocity/speed,omega/angular_speed)
    if scale>5:raise Rejection('lift_action_safe_retiming_exceeds_registered_bound')
    result=copy.deepcopy(command);result['time_from_start']=(times*scale).tolist()
    return result,{'scale':scale,'original_maximum_eef_speed_m_s':velocity,
        'original_maximum_eef_angular_speed_rad_s':omega,'selected_speed_bound_m_s':speed,
        'selected_angular_speed_bound_rad_s':angular_speed,'joint_positions_unchanged':True,
        'joint_velocity_acceleration_limits_modified':False}


def lift_teacher(base):
    class Teacher(base):
        def handle(self,request):
            active=request.get('comparison_arm')=='variant' if request.get('comparison_arm') is not None else self.registration.get('production') is True
            if request['action']!='plan' or request.get('phase')!='lift' or not active:
                return super().handle(request)
            policy=self.registration['load_aware_lift']
            if policy!={'distances_m':[.075,.15,.225,.30],'minimum_radius_m':.30,
                        'minimum_proxy_reduction':.25,'maximum_plan_candidates':4,
                        'maximum_planned_downward_excursion_m':.001}:
                raise ValueError('Unregistered load-aware lift policy')
            snapshot=request['snapshot'];self.initialize(read(request['scene']),snapshot)
            names=self.context['names'];planner,checker,updater,models=self.context['closed']
            measured=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
            updater.apply(models,{n:float(measured[n]) for n in updater.names})
            limits=np.array([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names])
            try:start=bounded_planning_start([measured[n] for n in names],limits)
            except Rejection:return super().handle(request)
            start=self.torch.as_tensor(start,device='cuda',dtype=self.torch.float32)
            body=np.asarray(snapshot['link_world']['body4']);inverse=np.linalg.inv(body)
            relative=np.linalg.inv(snapshot['link_world']['right_eef_link'])@np.asarray(snapshot['objects']['test_object']['world'])
            asset=next(a for a in self.registration['catalog']['objects']
                if a['asset_sha256']==request['visual_context']['asset']['asset_sha256'])
            candidates=[];rejected=[]
            for distance,desired in goals(request['goal_world'],body,policy['distances_m'],policy['minimum_radius_m']):
                try:
                    q=self.ik(desired,start,planner,checker,inverse)
                    score=load_score(self,planner,q,body,relative[:3,3],asset['mass_kg'],limits)
                    candidates.append({'distance_m':distance,'goal_world':desired.tolist(),**score})
                except Rejection as exc:rejected.append({'distance_m':distance,'rejected':str(exc)})
            original=next((c for c in candidates if c['distance_m']==0),None)
            if original is None:return super().handle(request)
            selected=[c for c in candidates if c['distance_m'] and c['moment_sensitivity_score']
                <=original['moment_sensitivity_score']*(1-policy['minimum_proxy_reduction'])]
            selected.sort(key=lambda c:(c['moment_sensitivity_score'],c['distance_m']))
            root=Path(request['output']);search=root/'lift_goal_search';search.mkdir(exist_ok=False)
            trace={'object_mass_kg':asset['mass_kg'],'baseline':original,'candidates':candidates,
                   'goal_rejections':rejected,'prospective_before_command':True,'physical_commands_in_search':0,
                   'robot_physics_calibration_and_original_evaluator_changed':False}
            for index,candidate in enumerate(selected[:policy['maximum_plan_candidates']]):
                folder=search/str(index);folder.mkdir();changed=copy.deepcopy(request)
                changed.update(goal_world=candidate['goal_world'],output=str(folder))
                result=super().handle(changed)
                write(folder/'virtual_planning_response.json',{'planning_only':True,'physical_commands':0,'result':result})
                trace.setdefault('plan_attempts',[]).append({'index':index,'candidate':candidate,
                    'directory':str(folder),'planning_rejected':result.get('rejected')})
                if result.get('rejected'):
                    trace.setdefault('plan_rejections',[]).append({'candidate':candidate,'reason':result['rejected']});continue
                command=result['command'];joint=self.torch.tensor(command['position'],device='cuda',dtype=self.torch.float32)
                poses=fk_world(self,planner,joint,body);target=self.visual_target(request)
                object_poses=poses@relative
                bottoms=[float((target.mesh.vertices@pose[2,:3]+pose[2,3]).min()) for pose in object_poses]
                radii=np.linalg.norm(poses[:,:2,3]-body[None,:2,3],axis=1)
                if min(bottoms)<bottoms[0]-policy['maximum_planned_downward_excursion_m'] or radii.min()<policy['minimum_radius_m']:
                    trace.setdefault('plan_rejections',[]).append({'candidate':candidate,'reason':'additional_carried_mesh_floor_or_body_radius_check'});continue
                actual_score=load_score(self,planner,joint[-1],body,relative[:3,3],asset['mass_kg'],limits)
                if actual_score['moment_sensitivity_score']>original['moment_sensitivity_score']*(1-policy['minimum_proxy_reduction']):continue
                try:command,timing=retime(command,poses)
                except Rejection as exc:
                    trace.setdefault('plan_rejections',[]).append({'candidate':candidate,'reason':str(exc)});continue
                files={str(p.relative_to(root)):sha(p) for p in folder.iterdir() if p.is_file()}
                trace.update(selected=candidate,selected_endpoint_score=actual_score,retiming=timing,
                             selected_plan_files=files,planned_bottom_excursion_m=float(min(bottoms)-bottoms[0]))
                for path in folder.iterdir():
                    if path.is_file():
                        dest=root/path.name
                        if dest.exists():raise RuntimeError('Lift planning artifact already exists; do not overwrite occurred evidence')
                        shutil.copy2(path,dest)
                opt=root/'lift_optimization.json';evidence=read(opt);evidence['prospective_motion_goal_selection']=trace;write(opt,evidence)
                result['command']=command;result['load_aware_lift']=trace
                return result
            result=super().handle(request)
            if (root/'lift_optimization.json').exists():
                evidence=read(root/'lift_optimization.json');evidence['prospective_motion_goal_selection']={**trace,'selected':'original_goal_fallback'};write(root/'lift_optimization.json',evidence)
            return result
    return Teacher
