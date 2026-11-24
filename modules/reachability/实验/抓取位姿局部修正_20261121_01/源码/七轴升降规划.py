"""Isolated seven-DOF reach planner and executor; original models stay immutable."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

CODE=Path(__file__).resolve().parent
MAP=CODE.parent/'reachability'
PILOT=MAP/'实验/升降联合规划_20261120_01'
SOURCE=MAP/'工具/验证升降联合规划.py'
ARM=[f'right_joint{i}' for i in range(21,27)]
NAMES=['joint4']+ARM
ACTIVE=None

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def origin(node):
    t=np.eye(4)
    if node is not None:
        t[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t

class BodySweep:
    """Exact convex swept volume of the lift-carried torso and locked left arm.

    These parts only translate along the authored lift axis. Their convex hull
    between the minimum and maximum lift is the full swept solid, not sparse
    waypoint sampling. Right-arm checking keeps the original cuRobo spheres.
    """
    def __init__(self):
        import fcl,trimesh
        self.fcl,self.trimesh=fcl,trimesh
        self.urdf=CODE/'assets/arx/robot.urdf';root=ET.parse(self.urdf).getroot()
        self.joints=root.findall('joint');self.parts=[]
        for link in root.findall('link'):
            name=link.get('name')
            if name!='body4' and not name.startswith('left_link'):continue
            for collision in link.findall('collision'):
                node=collision.find('geometry/mesh');path=self.urdf.parent/node.get('filename')
                mesh=trimesh.load_mesh(path,process=False).convex_hull
                vertices=np.asarray(mesh.vertices)*np.fromstring(node.get('scale','1 1 1'),sep=' ')
                t=origin(collision.find('origin'));vertices=vertices@t[:3,:3].T+t[:3,3]
                self.parts.append((name,vertices))
        assert len(self.parts)==9

    def fk(self,state,changes=None):
        q=dict(zip(state['joint_names'],state['qpos']));q.update(changes or {})
        transforms={'base_link':np.asarray(state['links']['base_link'])};pending=self.joints.copy()
        while pending:
            before=len(pending)
            for j in pending[:]:
                parent=j.find('parent').get('link')
                if parent not in transforms:continue
                t=origin(j.find('origin'));kind=j.get('type')
                if kind!='fixed':
                    axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');v=q[j.get('name')]
                    delta=np.eye(4)
                    if kind=='prismatic':delta[:3,3]=axis*v
                    else:delta[:3,:3]=Rotation.from_rotvec(axis*v).as_matrix()
                    t=t@delta
                transforms[j.find('child').get('link')]=transforms[parent]@t;pending.remove(j)
            assert len(pending)<before,'URDF tree could not be resolved'
        return transforms

    def convex(self,vertices):
        mesh=self.trimesh.convex.convex_hull(vertices)
        faces=np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).astype(np.int32).reshape(-1)
        return self.fcl.Convex(np.asarray(mesh.vertices),len(mesh.faces),faces)

    def check(self,state,lo,hi):
        fcl=self.fcl;now=state['qpos'][state['joint_names'].index('joint4')]
        lo,hi=float(lo),float(hi)
        assert 0<=lo<=hi<=.4600001
        poses=self.fk(state);obstacles=[]
        for name,g in state['scene_cuboids'].items():
            obstacles.append((name,fcl.CollisionObject(fcl.Box(*g['dims']),fcl.Transform(np.array(g['center'])))))
        for name,g in state.get('scene_meshes',{}).items():
            mesh=self.trimesh.load_mesh(g['file_path'],process=False)
            parts=mesh.split(only_watertight=False)
            assert len(parts)==g['native_convex_parts'],(name,len(parts),g['native_convex_parts'])
            t=np.asarray(g['world'])
            for part in parts:
                vertices=np.asarray(part.vertices)@t[:3,:3].T+t[:3,3]
                obstacles.append((name,fcl.CollisionObject(self.convex(vertices))))
        for name,vertices in self.parts:
            t=poses[name];world=vertices@t[:3,:3].T+t[:3,3]
            a=world+np.array([0,0,lo-now]);b=world+np.array([0,0,hi-now])
            swept=fcl.CollisionObject(self.convex(np.vstack([a,b])))
            for obstacle,obj in obstacles:
                if fcl.collide(swept,obj,fcl.CollisionRequest(),fcl.CollisionResult()):
                    return {'passed':False,'link':name,'obstacle':obstacle,'lift_interval_m':[lo,hi]}
        return {'passed':True,'links':[n for n,_ in self.parts],'lift_interval_m':[lo,hi],
            'method':'Exact convex translational sweep against native convex object parts and unchanged table/ground',
            'static_base_and_wheels':'Unchanged fixed-root geometry; initial reset/contact validation retained',
            'self_collision_policy':'Original disabled self-collision policy retained'}

def seven_class(root,state):
    """Reuse the already validated pilot implementation without editing it."""
    import torch,yaml
    from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    from curobo.types import GoalToolPose,JointState
    from curobo._src.types.robot import RobotCfg
    from curobo._src.graph_planner.graph_planner_prm import TrajInterpolationType
    from resident_planner import all_kinematics_params,collision_scene_config
    from optimize_path import optimize_path,flush_planning_artifacts
    sys.path.insert(0,str(MAP/'工具'))
    from 机器人几何 import RobotGeometry
    tcp=read(CODE.parent/'curobo_adapter/research/20261117_07_action_tcp/action_tcp_43mm.json')
    profile=np.array(tcp['T_eef_action']);geometry=RobotGeometry(profile,device='cuda')
    root.mkdir();cfg=yaml.safe_load((PILOT/'七关节.yml').read_text())
    (root/'七关节.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    sim=ET.parse(CODE/'assets/arx/robot.urdf').getroot();j=next(j for j in sim.findall('joint') if j.get('name')=='joint4')
    base_origin=origin(j.find('origin'))
    def scene(s,base):
        inv=np.linalg.inv(base);world={'cuboid':{},'mesh':{}}
        def pose(t):return [*t[:3,3].tolist(),*Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]].tolist()]
        for n,g in s['scene_cuboids'].items():
            t=np.eye(4);t[:3,3]=g['center'];world['cuboid'][n]={'pose':pose(inv@t),'dims':g['dims']}
        for n,g in s.get('scene_meshes',{}).items():
            if not g['target']:world['mesh'][n]={'pose':pose(inv@np.asarray(g['world'])),'file_path':g['file_path']}
        return collision_scene_config(world)
    parsed=ast.parse(SOURCE.read_text());run=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    cls=next(n for n in run.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    task=CODE.parent/'curobo/source/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml'
    scope={**locals(),**globals(),'out':root,'reg':{'lift_limits_m':[0.,.46000000834465027]},
        'origin':base_origin,'JOINTS':ARM,'TASK':task,'model_loads':{},'fk_checks':[]}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SOURCE), 'exec'),scope)
    model=scope['Planner'](7,state)
    # Freeze the source-level FK proof at startup before any execution.
    q=model.measured();model.check_fk(q[None]);assert model.validate_dense(q[None])
    path=model.path_to(q,q);assert path is not None
    model.optimize(path,root,'seven_startup')
    save(root/'预热核验.json',{'passed':True,'source':str(SOURCE),'dof':7,'physical_commands':0,
        'loads':scope['model_loads'],'fk':scope['fk_checks']})
    return model

def install_planner():
    import torch
    import fast_resident_planner as fast
    from short_lift_planner import ShortLiftPlanner
    from 比较升降高度 import shifted
    Base=fast.FastPlannerPool
    class Pool(Base):
        def __init__(self,root,warmup_trial):
            global ACTIVE
            super().__init__(root,warmup_trial)
            self.body_sweep=BodySweep();self.seven=seven_class(root/'joint7',read(warmup_trial/'state.json'));ACTIVE=self

        def request(self,directory,**kwargs):
            result=super().request(directory,**kwargs)
            if kwargs.get('approach') or kwargs.get('lift'):
                state=read(Path(directory)/('closed_state.json' if kwargs.get('lift') else 'pregrasp_state.json'))
                h=state['qpos'][state['joint_names'].index('joint4')];proof=self.body_sweep.check(state,h,h)
                save(Path(directory)/('抬升随动碰撞核验.json' if kwargs.get('lift') else '接近随动碰撞核验.json'),proof)
                if not proof['passed']:
                    result={'success':False,'reason':'lift_carried_body_scene_collision','body_sweep':proof}
                    save(Path(directory)/('lift_plan_result.json' if kwargs.get('lift') else 'approach_plan_result.json'),result)
            return result

    def select(self,state,rows,camera):
        pool=ACTIVE;model=pool.seven;model.bind(state);start=model.measured();base_out=self.out
        candidates=read(base_out.parent/'graspcl/candidate_pool.json')['candidates'];rejected=[]
        current=state['qpos'][state['joint_names'].index('joint4')]
        for index,candidate in enumerate(candidates):
            contact=np.array(candidate['pose']);g=contact.copy();g[:3,3]-=.043*g[:3,2]
            h=g.copy();h[:3,3]-=.08*g[:3,2];l=g.copy();l[2,3]+=.03
            qs=model.ik(g,start)
            if not len(qs):rejected.append({'index':index,'reason':'grasp_ik'});continue
            for branch,qg7 in enumerate(qs):
                height=float(qg7[0]);proof=pool.body_sweep.check(state,min(current,height),max(current,height))
                if not proof['passed']:rejected.append({'index':index,'branch':branch,'reason':'lift_carried_body_scene_collision',**proof});continue
                virtual=shifted(state,height,current)
                six=pool.bind(base_out/f'choice_{index}_{branch}',virtual)
                qg=qg7[1:].clone();qh,ql=six.ik(h,qg),six.ik(l,qg)
                if qh is None or ql is None:rejected.append({'index':index,'branch':branch,'reason':'pre_or_lift_ik'});continue
                qh7=torch.cat([qg7[:1],qh]);reach=model.path_to(start,qh7);approach=six.path(qh,qg)
                if reach is None or approach is None:rejected.append({'index':index,'branch':branch,'reason':'path'});continue
                # Optimization can overshoot the endpoint height; validate the entire result too.
                d=base_out/f'choice_{index}_{branch}';model.optimize(reach,d,'到预抓取')
                command=read(d/'到预抓取_指令.json');q=np.asarray(command['q'])
                proof=pool.body_sweep.check(state,float(q[:,0].min()),float(q[:,0].max()))
                if not proof['passed']:rejected.append({'index':index,'branch':branch,'reason':'optimized_body_sweep_collision',**proof});continue
                selected={'index':index,'score':candidate['score'],'candidate_metadata':candidate['metadata'],
                    'candidate_protocol':'arx_tcp43_native_post_truncation_lift30_hold10_v1','contact_pose_world':contact.tolist(),
                    'grasp_world':g.tolist(),'pregrasp_world':h.tolist(),'lift_world':l.tolist(),
                    'target_contact_world':contact[:3,3].tolist(),'table_clearance_raise_m':0.,
                    'q_grasp':qg.tolist(),'q_pre':qh.tolist(),'q_lift':ql.tolist(),
                    'selected_lift_m':height,'reach_joint_names':NAMES,'q_pre_7':qh7.tolist(),
                    'lift_fixed_during_approach_and_lift':True,'body_sweep':proof}
                save(base_out.parent/'七轴规划核验.json',{'passed':True,'initial_lift_m':current,'selected_lift_m':height,
                    'joint_names':NAMES,'candidate_index':index,'ik_branch':branch,'body_sweep':proof,
                    'native_candidate_goal_preserved':True,'physical_robot_configuration_edits':0})
                return {'success':True,'selected':selected,'rejected':rejected,'paths':{
                    'pregrasp':{'q':command['q'],'times':command['times_s'],'joint_names':NAMES},
                    'approach':six.optimize('approach',approach,5.)}}
        return {'success':False,'reason':'no_feasible_native_post_or_recalled_candidate','rejected':rejected,'count':len(candidates)}
    fast.FastPlannerPool=Pool;ShortLiftPlanner.select_tcp43=select

def install_execution(run):
    Base=run.Execution
    class Execution(Base):
        def segment(self,label,q,times,finger=.044,goal_world=None):
            q=np.asarray(q);names=NAMES if q.shape[1]==7 else ARM
            assert q.shape[1] in (6,7)
            if q.shape[1]==6:return super().segment(label,q,times,finger=finger,goal_world=goal_world)
            assert label=='pregrasp'
            indices=[self.names.index(n) for n in names];actual=self.env.agent.robot.qpos[0,indices].cpu().numpy();measured_start=actual.copy()
            assert abs(actual[0]-q[0,0])<.001 and np.max(np.abs(actual[1:]-q[0,1:]))<.005
            dt=np.diff(times);assert (dt>0).all() and np.min(q[:,0])>=0 and np.max(q[:,0])<=.4600001
            assert np.max(np.abs(np.diff(q[:,0])/dt))<=.1501
            self.label=label;steps=max(1,int(np.ceil(times[-1]*40)))
            for k in range(steps+1):
                t=min(k/40,times[-1]);self.command[indices]=[np.interp(t,times,q[:,j]) for j in range(7)]
                self.command[self.finger_indices]=finger;self.step()
            for _ in range(160):
                actual=self.env.agent.robot.qpos[0,indices].cpu().numpy()
                if abs(actual[0]-q[-1,0])<.001 and np.max(np.abs(actual[1:]-q[-1,1:]))<.005:break
                self.step()
            actual=self.env.agent.robot.qpos[0,indices].cpu().numpy();error=np.abs(actual-q[-1])
            stage={'joint_error_rad':float(error[1:].max()),'lift_error_m':float(error[0]),
                'settled':bool(error[0]<.001 and error[1:].max()<.005),'planned_seconds':float(times[-1]),
                'joint_names':names,'measured_start':measured_start.tolist(),'commanded_goal':q[-1].tolist()}
            if goal_world is not None:
                got=self.env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0];goal=np.asarray(goal_world)
                stage.update(tool_position_error_m=float(np.linalg.norm(got[:3,3]-goal[:3,3])),
                    tool_rotation_error_rad=float(Rotation.from_matrix(got[:3,:3]@goal[:3,:3].T).magnitude()))
            self.stages[label]=stage;save(self.out/'stages.json',self.stages)
            if not stage['settled']:raise RuntimeError(label+'_not_converged')
    run.Execution=Execution

def training_probe(env,run,directory):
    """Two bounded lift motions on an independent training scene, no grasp."""
    import torch
    names=[j.name for j in env.agent.robot.active_joints];index=names.index('joint4')
    command=np.array(env.initial_q,copy=True);events=[];samples=[]
    robot_entities={link._objs[0].entity for link in env.agent.robot.get_links()}
    def trace():
        for contact in env.scene.get_contacts():
            a,b=[v.entity for v in contact.bodies]
            if (a in robot_entities)!=(b in robot_entities):
                impulse=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
                if impulse>1e-8:events.append([a.name,b.name,impulse])
    env.trace_callback=trace
    fk=BodySweep.__new__(BodySweep);fk.joints=ET.parse(CODE/'assets/arx/robot.urdf').getroot().findall('joint')
    try:
        for target in [.442,.452]:
            start=float(env.agent.robot.qpos[0,index])
            for tick in range(81):
                command[index]=start+(target-start)*tick/80
                env._step_action(torch.tensor(command[None],device=env.device))
            for _ in range(40):env._step_action(torch.tensor(command[None],device=env.device))
            state=run.snapshot(env);actual=state['qpos'][index];computed=fk.fk(state)
            error=float(np.linalg.norm(computed['right_eef_link'][:3,3]-np.array(state['links']['right_eef_link'])[:3,3]))
            assert abs(actual-target)<.001 and error<2e-5 and not events
            samples.append({'target_m':target,'actual_m':actual,'independent_fk_error_m':error})
    finally:env.trace_callback=None
    save(directory,{'passed':True,'training_object':'014_lemon','physical_grasps':0,
        'bounded_lift_motion_segments':2,'samples':samples,'forbidden_contacts':events,
        'original_arm_commands_drives_and_physics_retained':True})
