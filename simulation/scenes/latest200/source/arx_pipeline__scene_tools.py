"""Scene geometry, camera admission and fixed-rule wrist observation preparation."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ARM = [f"right_joint{i}" for i in range(21,27)]
DB = json.loads((ROOT/"ManiSkill/assets/data/assets/mani_skill2_ycb/info_pick_v0.json").read_text())


def corners(info):
    lo,hi=np.array(info['bbox']['min']),np.array(info['bbox']['max'])
    return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])*info.get('scales',[1.])[0]


def export_geometry(env):
    cache=ROOT/'ManiSkill/cache/arx_scene_collision';cache.mkdir(exist_ok=True)
    exported={}
    for name,actor in env.objects.items():
        vertices=[];faces=[];offset=0
        shapes=actor._bodies[0].get_collision_shapes()
        for shape in shapes:
            v=np.asarray(shape.vertices,dtype=np.float64)*np.asarray(shape.scale)
            t=shape.local_pose.to_transformation_matrix()
            v=v@t[:3,:3].T+t[:3,3]
            f=np.asarray(shape.triangles,dtype=np.int64).reshape(-1,3)
            vertices.append(v);faces.append(f+offset);offset+=len(v)
        v=np.concatenate(vertices);f=np.concatenate(faces)
        key=hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()
        path=cache/(key+'.ply')
        header=f'ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n'
        payload=(header+''.join(' '.join(format(x,'.17g') for x in p)+'\n' for p in v)+
                 ''.join('3 '+' '.join(map(str,face))+'\n' for face in f)).encode()
        if path.exists() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(payload).digest():
            raise RuntimeError('Native collider cache integrity failure: '+str(path))
        if not path.exists():
            path.write_bytes(payload)
        exported[name]={'file_path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'geometry_sha256':key,'native_convex_parts':len(shapes),
                        'model_id':env.object_models[name],'target':name=='target',
                        'mass_kg':float(actor._bodies[0].mass)}
    env.exported_geometry=exported
    return exported


def mesh_state(env):
    geometry=getattr(env,'exported_geometry',None) or export_geometry(env)
    return {n:{**g,'world':env.objects[n].pose.to_transformation_matrix().cpu().numpy()[0].tolist()}
            for n,g in geometry.items()}


def visibility(env,obs=None):
    import sapien
    if obs is None:obs=env.get_obs()
    role=env.perception_camera
    params=obs['sensor_param'][role]
    gl=params['cam2world_gl'].cpu().numpy()[0]
    optical=gl@np.diag([1,-1,-1,1]);k=params['intrinsic_cv'].cpu().numpy()[0]
    world=env.obj.pose.to_transformation_matrix().cpu().numpy()[0]
    pts=corners(env.object_info['target'])@world[:3,:3].T+world[:3,3]
    cam=(pts-optical[:3,3])@optical[:3,:3]
    uv=(cam@k.T);uv=uv[:,:2]/uv[:,2:]
    seg=obs['sensor_data'][role]['segmentation'][0,:,:,0].cpu().numpy()
    mask=seg==env.obj._objs[0].per_scene_id
    full=bool(cam[:,2].min()>.05 and uv[:,0].min()>=10 and uv[:,0].max()<630 and uv[:,1].min()>=10 and uv[:,1].max()<470)
    hidden=[]
    try:
        for name,actor in env.objects.items():
            if name=='target':continue
            body=actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
            if body is not None:hidden.append((body,body.visibility));body.visibility=0
        isolated=env.get_obs()['sensor_data'][role]['segmentation'][0,:,:,0].cpu().numpy()
        isolated_count=int((isolated==env.obj._objs[0].per_scene_id).sum())
    finally:
        for body,value in hidden:body.visibility=value
    count=int(mask.sum())
    return {'target_mask_pixels':count,'isolated_mask_pixels':isolated_count,
            'occlusion_fraction':float(np.clip(1-count/max(1,isolated_count),0,1)),
            'projected_bbox_uv':[uv.min(0).tolist(),uv.max(0).tolist()],
            'fully_in_frame':full,'admitted_view':bool(full and count>=50)}


class ArmFK:
    def __init__(self):
        path=ROOT/'COOLAS-ARX/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
        root=ET.parse(path).getroot();bychild={j.find('child').get('link'):j for j in root.findall('joint')}
        chain=[];child='right_eef_link'
        while child in bychild:
            j=bychild[child];chain.append(j);child=j.find('parent').get('link')
        self.chain=[];self.lower=[];self.upper=[]
        for j in reversed(chain):
            origin=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(origin.get('xyz'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy'),sep=' ')).as_matrix()
            index=ARM.index(j.get('name')) if j.get('name') in ARM else None
            if index is not None:
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
                self.lower.append(float(j.find('limit').get('lower')));self.upper.append(float(j.find('limit').get('upper')))
            else:skew=None
            self.chain.append((t,index,skew))
        self.lower=np.array(self.lower);self.upper=np.array(self.upper)
        self.wrist_from_eef=np.eye(4);self.wrist_from_eef[:3,3]=[.11,0,-.006]
        self.wrist_from_eef[:3,:3]=Rotation.from_euler('y',np.pi/2).as_matrix()

    def fk(self,q):
        out=np.eye(4)
        for t,index,s in self.chain:
            out=out@t
            if index is not None:
                m=np.eye(4);m[:3,:3]+=np.sin(q[index])*s+(1-np.cos(q[index]))*(s@s);out=out@m
        return out

    def observation(self,env,seed):
        spec=env.camera_alignment['camera_definition']
        local_gl=np.asarray(spec['xform_rows']).T
        world=env.obj.pose.to_transformation_matrix().cpu().numpy()[0]
        box=corners(env.object_info['target']);center=world[:3,:3]@box.mean(0)+world[:3,3]
        base=env.agent.robot.links_map['body4'].pose.to_transformation_matrix().cpu().numpy()[0]
        invbase=np.linalg.inv(base)
        names=[j.name for j in env.agent.robot.active_joints]
        current=env.agent.robot.qpos[0].cpu().numpy()[[names.index(n) for n in ARM]]
        rng=np.random.default_rng(seed)
        offsets=([[-.30,-.08,.30],[-.22,-.16,.30],[-.34,.05,.28],[.08,-.22,.32]]
                 if env.scene_spec and env.scene_spec.get('layout')=='S5_occluded'
                 else [[-.14,-.03,.32],[-.10,-.12,.34],[-.20,-.08,.28],[.08,-.15,.30],[-.05,.10,.34]])
        for view,offset in enumerate(offsets):
            eye=center+offset;forward=center-eye;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            gl=np.eye(4);gl[:3,:3]=np.stack([right,up,-forward],axis=1);gl[:3,3]=eye
            goal=gl@np.linalg.inv(local_gl)@self.wrist_from_eef;target=invbase@goal
            def residual(q):
                actual=self.fk(q)
                return np.r_[5*(actual[:3,3]-target[:3,3]),Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()]
            seeds=[current,np.array([0,1.5,2.0,-.8,0,0])]+[rng.uniform(self.lower,self.upper) for _ in range(5)]
            for initial in seeds:
                result=least_squares(residual,np.clip(initial,self.lower+1e-5,self.upper-1e-5),bounds=(self.lower+1e-5,self.upper-1e-5),max_nfev=150)
                actual=self.fk(result.x);pe=np.linalg.norm(actual[:3,3]-target[:3,3]);re=Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).magnitude()
                if pe<.0005 and re<.005:
                    return {'q':result.x.tolist(),'goal_world':goal.tolist(),'view_index':view,
                            'policy':'Fixed camera offsets aimed at registered scene target bbox center; no grasp candidate or success screening',
                            'position_error_m':float(pe),'rotation_error_rad':float(re)}
        raise RuntimeError('No observation IK for declared scene')
