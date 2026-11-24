"""Derive the ARX planning chain and attach the current 871-sphere profile."""
import argparse,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from collision_profile import active_spheres, require_current_profile


def transform(pos,quat):
    out=np.eye(4); out[:3,:3]=Rotation.from_quat([*quat.GetImaginary(),quat.GetReal()]).as_matrix();out[:3,3]=pos
    return out


def origin(joint,matrix):
    ET.SubElement(joint,'origin',xyz=' '.join(map(str,matrix[:3,3])),rpy=' '.join(map(str,Rotation.from_matrix(matrix[:3,:3]).as_euler('xyz'))))


def build(asset,snapshot,out,fine_gripper=False,gripper_pitch=0.008):
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.Open(str(asset)); cache=UsdGeom.XformCache()
    selected={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and (p.GetName().startswith('right_') or p.GetName()=='body4')}
    robot=ET.Element('robot',name='arx_usd_right_arm')
    for name in selected: ET.SubElement(robot,'link',name=name)
    parents={}; active=[]; limits={}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint): continue
        joint=UsdPhysics.Joint(prim); p0=joint.GetBody0Rel().GetTargets();p1=joint.GetBody1Rel().GetTargets()
        if not p0 or not p1 or p0[0].name not in selected or p1[0].name not in selected: continue
        parent,child=p0[0].name,p1[0].name;parents[child]=parent
        kind='revolute' if prim.IsA(UsdPhysics.RevoluteJoint) else 'prismatic' if prim.IsA(UsdPhysics.PrismaticJoint) else 'fixed'
        virtual=child+'_joint_frame'; ET.SubElement(robot,'link',name=virtual)
        j=ET.SubElement(robot,'joint',name=prim.GetName(),type=kind)
        ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=virtual)
        origin(j,transform(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()))
        if kind!='fixed':
            axis=prim.GetAttribute('physics:axis').Get(); xyz=np.eye(3)['XYZ'.index(axis)]
            ET.SubElement(j,'axis',xyz=' '.join(map(str,xyz)))
            lo=float(prim.GetAttribute('physics:lowerLimit').Get()); hi=float(prim.GetAttribute('physics:upperLimit').Get())
            if kind=='revolute':lo,hi=math.radians(lo),math.radians(hi)
            ET.SubElement(j,'limit',lower=str(lo),upper=str(hi),effort='100',velocity='1.5')
            active.append(prim.GetName());limits[prim.GetName()]=[lo,hi]
        fixed=ET.SubElement(robot,'joint',name=child+'_frame_offset',type='fixed')
        ET.SubElement(fixed,'parent',link=virtual);ET.SubElement(fixed,'child',link=child)
        origin(fixed,np.linalg.inv(transform(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())))
    # Old callers may still supply fine_gripper/gripper_pitch; these no longer
    # select or regenerate a retired voxel model. Every new model uses 871.
    spheres=active_spheres()
    if not set(spheres).issubset(selected):raise ValueError('ARX links do not match the active collision profile')
    measured=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    measured={n:float(np.clip(v,limits[n][0],limits[n][1])) if n in limits else v for n,v in measured.items()}
    locks={n:measured[n] for n in active if n not in snapshot['right_arm']}
    ignore={name:[other for other in spheres if other!=name and (parents.get(name)==other or parents.get(other)==name or parents.get(name)==parents.get(other))] for name in spheres}
    # Wrist/gripper collision parts intentionally meet at adjacent fixed hardware.
    ignore['right_link23']=list(set(ignore['right_link23']+['right_link25']))
    ignore['right_link24']=list(set(ignore['right_link24']+['right_link26']))
    ignore['right_link25']=list(set(ignore['right_link25']+['right_link27','right_link28']))
    cfg={'robot_cfg':{'kinematics':{'format_version':2.0,'urdf_path':str((out/'arx.urdf').resolve()),'base_link':'body4','tool_frames':['right_eef_link'],
         'collision_link_names':list(spheres),'collision_spheres':spheres,'collision_sphere_buffer':0.0,'self_collision_ignore':ignore,
         'self_collision_buffer':{n:0.0 for n in spheres},'lock_joints':locks,
         'cspace':{'joint_names':active,'default_joint_position':[measured[n] for n in active],'cspace_distance_weight':[1.0]*len(active),
                   'null_space_weight':[1.0]*len(active),'max_acceleration':5.0,'max_jerk':100.0}}}}
    ET.indent(robot);ET.ElementTree(robot).write(out/'arx.urdf',encoding='unicode')
    (out/'arx.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    profile=require_current_profile(cfg)
    (out/'model_manifest.json').write_text(json.dumps({'source':str(asset),'spheres':{n:len(v) for n,v in spheres.items()},'locks':locks,'joint_limits':limits,'collision_profile':profile},indent=2))
    print('Generated',out/'arx.yml',sum(map(len,spheres.values())),'conservative collision spheres')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    build(a.asset,json.loads(a.snapshot.read_text()),a.out)
