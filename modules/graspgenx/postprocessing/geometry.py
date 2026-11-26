"""Observed-point and table checks using the authored ARX gripper colliders."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import ROOT, read_json
from modules.arx_core.arms import get_arm
from modules.graspgenx.arx_frames import TCP_M, generator_to_contact


def origin(node):
    result=np.eye(4)
    if node is not None:
        result[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        result[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return result


@dataclass
class ConvexPart:
    link: str
    vertices: np.ndarray
    equations: np.ndarray | None
    finger: bool

    def __post_init__(self):
        self.lower=self.vertices.min(0);self.upper=self.vertices.max(0)

    @classmethod
    def create(cls,link,vertices,finger=False):
        points=np.unique(np.asarray(vertices,dtype=float),axis=0)
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Invalid authored collider vertices')
        rank=np.linalg.matrix_rank(points-points.mean(0))
        hull=ConvexHull(points) if rank==3 else None
        return cls(link,points if hull is None else points[hull.vertices],
            None if hull is None else hull.equations.copy(),finger)

    def contains(self,points,margin=0.):
        """Positive margin erodes the solid; negative margin expands it."""
        values=np.asarray(points,dtype=float)
        if self.equations is None:return np.zeros(len(values),dtype=bool)
        broad=((values>=self.lower+margin)&(values<=self.upper-margin)).all(1)
        result=np.zeros(len(values),dtype=bool)
        q=values[broad]
        if len(q):result[broad]=(q@self.equations[:,:3].T+self.equations[:,3]<=-margin).all(1)
        return result


def mesh_parts(asset,link,transform=None):
    import trimesh
    result=[];paths=[]
    for collision in link.findall('collision'):
        mesh=collision.find('geometry/mesh')
        if mesh is None:raise ValueError('Expected the current authored mesh collider')
        path=Path(asset)/mesh.get('filename');paths.append(path)
        vertices=np.asarray(trimesh.load_mesh(path,process=False).vertices)
        vertices=vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        pose=origin(collision.find('origin'))
        if transform is not None:pose=transform@pose
        result.append(vertices@pose[:3,:3].T+pose[:3,3])
    return result,paths


def poses_from_candidates(candidates):
    poses=np.asarray([c['pose'] for c in candidates],dtype=float).reshape(-1,4,4)
    if not np.isfinite(poses).all():raise ValueError('Nonfinite candidate pose')
    if len(poses):
        if (not np.allclose(poses[:,3],[0,0,0,1],atol=1e-10,rtol=0)
                or np.max(np.abs(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3]-np.eye(3)))>2e-5
                or (np.linalg.det(poses[:,:3,:3])<.99998).any()):
            raise ValueError('Candidate poses must be rigid transforms')
    # Verify the original generator -> EEF -> TCP43 contract without changing it.
    for candidate,pose in zip(candidates,poses):
        meta=candidate.get('metadata',{})
        if 'source_generator_pose_world' in meta:
            if not np.allclose(generator_to_contact(meta['source_generator_pose_world']),pose,atol=2e-9,rtol=0):
                raise ValueError('Candidate lost the approved generator/TCP43 transform')
    eef=poses.copy();eef[:,:3,3]-=TCP_M*eef[:,:3,2]
    return poses,eef


class GripperGeometry:
    def __init__(self,arm='right',opening=.044,asset=None):
        if not 0<=opening<=.044:raise ValueError('Opening exceeds the frozen finger stroke')
        self.arm=get_arm(arm);self.asset=Path(asset) if asset is not None else resource_path('arx_robot')
        tree=ET.parse(self.asset/'robot.urdf').getroot()
        self.links={n.get('name'):n for n in tree.findall('link')}
        parents={n.find('child').get('link'):n for n in tree.findall('joint')}
        wrist=self.arm.links[5];cache={wrist:np.eye(4)}
        def frame(name):
            if name not in cache:
                j=parents[name];t=frame(j.find('parent').get('link'))@origin(j.find('origin'))
                if j.get('type')!='fixed':
                    if j.get('type')!='prismatic' or j.get('name') not in self.arm.finger_joints:
                        raise ValueError('Unexpected movable gripper transform')
                    shift=np.eye(4);shift[:3,3]=np.fromstring(j.find('axis').get('xyz'),sep=' ')*opening
                    t=t@shift
                cache[name]=t
            return cache[name]
        inverse=np.linalg.inv(frame(self.arm.eef));self.parts=[];self.source_paths=[self.asset/'robot.urdf']
        for name in [wrist,*self.arm.finger_links]:
            parts,paths=mesh_parts(self.asset,self.links[name],inverse@frame(name));self.source_paths+=paths
            self.parts.extend(ConvexPart.create(name,p,name in self.arm.finger_links) for p in parts)
        fingers=[p.vertices for p in self.parts if p.finger]
        self.tip=max(p[:,2].max() for p in fingers)
        pad=read_json(ROOT/'simulation/robots/arx/finger_pad_geometry.json')['parts']
        contact_low=max(v['pad_neutral_EEF_bounds_m'][0][2] for v in pad.values())
        groups=[]
        for name in self.arm.finger_links:
            points=np.concatenate([p.vertices for p in self.parts if p.link==name])
            groups.append(points[points[:,2]>=contact_low-1e-6])
        negative,positive=sorted(groups,key=lambda p:p[:,1].mean())
        lower=np.maximum(negative.min(0),positive.min(0));upper=np.minimum(negative.max(0),positive.max(0))
        lower[1],upper[1]=negative[:,1].max(),positive[:,1].min()
        if not (upper>lower).all():raise ValueError('Authored open gripper has no valid distal cavity')
        self.cavity=(lower,upper)
        vertices=np.concatenate([p.vertices for p in self.parts])
        self.center=(vertices.min(0)+vertices.max(0))/2
        self.radius=float(np.linalg.norm(vertices-self.center,axis=1).max())

    def table_rejections(self,eef,table,*,pre_distance=.08,tip_tolerance=.002,distal_length=.010):
        if not 0<=tip_tolerance<=.002 or not 0<distal_length<=.010:
            raise ValueError('Do not relax the current V2 table policy')
        lower=np.asarray(table['center'])-np.asarray(table['dims'])/2
        upper=np.asarray(table['center'])+np.asarray(table['dims'])/2
        rejected=np.zeros(len(eef),dtype=bool);reason=np.full(len(eef),'',dtype=object)
        for part in self.parts:
            v=part.vertices;world=np.einsum('kij,vj->kvi',eef[:,:3,:3],v)+eef[:,None,:3,3]
            pre=world-pre_distance*eef[:,None,:3,2]
            for stage,values in [('pregrasp',pre),('grasp',world)]:
                allowance=np.where(part.finger&(v[:,2]>=self.tip-distal_length),tip_tolerance,0.) if stage=='grasp' else np.zeros(len(v))
                inside_xy=((values[:,:,:2]>lower[:2]+1e-7)&(values[:,:,:2]<upper[:2]-1e-7)).all(2)
                inside_z=(values[:,:,2]>lower[2]+1e-7)&(values[:,:,2]<upper[2]-allowance[None]-1e-7)
                hit=(inside_xy&inside_z).any(1)&~rejected
                reason[hit]=stage+'_'+('finger' if part.finger else 'palm')+'_table_vertex'
                rejected|=hit
        return rejected,reason

    def cavity_counts_reference(self,eef,points,batch=64):
        lower,upper=self.cavity;result=[]
        for first in range(0,len(eef),batch):
            pose=eef[first:first+batch]
            local=np.einsum('kni,kij->knj',points[None]-pose[:,None,:3,3],pose[:,:3,:3])
            result.extend(((local>=lower)&(local<=upper)).all(2).sum(1).tolist())
        return np.asarray(result,dtype=int)

    def cavity_counts(self,eef,points):
        from .native_geometry import cavity_counts
        return cavity_counts(self,eef,points)

    def point_rejections(self,eef,points,*,margin=.001,pre_distance=.08,skip=None):
        if margin<.001:raise ValueError('Use the registered strict point penetration margin')
        from .native_geometry import point_rejections
        return point_rejections(self,eef,points,margin,pre_distance,skip)

    def point_rejections_reference(self,eef,points,*,margin=.001,pre_distance=.08,skip=None):
        """Check strict penetration at pre/mid/grasp, with exact convex solids."""
        if margin<.001:raise ValueError('Use the registered strict point penetration margin')
        tree=cKDTree(points);rejected=np.zeros(len(eef),dtype=bool);counts=np.zeros(len(eef),dtype=int)
        offsets=[-pre_distance,-pre_distance/2,0.]
        for index,pose in enumerate(eef):
            if skip is not None and skip[index]:continue
            center=pose[:3,:3]@self.center+pose[:3,3]-pre_distance/2*pose[:3,2]
            ids=tree.query_ball_point(center,self.radius+pre_distance/2,workers=1)
            if not ids:continue
            local=(points[ids]-pose[:3,3])@pose[:3,:3]
            for offset in offsets:
                shifted=local-np.array([0.,0.,offset])
                for part in self.parts:
                    inside=part.contains(shifted,margin)
                    if inside.any():
                        counts[index]=int(inside.sum());rejected[index]=True;break
                if rejected[index]:break
        return rejected,counts


def voxel_points(points,size):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Invalid observed point cloud')
    if not len(points):return points
    _,ids=np.unique(np.floor(points/size).astype(np.int64),axis=0,return_index=True)
    return points[np.sort(ids)]


def observed_cloud(state,depth,mask):
    k=np.asarray(state['intrinsics']);t=np.asarray(state['camera_world_optical'])
    if depth.ndim!=2 or mask.shape!=depth.shape:raise ValueError('Camera and SAM3 mask shapes differ')
    y,x=np.indices(depth.shape);valid=np.isfinite(depth)&(depth>=.1)&(depth<=2.)
    local=np.stack([(x-k[0,2])*depth/k[0,0],(y-k[1,2])*depth/k[1,1],depth],axis=-1)
    world=local[valid]@t[:3,:3].T+t[:3,3]
    return world,np.asarray(mask,dtype=bool)[valid]


class RobotPointFilter:
    def __init__(self,asset=None):
        self.asset=Path(asset) if asset is not None else resource_path('arx_robot')
        tree=ET.parse(self.asset/'robot.urdf').getroot();self.parts={};self.source_paths=[self.asset/'robot.urdf']
        for link in tree.findall('link'):
            if not link.findall('collision'):continue
            parts,paths=mesh_parts(self.asset,link);self.source_paths+=paths
            self.parts[link.get('name')]=[ConvexPart.create(link.get('name'),v) for v in parts]

    def remove(self,points,link_frames,margin=.002):
        """Use measured/verified FK frames, with no target-instance ground truth."""
        if set(self.parts)-set(link_frames):raise ValueError('Full measured robot link frames are required')
        index=cKDTree(points);removed=np.zeros(len(points),dtype=bool)
        for name,parts in self.parts.items():
            pose=np.asarray(link_frames[name])
            for part in parts:
                vertices=part.vertices;center=(vertices.min(0)+vertices.max(0))/2
                radius=float(np.linalg.norm(vertices-center,axis=1).max())+margin
                ids=np.asarray(index.query_ball_point(pose[:3,:3]@center+pose[:3,3],radius),dtype=int)
                if len(ids):removed[ids]|=part.contains((points[ids]-pose[:3,3])@pose[:3,:3],-margin)
        return ~removed,{'removed_robot_points':int(removed.sum()),'measured_links_checked':len(self.parts)}
