"""Lookup physical joint witnesses without modifying an input grasp pose."""
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, read_json, sha256
from modules.curobo.indexed_planning.geometric.kinematics import ArmChain


MAP = ROOT/'tmp/reachability/products/前方精细_20261121_01'


def quaternion_errors(quaternions_wxyz, target):
    values=np.asarray(quaternions_wxyz,dtype=float)
    if values.ndim!=2 or values.shape[1]!=4 or not np.isfinite(values).all():
        raise ValueError('Invalid stored quaternion array')
    norms=np.linalg.norm(values,axis=1)
    if np.any(norms<1e-8):raise ValueError('Zero map quaternion')
    return 2*np.arccos(np.clip(np.abs((values/norms[:,None])@target),0,1))


class FullPoseIndex:
    def __init__(self,directory=MAP):
        self.directory=Path(directory).resolve();began=time.perf_counter()
        complete=read_json(self.directory/'完成.json')
        if not complete.get('passed'):raise ValueError('Map has not passed its original verification')
        self.meta=read_json(self.directory/'地图元数据.json')
        for name in ['地图元数据.json',self.meta['witness_file']]:
            path=self.directory/name
            if path.resolve().parent!=self.directory or sha256(path)!=complete['file_sha256'][name]:
                raise ValueError('Map input changed: '+name)
        if self.meta['frame']!='body4' or self.meta['pose_frame']!='action_tcp43':
            raise ValueError('Wrong witness frame')
        if self.meta['fields'][:13]!=['q21','q22','q23','q24','q25','q26',
                                      'tcp_x','tcp_y','tcp_z','qw','qx','qy','qz']:
            raise ValueError('Unexpected map schema')
        self.chain=ArmChain('right')
        if (self.meta['joint_names']!=list(self.chain.arm.joints)
                or not np.allclose(self.meta['joint_limits'],self.chain.limits,atol=1e-10,rtol=0)):
            raise ValueError('Map joint names or bounds differ from the current robot')
        self.tcp=np.asarray(self.meta['T_eef_action'],dtype=float)
        expected=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,.043],[0,0,0,1.]])
        if not np.allclose(self.tcp,expected,atol=1e-12,rtol=0):raise ValueError('Map TCP differs from TCP43')
        shape=(self.meta['witnesses'],len(self.meta['fields']))
        self.bank=np.memmap(self.directory/self.meta['witness_file'],mode='r',
            dtype=self.meta['dtype'],shape=shape)
        if self.bank.nbytes!=(self.directory/self.meta['witness_file']).stat().st_size:
            raise ValueError('Unexpected map byte count')
        self.tree=cKDTree(np.asarray(self.bank[:,6:9],dtype=float),leafsize=32,
            compact_nodes=True,balanced_tree=True)
        self.position_tolerance=.002;self.rotation_tolerance=.03
        # Index TCP positions while accepting the original EEF pose tolerance.
        # Rotation about a 43 mm offset can move the TCP another 1.29 mm.
        self.radius=self.position_tolerance+2*np.linalg.norm(self.tcp[:3,3])*np.sin(self.rotation_tolerance/2)+2e-6
        self.load_seconds=time.perf_counter()-began
        self.witness_sha256=complete['file_sha256'][self.meta['witness_file']]

    def action_pose(self,eef):
        pose=np.asarray(eef,dtype=float)
        if (pose.shape!=(4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3],[0,0,0,1],atol=1e-10,rtol=0)
                or np.linalg.norm(pose[:3,:3].T@pose[:3,:3]-np.eye(3))>2e-5
                or np.linalg.det(pose[:3,:3])<.99998):
            raise ValueError('Expected a rigid body4 EEF target')
        return pose@self.tcp

    def direct(self,eef,current_q,limit=32):
        current_q=np.asarray(current_q,dtype=float)
        if current_q.shape!=(6,) or not np.isfinite(current_q).all() or not 1<=limit<=32:
            raise ValueError('Invalid current joints or witness return budget')
        began=time.perf_counter();goal=self.action_pose(eef)
        indices=np.asarray(self.tree.query_ball_point(goal[:3,3],self.radius,workers=1),dtype=np.int64)
        position_neighbors=len(indices)
        if len(indices):
            rows=np.asarray(self.bank[indices])
            target=Rotation.from_matrix(goal[:3,:3]).as_quat()[[3,0,1,2]]
            angles=quaternion_errors(rows[:,9:13],target)
            indices=indices[angles<=self.rotation_tolerance+1e-6]
        matched=len(indices);result=[]
        if matched:
            q=np.asarray(self.bank[indices,:6],dtype=float)
            actual=self.chain.fk(q)
            pe=np.linalg.norm(actual[:,:3,3]-np.asarray(eef)[:3,3],axis=1)
            re=Rotation.from_matrix(actual[:,:3,:3]@np.asarray(eef)[:3,:3].T).magnitude()
            valid=((q>=self.chain.limits[:,0])&(q<=self.chain.limits[:,1])).all(1)&(pe<=.002)&(re<=.03)
            eligible=np.flatnonzero(valid)
            order=np.argsort(np.sum((q[eligible]-current_q)**2,axis=1),kind='stable')
            for row in eligible[order[:limit]]:
                result.append({'q':q[row].tolist(),'bank_row':int(indices[row]),
                    'independent_position_error_m':float(pe[row]),'independent_rotation_error_rad':float(re[row])})
        return {'witnesses':result,'position_neighbors':position_neighbors,'orientation_neighbors':matched,
            'seconds':time.perf_counter()-began,'map_miss_is_unreachable':False,'scene_checked':False}

    def seeds(self,eef,current_q,count=32,neighbors=8192,diversify=True):
        began=time.perf_counter();goal=self.action_pose(eef)
        if not 1<=count<=32 or not count<=neighbors<=16384:raise ValueError('Invalid map seed budget')
        current_q=np.asarray(current_q,dtype=np.float32)
        if current_q.shape!=(6,) or not np.isfinite(current_q).all():raise ValueError('Expected six current joints')
        distances,indices=self.tree.query(goal[:3,3],k=min(neighbors,len(self.bank)),workers=1)
        distances,indices=np.atleast_1d(distances),np.atleast_1d(indices)
        rows=np.asarray(self.bank[indices])
        target=Rotation.from_matrix(goal[:3,:3]).as_quat()[[3,0,1,2]]
        angles=quaternion_errors(rows[:,9:13],target)
        scale=np.ptp(self.chain.limits,axis=1)
        score=distances/.02+angles/.35-.02*rows[:,14]
        score+=.03*np.linalg.norm((rows[:,:6]-current_q)/scale,axis=1)
        order=np.argsort(score,kind='stable');chosen=[];seeds=[current_q.tolist()]
        normalized=rows[:,:6]/scale
        separation=np.linalg.norm(normalized-current_q/scale,axis=1)
        for row in order:
            if len(seeds)>=count:break
            if diversify and separation[row]<.04:continue
            if not diversify and np.min(np.linalg.norm(np.asarray(seeds)-rows[row,:6],axis=1))<1e-7:continue
            seeds.append(rows[row,:6].tolist());chosen.append(int(row))
            if diversify:separation=np.minimum(separation,np.linalg.norm(normalized-normalized[row],axis=1))
            if len(seeds)>=count:break
        for row in order:
            if len(seeds)>=count:break
            if int(row) in chosen or np.min(np.linalg.norm(np.asarray(seeds)-rows[row,:6],axis=1))<1e-7:continue
            seeds.append(rows[row,:6].tolist());chosen.append(int(row))
        seed_rows=[None]+indices[chosen].astype(int).tolist()
        while len(seeds)<count:
            seeds.append(seeds[-1]);seed_rows.append(seed_rows[-1])
        best=int(order[0])
        return {'seeds':seeds[:count],'bank_rows':seed_rows[:count],
            'best_position_error_m':float(distances[best]),'best_rotation_error_rad':float(angles[best]),
            'position_neighbors':len(rows),'seconds':time.perf_counter()-began,
            'best_score':float(score[best]),'target_pose_changed':False}

    def neighborhood(self,eef,current_q,radius=.02,angle=.17453292519943295,count=9):
        """Broader seed neighborhood; it never changes final pose tolerances."""
        began=time.perf_counter();goal=self.action_pose(eef)
        current=np.asarray(current_q,dtype=np.float32)
        if (current.shape!=(6,) or not np.isfinite(current).all() or not 1<=count<=32
                or not 0<radius<=.05 or not 0<angle<=np.deg2rad(20)):
            raise ValueError('Invalid seed neighborhood')
        indices=np.asarray(self.tree.query_ball_point(goal[:3,3],radius,workers=1),dtype=np.int64)
        spatial=len(indices);histogram={str(a):0 for a in [.03,*np.deg2rad([3,5,10])]}
        seeds=[current.tolist()];sources=[None];best=None
        if spatial:
            rows=np.asarray(self.bank[indices]);delta=rows[:,6:9]-goal[:3,3]
            distances=np.linalg.norm(delta,axis=1)
            target=Rotation.from_matrix(goal[:3,:3]).as_quat()[[3,0,1,2]]
            angles=quaternion_errors(rows[:,9:13],target)
            histogram={str(a):int(np.count_nonzero(angles<=a)) for a in [.03,*np.deg2rad([3,5,10])]}
            eligible=np.flatnonzero(angles<=angle)
            scale=np.ptp(self.chain.limits,axis=1)
            score=distances/radius+angles/angle+.03*np.linalg.norm((rows[:,:6]-current)/scale,axis=1)
            order=eligible[np.argsort(score[eligible],kind='stable')]
            if len(order):best={'position_error_m':float(distances[order[0]]),'rotation_error_rad':float(angles[order[0]])}
            for row in order:
                if len(seeds)>=count:break
                if np.min(np.linalg.norm((np.asarray(seeds)-rows[row,:6])/scale,axis=1))<.04:continue
                seeds.append(rows[row,:6].tolist());sources.append(int(indices[row]))
        return {'seeds':seeds,'bank_rows':sources,'position_neighbors':spatial,
            'angular_counts':histogram,'best':best,'seconds':time.perf_counter()-began,
            'target_pose_changed':False,'final_position_tolerance_m':.002,'final_rotation_tolerance_rad':.03}
