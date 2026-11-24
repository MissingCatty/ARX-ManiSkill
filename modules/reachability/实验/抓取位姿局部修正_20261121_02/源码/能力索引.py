"""Load a completed map once; return advisory coverage and FK-based IK seeds."""
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from 公共 import read, sha, voxel_ids, orientation_ids, ORIENTATIONS, DOWN_ORIENTATIONS


class CapabilityMap:
    def __init__(self, directory, verify=True):
        self.directory=Path(directory).resolve();started=time.monotonic()
        complete=read(self.directory/'完成.json')
        if not complete.get('passed'):raise ValueError('Map is not complete and validated')
        if verify:
            for name,digest in complete['file_sha256'].items():
                p=(self.directory/name).resolve()
                if p.parent!=self.directory or sha(p)!=digest:raise ValueError('Map integrity mismatch: '+name)
        self.meta=read(self.directory/'地图元数据.json');self.bounds=np.asarray(self.meta['joint_limits'])
        n=self.meta['witnesses'];width=len(self.meta['fields'])
        p=self.directory/self.meta['witness_file']
        if p.stat().st_size!=n*width*4:raise ValueError('Witness dimensions do not match metadata')
        self.bank=np.memmap(p,dtype=self.meta['dtype'],mode='r',shape=(n,width))
        self.tree=cKDTree(self.bank[:,6:9],balanced_tree=True)
        self.cells=dict(np.load(self.directory/self.meta['fine']['file'],allow_pickle=False))
        self.seen=np.load(self.directory/'已覆盖位姿格.npy',mmap_mode='r',allow_pickle=False)
        self.tcp=np.asarray(self.meta['T_eef_action']);self.base=np.asarray(self.meta['reference_base_world'])
        self.load_seconds=time.monotonic()-started

    def coverage(self, position, rotation=None):
        position=np.asarray(position,dtype=float)
        ids,_=voxel_ids(position[None],self.meta['fine']['resolution_m']);index=int(np.searchsorted(self.cells['voxel_id'],ids[0]))
        present=index<len(self.cells['voxel_id']) and self.cells['voxel_id'][index]==ids[0]
        result={'position_body4_m':position.tolist(),'covered_position_voxel':bool(present),
                'status':'位置已覆盖' if present else '未覆盖，未知',
                'uncovered_is_unreachable':False,'hard_candidate_rejection':False,
                'current_scene_checked':False,'voxel_resolution_m':self.meta['fine']['resolution_m']}
        if present:
            count=int(self.cells['orientation_count'][index]);down=int(self.cells['down_orientation_count'][index])
            result.update(observed_orientation_bins=count,observed_orientation_fraction=count/ORIENTATIONS,
                          observed_downward_orientation_fraction=down/DOWN_ORIENTATIONS,
                          sample_count=int(self.cells['sample_count'][index]),
                          mean_joint_margin_rad=float(self.cells['margin_mean_rad'][index]),
                          mean_jacobian_condition_index=float(self.cells['condition_mean'][index]))
        if rotation is not None:
            orientation,_=orientation_ids(np.asarray(rotation)[None]);key=int(ids[0]*ORIENTATIONS+orientation[0])
            k=int(np.searchsorted(self.seen,key));seen=k<len(self.seen) and int(self.seen[k])==key
            result['covered_orientation_bin']=bool(seen)
            if present:result['status']='位置及朝向格已覆盖' if seen else '位置已覆盖，朝向覆盖未知'
        return result

    def query_pose(self, pose, current_q=None, num_seeds=32, neighbors=8192):
        started=time.monotonic();pose=np.asarray(pose,dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Expected finite 4x4 action TCP pose in body4')
        if np.abs(pose[:3,:3].T@pose[:3,:3]-np.eye(3)).max()>.002:raise ValueError('Invalid orientation matrix')
        result=self.coverage(pose[:3,3],pose[:3,:3])
        k=min(int(neighbors),len(self.bank));distance,indices=self.tree.query(pose[:3,3],k=k,workers=1)
        distance=np.atleast_1d(distance);indices=np.atleast_1d(indices)
        rows=np.asarray(self.bank[indices]);target=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
        quaternions=rows[:,9:13].astype(float);quaternions/=np.linalg.norm(quaternions,axis=1,keepdims=True)
        angles=2*np.arccos(np.clip(np.abs(quaternions@target),0,1))
        score=distance/.02+angles/.35-.02*rows[:,14]
        current=None if current_q is None else np.asarray(current_q,dtype=np.float32)
        if current is not None:
            if current.shape!=(6,) or not np.isfinite(current).all():raise ValueError('Expected six measured arm joints')
            score+=.03*np.linalg.norm((rows[:,:6]-current)/(self.bounds[:,1]-self.bounds[:,0]),axis=1)
        order=np.argsort(score,kind='stable');seeds=[];chosen=[];seed_rows=[]
        if current is not None:seeds.append(current.tolist());seed_rows.append(None)
        scale=self.bounds[:,1]-self.bounds[:,0]
        normalized=rows[:,:6]/scale
        separation=np.full(len(rows),np.inf)
        if current is not None:separation=np.linalg.norm(normalized-current/scale,axis=1)
        for index in order:
            q=rows[index,:6]
            if separation[index]<.04:continue
            seeds.append(q.tolist());chosen.append(int(index));seed_rows.append(int(indices[index]))
            separation=np.minimum(separation,np.linalg.norm(normalized-normalized[index],axis=1))
            if len(seeds)>=num_seeds:break
        if len(seeds)<num_seeds:
            for index in order:
                if int(index) in chosen:continue
                q=rows[index,:6]
                if seeds and min(np.linalg.norm(q-np.asarray(p)) for p in seeds)<1e-7:continue
                seeds.append(q.tolist());chosen.append(int(index));seed_rows.append(int(indices[index]))
                if len(seeds)>=num_seeds:break
        if not seeds:raise RuntimeError('Completed map contains no searchable witnesses')
        while len(seeds)<num_seeds:seeds.append(seeds[-1]);seed_rows.append(seed_rows[-1])
        best=int(order[0]);exact=(distance<=.002)&(angles<=.03)
        result.update(seeds=seeds[:num_seeds],seed_count=min(len(seeds),num_seeds),
             seed_bank_rows=seed_rows[:num_seeds],
             current_state_included_as_seed=current is not None,nearest_position_sample_distance_m=float(distance.min()),
             best_pose_sample={'bank_row':int(indices[best]),'translation_error_m':float(distance[best]),
                 'orientation_error_rad':float(angles[best]),'q':rows[best,:6].tolist(),
                 'joint_limit_margin_rad':float(rows[best,13])},
             direct_fk_witness_within_original_pose_tolerances=bool(exact.any()),
             search_seconds=time.monotonic()-started,
             seed_ranking='position / 2cm + orientation / 0.35rad; small margin and current-joint preferences',
             acceptance='Seeds only. Original current-scene IK/checker and path validation must decide execution.')
        return result

    def query(self, request):
        pose=np.asarray(request['matrix'],dtype=float)
        frame=request.get('frame','body4');tool=request.get('tool','action_tcp43')
        if frame=='world':pose=np.linalg.inv(np.asarray(request.get('robot_base_world',self.base)))@pose
        elif frame!='body4':raise ValueError('Unknown coordinate frame')
        if tool=='right_eef_link':pose=pose@self.tcp
        elif tool!='action_tcp43':raise ValueError('Unknown tool frame')
        result=self.query_pose(pose,request.get('current_q'),request.get('num_seeds',32))
        result.update(input_frame=frame,input_tool=tool,query_pose_body4_action=pose.tolist())
        return result
