"""Compare unchanged candidate IDs after authored-geometry checks and NMS."""
import time

import numpy as np

from .geometry import poses_from_candidates,voxel_points,observed_cloud
from .nms import native_nms


class CandidatePostprocessor:
    def __init__(self,grippers,robot_points,thresholds):
        self.grippers,self.robot_points,self.thresholds=grippers,robot_points,thresholds

    def observations(self,state,depth,mask,completed,link_frames):
        began=time.perf_counter();points,labels=observed_cloud(state,depth,mask)
        _,ids=np.unique(np.floor(points/.002).astype(np.int64),axis=0,return_index=True)
        ids=np.sort(ids);points,labels=points[ids],labels[ids]
        keep,robot=self.robot_points.remove(points,link_frames)
        points,labels=points[keep],labels[keep]
        table=state['scene_cuboids']['table'];center=np.asarray(table['center']);half=np.asarray(table['dims'])/2
        # The analytic table check owns V2 tip contact; do not reintroduce the
        # same table surface as a generic object point collision.
        on_table=(np.abs(points[:,2]-(center[2]+half[2]))<=.001)&(np.abs(points[:,:2]-center[:2])<=half[:2]).all(1)
        points,labels=points[~on_table],labels[~on_table]
        target=voxel_points(completed,.002)
        if not len(target):raise ValueError('The saved RaySt3R target cloud is empty')
        return points,target,{'seconds':time.perf_counter()-began,'observed_non_table_points':len(points),
            'observed_target_points':int(labels.sum()),'completed_target_voxels':len(target),
            'table_surface_points_removed':int(on_table.sum()),**robot}

    def process(self,candidates,state,observed,completed,repeats=3):
        began=time.perf_counter();poses,eef=poses_from_candidates(candidates);count=len(poses)
        coordinate_seconds=time.perf_counter()-began
        widths=np.array([c['width'] for c in candidates]);scores=np.array([c['score'] for c in candidates])
        if not np.allclose(widths,.088,atol=1e-12,rtol=0):
            raise ValueError('This registered comparison retains the original 88mm planned opening')
        if (np.diff(scores)>1e-12).any():raise ValueError('Original native-score order changed')
        geometry={};geometry_seconds=0.
        for side,model in self.grippers.items():
            began=time.perf_counter()
            table_bad,reasons=model.table_rejections(eef,state['scene_cuboids']['table'])
            table_seconds=time.perf_counter()-began;began=time.perf_counter()
            point_bad,point_counts=model.point_rejections(eef,observed,skip=table_bad)
            point_seconds=time.perf_counter()-began;began=time.perf_counter()
            cavity=model.cavity_counts(eef,completed)
            cavity_seconds=time.perf_counter()-began
            geometry_seconds+=table_seconds+point_seconds+cavity_seconds
            geometry[side]={'table_bad':table_bad,'point_bad':point_bad,'point_counts':point_counts,
                'table_reasons':reasons.tolist(),'cavity_counts':cavity,
                'kept':~(table_bad|point_bad),'kept_nonempty':~(table_bad|point_bad)&(cavity>0),
                'timing':{'table_seconds':table_seconds,'point_seconds':point_seconds,'cavity_seconds':cavity_seconds}}
        union=geometry['right']['kept']|geometry['left']['kept']
        nonempty=geometry['right']['kept_nonempty']|geometry['left']['kept_nonempty']
        variants={'control':{'representatives':list(range(count)),'deferred':[],
            'representative_for_input':list(range(count)),'input_indices':list(range(count)),
            'postprocessing_seconds':0.}}
        for prefix,mask in [('nms',np.ones(count,dtype=bool)),('geometry_nms',union),('geometry_empty_nms',nonempty)]:
            ids=np.flatnonzero(mask)
            for value in self.thresholds:
                records=[native_nms(poses[ids],scores[ids],widths[ids],value['position_m'],
                    np.deg2rad(value['rotation_deg']),ids) for _ in range(repeats)]
                for row in records[1:]:
                    if (row['representatives']!=records[0]['representatives']
                            or row['representative_for_input']!=records[0]['representative_for_input']):
                        raise RuntimeError('Native NMS replay changed source identity')
                result=records[0]
                result.update(input_indices=ids.tolist(),threshold=value,
                    native_seconds_median=float(np.median([r['native_seconds'] for r in records])),
                    provenance_seconds_median=float(np.median([r['provenance_seconds'] for r in records])))
                result['postprocessing_seconds']=(coordinate_seconds+(0 if prefix=='nms' else geometry_seconds)
                    +result['native_seconds_median']+result['provenance_seconds_median'])
                variants[prefix+'_'+value['id']]=result
        return {'raw_count':count,'geometry':geometry,'variants':variants,
            'geometry_seconds':geometry_seconds,'coordinate_validation_seconds':coordinate_seconds,
            'candidate_poses_scores_and_widths_unchanged':True}
