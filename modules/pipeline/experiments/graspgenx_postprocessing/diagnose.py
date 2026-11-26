"""Locate the observed points that removed historical selected grasps."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter,poses_from_candidates,observed_cloud
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor
from .offline import raw,verify


def first_collision(model,pose,points,raw_tree,raw_labels):
    center=pose[:3,:3]@model.center+pose[:3,3]-.04*pose[:3,2]
    ids=np.asarray(cKDTree(points).query_ball_point(center,model.radius+.04),dtype=int)
    local=(points[ids]-pose[:3,3])@pose[:3,:3]
    for stage,offset in [('pregrasp',-.08),('mid',-.04),('grasp',0.)]:
        shifted=local-np.array([0.,0.,offset])
        for number,part in enumerate(model.parts):
            inside=part.contains(shifted,.001)
            if not inside.any():continue
            cloud=points[ids[inside]]
            separation,nearest=raw_tree.query(cloud)
            if (separation>2e-9).any():raise ValueError('Diagnostic points lost their measured pixel source')
            penetration=-(shifted[inside]@part.equations[:,:3].T+part.equations[:,3]).max(1)
            return {'stage':stage,'link':part.link,'convex_part':number,'collision_points':len(cloud),
                'SAM_target_points':int(raw_labels[nearest].sum()),
                'other_observed_points':int((~raw_labels[nearest]).sum()),
                'strict_penetration_mm':{'min':float(penetration.min()*1000),
                    'median':float(np.median(penetration)*1000),'max':float(penetration.max()*1000)},
                'world_points_m':cloud[:20]}
    raise ValueError('Registered point rejection could not be independently localized')


def main(source,root):
    if root.exists():raise ValueError('Keep previous selected-grasp diagnosis')
    reg=raw(source/'registration.json');verify(reg)
    if not raw(source/'independent_review.json')['passed']:raise ValueError('V7 record review must finish first')
    cases=[]
    for case in reg['cases']:
        row=raw(source/'cases'/case['id']/'result.json')
        if any(not v['retention']['original_selection_same_arm_available'] for v in row['variants'].values()):
            cases.append((case,row))
    root.mkdir(parents=True)
    write_json(root/'registration.json',{'at':now(),'source':str(source),'cases':[case['id'] for case,row in cases],
        'source_sha256':{str(Path(__file__)):sha256(__file__)},'offline_acceptance_sha256':sha256(source/'acceptance.json'),
        'new_IK_calls':0,'planning_requests':0,'physical_grasps':0})
    model=ArmModel();robot=RobotPointFilter();grippers={side:GripperGeometry(side) for side in ['right','left']}
    engine=CandidatePostprocessor(grippers,robot,[]);rows=[]
    for case,row in cases:
        directory=Path(case['source_directory']);state=read_json(directory/'state.json')
        depth=np.load(directory/'camera/depth_m.npy');mask=np.asarray(Image.open(directory/'camera/mask.png'))>0
        completed=np.load(directory/'completed_points_world.npy');side=row['original_selected']['arm'];index=row['original_selected']['index']
        candidate=raw(directory/'candidates/candidate_pool.json')['candidates'][index]
        points,target,observation=engine.observations(state,depth,mask,completed,model.frames(state))
        raw_points,labels=observed_cloud(state,depth,mask)
        _,eef=poses_from_candidates([candidate]);metrics=row['arms'][side]
        result={'id':case['id'],'historical_physical_success':row['original_physical_grasp_success'],
            'selected':row['original_selected'],'table_reason':metrics['selected_table_reason'],
            'point_collision':metrics['selected_observed_collision'],'cavity_points':metrics['selected_cavity_points'],
            'rejected_by':[name for name,v in row['variants'].items() if not v['retention']['original_selection_same_arm_available']]}
        if result['point_collision']:
            result['point_detail']=first_collision(grippers[side],eef[0],points,cKDTree(raw_points),labels)
            with np.load(source/'cases'/case['id']/'geometry_masks.npz') as saved:
                if result['point_detail']['collision_points']!=int(saved[side+'_point_counts'][index]):
                    raise ValueError('Independent point count differs from registered batch result')
        write_json(root/'cases'/(case['id']+'.json'),result);rows.append(result)
        print('DIAGNOSE',case['id'],result.get('point_detail',{}).get('stage'),flush=True)
    verify(reg)
    write_json(root/'acceptance.json',{'at':now(),'passed':True,'cases':rows,
        'successful_grasps_rejected':sum(r['historical_physical_success'] for r in rows),
        'new_IK_calls':0,'planning_requests':0,'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();main(ensure_run_path(args.source),ensure_run_path(args.root))
