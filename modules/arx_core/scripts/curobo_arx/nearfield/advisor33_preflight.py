"""Finite canonical Spam IK/closed-model screen, with no robot execution."""
import argparse
import copy
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha
from scripts.curobo_arx.data_factory.teacher import Rejection
from .advisor33_teacher import Teacher


def run(root,gpu):
    root=Path(root).resolve();out=root/'spam_model_preflight';out.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/'runs/nearfield_closedloop30_20261110_01/registration.json')
    geometry=read(ROOT/'runs/nearfield_advisor32_20261111_01/diagnosis_complete/geometry.json')
    geometry=next(g for g in geometry if g['object_id']=='household_ycb_spam_can')
    normal=next(t for t in old['seed_trials'] if t['method']=='D' and t['object_id']=='household_ycb_spam_can' and t['condition']=='normal')
    offset=next(t for t in old['seed_trials'] if t['method']=='D' and t['object_id']=='household_ycb_spam_can' and t['condition']=='entry_offset')
    seed=old['seeds'][normal['seed_id']];snapshot=seed['anchor_snapshot']
    ep=next((ROOT/'runs/nearfield_advisor31_20261110_01').glob('groups/*/trials/a31_regression_r30_21_03_normal_C/episodes/*'))
    scene=ep/'pregrasp_scene.json'
    canonical=np.array(geometry['canonical_proposal']['Gstar_canon_world'])
    box=np.array(geometry['OBB_world']);vertical=geometry['vertical_axis'];up=box[:3,vertical].copy()
    if up[2]<0:up*=-1
    height=geometry['OBB_extents_mm'][vertical]/1000
    pad=np.array(geometry['frozen_pad_center_EE_m']);variants=[]
    for fraction in (.70,.60,.50,.75):
        for tilt in (0.,-10.,10.):
            g=canonical.copy();g[:3,:3]=canonical[:3,:3]@Rotation.from_euler('y',tilt,degrees=True).as_matrix()
            center=box[:3,3]+(fraction-.5)*height*up;g[:3,3]=center-g[:3,:3]@pad
            variants.append({'height_fraction':fraction,'approach_change_degrees':tilt,'goal_world':g.tolist()})
    write(out/'registration.json',{'physical_trials':0,'GPU_uuid':gpu,'maximum_candidates':12,'wall_seconds':900,
        'variants':variants,'scene':str(scene),'scene_sha256':sha(scene),
        'source_files':{str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('advisor33_teacher.py')):sha(Path(__file__).with_name('advisor33_teacher.py'))},
        'unchanged_robot_scene_physics':True,'authority':str(ROOT/'research_plan/chat/gemini回复/gemini回复33.md')})
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    teacher=Teacher(old,out);teacher.current_output=out;teacher.current_label='canonical'
    teacher.initialize(read(scene),snapshot)
    names=teacher.context['names'];start=teacher.torch.tensor([snapshot['joint_positions'][snapshot['joint_names'].index(n)] for n in names],device='cuda',dtype=teacher.torch.float32)
    limits=np.array([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in names])
    inverse=np.linalg.inv(snapshot['link_world']['body4']);planner,checker,_,_=teacher.context['open'];rows=[];began=time.monotonic()
    for index,variant in enumerate(variants):
        if time.monotonic()-began>900:raise TimeoutError('Canonical model-only screen budget')
        g=np.array(variant['goal_world']);record={**variant,'index':index,'pass':False};rows.append(record)
        try:
            cert=teacher.nominal_closed(g,start,inverse);qs=[cert['q_grasp'],cert['q_lift']];entries=[]
            for t in (normal,offset):
                H=g@np.linalg.inv(np.array(t['G0_world']))@np.array(t['entry_world'])
                q=teacher.ik(H,start,planner,checker,inverse)
                qs.append(q.tolist());entries.append({'condition':t['condition'],'world':H.tolist(),'q':q.tolist()})
            values=np.array(qs);margin=float(np.min(np.minimum(values-limits[:,0],limits[:,1]-values)))
            record.update(pass_=True,certificate=cert,entries=entries,minimum_joint_limit_margin_rad=margin)
            record['pass']=True;record.pop('pass_')
        except Rejection as exc:record.update(rejected=str(exc),details=exc.details)
        write(out/'candidates.json',rows)
    valid=[r for r in rows if r['pass']]
    chosen=max(valid,key=lambda r:r['minimum_joint_limit_margin_rad']) if valid else None
    result={'pass':chosen is not None,'selected':chosen,'candidates':rows,'physical_trials':0,
            'planning_model_only':True,'approach_and_lift_full_paths_checked_in_each_physical_slot':True,
            'seconds':time.monotonic()-began,'GPU_uuid':gpu}
    write(out/'result.json',result)
    if chosen is None:raise RuntimeError('No frozen-model feasible canonical candidate')
    return {'pass':True,'selected_index':chosen['index'],'valid_candidates':len(valid),'physical_trials':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--gpu',required=True);a=p.parse_args();print(run(a.root,a.gpu))
