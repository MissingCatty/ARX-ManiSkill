"""Frozen-model path qualification of the selected canonical Spam proposal."""
import argparse
import copy
import os
from pathlib import Path
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha
from .advisor33_teacher import Teacher


def run(root, gpu):
    root=Path(root).resolve(); out=root/'spam_path_preflight'; out.mkdir(exist_ok=False)
    screen=read(root/'spam_model_preflight/result.json'); chosen=screen['selected']
    reg=read(ROOT/'runs/nearfield_closedloop30_20261110_01/registration.json')
    seed=next(s for s in reg['seeds'].values() if s['object_id']=='household_ycb_spam_can')
    scene=Path(read(root/'spam_model_preflight/registration.json')['scene'])
    old=next(scene.parent.glob('policy/*_request.json')); req=read(old); row=read(req['observation'])
    asset=next(a for a in reg['catalog']['objects'] if a['object_id']==seed['object_id'])
    context={'asset':{'usd_path':asset['usd_path'],'asset_sha256':asset['asset_sha256']},
        'eef_from_camera':(np.linalg.inv(row['snapshot']['link_world']['right_eef_link'])@np.array(row['camera_world_optical'])).tolist(),
        'intrinsics':row['intrinsics'],'shape':[480,640], 'clipping_range_m':row['camera_clipping_range_m']}
    write(out/'registration.json',{'physical_trials':0,'maximum_paths':4,'GPU_uuid':gpu,
        'source_files':{str(Path(__file__)):sha(__file__)},'screen_sha256':sha(root/'spam_model_preflight/result.json'),
        'scope':'Two entry paths to canonical G, plus nominal closed lift, with original dense collision and visual checks'})
    os.environ['CUDA_VISIBLE_DEVICES']=gpu; teacher=Teacher(reg,out)
    teacher.initialize(read(scene),seed['anchor_snapshot']); rows=[]

    def snapshot(q, closed=False):
        snap=copy.deepcopy(seed['anchor_snapshot']); names=teacher.context['names']
        qt=teacher.torch.tensor(q,dtype=teacher.torch.float32,device='cuda')
        fk=teacher.context['closed' if closed else 'open'][0].compute_kinematics(teacher.js(qt)).tool_poses
        local=np.eye(4);local[:3,3]=fk.position.detach().cpu().numpy().reshape(3)
        quat=fk.quaternion.detach().cpu().numpy().reshape(4)
        local[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        snap['link_world']['right_eef_link']=(np.array(snap['link_world']['body4'])@local).tolist()
        for n,v in zip(names,q):snap['joint_positions'][snap['joint_names'].index(n)]=float(v)
        if closed:
            for n in ('right_joint27','right_joint28'):snap['joint_positions'][snap['joint_names'].index(n)]=.022
        snap['joint_velocities']=[0.]*len(snap['joint_names']);return snap

    for entry in chosen['entries']:
        folder=out/entry['condition']; folder.mkdir()
        request={'action':'plan','phase':'local','label':'canonical_approach','snapshot':snapshot(entry['q']),
            'scene':str(scene),'goal_world':chosen['goal_world'],'grasp_world':chosen['goal_world'],
            'output':str(folder),'visual_context':context,'command_id':uuid.uuid4().hex,'minimum_duration':3.}
        write(folder/'request.json',request);reply=teacher.handle(request);write(folder/'reply.json',reply)
        rows.append({'condition':entry['condition'],'stage':'approach','pass':not bool(reply.get('rejected')),'rejected':reply.get('rejected')})
        folder=out/(entry['condition']+'_lift');folder.mkdir()
        request.update(phase='lift',label='lift',snapshot=snapshot(chosen['certificate']['q_grasp'],True),output=str(folder),
            goal_world=chosen['certificate']['selected_lift_world'],command_id=uuid.uuid4().hex)
        write(folder/'request.json',request);reply=teacher.handle(request);write(folder/'reply.json',reply)
        rows.append({'condition':entry['condition'],'stage':'lift','pass':not bool(reply.get('rejected')),'rejected':reply.get('rejected')})
    result={'pass':all(r['pass'] for r in rows),'paths':rows,'physical_trials':0,'actual_grasp_success_proven':False}
    write(out/'result.json',result);print(result,flush=True)
    if not result['pass']:raise RuntimeError('Selected canonical path qualification rejected')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--gpu',required=True);a=p.parse_args();run(a.root,a.gpu)
