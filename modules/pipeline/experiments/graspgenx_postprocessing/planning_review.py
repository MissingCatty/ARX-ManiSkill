"""Check every planning receipt, original candidate identity and physical-arm FK."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import now,read_json,sha256,write_json
from arx_project.artifacts import encode
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.ordering import validate_order
from .planning import raw,verify


def check_receipt(directory,claim):
    if claim['state']!='complete':raise ValueError('Planning request is not complete')
    for key,name in [('metrics_sha256','metrics.json'),('plan_sha256','plan_result.json'),('order_sha256','candidate_order.json')]:
        if sha256(directory/name)!=claim[key]:raise ValueError('Missing or changed planning receipt: '+name)
    commit=raw(directory/'candidate_snapshot_commit.json')
    if not commit['persisted']:raise ValueError('Candidate snapshot was not committed')


def review(root):
    reg=raw(root/'registration.json');verify(reg);accepted=raw(root/'acceptance.json')
    if not accepted['task_complete'] or not accepted['record_integrity']:raise ValueError('Planning is incomplete')
    model=ArmModel();endpoints=[];checked_rows=0;requests=0
    for case in reg['cases']:
        original=raw(Path(case['source_directory'])/'candidates/candidate_pool.json')['candidates']
        for mode in reg['variants']:
            directory=root/'plans'/case['id']/mode
            check_receipt(directory,raw(root/'claims'/(case['id']+'_'+mode+'.json')))
            manifest=raw(directory/'candidate_order.json');queue=raw(directory/'candidates/candidate_pool.json')
            order=validate_order(len(original),manifest['representatives'],manifest['deferred'])
            if order!=manifest['priority_to_original_source'] or len(queue['candidates'])!=len(original):
                raise ValueError('Candidate mapping or count changed')
            for rank,index in enumerate(order):
                if queue['candidates'][rank]!=original[index]:raise ValueError('Original candidate geometry/score/metadata changed')
            checked_rows+=len(order);requests+=1
            result=raw(directory/'plan_result.json');metrics=raw(directory/'metrics.json')
            if metrics['success']!=result['success']:raise ValueError('Plan and metrics disagree')
            if not result['success']:continue
            selected=result['selected'];rank=selected['index'];source=order[rank]
            if selected['original_source_index']!=source or metrics['selected_original_source_index']!=source:
                raise ValueError('Selected grasp does not trace to its original candidate')
            if not np.array_equal(np.asarray(original[source]['pose']),np.asarray(selected['contact_pose_world'])):
                raise ValueError('Selected original grasp pose changed')
            arm=get_arm(selected['arm']);state=read_json(directory/'state.json')
            for stage,key in [('grasp','grasp_world'),('pre','pregrasp_world'),('lift','lift_world')]:
                q=np.asarray(selected['q_'+stage]);actual=deepcopy(state)
                actual['links']={'base_link':state['links']['base_link']};clear=True
                for name,value in zip(arm.joints,q):
                    i=state['joint_names'].index(name);lo,hi=state['joint_limits'][i]
                    clear=clear and bool(lo<=value<=hi);actual['qpos'][i]=float(value)
                reached=model.frames(actual)[arm.eef];goal=np.asarray(selected[key])
                position=float(np.linalg.norm(reached[:3,3]-goal[:3,3]))
                rotation=float(Rotation.from_matrix(reached[:3,:3]@goal[:3,:3].T).magnitude())
                if not clear or not np.isfinite([position,rotation]).all() or position>.002 or rotation>.03:
                    raise ValueError('Independent physical-arm FK or joint limits failed')
                endpoints.append({'case':case['id'],'variant':mode,'arm':arm.name,'stage':stage,
                    'source_index':source,'position_error_m':position,'rotation_error_rad':rotation,'joint_limits_clear':clear})
    verify(reg)
    result={'at':now(),'passed':True,'record_integrity':True,'checked_requests':requests,
        'checked_original_candidate_rows':checked_rows,'checked_endpoints':len(endpoints),'endpoints':endpoints,
        'control_successes_retained':accepted['control_successes_retained'],
        'physical_grasps':0,'automatic_default_activation':False}
    write_json(root/'planning_review.json',result);return result
