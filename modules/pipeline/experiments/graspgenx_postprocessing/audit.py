"""Independently validate saved V7 partitions, source IDs and NMS relations."""
import argparse
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT,ensure_run_path,now,sha256,write_json
from .offline import raw,verify


def check(condition,message):
    if not condition:raise ValueError(message)


def audit(root):
    reg=raw(root/'registration.json');verify(reg)
    check(not (root/'STOP').exists(),'Offline execution stopped')
    accepted=raw(root/'acceptance.json');progress=raw(root/'progress.json')
    check(progress['phase']=='complete' and progress['completed_cases']==len(reg['cases']),'Incomplete V7 execution')
    checked_variants=0;relations=0;all_counts=0;summary={}
    for ordinal,case in enumerate(reg['cases']):
        out=root/'cases'/case['id'];row=raw(out/'result.json');count=reg['candidate_counts'][ordinal]
        claim=raw(root/'claims'/(case['id']+'.json'))
        check(claim['state']=='complete' and claim['result_sha256']==sha256(out/'result.json'),'Invalid completion receipt')
        check(row['id']==case['id'] and row['raw_count']==count,'Wrong source case/count')
        original=Path(case['source_directory']);pool=raw(original/'candidates/candidate_pool.json')['candidates']
        poses=np.asarray([v['pose'] for v in pool]);selected=raw(original/'plan_result.json')['selected']
        physical=raw(original/'result.json')['success'];index=int(selected['index']);side=selected['arm']
        masks=np.load(out/'geometry_masks.npz',allow_pickle=False)
        for arm in ['right','left']:
            table,point=masks[arm+'_table_bad'],masks[arm+'_point_bad']
            check(table.shape==(count,) and point.shape==(count,),'Invalid geometry mask shape')
            check(np.array_equal(masks[arm+'_kept'],~(table|point)),'Geometry partition mismatch')
            check(np.array_equal(masks[arm+'_kept_nonempty'],~(table|point)&(masks[arm+'_cavity_counts']>0)),
                'Empty-cavity partition mismatch')
        expected_names=['control']+[prefix+'_'+t['id'] for prefix in ['nms','geometry_nms','geometry_empty_nms'] for t in reg['thresholds']]
        check(set(row['variants'])==set(expected_names),'Missing variant records')
        for name in expected_names:
            value=raw(out/(name+'.json'));ids=value['input_indices'];reps=value['representatives'];deferred=value['deferred']
            mapping=value['representative_for_input']
            if name.startswith('geometry'):
                key='_kept_nonempty' if name.startswith('geometry_empty') else '_kept'
                allowed=masks['right'+key]|masks['left'+key]
                same_arm_allowed=bool(masks[side+key][index])
            else:allowed=np.ones(count,dtype=bool);same_arm_allowed=True
            check(ids==np.flatnonzero(allowed).tolist(),'Candidate partition does not match arm geometry')
            check(len(set(reps))==len(reps) and len(set(deferred))==len(deferred),'Duplicate source candidate')
            check(set(reps).isdisjoint(deferred) and set(reps)|set(deferred)==set(ids),'Lost candidate/NMS fallback record')
            check(reps==sorted(reps) and deferred==sorted(deferred),'Native score order changed')
            check(len(mapping)==len(ids) and set(mapping)<=set(reps),'Missing suppression provenance')
            if name=='control':check(mapping==ids and reps==ids and not deferred,'Control changed')
            elif ids:
                original_pose=poses[ids];reference=poses[mapping]
                separation=np.linalg.norm(original_pose[:,:3,3]-reference[:,:3,3],axis=1)
                rotation=reference[:,:3,:3]@original_pose[:,:3,:3].transpose(0,2,1)
                angles=np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))
                check((separation<=value['threshold']['position_m']+1e-7).all(),'NMS source position mismatch')
                check((angles<=np.deg2rad(value['threshold']['rotation_deg'])+1e-6).all(),'NMS source orientation mismatch')
                relations+=len(ids)
            retained=index in ids;same_arm=retained and same_arm_allowed
            small=row['variants'][name];retention=small['retention']
            check(retention['selected_retained_including_deferred']==retained,'Incorrect source availability')
            check(retention['original_selection_same_arm_available']==same_arm,'Incorrect same-arm grasp retention')
            check(retention['selected_is_priority_representative']==(index in reps),'Incorrect priority retention')
            check(retention['original_physical_grasp_success']==physical,'Incorrect physical history')
            check(small['representatives']==len(reps) and small['deferred']==len(deferred),'Incorrect candidate totals')
            summary.setdefault(name,[]).append({'representatives':len(reps),'available':retained,'same_arm':same_arm,
                'successful_available':bool(physical and retained),'successful_same_arm':bool(physical and same_arm),
                'was_successful':bool(physical),'seconds':small['postprocessing_seconds']})
            checked_variants+=1
        all_counts+=count
    for name,rows in summary.items():
        recorded=accepted['summaries'][name]
        expected={'representatives_total':sum(r['representatives'] for r in rows),
            'original_selected_available':sum(r['available'] for r in rows),
            'original_selected_same_arm_available':sum(r['same_arm'] for r in rows),
            'original_successful_selected_available':sum(r['successful_available'] for r in rows),
            'original_successful_same_arm_available':sum(r['successful_same_arm'] for r in rows),
            'original_successful_cases':sum(r['was_successful'] for r in rows)}
        check(all(recorded[k]==v for k,v in expected.items()),'Acceptance aggregate mismatch')
    verify(reg)
    write_json(root/'independent_review.json',{'at':now(),'passed':True,'record_integrity':True,
        'cases':len(reg['cases']),'raw_candidates':all_counts,'checked_variants':checked_variants,
        'checked_source_to_representative_relations':relations,'no_candidate_poses_scores_or_widths_changed':True,
        'historical_success_count':sum(v['was_successful'] for v in next(iter(summary.values()))),
        'new_IK_calls':0,'planning_requests':0,'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    audit(ensure_run_path(parser.parse_args().root))
