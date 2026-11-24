"""Inspect paired data, neural-action provenance and real ARX P2 outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from observations import encode_state
from egad_evaluate import evaluate
from summarize_chunks import ROOT,read,sha,write


def policy_trial(trial,result):
    report=read(trial/'grasp_report.json') if (trial/'grasp_report.json').exists() else {}
    selected=read(trial/'selected_grasp.json') if (trial/'selected_grasp.json').exists() else {}
    convention=report.get('state_convention','object_relative_v1')
    errors=[]
    for source in report.get('action_sources',[]):
        saved=read(trial/f"local_{source['step']:04d}_prediction.json")
        original=read(trial/source['prediction_file'])
        snapshot=read(trial/f"local_{source['step']:04d}_state.json")
        expected=encode_state(snapshot,convention,selected.get('grasp_world'))
        if source['executed_policy_action_index']!=0 or saved['actions']!=original['actions']:
            errors.append({'step':source['step'],'reason':'Action changed or chunk index differs from K1'})
        if not np.allclose(expected,original['raw_state'],rtol=0,atol=1e-10):
            errors.append({'step':source['step'],'reason':'Dataset and runtime state codec differ'})
    predictions=sorted(trial.glob('policy_*_prediction.json'))
    if len(predictions)!=report.get('local_policy_calls',0):errors.append({'reason':'Policy call accounting differs'})
    if len(list(trial.glob('local_*_execution.json')))!=report.get('local_action_count',0):
        errors.append({'reason':'Actual action count differs'})
    frozen={}
    for name in ('model','model_closed'):
        dest=trial/name;base=ROOT/'scripts/curobo_arx/static_pick/reference'/name
        if not (dest/'arx.yml').exists():continue
        actual=yaml.safe_load((dest/'arx.yml').read_text());reference=yaml.safe_load((base/'arx.yml').read_text())
        a=actual['robot_cfg']['kinematics'];b=reference['robot_cfg']['kinematics']
        # These three fields describe relocated files and measured joint state;
        # frozen geometry, dynamics, limits and collision settings must match.
        for key in ('urdf_path','lock_joints'):a[key]=b[key]
        a['cspace']['default_joint_position']=b['cspace']['default_joint_position']
        frozen[name]=actual==reference and sha(dest/'arx.urdf')==sha(base/'arx.urdf')
    first=read(predictions[0]) if predictions else {}
    observed=[read(p) for p in sorted((trial/'lilo_observations').glob('*/observation.json'))]
    ages=[abs(x['capture_sim_time']-x['frame_time']) for x in observed]
    closure=None
    if (trial/'close_policy_source.json').exists() and selected:
        decision=read(trial/'close_policy_source.json')
        state_file=trial/decision['prediction'].replace('_prediction.json','_state.json')
        before=np.asarray(read(state_file)['link_world']['right_eef_link'])
        goal=np.asarray(selected['grasp_world'])
        closure={'source_prediction':decision['prediction'],
            'decision_position_error_to_planned_grasp_m':float(np.linalg.norm(before[:3,3]-goal[:3,3])),
            'decision_rotation_error_to_planned_grasp_rad':float(Rotation.from_matrix(before[:3,:3]@goal[:3,:3].T).magnitude())}
        if (trial/'close_execution.json').exists():
            after=np.asarray(read(trial/'close_execution.json')['final']['link_world']['right_eef_link'])
            closure['after_closure_position_error_to_planned_grasp_m']=float(np.linalg.norm(after[:3,3]-goal[:3,3]))
    entry=read(trial/'handoff_actual_entry.json') if (trial/'handoff_actual_entry.json').exists() else {}
    video=read(trial/'video_receipt.json') if (trial/'video_receipt.json').exists() else None
    return {**result,'actual_entry_pass':entry.get('pass'),'action_provenance_pass':not errors,
        'provenance_errors':errors,'frozen_planning_models':frozen,
        'first_raw_state':first.get('raw_state'),'first_normalized_state':first.get('normalized_state'),
        'first_action_world_delta_z_m':float(first['actions'][0][2]*.05) if first else None,
        'actual_frame_state_alignment':{'observations':len(ages),'max_abs_time_difference_seconds':max(ages) if ages else None,
            'all_within_0_1ms':all(x<=1e-4 for x in ages) if ages else None},
        'closure_geometry':closure,
        'video':video,'planning_certificate_is_not_grasp_success':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();collection=args.collection.resolve();comparison=args.comparison.resolve()
    attempts=read(collection/('all_attempts.json' if (collection/'all_attempts.json').exists() else 'results.json'))
    probe=ROOT/'runs/lilo_P2_teacher_probe_20261028_01'
    teacher=[]
    for name in ('A1','bottle'):
        trial=probe/(name+'_grasp_frame_r1')
        teacher.append({'scene':name,'evaluation':evaluate(trial),'source_verification':read(trial/'verification.json'),
            'source':str(trial),'A1_excluded_from_datasets':name=='A1'})
    groups=[]
    for name in sorted({r['object_name'] for r in attempts}):
        subset=[r for r in attempts if r['object_name']==name]
        groups.append({'object_name':name,'attempts':len(subset),'successes':sum(r['pass'] for r in subset),
            'demonstration_ready':sum(r['demonstration_ready'] for r in subset),
            'failures':[{'label':r['label'],'error':r['evaluation'].get('reported_error')} for r in subset if not r['pass']]})
    variants=[]
    for label in ('old','object','goal'):
        folder=comparison/(label+'_physical')
        if not (folder/'results.json').exists():continue
        rows=[policy_trial(folder/r['scene'],r) for r in read(folder/'results.json')]
        variants.append({'variant':label,'rows':rows,'attempts':len(rows),'successes':sum(r['pass'] for r in rows),
            'integrity':read(folder/'final_integrity.json') if (folder/'final_integrity.json').exists() else None})
    dataset=[]
    for label in ('object','goal'):
        folder=comparison/(label+'_dataset')
        if not (folder/'quality_report.json').exists():continue
        spec=read(folder/'dataset.json');quality=read(folder/'quality_report.json')
        dataset.append({'variant':label,'state_convention':spec['state_convention'],
            'action_alignment_version':spec.get('action_alignment_version'),
            'exclude_controller_closure_dwell_and_hold_chunk_starts':spec.get('exclude_controller_closure_dwell_and_hold_chunk_starts',False),
            'objects':{k:spec[k+'_objects'] for k in ('train','validation')},
            'episodes':dict(Counter(e['split'] for e in spec['episodes'])),
            'transitions':{split:sum(len(e['items']) for e in spec['episodes'] if e['split']==split) for split in ('train','validation')},
            'chunks':{split:sum(len(e['chunk_starts']) for e in spec['episodes'] if e['split']==split) for split in ('train','validation')},
            'quality':quality,'manifest_sha256':sha(folder/'dataset.json')})
    adapters={}
    for label in ('object','goal'):
        folder=comparison/(label+'_adapter')
        if not (folder/'adapter_manifest.json').exists():continue
        training=read(folder/'training_report.json')
        selected=next((r for r in training['history'] if r['step']==training['selected_step']),training['baseline'])
        adapters[label]={'manifest':read(folder/'adapter_manifest.json'),'baseline':training['baseline'],
            'selected_error':selected,'action_head_sha256':sha(folder/'action_head.pt')}
    chain=[];cursor=collection
    while cursor:
        registered=read(cursor/'registration.json')
        chain.append({'directory':str(cursor),'state':read(cursor/'state.json'),
            'registered_new_attempts':registered['new_attempts'],
            'recorded_attempts':len(read(cursor/'results.json')),
            'candidate_handoff_filter':registered.get('candidate_handoff_filter',False),
            'physical_sources_still_equal':all(sha(p)==digest for p,digest in registered['physical_source_sha256'].items()),
            'executed_source_snapshot_matches_registration':all(sha(cursor/'registered_source_snapshot'/Path(p).name)==digest for p,digest in registered['algorithm_sha256'].items()),
            'completion_integrity':read(cursor/'completion_integrity.json') if (cursor/'completion_integrity.json').exists() else None,
            'interruption_recovery':read(cursor/'interruption_recovery.json') if (cursor/'interruption_recovery.json').exists() else None})
        cursor=Path(registered['previous_collection']) if registered.get('previous_collection') else None
    result={'scope':'ARX P2 development comparison, head-only adaptation; not full LiLo fine-tuning or an EGAD49 score',
        'teacher_qualification':teacher,'new_collection_attempts':len(attempts),'new_collection_successes':sum(r['pass'] for r in attempts),
        'reused_bottle_success':1,'collection_groups':groups,'datasets':dataset,'adapters':adapters,
        'collection_chain':chain,
        'paired_data':read(comparison/'paired_dataset_verification.json') if (comparison/'paired_dataset_verification.json').exists() else None,
        'physical_policy_variants':variants,'actual_policy_attempts':sum(v['attempts'] for v in variants),
        'actual_policy_successes':sum(v['successes'] for v in variants),
        'all_recorded_actions_from_versioned_neural_output':all(r['action_provenance_pass'] for v in variants for r in v['rows']),
        'collection_state':read(collection/'state.json'),'comparison_state':read(comparison/'state.json'),
        'default_execute_steps':1,'global_policy_changed':False,'full_EGAD49_resumed':False,
        'issue':'LILO-ARX-001 remains unresolved pending repeated failure-case and positive-control validation'}
    write(args.output.resolve(),result)
    print(json.dumps({'output':str(args.output),'collection_attempts':len(attempts),'collection_successes':result['new_collection_successes'],
        'physical_variants':[{k:v[k] for k in ('variant','attempts','successes')} for v in variants]},indent=2))


if __name__=='__main__':main()
