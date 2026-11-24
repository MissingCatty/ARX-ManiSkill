"""A case-scoped admission contract from repeated actual ARX successes.

This is an empirical guard for pinned development goals, never an assertion
that a normalization box or an untested object is executable by the policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .egad49 import compare_configuration
else:
    from egad49 import compare_configuration


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def scenario_key(scenario):
    placement=scenario['placement']
    return digest({'asset':scenario['asset'],'target':scenario['target'],
        'position':placement['position'],'orientation':placement.get('orientation',[1.,0.,0.,0.])})


def verify_policy(profile,health):
    if profile.get('qualified') is not True:raise ValueError('Handoff profile is not qualified')
    if Path(health.get('adapter','')).resolve()!=Path(profile['adapter']).resolve():raise ValueError('Profile adapter mismatch')
    if health.get('state_convention')!=profile['state_convention']:raise ValueError('Profile state convention mismatch')
    if health.get('policy_artifact_sha256')!=profile['policy_artifact_sha256']:
        raise ValueError('Loaded policy artifact identity does not match the qualified profile')
    if not health.get('checkpoint_verified') or health.get('checkpoint_revision')!=profile['checkpoint_revision']:
        raise ValueError('Profile base checkpoint mismatch')


def verify_case(profile,scenario,selected,*,check_rule=True):
    candidates=[(name,case) for name,case in profile['cases'].items() if case['scenario_sha256']==scenario_key(scenario)]
    if len(candidates)!=1:raise ValueError('Scenario is outside the measured handoff profile')
    name,case=candidates[0]
    if digest(selected['grasp_world'])!=case['planned_grasp_sha256']:
        raise ValueError('Selected grasp is outside the measured handoff profile')
    if check_rule and (selected.get('handoff_rule')!=profile['rule']['version']
        or selected.get('handoff_backoff_m')!=profile['rule']['ordered_backoff_m'][0]):
        raise ValueError('Entry rule does not match the measured handoff profile')
    return name,case


def admit(profile,health,scenario,selected,observation,certificate,setup,instruction):
    verify_policy(profile,health);name,case=verify_case(profile,scenario,selected)
    if instruction.strip().lower()!=case['instruction'].strip().lower():raise ValueError('Skill instruction does not match the qualified case')
    mismatch=compare_configuration(case['reference_fingerprint'],setup['fingerprint'])
    if mismatch:raise ValueError('Profile frozen configuration mismatch: '+str(mismatch[:5]))
    for key in ('joint_names','joint_positions'):
        if setup['initial_conditions'][key]!=case['initial_robot_state'][key]:
            raise ValueError('Profile historical robot reset mismatch')
    initial=setup['initial_conditions']['objects']['asset:'+scenario['asset']]
    if initial['position']!=scenario['placement']['position'] or initial['quaternion_wxyz']!=scenario['placement'].get('orientation',[1.,0.,0.,0.]):
        raise ValueError('Profile object placement mismatch')
    if not certificate.get('pass') or not certificate.get('frustum',{}).get('pass') \
        or certificate.get('pregrasp_target_contact_events')!=0 or not certificate.get('continuation',{}).get('pass'):
        raise ValueError('Existing actual-entry contact/view/continuation gates rejected this handoff')
    if observation.get('camera_role')!='right_wrist':raise ValueError('Profile requires the measured right wrist observation')
    age=abs(observation['capture_sim_time']-observation['frame_time'])
    if not np.isfinite(age) or age>profile['frame_state_tolerance_seconds']:raise ValueError('Profile observation and state are not synchronized')
    actual=np.asarray(observation['snapshot']['link_world']['right_eef_link'])
    if actual.shape!=(4,4) or not np.isfinite(actual).all():raise ValueError('Invalid measured entry pose')
    center=np.asarray(case['measured_entry_center_world'])
    position=float(np.linalg.norm(actual[:3,3]-center[:3,3]))
    angle=float(Rotation.from_matrix(actual[:3,:3]@center[:3,:3].T).magnitude())
    if position>profile['entry_position_tolerance_m'] or angle>profile['entry_rotation_tolerance_rad']:
        raise ValueError('Actual entry is outside the tested point tracking neighborhood')
    snapshot=observation['snapshot'];joints=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    fingers=np.array([joints[n] for n in ('right_joint27','right_joint28')])
    if not np.isfinite(fingers).all() or np.max(np.abs(fingers-case['measured_entry_fingers_m']))>profile['entry_gripper_tolerance_m']:
        raise ValueError('Measured finger state does not match the tested open-gripper entry')
    return {'pass':True,'case':name,'position_distance_from_measured_center_m':position,
        'rotation_distance_from_measured_center_rad':angle,'frame_state_age_seconds':age,
        'rule_version':profile['rule']['version'],'policy_identity_verified':True,
        'scope':'Pinned policy, frozen scenario and planned goal; empirical tracking neighborhood only, no unseen-object guarantee'}


def create(campaign,output,summary_file=None):
    campaign=Path(campaign).resolve()
    if read(campaign/'state.json')['status']!='completed':raise ValueError('P3 qualification is not complete')
    summary_path=Path(summary_file).resolve() if summary_file else campaign/'summary.json'
    summary=read(summary_path);registration=read(campaign/'registration.json')
    if summary_path!=campaign/'summary.json':
        original=read(campaign/'summary.json')
        if summary.get('original_report_sha256')!=hashlib.sha256((campaign/'summary.json').read_bytes()).hexdigest():
            raise ValueError('Rechecked report is not bound to this original campaign')
        if not summary.get('physical_outcomes_unchanged') or not summary.get('recheck_environment',{}).get('strict_state_check_pass'):
            raise ValueError('Runtime recheck did not preserve physical outcomes and strict state precision')
        before=[(r['trial'],r['pass'],r['evaluation']) for r in original['rows']]
        after=[(r['trial'],r['pass'],r['evaluation']) for r in summary['rows']]
        if before!=after:raise ValueError('Recheck changed physical outcome records')
    integrity=read(campaign/'completion_integrity.json')
    if not integrity.get('registered_sources_and_policy_unchanged') or not integrity.get('source_archive_matches'):
        raise ValueError('P3 execution integrity evidence is incomplete')
    selected=summary['selected_for_repeats']
    if selected not in summary['qualified_development_points']:raise ValueError('No point passed all three repeats in both cases')
    artifacts={Path(p).name:value for p,value in registration['adapter_sha256'].items()}
    adapter=Path(registration['adapter']);manifest=read(adapter/'adapter_manifest.json')
    for name,value in artifacts.items():
        if hashlib.sha256((adapter/name).read_bytes()).hexdigest()!=value:raise ValueError('Qualified policy artifact changed')
    profile={'version':'arx_lilo_measured_entry_v1','qualified':True,'adapter':str(adapter),
        'policy_artifact_sha256':artifacts,'state_convention':manifest['state_convention'],
        'checkpoint_revision':manifest['base_checkpoint_revision'],'rule':registration['rules'][selected],
        'entry_position_tolerance_m':.002,'entry_rotation_tolerance_rad':.03,'entry_gripper_tolerance_m':.001,
        'frame_state_tolerance_seconds':1e-4,
        'qualification_campaign':str(campaign),'qualification_summary':str(summary_path),
        'qualification_summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'cases':{},'default_enabled':False,'max_automatic_repositions':0,
        'scope':'Two pinned development goals; do not infer a continuous capture range or unseen-object support'}
    for scene in ('A1','bottle'):
        rows=[r for r in summary['rows'] if r['entry_group']==selected and r['scene']==scene]
        if len(rows)!=3 or not all(r['pass'] and r['action_provenance_pass'] and r['actual_entry_pass']
            and len(r['frozen_planning_models'])==2 and all(r['frozen_planning_models'].values()) for r in rows):
            raise ValueError('Missing independent physical repeat evidence')
        poses=[];scenarios=[];goals=[];fingers=[]
        for row in rows:
            trial=Path(row['trial']);observation=read(trial/'handoff_actual_observation/observation.json')
            poses.append(np.asarray(observation['snapshot']['link_world']['right_eef_link']))
            snapshot=observation['snapshot'];q=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
            fingers.append([q[n] for n in ('right_joint27','right_joint28')])
            scenarios.append(scenario_key(read(trial/'scenario.json')))
            goals.append(digest(read(trial/'selected_grasp.json')['grasp_world']))
        if len(set(scenarios))!=1 or len(set(goals))!=1:raise ValueError('Qualification mixed different scenarios or grasp goals')
        center=np.eye(4);center[:3,3]=np.mean([p[:3,3] for p in poses],axis=0)
        center[:3,:3]=Rotation.from_matrix(np.stack([p[:3,:3] for p in poses])).mean().as_matrix()
        trial=Path(rows[0]['trial']);setup=read(trial/'setup.json')
        profile['cases'][scene]={'scenario_sha256':scenarios[0],'planned_grasp_sha256':goals[0],
            'instruction':read(trial/'grasp_report.json')['instruction'],
            'measured_entry_center_world':center.tolist(),'reference_fingerprint':setup['fingerprint'],
            'measured_entry_fingers_m':np.mean(fingers,axis=0).tolist(),
            'initial_robot_state':{k:setup['initial_conditions'][k] for k in ('joint_names','joint_positions')},
            'qualification_trials':[r['trial'] for r in rows]}
    health={'adapter':str(adapter),'state_convention':profile['state_convention'],
        'policy_artifact_sha256':artifacts,'checkpoint_verified':True,'checkpoint_revision':profile['checkpoint_revision']}
    for case in profile['cases'].values():
        for folder in case['qualification_trials']:
            trial=Path(folder)
            admit(profile,health,read(trial/'scenario.json'),read(trial/'selected_grasp.json'),
                read(trial/'handoff_actual_observation/observation.json'),read(trial/'handoff_actual_entry.json'),read(trial/'setup.json'),
                read(trial/'grasp_report.json')['instruction'])
    Path(output).write_text(json.dumps(profile,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'profile':str(output),'rule':profile['rule']['version'],'cases':list(profile['cases'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,help='Optional strict recheck from the action-execution Python environment')
    args=parser.parse_args();create(args.campaign,args.output,args.summary)
