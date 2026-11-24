"""New declared target placements, with an unchanged prior and entrance."""
import copy
import math
import random
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, digest
from training.nearfield_arx.contract import pose_error
from training.nearfield_arx.lateral_recovery import local_features
from .attribution_spec import geometry_prediction
from .verify_validation import reset_signature

VERSION='arx_visual_grounding_physical_v1'
ARMS=('g0','fresh','stale_first')
OBJECTS=('household_hope_milk_carton','household_hot3d_bbq_sauce_bottle')


def make_trials(old):
    originals={t['object_id']:t for t in old['seed_trials'] if t['method']=='D' and t['condition']=='normal'}
    seeds=copy.deepcopy(old['seeds']);cases=[]
    for obj in OBJECTS:
        source=originals[obj]
        for shift in (-.015,.015):
            t=copy.deepcopy(source);placement=copy.deepcopy(source['placement'])
            placement['placement']['position'][0]+=shift;placement['offset_xy_m'][0]+=shift
            placement['minimum_support_border_m']-=abs(shift)
            if placement['minimum_support_border_m']<=0:raise ValueError('Target shift exceeds original support')
            placement['placement_id']=digest([VERSION,source['placement']['placement_id'],shift])[:20]
            placement['registered_counterfactual_delta_world_m']=[shift,0.,0.]
            ident=digest([source['seed_id'],placement['placement_id']])[:32]
            seed=copy.deepcopy(seeds[source['seed_id']]);seed.update(seed_id=ident,placement=placement,
                 source_seed_id=source['seed_id']);seeds[ident]=seed
            t.update(seed_id=ident,source_seed_id=source['seed_id'],placement=placement,
                case_id=obj+('_minus15' if shift<0 else '_plus15'),condition='target_x_shift',
                target_delta_world_m=[shift,0.,0.],policy=VERSION,method='D',no_retry=True,
                fixed_entry_backoff_m=.08,historical_trial_id=source['trial_id'],
                source_Gstar_role='Translated historical pose for diagnosis only; never a policy input or command target')
            t['source_Gstar_world'][0][3]+=shift
            cases.append(t)
    rng=random.Random(20261110);trials=[]
    orders=[('g0','fresh','stale_first'),('fresh','stale_first','g0'),('stale_first','g0','fresh'),
            ('g0','stale_first','fresh'),('stale_first','fresh','g0'),('fresh','g0','stale_first')]
    for repeat in (1,2):
        ordered=copy.deepcopy(cases);rng.shuffle(ordered)
        for index,c in enumerate(ordered):
            for arm in orders[(index+2*(repeat-1))%6]:
                t=copy.deepcopy(c);t.update(trial_id=f'v{len(trials):02d}_r{repeat}_{c["case_id"]}_{arm}',
                    repeat=repeat,vision_arm=arm,attribution_arm='g0' if arm=='g0' else 'trained')
                trials.append(t)
    validate_trials(trials)
    return trials,seeds


def validate_trials(trials):
    if len(trials)!=24 or len({t['trial_id'] for t in trials})!=24:raise ValueError('Exactly24 unique slots')
    groups={}
    for t in trials:
        if t['vision_arm'] not in ARMS or t['repeat'] not in (1,2) or not t['no_retry']:
            raise ValueError('Unregistered branch or retry')
        groups.setdefault((t['case_id'],t['repeat']),[]).append(t)
    if len(groups)!=8:raise ValueError('Need eight matched triplets')
    for rows in groups.values():
        if {t['vision_arm'] for t in rows}!=set(ARMS) or len(rows)!=3:raise ValueError('Missing arm')
        for t in rows[1:]:
            for key in ('placement','seed_id','G0_world','entry_world','renderer','target_delta_world_m'):
                if t[key]!=rows[0][key]:raise ValueError('Unmatched counterfactual condition: '+key)
    for obj in OBJECTS:
        rows=[t for t in trials if t['object_id']==obj]
        if any(t['G0_world']!=rows[0]['G0_world'] or t['entry_world']!=rows[0]['entry_world'] for t in rows):
            raise ValueError('Prior or entrance followed the displaced object')


def expected_reset(seed,trial):
    value=reset_signature(read(Path(seed['episode'])/'initial_conditions.json'))
    value=copy.deepcopy(value);key='asset:'+trial['object_id']
    if key not in value['objects']:raise ValueError('Missing target in original reset')
    value['objects'][key]['position']=[float(a+b) for a,b in zip(value['objects'][key]['position'],trial['target_delta_world_m'])]
    value['collection_placement']=trial['placement']
    return value


def binding(reg,request):
    rows=[t for t in reg['seed_trials'] if t['trial_id']==request['trial_id']]
    if len(rows)!=1:raise ValueError('Unknown trial')
    t=rows[0]
    if request['method']!='D' or request['inputs']['G0_world']!=t['G0_world']:
        raise ValueError('Method or prior changed')
    return t


def first_decision(request):
    return request['inputs']['decision_history_by_method']['D']['calls_completed']==0


def alignment(request,donor):
    old=read(donor['request']);a=request['inputs'];b=old['inputs']
    actual_observation=read(request['observation']);donor_observation=read(old['observation'])
    ee=pose_error(a['eef_world'],b['eef_world'])
    camera=pose_error(actual_observation['camera_world_optical'],donor_observation['camera_world_optical'])
    state=local_features(a,'D')==local_features(b,'D')
    intrinsics=np.allclose(actual_observation['intrinsics'],donor_observation['intrinsics'],atol=1e-7,rtol=0)
    fingers=max(abs(a['joint_positions'][a['joint_names'].index(n)]-b['joint_positions'][b['joint_names'].index(n)])
                for n in ('right_joint27','right_joint28'))
    passed=state and intrinsics and fingers<=.0005 and all(e['translation_m']<=.005 and e['rotation_rad']<=math.radians(2) for e in (ee,camera))
    return {'pass':bool(passed),'same_nonvisual_state':state,'same_intrinsics':bool(intrinsics),
            'eef_pose_error':ee,'camera_pose_error':camera,'maximum_finger_joint_difference_m':float(fingers),
            'pose_limits_m_degrees':[.005,2.],'donor_request':donor['request']}


def verify_reply(reg,request,reply):
    t=binding(reg,request); arm=t['vision_arm']; applicable=reply['visual_control']['applicable']
    expected_swap=applicable and arm=='stale_first' and first_decision(request)
    if reply['vision_arm']!=arm or reply['visual_control']['replaced']!=expected_swap:
        raise ValueError('Wrong registered visual intervention')
    expected_model=applicable and arm!='g0'
    if reply['model_inference']!=expected_model:raise ValueError('Model-use identity mismatch')
    if expected_model and reply['checkpoint']!=reg['learned_policy']['checkpoints']['D']:
        raise ValueError('Wrong frozen D model')
    if applicable and arm=='g0' and reply['prediction']!=geometry_prediction(request['inputs']):
        raise ValueError('G0 branch did not use its registered geometric control')
    if first_decision(request):
        expected=alignment(request,reg['visual_donors'][t['object_id']])
        # Same host/container arithmetic may differ in final rotation bits.
        from training.nearfield_arx.receipt_numbers import same_pose_diagnostics
        if not same_pose_diagnostics(expected,reply['visual_control']['alignment']) or expected['pass']!=applicable:
            raise ValueError('Unbound or changed donor-pose matching')
    if expected_swap:
        if reply['visual_control']['model_images']!=reg['visual_donors'][t['object_id']]['images']:
            raise ValueError('Unregistered wrong image supplied')
    else:
        directory=Path(request['observation']).parent
        expected=[str(directory/(request['request_id']+'_'+name+'.png')) for name in ('rgb','depth')]
        if reply['visual_control']['model_images']!=expected:
            raise ValueError('Fresh-observation arm received a substituted image')
    return t
