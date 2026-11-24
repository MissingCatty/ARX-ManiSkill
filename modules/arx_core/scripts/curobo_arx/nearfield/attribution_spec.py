"""Matched G0 / trained D / foundation-model experiment; no physical mutations."""
import copy
import random
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, digest
from training.nearfield_arx.eef_fit import encode

VERSION = 'arx_g0_attribution_v1'
ARMS = ('g0', 'trained', 'base')
REPEATS = 2
SCHEDULE_SEED = 20261110
MATCHED = ('case_id', 'repeat', 'seed_id', 'object_id', 'placement', 'renderer', 'source_split',
           'G0_world', 'entry_world', 'offset_goal_world', 'perturbation', 'fixed_entry_backoff_m')


def make_trials(previous, verification):
    originals = {t['case_id']: t for t in previous['seed_trials'] if t['method'] == 'D'}
    results = {t['case_id']: t for t in verification['trials'] if t['method'] == 'D'}
    if len(originals) != 12 or set(results) != set(originals):
        raise ValueError('Need all twelve historical D conditions')
    orders = [(a,b,c) for a in ARMS for b in ARMS for c in ARMS if len({a,b,c}) == 3]
    rng = random.Random(SCHEDULE_SEED); trials = []
    for repeat in range(REPEATS):
        cases = sorted(originals); rng.shuffle(cases)
        for index, case in enumerate(cases):
            source = originals[case]; selected = results[case]['selected_entry']
            entry = selected['selected'] if selected else {'goal_world': source['entry_world'], 'backoff_m': .08}
            if not entry: raise ValueError('Historical selected entry missing')
            for arm in orders[(index + repeat*3) % len(orders)]:
                t = copy.deepcopy(source)
                t.update(trial_id=f'a{len(trials):02d}_r{repeat+1}_{case}_{arm}',
                         historical_trial_id=source['trial_id'], attribution_arm=arm,
                         repeat=repeat+1, policy=VERSION+'_'+arm, method='D',
                         entry_world=copy.deepcopy(entry['goal_world']),
                         offset_goal_world=copy.deepcopy(entry['goal_world']),
                         original_80mm_entry_world=copy.deepcopy(source['entry_world']),
                         fixed_entry_backoff_m=entry['backoff_m'], no_retry=True)
                trials.append(t)
    validate_trials(trials)
    return trials


def validate_trials(trials):
    if len(trials) != 72 or len({t['trial_id'] for t in trials}) != 72:
        raise ValueError('Exactly72 unique prospectively scheduled slots')
    groups = {}
    for t in trials:
        if t['method'] != 'D' or t['attribution_arm'] not in ARMS or not t['no_retry'] or t['repeat'] not in (1,2):
            raise ValueError('Unregistered controller or retry')
        groups.setdefault((t['case_id'], t['repeat']), []).append(t)
    if len(groups) != 24: raise ValueError('Exactly24 matched triplets')
    for group in groups.values():
        if len(group) != 3 or {t['attribution_arm'] for t in group} != set(ARMS):
            raise ValueError('Missing or repeated comparison arm')
        for t in group[1:]:
            if any(t[k] != group[0][k] for k in MATCHED):
                raise ValueError('Unmatched G0, entry, reset or rendering')


def fixed_entry_candidates(trial):
    return [{'backoff_m': trial['fixed_entry_backoff_m'], 'goal_world': trial['entry_world']}]


def identity(reg, arm):
    if arm == 'g0':
        return {'generator': 'g0', 'generation_kind': 'deterministic_geometry', 'checkpoint': None,
                'ARX_finetuned': False, 'model_inference': False}
    if arm not in ARMS: raise ValueError('Unknown comparison arm')
    path = reg['learned_policy']['checkpoints']['D'] if arm == 'trained' else reg['learned_policy']['base_model']
    return {'generator': arm, 'generation_kind': 'VLM_inference', 'checkpoint': path,
            'ARX_finetuned': arm == 'trained', 'model_inference': True}


def bind_arm(reg, request):
    matches = [t for t in reg['seed_trials'] if t['trial_id'] == request['trial_id']]
    if len(matches) != 1: raise ValueError('Unknown or duplicate trial request')
    trial = matches[0]
    if request['method'] != 'D' or request['inputs']['G0_world'] != trial['G0_world']:
        raise ValueError('Method or fixed G0 changed')
    return trial['attribution_arm']


def geometry_prediction(inputs):
    # Only the measured current pose and fixed front-end proposal enter this controller.
    return encode('D', inputs['eef_world'], inputs['G0_world'], inputs['G0_world'])


def verify_generation(reg, request, reply):
    arm = bind_arm(reg, request); expected = identity(reg, arm)
    if any(reply.get(k) != v for k,v in expected.items()):
        raise ValueError('Generator/model identity differs from registered arm')
    if arm == 'g0' and reply['prediction'] != geometry_prediction(request['inputs']):
        raise ValueError('G0 control used a different target or external prediction')
    return arm
