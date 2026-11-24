"""Exactly two diagnostic regressions followed by the original 24 shift slots."""
import copy
from .visual_grounding_spec import make_trials as shift_trials, validate_trials as validate_shifts

REGRESSIONS = ('r30_21_03_normal_C', 'r30_04_02_entry_offset_D')


def make_trials(previous):
    shifts, seeds = shift_trials(previous); trials = []
    for ident in REGRESSIONS:
        t = copy.deepcopy(next(t for t in previous['seed_trials'] if t['trial_id'] == ident))
        t.update(trial_id='a31_regression_'+ident, historical_trial_id=ident, phase='regression',
                 vision_arm='fresh', repeat=1, target_delta_world_m=[0., 0., 0.],
                 fixed_entry_backoff_m=.08, no_retry=True)
        trials.append(t)
    for t in shifts:
        t['prior_unused_slot_id'] = t['trial_id']
        t.update(trial_id='a31_'+t['trial_id'], phase='target_shift')
        trials.append(t)
    validate_trials(trials)
    return trials, {k: v for k, v in seeds.items() if k in {t['seed_id'] for t in trials}}


def validate_trials(trials):
    if len(trials) != 26 or len({t['trial_id'] for t in trials}) != 26:
        raise ValueError('Exactly26 new slots, without automatic retries')
    if tuple(t['historical_trial_id'] for t in trials[:2]) != REGRESSIONS:
        raise ValueError('Specified two diagnostic regressions must run first')
    if any(t['phase'] != 'regression' or t['target_delta_world_m'] != [0., 0., 0.] for t in trials[:2]):
        raise ValueError('Historical regression reset changed')
    if any(t['phase'] != 'target_shift' for t in trials[2:]): raise ValueError('Unexpected extra phase')
    validate_shifts(trials[2:])
    if not all(t['no_retry'] for t in trials): raise ValueError('No automatic retries')
