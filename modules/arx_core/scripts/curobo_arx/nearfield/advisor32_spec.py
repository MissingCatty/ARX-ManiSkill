"""Prospective 24+6 slots and full-stale intervention, not a launch authority."""
import copy
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import digest
from .visual_grounding_spec import make_trials as old_shifts, validate_trials as validate_old_shifts


ARMS = ('g0', 'fresh', 'full_stale')


def make_trials(old, canonical):
    shifts, _ = old_shifts(old)
    result = []
    for trial in shifts:
        row = copy.deepcopy(trial)
        if row['vision_arm'] == 'stale_first':
            row['vision_arm'] = 'full_stale'
        row.update(trial_id='a32_'+row['trial_id'].replace('stale_first', 'full_stale'),
                   phase='target_shift', no_retry=True, policy='advisor32_proposed',
                   source_Gstar_role='Offline diagnostic only, never supplied to D')
        result.append(row)
    canon = np.asarray(canonical['Gstar_canon_world'], dtype=float)
    for condition in ('normal', 'entry_offset'):
        for arm in ('g0', 'C', 'D'):
            method = 'C' if arm == 'C' else 'D'
            matches = [t for t in old['seed_trials'] if t['object_id']=='household_ycb_spam_can'
                       and t['condition']==condition and t['method']==method]
            if len(matches) != 1:
                raise ValueError('Ambiguous historical Spam entry')
            original = matches[0]; row = copy.deepcopy(original)
            relative = np.linalg.inv(np.asarray(original['G0_world'])) @ np.asarray(original['entry_world'])
            entry = canon @ relative
            row.update(trial_id=f'a32_spam_canon_{condition}_{arm}', phase='spam_canonical',
                source_trial_id=original['trial_id'], method=method,
                vision_arm='g0' if arm=='g0' else 'fresh', comparison_arm=arm,
                G0_world=canon.tolist(), source_Gstar_world=canon.tolist(),
                entry_world=entry.tolist(), offset_goal_world=entry.tolist(),
                target_delta_world_m=[0., 0., 0.], repeat=1, no_retry=True,
                canonical_reference_is_a_proposal_not_successful_teacher=True,
                canonical_IK_and_collision_preflight_required=True,
                source_Gstar_role='Offline proposed canonical reference; not previously demonstrated success',
                policy='advisor32_proposed')
            row['canonical_entry_relative_to_grasp'] = relative.tolist()
            result.append(row)
    validate_trials(result)
    return result


def validate_trials(rows):
    if len(rows) != 30 or len({r['trial_id'] for r in rows}) != 30:
        raise ValueError('Exactly24 target-shift and6 canonical Spam slots')
    shifts = copy.deepcopy([r for r in rows if r['phase']=='target_shift'])
    for row in shifts:
        if row['vision_arm'] == 'full_stale':
            row['vision_arm'] = 'stale_first'
    validate_old_shifts(shifts)
    spam = [r for r in rows if r['phase']=='spam_canonical']
    if len(spam) != 6 or {(r['condition'], r['comparison_arm']) for r in spam} != {
            (c, a) for c in ('normal', 'entry_offset') for a in ('g0', 'C', 'D')}:
        raise ValueError('Missing or extra canonical Spam comparison')
    for condition in ('normal', 'entry_offset'):
        group = [r for r in spam if r['condition']==condition]
        for r in group[1:]:
            for field in ('placement', 'G0_world', 'entry_world', 'renderer', 'seed_id'):
                if r[field] != group[0][field]:
                    raise ValueError('Unmatched canonical condition: '+field)
    if not all(r['no_retry'] for r in rows):
        raise ValueError('No implicit retry slots')
    return {'pass': True, 'target_shift': len(shifts), 'spam_canonical': len(spam),
            'total': len(rows), 'physical_trials_executed': 0,
            'manifest_sha256': digest(rows), 'launch_ready': False}


def model_images(arm, decision_index, current_images, historical_by_plane):
    """Both stale decisions use archived images; never silently use fresh data."""
    if arm not in ARMS or type(decision_index) is not int or decision_index not in (0, 1):
        raise ValueError('Expected one of exactly two registered decisions')
    if arm == 'g0':
        return {'model_inference': False, 'images': [], 'replaced': False}
    if len(current_images) != 2:
        raise ValueError('Expected a current RGB/depth pair')
    if arm == 'full_stale':
        if set(historical_by_plane) != {'0', '1'} or len(historical_by_plane[str(decision_index)]) != 2:
            raise ValueError('Missing stage-matched historical pair; no fresh fallback')
        images = historical_by_plane[str(decision_index)]
    else:
        images = current_images
    if not all(Path(p).is_file() for p in images):
        raise FileNotFoundError('Missing registered model image; no replacement')
    return {'model_inference': True, 'images': list(map(str, images)),
            'replaced': arm == 'full_stale',
            'scope': 'Neural model images only; execution checks use measured current observations'}
