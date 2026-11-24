"""Registered real-ARX entry perturbations and bounded teacher motion goals."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from training.show_harness_arx.contract import rigid
from training.nearfield_arx.contract import encode, decode

VERSION = 'arx_physical_entry_variance_v1'
MAX_TEACHER_STEPS = 12


def physical_trials(sources):
    """Four physical entries per source; one actual G0 per fresh trajectory."""
    trials = []
    for index, source in enumerate(sources):
        star = rigid(source['source_Gstar_world'])
        axis = index % 3
        sign = 1 if index % 2 == 0 else -1
        for slot, (distance, side, angle) in enumerate(
                ((.100, .010, 5.), (.080, .015, 10.), (.060, .020, 15.),
                 ((.005, .010, .020)[index % 3], .005, 5.))):
            dp = np.zeros(3); dp[(index + slot) % 2] = sign * side
            dr = np.zeros(3); dr[(axis + slot) % 3] = sign * angle
            offset = np.eye(4); offset[:3, 3] = dp
            offset[:3, :3] = Rotation.from_rotvec(np.deg2rad(dr)).as_matrix()
            prior = star @ offset
            back = np.eye(4); back[2, 3] = -distance
            entry = prior @ back
            # This is also an early range check of the unchanged D interface.
            encode('D', entry, star, prior)
            trial = copy.deepcopy(source)
            trial.update(trial_id=f'{len(trials):02d}__{source["object_id"]}__physical_{slot}',
                method='recovery', policy='real_entry_teacher', entry_world=entry.tolist(),
                offset_goal_world=entry.tolist(), G0_world=prior.tolist(),
                source_split=source['source_split'], no_retry=True,
                physical_entry_kind='biased_entry' if slot < 3 else 'near_stopped_recovery',
                C_waypoints=[], C_waypoint_raw_indices=[],
                perturbation={'type':'physically_reached_G0_backoff',
                    'translation_Gstar_axes_m':dp.tolist(), 'rotation_Gstar_axes_degrees':dr.tolist(),
                    'backoff_m':distance, 'rotation_center':'source Gstar TCP origin',
                    'formula':'G0=Gstar@offset; H=G0@Tz(-distance)',
                    'prior_bound_to_physical_entry':True, 'virtual_prior_variants':0})
            trials.append(trial)
    if len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('Duplicate registered trial')
    return trials


def bounded_teacher_goal(current, star):
    """Offline teacher only; use unchanged C limits and original path checks."""
    current, star = rigid(current), rigid(star)
    dp = star[:3, 3] - current[:3, 3]
    dr = Rotation.from_matrix(current[:3, :3].T @ star[:3, :3]).as_rotvec()
    scale = min(1., .020 / max(float(np.linalg.norm(dp)), 1e-12),
                np.deg2rad(4.5) / max(float(np.linalg.norm(dr)), 1e-12))
    goal = current.copy(); goal[:3, 3] += dp * scale
    goal[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(dr * scale).as_matrix()
    token = encode('C', current, goal)
    return token, decode('C', current, token), bool(scale == 1.)
