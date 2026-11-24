"""Frozen 4-object x 3-condition x 4-method comparison; no result selection."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from training.nearfield_arx.contract import pose_error

VERSION = 'nearfield_original48_v1'
OBJECTS = ('household_hope_milk_carton', 'mustard_bottle',
           'household_hot3d_bbq_sauce_bottle', 'household_ycb_spam_can')
GEOMETRY = {
    'max_queries': 2, 'request_timeout_seconds': 120,
    'translation_neighborhood_m': .04, 'rotation_neighborhood_degrees': 30.,
    'target_surface_distance_m': .04, 'cloud_context_padding_m': .06,
    'minimum_depth_m': .001, 'maximum_depth_m': 10.,
    'candidate_selection': 'highest pretrained score after collision/NMS, visible-target association and G0 neighborhood; stable ties',
    'symmetry': 'actual ARX TCP rotation about local approach Z by180degrees; same frozen40mm tool offset',
    'no_candidate': 'ordinary failure, no alternative algorithm or retry',
    'mask_source': 'same frozen visible-mesh/depth projection as C/D; no instance truth mask',
    'Gstar_input': False, 'ICP_fallback': False,
}


def make_trials(sources, seeds):
    selected = [min((s for s in sources if s['object_id'] == obj), key=lambda s: s['seed_id']) for obj in OBJECTS]
    trials = []
    for oi, source in enumerate(selected):
        # Gbase is the frozen front-end proposal, not the measured historical Gstar.
        base = np.asarray(seeds[source['seed_id']]['goal']['grasp_world'], dtype=float)
        offset = np.eye(4)
        offset[oi % 2, 3] = (.02 if oi % 2 == 0 else -.02)
        rot = np.zeros(3); rot[(oi + 1) % 3] = (10 if oi % 2 == 0 else -10)
        offset[:3, :3] = Rotation.from_rotvec(np.deg2rad(rot)).as_matrix()
        backoff = np.eye(4); backoff[2, 3] = -.08
        for ci, condition in enumerate(('normal', 'entry_offset', 'joint_prior_entry_offset')):
            g0 = base @ offset if ci == 2 else base.copy()
            entry = base @ backoff @ offset if ci == 1 else g0 @ backoff
            # Cyclic order balances all methods across the12 cases.
            order = 'ABCD'[(oi * 3 + ci) % 4:] + 'ABCD'[:(oi * 3 + ci) % 4]
            for method in order:
                trial = copy.deepcopy(source)
                trial.update(trial_id=f'{len(trials):02d}_{source["object_id"]}_{condition}_{method}',
                    case_id=f'{oi:02d}_{condition}', condition=condition, method=method,
                    policy='main48_' + method, Gbase_world=base.tolist(), G0_world=g0.tolist(),
                    entry_world=entry.tolist(), offset_goal_world=entry.tolist(),
                    C_waypoints=[], C_waypoint_raw_indices=[], no_retry=True,
                    physical_entry_kind=condition, truth_replay_only_no_policy_training=False,
                    perturbation={'type': condition, 'backoff_m': .08,
                        'translation_local_m': offset[:3, 3].tolist() if ci else [0., 0., 0.],
                        'rotation_local_degrees': rot.tolist() if ci else [0., 0., 0.],
                        'formula': ('G0=Gbase; H=Gbase@Tz(-.08)' if ci == 0 else
                            'G0=Gbase; H=Gbase@Tz(-.08)@offset' if ci == 1 else
                            'G0=Gbase@offset; H=G0@Tz(-.08)'),
                        'prior_bound_to_physical_entry': ci == 2, 'virtual_prior_variants': 0})
                trials.append(trial)
    validate_trials(trials)
    return trials


def validate_trials(trials):
    if len(trials) != 48 or len({t['trial_id'] for t in trials}) != 48:
        raise ValueError('Exactly48 unique original slots required')
    for case in {t['case_id'] for t in trials}:
        rows = [t for t in trials if t['case_id'] == case]
        if len(rows) != 4 or {r['method'] for r in rows} != set('ABCD'):
            raise ValueError('Four methods per case required')
        for r in rows[1:]:
            for key in ('seed_id','placement','renderer','Gbase_world','G0_world','entry_world','perturbation'):
                if r[key] != rows[0][key]: raise ValueError('Unequal comparison condition: ' + key)


def validate_schedule(reg):
    full=reg.get('registered_trials',reg['seed_trials']);validate_trials(full)
    by_id={t['trial_id']:t for t in full}
    pending=reg['seed_trials'];carried=reg.get('carried_details',[]);deferred=reg.get('deferred_trials',[])
    pids=[t['trial_id'] for t in pending];cids=[t['trial_id'] for t in carried];dids=[t['trial_id'] for t in deferred]
    groups=[set(pids),set(cids),set(dids)]
    if any(len(ids)!=len(set(ids)) for ids in (pids,cids,dids)) or any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)) or set().union(*groups)!=set(by_id):
        raise ValueError('Original48 consumed/pending partition is damaged')
    if deferred:
        amendment=reg.get('execution_amendment',{})
        if amendment.get('kind')!='user_defer_remaining_D' or amendment.get('deferred_methods')!=['D'] or not amendment.get('user_message'):
            raise ValueError('D deferral requires explicit user scope record')
        if deferred!=[t for t in full if t['trial_id'] not in set(cids) and t['method']=='D']:
            raise ValueError('Deferred slots differ from the unstarted original D cases')
    if pending!=[t for t in full if t['trial_id'] not in set(cids)|set(dids)]:
        raise ValueError('Remaining case content or order changed')
    for d in carried:
        if not d['record_integrity_pass'] or d['method']!=by_id[d['trial_id']]['method']:
            raise ValueError('Unverified or foreign carried result')


def point_cloud(depth, intrinsics, masks, config=GEOMETRY):
    depth = np.asarray(depth)
    if depth.ndim != 2 or set(masks) != {'target','right_link27','right_link28'}:
        raise ValueError('Expected wrist linear-Z depth and three visible roles')
    if any(np.asarray(m).shape != depth.shape for m in masks.values()): raise ValueError('Mask shape mismatch')
    if np.any(sum(np.asarray(m, dtype=np.uint8) for m in masks.values()) > 1): raise ValueError('Overlapping role masks')
    k = np.asarray(intrinsics); y, x = np.indices(depth.shape)
    xyz = np.stack(((x-k[0,2])*depth/k[0,0], (y-k[1,2])*depth/k[1,1], depth), axis=-1)
    valid = np.isfinite(depth) & (depth > config['minimum_depth_m']) & (depth < config['maximum_depth_m'])
    target = xyz[valid & masks['target']]
    if len(target) < 25: return np.empty((0,3),np.float32), target
    pad = config['cloud_context_padding_m']
    region = ((xyz >= target.min(0)-pad) & (xyz <= target.max(0)+pad)).all(-1)
    include = valid & region & ~masks['right_link27'] & ~masks['right_link28']
    return xyz[include].astype(np.float32), target.astype(np.float32)


def select_candidate(rows, camera_world, g0, target_points, config=GEOMETRY):
    """Convert official GraspNet predictions using the existing ARX calibration."""
    from scipy.spatial import cKDTree
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 17: raise ValueError('Expected Nx17 GraspNet predictions')
    camera = np.asarray(camera_world); prior = np.asarray(g0)
    if camera.shape != (4,4) or prior.shape != (4,4): raise ValueError('Invalid camera/prior pose')
    axes = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    tree = cKDTree(target_points) if len(target_points) else None
    accepted = []; rejected = []
    for i, row in enumerate(rows):
        if not np.isfinite(row).all() or not 0 < row[1] <= .088:
            rejected.append({'index':i,'reason':'nonfinite_or_width'}); continue
        rotation = row[4:13].reshape(3,3)
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4) or abs(np.linalg.det(rotation)-1) > 1e-4:
            rejected.append({'index':i,'reason':'invalid_rotation'}); continue
        contact_cam = row[13:16] + row[3]*rotation[:,0]
        if tree is None or tree.query(contact_cam)[0] > config['target_surface_distance_m']:
            rejected.append({'index':i,'reason':'no_visible_target_support'}); continue
        rg = camera[:3,:3] @ rotation
        contact = camera[:3,:3] @ row[13:16] + camera[:3,3] + row[3]*rg[:,0]
        goal = np.eye(4); goal[:3,:3] = rg@axes; goal[:3,3] = contact - .040*rg[:,0]
        flipped = goal.copy(); flipped[:3,:3] = goal[:3,:3] @ np.diag([-1.,-1.,1.])
        branches = [(pose_error(g,prior), f, g) for f,g in ((False,goal),(True,flipped))]
        error, flip, goal = min(branches, key=lambda e:e[0]['rotation_rad'])
        if error['translation_m'] >= config['translation_neighborhood_m'] or error['rotation_rad'] >= np.deg2rad(config['rotation_neighborhood_degrees']):
            rejected.append({'index':i,'reason':'outside_G0_neighborhood','error':error}); continue
        accepted.append({'index':i,'score':float(row[0]),'width_m':float(row[1]),
            'goal_world':goal.tolist(),'symmetry_flip':flip,'to_G0':error})
    accepted.sort(key=lambda x:(-x['score'],x['index']))
    return {'selected':accepted[0] if accepted else None,'accepted':accepted,'rejected':rejected}
