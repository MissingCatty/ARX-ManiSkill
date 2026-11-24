"""Project the optimized terminal region onto a continuous approach corridor."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

import arx_direct_grasp_20261120 as core

ORIGINAL = core.inspect_route


def project(context, route, goal):
    import torch
    q = np.asarray(route['q'], dtype=float)
    xyz, rotations = core.world_poses(context, q)
    tcp = xyz + .043*rotations[:, :, 2]
    local = (tcp-tcp[-1]) @ goal[:3, :3]
    distance = np.linalg.norm(local, axis=1)
    outside = np.flatnonzero(distance > .050)
    start = int(outside[-1])+1 if len(outside) else 1
    indices = np.arange(max(1, start), len(q)-1)
    if not len(indices):
        return route, {'applied': False, 'reason': 'No terminal interior samples'}
    weight = np.clip((.050-distance[indices])/.025, 0., 1.)
    weight = weight**3*(10.-15.*weight+6.*weight**2)
    lateral = local[indices].copy();lateral[:, 2] = 0.
    desired_tcp = tcp[indices] - weight[:, None]*(lateral @ goal[:3, :3].T)
    delta = Rotation.from_matrix(rotations[-1] @ rotations[indices].transpose(0, 2, 1)).as_rotvec()
    desired_rotation = Rotation.from_rotvec(weight[:, None]*delta).as_matrix() @ rotations[indices]
    positions = q[indices].copy()
    for iteration in range(20):
        probes = np.concatenate([positions[:, None], positions[:, None]+np.eye(6)[None]*.0005], axis=1)
        p, r = core.world_poses(context, probes.reshape(-1, 6))
        p = p.reshape(-1, 7, 3);r = r.reshape(-1, 7, 3, 3)
        t = p+.043*r[:, :, :, 2]
        error_p = desired_tcp-t[:, 0]
        error_r = Rotation.from_matrix(desired_rotation @ r[:, 0].transpose(0, 2, 1)).as_rotvec()
        if np.linalg.norm(error_p, axis=1).max() < .00012 and np.linalg.norm(error_r, axis=1).max() < .001:
            break
        jp = ((t[:, 1:]-t[:, :1])/.0005).transpose(0, 2, 1)
        jr = Rotation.from_matrix((r[:, 1:] @ r[:, :1].transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).as_rotvec()
        jr = (jr.reshape(-1, 6, 3)/.0005).transpose(0, 2, 1)
        jac = np.concatenate([jp, .2*jr], axis=1)
        error = np.concatenate([error_p, .2*error_r], axis=1)
        jt = jac.transpose(0, 2, 1)
        step = np.linalg.solve(jt @ jac + np.eye(6)*1e-6, (jt @ error[:, :, None])).squeeze(-1)
        positions += step*np.minimum(1., .04/np.maximum(np.abs(step).max(axis=1), 1e-9))[:, None]
    changed = q.copy();changed[indices] = positions
    # Same joint/scene acceptance as the optimized route, including ten samples
    # per interval. The constraints do not override collision or joint limits.
    a = (np.arange(10)/10)[None, :, None]
    dense = np.vstack([((1-a)*changed[:-1, None]+a*changed[1:, None]).reshape(-1, 6), changed[-1]])
    tensor = torch.as_tensor(dense, device='cuda', dtype=torch.float32)
    valid = all(bool(context.checker.validate(part[:, None]).all()) for part in tensor.split(64))
    if not valid:
        return route, {'applied': False, 'reason': 'Projected route failed original scene/limit checks', 'dense_samples': len(dense)}
    times = np.asarray(route['times']);dt = float(np.diff(times).min());scale = 1.;fd = changed
    maxima = {}
    for order, bound in [(1, 1.5), (2, 5.), (3, 100.)]:
        fd = np.diff(fd, axis=0)/dt
        peak = float(np.abs(fd).max(initial=0.));maxima[str(order)] = peak
        scale = max(scale, (peak/bound)**(1/order))
    result = deepcopy(route);result.update(q=changed.tolist(), times=(times*scale).tolist())
    return result, {'applied': True, 'method': 'Continuous terminal corridor projection after global optimization',
        'fixed_pregrasp_goal': False, 'start_radius_m': .050, 'full_constraint_radius_m': .025,
        'c2_blend': True, 'iterations': iteration+1, 'dense_collision_samples': len(dense),
        'max_joint_adjustment_rad': float(np.abs(changed-q).max()),
        'derivative_peaks_before_retime': maxima, 'time_scale': scale,
        'original_start_and_goal_unchanged': bool(np.array_equal(changed[[0,-1]], q[[0,-1]]))}


def inspect(context, route, goal):
    result, proof, dense, tcp = ORIGINAL(context, route, goal)
    if not proof['passed']:
        fixed, projection = project(context, route, goal)
        result, proof, dense, tcp = ORIGINAL(context, fixed, goal)
        proof['terminal_constraint_projection'] = projection
    return result, proof, dense, tcp


def install():
    core.inspect_route = inspect
