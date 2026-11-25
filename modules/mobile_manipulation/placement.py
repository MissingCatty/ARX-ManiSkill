"""SE(2) station proposals and full-robot swept-volume checks."""
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def pose_matrix(x, y, yaw, z):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    out[:3, 3] = [x, y, z]
    return out


def hypothetical_state(model, state, base, lift):
    out = deepcopy(state)
    out['qpos'][out['joint_names'].index('joint4')] = float(lift)
    out['links'] = {'base_link': np.asarray(base).tolist()}
    frames = model.frames(out)
    out['links'] = {n: frames[n].tolist() for n in state['links']}
    return out


def backproject(mask, depth, intrinsics, camera):
    mask, depth = np.asarray(mask, bool), np.asarray(depth)
    v, u = np.where(mask & np.isfinite(depth) & (depth > .05) & (depth < 2.))
    if len(u) < 50:
        raise ValueError('insufficient_valid_target_depth')
    k, t = np.asarray(intrinsics), np.asarray(camera)
    z = depth[v, u]
    points = np.stack([(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z], axis=-1)
    return points@t[:3, :3].T+t[:3, 3]


def station_proposals(state, target, settings):
    """The target is a measured RGB-D point, never a simulator object pose."""
    base = np.asarray(state['links']['base_link'])
    start_yaw = float(Rotation.from_matrix(base[:3, :3]).as_euler('xyz')[2])
    lower, upper = state['joint_limits'][state['joint_names'].index('joint4')]
    proposals = []
    for dx, dy, dz, degrees in product(settings['reference_x_m'], settings['reference_y_m'],
                                     settings['reference_z_m'], settings['yaw_offsets_degrees']):
        yaw = start_yaw+np.deg2rad(degrees)
        # Actual full URDF: base_link -> body4 translation, verified at runtime.
        lift = float(target[2]-base[2, 3]-.27000001072883606-dz)
        if not lower+.002 <= lift <= upper-.002:
            continue
        offset = Rotation.from_euler('z', yaw).apply([.09411799907684326+dx, .001006699982099235+dy, 0.])
        xy = np.asarray(target[:2])-offset[:2]
        distance = float(np.linalg.norm(xy-base[:2, 3]))
        if distance > settings['maximum_base_travel_m']:
            continue
        goal = pose_matrix(*xy, yaw, base[2, 3])
        cost = distance+.12*abs(yaw-start_yaw)+.3*abs(lift-state['qpos'][state['joint_names'].index('joint4')])
        proposals.append({'base_world': goal.tolist(), 'lift_m': lift,
                          'reference_body4_m': [dx, dy, dz], 'cost': float(cost)})
    return sorted(proposals, key=lambda row: row['cost'])


def interpolate(start, goal, settings):
    a, b = np.asarray(start['links']['base_link']), np.asarray(goal['base_world'])
    if abs(a[2, 3]-b[2, 3]) > 1e-5:
        raise ValueError('SE2 base cannot change ground height')
    ha = start['qpos'][start['joint_names'].index('joint4')]
    hb = goal['lift_m']
    ya, yb = [Rotation.from_matrix(t[:3, :3]).as_euler('xyz')[2] for t in [a, b]]
    angle = float(np.arctan2(np.sin(yb-ya), np.cos(yb-ya)))
    duration = max(np.linalg.norm(b[:2, 3]-a[:2, 3])/settings['base_speed_mps'],
                   abs(angle)/np.deg2rad(settings['yaw_speed_degrees_per_second']),
                   abs(hb-ha)/settings['lift_speed_mps'], 1.)
    count = max(2, int(np.ceil(duration*40))+1)
    # Smooth endpoint velocity; all safety checks inspect this same path.
    for time in np.linspace(0., duration, count):
        f = time/duration
        u = 3*f*f-2*f*f*f
        # Speeds peak at 1.5x mean; duration includes that factor below.
        yield {'time_s': float(time*1.5), 'base_world': pose_matrix(
            *(a[:2, 3]+u*(b[:2, 3]-a[:2, 3])), ya+u*angle, a[2, 3]+u*(b[2, 3]-a[2, 3])).tolist(),
            'lift_m': float(ha+u*(hb-ha))}


class SweepChecker:
    """Original convex robot colliders vs tabletop and object bounding boxes.

    Object boxes are conservative. Robot parts keep their authored convex
    decomposition. Separating axes handle oblique surfaces and planar pieces.
    """
    def __init__(self, model, state, clearance=.003):
        self.model, self.state, self.clearance = model, state, clearance
        self.parts = []
        cached = getattr(model, '_transit_convex_parts', None)
        for name in ([] if cached is not None else model.links):
            if not model.links[name].findall('collision'):
                continue
            for v, faces, _ in model.parts(name):
                v = np.unique(v, axis=0)
                rank = np.linalg.matrix_rank(v-v.mean(0))
                if rank == 3:
                    hull = ConvexHull(v)
                    normal = hull.equations[:, :3]
                    tri = v[hull.simplices]
                    edges = (tri[:, [1, 2, 0]]-tri).reshape(-1, 3)
                    v = v[hull.vertices]
                else:
                    _, _, basis = np.linalg.svd(v-v.mean(0), full_matrices=True)
                    normal = basis[rank:]
                    edges = (v[:, None]-v[None, :]).reshape(-1, 3)
                self.parts.append((name, v, normal, edges))
        if cached is None:
            model._transit_convex_parts = tuple(self.parts)
        self.parts = model._transit_convex_parts
        self.obstacles = []
        table = state['scene_cuboids']['table']
        c, h = np.asarray(table['center']), np.asarray(table['dims'])/2
        self.obstacles.append(('table', c-h, c+h))
        import trimesh
        for name, record in state.get('scene_meshes', {}).items():
            mesh = trimesh.load_mesh(record['file_path'], process=False)
            t = np.asarray(record['world'])
            v = np.asarray(mesh.vertices)@t[:3, :3].T+t[:3, 3]
            self.obstacles.append((name, v.min(0), v.max(0)))

    def check(self, base, lift):
        state = hypothetical_state(self.model, self.state, base, lift)
        frames = self.model.frames(state)
        for name, vertices, normals, edges in self.parts:
            t = frames[name]
            v = vertices@t[:3, :3].T+t[:3, 3]
            # Preserve the original wheel/ground fixture. No ground allowance
            # applies to the upper body or either arm.
            if name not in ['wheel1', 'wheel2', 'wheel3', 'base_link'] and v[:, 2].min() < -.499:
                return {'passed': False, 'link': name, 'obstacle': 'ground'}
            for obstacle, lo, hi in self.obstacles:
                lo, hi = lo-self.clearance, hi+self.clearance
                if (v.max(0) < lo).any() or (v.min(0) > hi).any():
                    continue
                r = t[:3, :3]
                e = edges@r.T
                axes = np.vstack([np.eye(3), normals@r.T,
                                  *[np.cross(e, axis) for axis in np.eye(3)]])
                axes = axes[np.linalg.norm(axes, axis=1) > 1e-8]
                axes /= np.linalg.norm(axes, axis=1)[:, None]
                axes = np.unique(np.round(axes, 7), axis=0)
                projected = v@axes.T
                center, half = (lo+hi)/2, (hi-lo)/2
                mid, radius = center@axes.T, half@np.abs(axes).T
                separated = ((projected.max(0) < mid-radius) | (projected.min(0) > mid+radius)).any()
                if not separated:
                    return {'passed': False, 'link': name, 'obstacle': obstacle}
        return {'passed': True}

    def path(self, goal, settings):
        samples = list(interpolate(self.state, goal, settings))
        for i, row in enumerate(samples):
            result = self.check(row['base_world'], row['lift_m'])
            if not result['passed']:
                return {**result, 'sample': i, 'samples': len(samples)}, samples
        return {'passed': True, 'samples': len(samples), 'clearance_m': self.clearance}, samples
