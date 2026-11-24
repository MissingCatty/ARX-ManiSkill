"""G-local SE(3) sampling and lossless environment coordinate/role mapping."""
import ast
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from .common import digest


def rigid(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all() or not np.allclose(value[3], [0, 0, 0, 1]):
        raise ValueError('Expected a finite rigid 4x4 transform')
    if not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(value[:3, :3]) < .9999:
        raise ValueError('Invalid rotation')
    return value


def translated(value, origin, inverse=False):
    value = rigid(value).copy()
    value[:3, 3] += (-1 if inverse else 1) * np.asarray(origin)
    return value


def handoff(goal, distance_mm, sample, seed):
    """Sample zero first, then a deterministic ball perturbation in G axes."""
    g = rigid(goal)
    rng = np.random.default_rng(np.random.SeedSequence([seed, distance_mm, sample]))
    d = distance_mm / 1000.
    delta = np.zeros(3)
    omega = np.zeros(3)
    if sample:
        angle = rng.uniform(0, 2*np.pi)
        radius = .020 * np.sqrt(rng.random())
        delta = np.array([radius*np.cos(angle), radius*np.sin(angle), rng.uniform(-.25*d, .25*d)])
        axis = rng.normal(size=3)
        omega = axis/np.linalg.norm(axis) * rng.uniform(0, np.deg2rad(15))
    offset = np.eye(4)
    offset[:3, 3] = delta - [0, 0, d]
    offset[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    return {'world': (g @ offset).tolist(), 'distance_mm': distance_mm, 'sample': sample, 'seed': seed,
            'local_translation_m': offset[:3, 3].tolist(), 'local_rotation_vector_rad': omega.tolist(),
            'formula': 'H = G @ Translate(dx,dy,-d+dz) @ Exp(omega)'}


def pose_equal(a, b, position=.001, angle=np.deg2rad(1), symmetric=False):
    a, b = rigid(a), rigid(b)
    if np.linalg.norm(a[:3, 3]-b[:3, 3]) > position:
        return False
    rotations = [b[:3, :3]]
    if symmetric:
        # Parallel fingers: exchanging +Y/-Y with the same approach axis.
        rotations.append(b[:3, :3] @ np.diag([-1, -1, 1]))
    return min(Rotation.from_matrix(a[:3, :3] @ r.T).magnitude() for r in rotations) <= angle


def grasp_comparison_frame(record):
    """Compare pad-aware grasps at contact center; legacy goals retain TCP semantics."""
    frame = rigid(record['grasp_world']).copy()
    if 'pad_center_eef_m' in record:
        pad = np.asarray(record['pad_center_eef_m'], dtype=float)
        if pad.shape != (3,) or not np.isfinite(pad).all():
            raise ValueError('Invalid grasp contact offset')
        frame[:3, 3] += frame[:3, :3] @ pad
    return frame


def grasp_equal(a, b):
    if ('pad_center_eef_m' in a) != ('pad_center_eef_m' in b):
        return False
    return pose_equal(grasp_comparison_frame(a), grasp_comparison_frame(b), .005, np.deg2rad(10), True)


def equivalent_grasp(record, flip):
    """Exchange parallel fingers about the contact, respecting the frozen TCP offset."""
    goal = rigid(record['grasp_world']).copy()
    if flip:
        center = grasp_comparison_frame(record)[:3, 3]
        goal[:3, :3] = goal[:3, :3] @ np.diag([-1., -1., 1.])
        if 'pad_center_eef_m' in record:
            goal[:3, 3] = center - goal[:3, :3] @ record['pad_center_eef_m']
    return goal


def balanced_group_order(groups):
    """Avoid losing whole surface regions to a lexicographically truncated pass."""
    remaining=set(groups)
    region_counts,direction_counts,roll_counts=Counter(),Counter(),Counter()
    metadata={key:ast.literal_eval(key) for key in remaining}
    order=[]
    while remaining:
        def priority(key):
            region,direction,roll=metadata[key]
            return (region_counts[region],direction_counts[direction],roll_counts[roll],
                    -groups[key][0]['score'],key)
        key=min(remaining,key=priority)
        remaining.remove(key);order.append(key)
        region,direction,roll=metadata[key]
        region_counts[region]+=1;direction_counts[direction]+=1;roll_counts[roll]+=1
    return order


def grasp_pool(rows, camera, object_world, size, maximum=64):
    """Keep spatial/approach/roll diversity; no old downward/height heuristics."""
    cam, obj = rigid(camera), rigid(object_world)
    axes = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    groups, rejected = {}, []
    for index, row in enumerate(np.asarray(rows)):
        if len(row) < 16 or not np.isfinite(row).all():
            rejected.append({'index': index, 'reason': 'malformed_network_row'})
            continue
        rg = cam[:3, :3] @ row[4:13].reshape(3, 3)
        contact = cam[:3, :3] @ row[13:16] + cam[:3, 3] + row[3]*rg[:, 0]
        g = np.eye(4)
        g[:3, :3] = rg @ axes
        g[:3, 3] = contact - .040*rg[:, 0]  # Existing, frozen ARX pad/tool calibration.
        try:
            rigid(g)
        except ValueError:
            rejected.append({'index': index, 'reason': 'nonrigid_network_pose'})
            continue
        local = obj[:3, :3].T @ (contact-obj[:3, 3])
        if (np.abs(local) > np.asarray(size)/2 + .015).any():
            rejected.append({'index': index, 'reason': 'outside_actual_target_bounds'})
            continue
        scaled = local / np.maximum(np.asarray(size)/2, .001)
        region = (int(np.argmax(np.abs(scaled))), int(np.sign(scaled[np.argmax(np.abs(scaled))])))
        approach = g[:3, 2]
        direction = (int(np.argmax(np.abs(approach))), int(np.sign(approach[np.argmax(np.abs(approach))])))
        # Roll modulo pi respects the parallel-jaw symmetry.
        roll = int(np.floor((np.arctan2(g[1, 1], g[0, 1]) % np.pi)/(np.pi/4)))
        group = str((region, direction, roll))
        candidate = {'raw_index': index, 'score': float(row[0]), 'width_m': float(row[1]),
                     'network_row': row.tolist(), 'grasp_world': g.tolist(), 'group': group,
                     'target_contact_world': contact.tolist()}
        candidate['goal_id'] = digest({'g': np.round(g, 7).tolist()})[:20]
        groups.setdefault(group, []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda c: (-c['score'], c['raw_index']))
    group_order=balanced_group_order(groups)
    selected = []
    while groups and len(selected) < maximum:
        for key in group_order:
            if key not in groups:continue
            candidate = groups[key].pop(0)
            if any(pose_equal(candidate['grasp_world'], old['grasp_world'], .005, np.deg2rad(10), True) for old in selected):
                rejected.append({'index': candidate['raw_index'], 'reason': 'parallel_jaw_equivalent_duplicate'})
            else:
                selected.append(candidate)
            if not groups[key]:
                del groups[key]
            if len(selected) >= maximum:
                break
    return {'candidates': selected, 'rejected': rejected, 'raw_candidates': len(rows),
            'group_selection':'Round-robin over groups ordered by deficient surface region, approach direction and roll, then score',
            'reserve_limit': maximum, 'remaining_unselected': sum(map(len, groups.values()))}


class Roles:
    """Canonical evaluator view with the original path always retained separately."""
    def __init__(self, root, origin):
        self.root, self.origin = root, np.asarray(origin, dtype=float)

    def owns(self, path):
        return path == self.root or path.startswith(self.root+'/')

    def path(self, path):
        if not self.owns(path):
            raise ValueError('Cross-environment or unowned contact/path: ' + path)
        return '/World' + path[len(self.root):]

    def event(self, event):
        return {**event, 'a': self.path(event['a']), 'b': self.path(event['b']),
                'raw_a': event['a'], 'raw_b': event['b'], 'environment_root': self.root}
