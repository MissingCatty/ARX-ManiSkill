"""Identify link and obstacle pairs for the six archived open-grasp replay cases."""
from collections import defaultdict
import ast
import xml.etree.ElementTree as ET
import hashlib
import json
from pathlib import Path

import fcl
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

# Reuse the exact CPU geometry functions without importing the CUDA checker.
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/check_arx_collision_spheres.py'
tree = ast.parse(source.read_text(), filename=str(source))
names = {'fk', 'convex', 'manager', 'object_at', 'read', 'LINKS', 'SIM'}
nodes = [
    node for node in tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in names)
    or (isinstance(node, ast.Assign) and node.targets[0].id in names)
]
from scipy.spatial import ConvexHull
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))

WORK = ROOT / 'outputs/arx_collision_spheres_20261119_01'
AUDIT = ROOT / 'outputs/arx_mpc_grasp_ik_audit_20261119_01'


def get_obstacles(scene, state):
    obstacles = {}
    for kind, specs in scene.items():
        for name, spec in specs.items():
            t = np.eye(4)
            p = spec['pose']
            t[:3, :3] = Rotation.from_quat([*p[4:], p[3]]).as_matrix()
            t[:3, 3] = p[:3]
            if kind == 'cuboid':
                solids = [fcl.Box(*spec['dims'])]
                label = name
            else:
                path = Path(spec['file_path'])
                meta = state['scene_meshes'][name]
                assert not meta['target'] and hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
                m = trimesh.load_mesh(path, process=False)
                parts = trimesh.graph.connected_components(m.edges, nodes=np.arange(len(m.vertices)))
                assert len(parts) == meta['native_convex_parts']
                solids = [convex(np.asarray(m.vertices[indices])) for indices in parts]
                label = meta['model_id']
            obstacles[name] = (label, [object_at(solid, t) for solid in solids])
    return obstacles


def local_robot_objects():
    arms = {}
    for link in LINKS:
        with np.load(WORK / (link + '.npz')) as data:
            arms[link] = [object_at(convex(data[k] / 1000.), np.eye(4)) for k in data.files if k.startswith('convex_')]
    return arms


def maximum_overlap(arms, obstacles):
    request = fcl.CollisionRequest(enable_contact=True)
    depths = []
    for a in arms:
        for b in obstacles:
            result = fcl.CollisionResult()
            if fcl.collide(a, b, request, result):
                depths.extend(c.penetration_depth for c in result.contacts)
    return float(max(depths, default=0.)*1000.)


def group_margin_hits(hits):
    groups = defaultdict(list)
    for hit in hits:
        groups[hit['candidate_1_based'], hit['link'], hit['model']].append(hit)
    return groups


def diagnose(group):
    paired = np.load(WORK / (group.name + '_paired.npz'))
    with np.load(group / 'full_pose_solutions.npz') as src:
        table_clear = src['table_clear'][src['cpu_pose_valid']]
        ground_clear = src['ground_clear'][src['cpu_pose_valid']]
    candidate = paired['candidate']
    branch = paired['branch']
    # Solve results often contain repeated joint vectors. Deduplicate exactly for collision only.
    q, inv = np.unique(paired['q'], axis=0, return_inverse=True)
    kin = yaml.safe_load((group/'model/arx.yml').read_text())['robot_cfg']['kinematics']
    poses = fk(q, kin['lock_joints'], SIM)
    state = read(Path(read(group/'audit.json')['trial']) / 'state.json')
    obstacles = get_obstacles(read(group/'model/scene.json'), state)
    world = {name: manager(objs) for name, (_, objs) in obstacles.items()}
    robot = local_robot_objects()
    arm = {name: manager(objs) for name, objs in robot.items()}
    hits = np.zeros((len(q), len(LINKS), len(world)), dtype=bool)
    for i in range(len(q)):
        for link_index, link in enumerate(LINKS):
            t = poses[link][i]
            tf = fcl.Transform(t[:3, :3], t[:3, 3])
            for obj in robot[link]:
                obj.setTransform(tf)
            arm[link].update()
            for j, obstacle in enumerate(world.values()):
                data = fcl.CollisionData()
                arm[link].collide(obstacle, data, fcl.defaultCollisionCallback)
                hits[i, link_index, j] = data.result.is_collision
    hits = hits[inv]
    any_collision = hits.any(axis=(1,2))
    assert np.array_equal(~any_collision, paired['solid'])
    all_candidates = np.unique(candidate)
    per_pair, per_object, per_link, examples = [], [], [], []
    for j, (name, (label, objs)) in enumerate(obstacles.items()):
        obstacle_hits = hits[:, :, j].any(axis=1)
        per_object.append({'obstacle': name, 'model': label,
                           'candidates_any_ik': int(len(np.unique(candidate[obstacle_hits]))),
                           'candidates_all_ik': int(sum(obstacle_hits[candidate==c].all() for c in all_candidates)),
                           'branches': int(obstacle_hits.sum())})
        for i, link in enumerate(LINKS):
            mask = hits[:, i, j]
            if not mask.any():
                continue
            per_pair.append({'link': link, 'obstacle': name, 'model': label,
                             'candidates_any_ik': int(len(np.unique(candidate[mask]))),
                             'branches': int(mask.sum())})
            if name in ('table', 'ground'):
                continue
            w = int(np.flatnonzero(mask)[0])
            t = poses[link][inv[w]]
            for arm_part in robot[link]:
                arm_part.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
            examples.append({'link': link, 'obstacle': name, 'model': label,
                             'candidate_1_based': int(candidate[w]+1), 'ik_branch_1_based': int(branch[w]+1),
                             'maximum_convex_pair_penetration_mm': maximum_overlap(robot[link], objs)})
    for i, link in enumerate(LINKS):
        mask = hits[:, i, :].any(axis=1)
        per_link.append({'link':link, 'candidates_any_ik': int(len(np.unique(candidate[mask]))),
                         'branches': int(mask.sum())})
    margin_hits = []
    miss = np.flatnonzero(paired['solid'] & ~paired['new'])
    if len(miss):
        spheres = yaml.safe_load((ROOT/'assets/arx_collision_v2/model/arx.yml').read_text())['robot_cfg']['kinematics']['collision_spheres']
        for i in np.unique(inv[miss]):
            w = miss[inv[miss] == i][0]
            for link in LINKS:
                t = poses[link][i]
                balls = [fcl.CollisionObject(fcl.Sphere(b['radius']), fcl.Transform(t[:3,:3] @ b['center'] + t[:3,3]))
                         for b in spheres[link]]
                a = manager(balls)
                for name, (label, objs) in obstacles.items():
                    data = fcl.CollisionData()
                    a.collide(world[name], data, fcl.defaultCollisionCallback)
                    if data.result.is_collision:
                        for arm_part in robot[link]:
                            arm_part.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
                        request = fcl.DistanceRequest()
                        gap = min(fcl.distance(part, ob, request, fcl.DistanceResult())
                                  for part in robot[link] for ob in objs)
                        assert gap >= 0
                        margin_hits.append({'candidate_1_based': int(candidate[w]+1), 'link':link,
                                            'obstacle':name,'model':label,
                                            'solid_clearance_mm': float(gap*1000),
                                            'sphere_overlap_mm':maximum_overlap(balls,objs)})
    table_or_ground = hits[:, :, [list(obstacles).index(n) for n in ('table','ground')]].any(axis=(1,2))
    clear_support = ~table_or_ground
    hit_distractor = hits[:, :, [j for j, n in enumerate(obstacles) if n not in ('table','ground')]].any(axis=(1,2))
    fingers = hits[:, -2:, :].any(axis=(1,2))
    return {'case':group.name, 'target':state['scene_meshes']['target']['model_id'],
            'candidates': int(len(all_candidates)), 'branches':len(candidate),
            'all_ik_collide': int(sum(any_collision[candidate==c].all() for c in all_candidates)),
            'distractor_ever': int(len(np.unique(candidate[hit_distractor]))),
            'distractor_all': int(sum(hit_distractor[candidate==c].all() for c in all_candidates)),
            'all_ik_collide_and_support_clear_branch_hits_distractor':
                int(sum(any_collision[candidate==c].all() and (clear_support & hit_distractor & (candidate==c)).any()
                        for c in all_candidates)),
            'fingers_ever': int(len(np.unique(candidate[fingers]))),
            'fingers_all': int(sum(fingers[candidate==c].all() for c in all_candidates)),
            'support_only_branches': int((table_or_ground & ~hit_distractor).sum()),
            'old_sphere_support_failed': int((~(table_clear & ground_clear)).sum()),
            'contacts_by_obstacle': per_object, 'contacts_by_link': per_link,
            'contacts_by_pair': per_pair, 'examples': examples, 'new_sphere_only_contact_ranges': {
                '|'.join(map(str,k)):{'ik_states':len(ms),'min_solid_clearance_mm':min(m['solid_clearance_mm'] for m in ms),
                    'max_solid_clearance_mm':max(m['solid_clearance_mm'] for m in ms),
                    'max_sphere_overlap_mm':max(m['sphere_overlap_mm'] for m in ms)}
                for k, ms in group_margin_hits(margin_hits).items()}}


if __name__ == '__main__':
    result = []
    for f in sorted(AUDIT.glob('I*/full_pose_solutions.npz')):
        case = diagnose(f.parent)
        result.append(case)
        print(case['case'], case['contacts_by_obstacle'], 'margin', case['new_sphere_only_contact_ranges'], flush=True)
    (WORK/'endpoint_contact_pairs.json').write_text(json.dumps({'scope':'Saved open-grasp IK endpoints against simulated obstacle convex parts and table/ground cuboids; target excluded; no self collision or dynamic execution.',
            'case_results':result}, indent=2)+'\n')
