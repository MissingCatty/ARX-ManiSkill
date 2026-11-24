"""Paired cuRobo endpoint checks and an independent solid-convex FCL reference."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

ROOT = Path(__file__).resolve().parents[1]
LINKS = [f'right_link{i}' for i in range(21, 29)]
SIM = ROOT/'assets/arx/robot.urdf'


def read(file):
    return json.loads(file.read_text())


def fk(q, lock, urdf):
    transforms = {'body4': np.broadcast_to(np.eye(4), (len(q), 4, 4))}
    by_child = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
    needed = set()
    for name in LINKS:
        while name != 'body4':
            j = by_child[name]
            needed.add(j)
            name = j.find('parent').get('link')
    todo = [j for j in by_child.values() if j in needed]
    while any(link not in transforms for link in LINKS):
        next_round = []
        for joint in todo:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                next_round.append(joint)
                continue
            origin = joint.find('origin')
            local = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            local[:, :3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
            local[:, :3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                name = joint.get('name')
                if name in lock:
                    local[:, :3, 3] += local[:, :3, :3] @ (axis*lock[name])
                else:
                    j = int(name.removeprefix('right_joint'))-21
                    turn = Rotation.from_rotvec(q[:, j, None]*axis).as_matrix()
                    local[:, :3, :3] = local[:, :3, :3] @ turn
            transforms[joint.find('child').get('link')] = transforms[parent] @ local
        if len(next_round) == len(todo):
            raise ValueError('disconnected ARX chain')
        todo = next_round
    return {key: transforms[key] for key in LINKS}


def convex(vertices):
    h = ConvexHull(vertices)
    faces = h.simplices.copy()
    v = vertices[faces]
    flip = (np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])*h.equations[:, :3]).sum(axis=1)<0
    faces[flip] = faces[flip, ::-1]
    encoded = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    return fcl.Convex(vertices, len(faces), encoded)


def object_at(solid, matrix):
    return fcl.CollisionObject(solid, fcl.Transform(matrix[:3, :3], matrix[:3, 3]))


def manager(objects):
    m = fcl.DynamicAABBTreeCollisionManager()
    m.registerObjects(objects)
    m.setup()
    return m


def solid_checks(poses, cooked, scene, state):
    arm_objects, link_indices, obstacles = [], [], []
    for link in LINKS:
        for part in cooked[link]:
            arm_objects.append(object_at(convex(part), np.eye(4)))
            link_indices.append(link)
    for kind, specs in scene.items():
        for name, obj in specs.items():
            t = np.eye(4)
            p = obj['pose']
            t[:3, :3] = Rotation.from_quat([*p[4:], p[3]]).as_matrix()
            t[:3, 3] = p[:3]
            if kind == 'cuboid':
                obstacles.append(object_at(fcl.Box(*obj['dims']), t))
                continue
            meta = state['scene_meshes'][name]
            path = Path(obj['file_path'])
            assert not meta['target'] and hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
            mesh = trimesh.load_mesh(path, process=False)
            # The scene exporter concatenates disjoint vertex-index ranges for the native convex parts.
            groups = trimesh.graph.connected_components(mesh.edges, nodes=np.arange(len(mesh.vertices)))
            assert len(groups) == meta['native_convex_parts']
            for indices in groups:
                obstacles.append(object_at(convex(np.asarray(mesh.vertices[indices])), t))
    arm, world = manager(arm_objects), manager(obstacles)
    clear = []
    for i in range(len(poses[LINKS[0]])):
        for obj, link in zip(arm_objects, link_indices):
            t = poses[link][i]
            obj.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
        arm.update()
        data = fcl.CollisionData()
        arm.collide(world, data, fcl.defaultCollisionCallback)
        clear.append(not data.result.is_collision)
    return np.asarray(clear)


def curobo_checks(cfg, scene, q, poses):
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp)/'arx.yml'
        model.write_text(yaml.safe_dump(cfg, sort_keys=False))
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=str(model), scene_model=scene, n_meshes=8, n_cuboids=8, collision_activation_distance=0.))
    checker.self_collision_cost = None
    x = torch.as_tensor(q, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        results = torch.cat([checker.validate(chunk[:, None]).reshape(-1) for chunk in x.split(128)]).cpu().numpy()
        loaded = checker.get_kinematics(x[:1, None]).robot_spheres.reshape(-1, 4).cpu().numpy()
    loaded = loaded[loaded[:, 3]>0]
    kin = cfg['robot_cfg']['kinematics']
    expected = []
    for link in LINKS:
        t = poses[link][0]
        for ball in kin['collision_spheres'][link]:
            expected.append([*(t[:3, :3]@ball['center']+t[:3, 3]), ball['radius']])
    assert len(expected) == len(loaded)
    d, idx = cKDTree(loaded).query(expected)
    assert max(d) < 1e-6 and cKDTree(expected).query(loaded)[0].max()<1e-6, max(d)
    return results.astype(bool), float(max(d))


def run(work, models, source):
    cooked = {}
    for link in LINKS:
        with np.load(work/f'{link}.npz') as z:
            cooked[link] = [z[k]/1000 for k in z.files if k.startswith('convex_')]
    modes = {r: yaml.safe_load((models/r/'arx.yml').read_text()) for r in ('model','model_closed')}
    counts = []
    for case in sorted(source.glob('I*/full_pose_solutions.npz')):
        group = case.parent
        state = read(Path(read(group/'audit.json')['trial'])/'state.json')
        scene = read(group/'model/scene.json')
        cfg = yaml.safe_load((group/'model/arx.yml').read_text())
        kin = cfg['robot_cfg']['kinematics']
        solutions = np.load(case)
        mask = solutions['cpu_pose_valid']
        candidate, branch = np.nonzero(mask)
        start = read(group/'model/start_state.json')['planning_start']
        q = np.vstack([start, solutions['q'][mask]])
        poses = fk(q, kin['lock_joints'], SIM)
        exact = solid_checks(poses, cooked, scene, state)
        old, e1 = curobo_checks(cfg, scene, q, poses)
        assert np.array_equal(old[1:], solutions['open_scene_clear'][mask])
        new_cfg = modes['model']
        new_kin = new_cfg['robot_cfg']['kinematics']
        new_kin['lock_joints'] = kin['lock_joints']
        assert {k:v for k,v in new_kin.items() if k!='collision_spheres'} == {k:v for k,v in kin.items() if k!='collision_spheres'}
        new, e2 = curobo_checks(new_cfg, scene, q, poses)
        if not counts:
            closed_cfg = modes['model_closed']
            closed_kin = closed_cfg['robot_cfg']['kinematics']
            closed_poses = fk(q[:1], closed_kin['lock_joints'], SIM)
            _, e3 = curobo_checks(closed_cfg, scene, q[:1], closed_poses)
        assert old[0] and new[0] and exact[0], 'recorded start control'
        old, new, exact = old[1:], new[1:], exact[1:]
        report = {'case': group.name, 'candidates_with_ik': len(np.unique(candidate)), 'ik_branches': len(q)-1,
                  'old_clear_candidates': len(np.unique(candidate[old])), 'new_clear_candidates': len(np.unique(candidate[new])),
                  'convex_clear_candidates': len(np.unique(candidate[exact])),
                  'new_clear_branches': int(new.sum()), 'convex_clear_branches': int(exact.sum()),
                  'new_clear_but_solid_collision_branches': int((new&~exact).sum()),
                  'solid_clear_but_new_collision_branches': int((exact&~new).sum()),
                  'old_and_new_passed_measured_start': True, 'max_sphere_fk_discrepancy_m': max(e1, e2)}
        np.savez_compressed(work/(group.name+'_paired.npz'), candidate=candidate, branch=branch, q=q[1:], old=old, new=new, solid=exact)
        counts.append(report)
        print(report, flush=True)
        (work/'paired_checks.json').write_text(json.dumps({'cases':counts,'reference':str(source),
            'closed_smoke_sphere_fk_max_error_m':e3,'physical_grasps':0,'complete':len(counts)==6},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('work','models','source'):
        p.add_argument(arg,type=Path)
    args=p.parse_args()
    run(args.work,args.models,args.source)
