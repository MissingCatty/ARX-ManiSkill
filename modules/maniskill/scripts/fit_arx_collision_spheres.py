"""Fit link-local cuRobo spheres against cooked ManiSkill/PhysX convex solids (mm internally)."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import yaml
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'assets/arx/robot.urdf'
REF = ROOT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference'
LINKS = [f'right_link{i}' for i in range(21, 29)]
EDGES = np.array([(a, b) for a in range(4) for b in range(a + 1, 4)])


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def chain(root, link):
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    path = []
    while link != 'body4':
        j = by_child[link]
        path.append(j)
        link = j.find('parent').get('link')
    return list(reversed(path))


def check_frames(sim, planner):
    rng = np.random.default_rng(43)
    for link in LINKS:
        for _ in range(10):
            q = {f'right_joint{i}': rng.uniform(0, .044) if i > 26 else rng.uniform(-2, 2) for i in range(21, 29)}
            transforms = []
            for robot in (sim, planner):
                t = np.eye(4)
                for j in chain(robot, link):
                    t = t @ origin(j.find('origin'))
                    if j.get('type') != 'fixed':
                        axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                        motion = np.eye(4)
                        value = axis * q[j.get('name')]
                        if j.get('type') == 'prismatic':
                            motion[:3, 3] = value
                        else:
                            motion[:3, :3] = Rotation.from_rotvec(value).as_matrix()
                        t = t @ motion
                transforms.append(t)
            np.testing.assert_allclose(*transforms, rtol=0, atol=2e-7)


def cooked_parts(urdf, name):
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    pieces = []
    inputs = {}
    for col in urdf.find(f"link[@name='{name}']").findall('collision'):
        mesh = col.find('geometry/mesh')
        path = SIM.parent / mesh.get('filename')
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        shape = sapien.physx.PhysxCollisionShapeConvexMesh(str(path), scale, material)
        t = origin(col.find('origin'))
        vertices = (shape.vertices * shape.scale @ t[:3, :3].T + t[:3, 3]) * 1000
        pieces.append(ConvexHull(vertices))
    return pieces, inputs


def grid(lo, hi, step):
    axes = [np.arange(a, b + step, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def samples(hull, spacing):
    points = []
    for a, b, c in hull.points[hull.simplices]:
        n = max(1, int(np.ceil(max(np.linalg.norm(a-b), np.linalg.norm(c-b), np.linalg.norm(a-c))/spacing)))
        u, v = np.triu_indices(n+1)
        points.append(a + (b-a)*(u/n)[:, None] + (c-a)*((v-u)/n)[:, None])
    return np.unique(np.vstack(points).round(6), axis=0)


def planes(hull):
    return np.unique(hull.equations.round(11), axis=0)


def depth(points, hull):
    eq = planes(hull)
    return np.concatenate([-(p @ eq[:, :3].T + eq[:, 3]).max(axis=1) for p in np.array_split(points, 1+len(points)//8192)])


def candidate_pool(parts, margin, step):
    balls, targets = [], []
    for h in parts:
        boundary = samples(h, step * .9)
        targets.append(boundary)
        nodes = grid(h.points.min(axis=0), h.points.max(axis=0), step)
        d = depth(nodes, h)
        inside = nodes[d >= 0]
        targets.append(inside[::5])
        # Interior balls touch a supporting plane before adding the allowed outer margin.
        # In thin slivers, face-to-centroid interpolation supplies centers a lattice can miss.
        inner = np.vstack([inside, h.points.mean(axis=0), (boundary[::4] + h.points.mean(axis=0)) / 2])
        balls.append(np.column_stack([inner, np.maximum(0., depth(inner, h)) + margin]))
        balls.append(np.column_stack([boundary, np.full(len(boundary), margin)]))
    if len(parts) > 1:
        # A union of whole voxels gives an additional certified inner approximation across
        # overlapping convex pieces. Distance to a non-full voxel bounds an inscribed ball.
        pitch = step/2
        vertices = np.vstack([h.points for h in parts])
        nodes = grid(vertices.min(axis=0)-2*pitch, vertices.max(axis=0)+2*pitch, pitch)
        shape = [len(np.unique(nodes[:, k])) for k in range(3)]
        full = np.zeros(len(nodes), dtype=bool)
        for h in parts:
            idx = np.where(((nodes >= h.points.min(axis=0)) & (nodes <= h.points.max(axis=0))).all(axis=1))[0]
            eq = planes(h)
            cube = np.abs(eq[:, :3]).sum(axis=1)*pitch/2
            for batch in np.array_split(idx, 1 + len(idx)//8192):
                full[batch] |= (nodes[batch] @ eq[:, :3].T + eq[:, 3] + cube <= -1e-7).all(axis=1)
        depths = distance_transform_edt(full.reshape(shape))*pitch - pitch*np.sqrt(3)/2
        lattice = np.zeros(shape, dtype=bool)
        lattice[::2, ::2, ::2] = True
        use = (lattice & (depths > 0)).ravel()
        balls.append(np.column_stack([nodes[use], depths.ravel()[use]+margin]))
    pool = np.unique(np.vstack(balls).round(6), axis=0)
    pool = pool[np.argsort(-pool[:, 3], kind='stable')]
    return pool, np.unique(np.vstack(targets).round(6), axis=0)


def cover(points, balls):
    tree = cKDTree(points)
    rows = tree.query_ball_point(balls[:, :3], balls[:, 3] - 1e-6, return_sorted=False)
    sizes = np.array([len(row) for row in rows])
    indices = np.concatenate([row for row in rows if len(row)]).astype(np.int32)
    matrix = csr_matrix((np.ones(len(indices), np.int32), indices, np.r_[0, sizes.cumsum()]), shape=(len(balls), len(points)))
    transpose = matrix.T.tocsr()
    gains = sizes.copy()
    uncovered = np.ones(len(points), dtype=bool)
    selected = []
    while uncovered.any():
        i = gains.argmax()
        assert gains[i] > 0, 'candidate library leaves training points uncovered'
        selected.append(i)
        row = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        changed = row[uncovered[row]]
        uncovered[changed] = False
        gains -= np.bincount(transpose[changed].indices, minlength=len(gains))
    return balls[selected]


def certify(parts, balls, first_witness=False):
    # A ball containing all four vertices contains their entire solid tetrahedron.
    # Bisect ambiguous tetrahedra; return any actually uncovered vertices as fitting constraints.
    tetra = np.vstack([np.concatenate([h.points[h.simplices], np.broadcast_to(h.points.mean(axis=0),
                       (len(h.simplices), 1, 3))], axis=1) for h in parts])
    violations = []
    total = 0
    for _ in range(180):
        children = []
        for i in range(0, len(tetra), 128):
            t = tetra[i:i+128]
            total += len(t)
            inside = (cdist(t.reshape(-1, 3), balls[:, :3], 'sqeuclidean') <= (balls[:, 3] - 1e-6)**2).reshape(len(t), 4, -1)
            missing = ~inside.any(axis=2)
            if missing.any():
                if first_witness:
                    return t[missing][:1], total
                violations.append(t[missing])
            t = t[~(inside.all(axis=1).any(axis=1) | missing.any(axis=1))]
            if not len(t):
                continue
            lengths = ((t[:, EDGES[:, 0]]-t[:, EDGES[:, 1]])**2).sum(axis=-1)
            large = lengths.max(axis=1) > .001**2
            # Any vertex plus a 1 micron radius increment covers its entire remaining tetrahedron.
            t, lengths = t[large], lengths[large]
            if not len(t):
                continue
            a, b = EDGES[lengths.argmax(axis=1)].T
            k = np.arange(len(t))
            mid = (t[k, a]+t[k, b])/2
            left = t.copy()
            left[k, a] = mid
            t[k, b] = mid
            children.extend([left, t])
        if not children:
            break
        tetra = np.concatenate(children)
    else:
        raise RuntimeError('unresolved tetrahedra in solid coverage test')
    return np.unique(np.vstack(violations).round(6), axis=0) if violations else np.empty((0, 3)), total


def fit_link(index, work):
    link = LINKS[index]
    parts, inputs = cooked_parts(ET.parse(SIM).getroot(), link)
    margin, grid_step, guard = ((6.5, 5.0, .8) if index < 5 else
                                (3.45, 3.2, .45) if index == 5 else (2.6, 2.5, .4))
    balls, points = candidate_pool(parts, margin, grid_step)
    for iteration in range(12):
        chosen = cover(points, balls)
        chosen[:, 3] += guard
        extra, ntet = certify(parts, chosen)
        if not len(extra):
            break
        points = np.unique(np.vstack([points, extra]), axis=0)
        balls = np.vstack([balls, np.column_stack([extra, np.full(len(extra), margin)])])
    else:
        raise RuntimeError('solid coverage did not converge')
    # Remove any sphere whose deletion still covers the complete convex solids.
    # Exclusive training points give inexpensive, exact witnesses against deletion.
    covered = cdist(points, chosen[:, :3]) <= chosen[:, 3]
    counts = covered.sum(axis=1)
    keep = np.ones(len(chosen), dtype=bool)
    for j in np.argsort(chosen[:, 3]):
        mask = covered[:, j]
        if np.any(counts[mask] == 1):
            continue
        keep[j] = False
        probe, _ = certify(parts, chosen[keep], first_witness=True)
        if len(probe):
            keep[j] = True
        else:
            counts -= mask
    chosen = chosen[keep]
    missing, ntet = certify(parts, chosen)
    assert not len(missing)
    chosen[:, 3] += .001001
    np.savez(work/f'{link}.npz', balls=chosen, **{f'convex_{i}': p.points for i, p in enumerate(parts)})
    info = {'balls': len(chosen), 'convex_parts': len(parts), 'solid_coverage_verified': True,
            'tetrahedra_checked': ntet, 'outward_bound_mm': margin+guard+.00101, 'source_sha256': inputs}
    (work/f'{link}.json').write_text(json.dumps(info, indent=2))
    print(link, len(chosen), ntet, flush=True)


def assemble(work, output):
    sim = ET.parse(SIM).getroot()
    result = {}
    all_spheres = {}
    for link in LINKS:
        result[link] = json.loads((work/f'{link}.json').read_text())
        balls = np.load(work/f'{link}.npz')['balls']/1000
        all_spheres[link] = [dict(center=row[:3].tolist(), radius=float(row[3])) for row in balls]
    for role in ('model', 'model_closed'):
        check_frames(sim, ET.parse(REF/role/'arx.urdf').getroot())
        source = REF/role/'arx.yml'
        cfg = yaml.safe_load(source.read_text())
        kin = cfg['robot_cfg']['kinematics']
        assert kin['collision_link_names'] == LINKS and kin['collision_sphere_buffer'] == 0
        for l in LINKS:
            result[l][role+'_old_count'] = len(kin['collision_spheres'][l])
        kin['collision_spheres'] = all_spheres
        kin['urdf_path'] = str(REF/role/'arx.urdf')
        target = output/role/'arx.yml'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (work/'geometry.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--link', type=int, choices=range(21, 29))
    parser.add_argument('--assemble', type=Path)
    args = parser.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    if args.link is not None:
        fit_link(args.link-21, args.work)
    if args.assemble:
        assemble(args.work, args.assemble)
