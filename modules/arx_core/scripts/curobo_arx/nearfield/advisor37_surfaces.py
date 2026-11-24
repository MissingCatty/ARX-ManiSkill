"""Read original PhysX convex pieces without changing ARX collision settings.

The synchronous API reads/cooks the representation of the existing collider
with its authored settings. It never forces physics loading, reattaches a World,
edits a mesh, or substitutes a whole-object hull for convex decomposition.
"""
import numpy as np
from scripts.curobo_arx.data_factory.common import digest
from .advisor36_surfaces import SurfaceChecker as OriginalChecker


def cooked_convexes(stage, path):
    from pxr import UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    result = []

    def receive(status, convexes):
        if status != PhysxCollisionRepresentationResult.RESULT_VALID:
            raise RuntimeError(f'Original PhysX collider unavailable: {path}: {status}')
        for convex in convexes:
            vertices = [[float(v.x), float(v.y), float(v.z)] for v in convex.vertices]
            indices = list(map(int, convex.indices))
            faces = [indices[int(p.index_base):int(p.index_base) + int(p.num_vertices)]
                     for p in convex.polygons]
            result.append({'points_local': vertices, 'faces': faces})

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    if stage_id <= 0:
        raise RuntimeError('Existing ARX stage is not registered; do not attach/reload physics')
    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(path)),
        run_asynchronously=False, on_result=receive)
    if not result:
        raise RuntimeError('No original cooked convexes returned: ' + str(path))
    # PhysX does not promise a deterministic decomposition enumeration order.
    return sorted(result, key=digest)


def export_body(stage, path, reader=cooked_convexes, *, require_native=True):
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError('Missing ARX collider body: ' + str(path))
    cache = UsdGeom.XformCache()
    frame = np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy()
    scales = np.linalg.norm(frame[:3, :3], axis=0)
    if np.any(scales <= 0):
        raise RuntimeError('Degenerate collision body')
    frame[:3, :3] /= scales
    if (not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)
        or np.linalg.det(frame[:3, :3]) <= 0):
        raise RuntimeError('Unsupported collision body shear/reflection')
    inverse = np.linalg.inv(frame)
    shapes = []
    for child in Usd.PrimRange(prim):
        if (not child.HasAPI(UsdPhysics.CollisionAPI) or
            not UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()):
            continue
        if not child.IsA(UsdGeom.Mesh):
            raise RuntimeError('Unaccounted collider: ' + str(child.GetPath()))
        kind = UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
        matrix = inverse @ np.asarray(cache.GetLocalToWorldTransform(child)).T
        if np.linalg.det(matrix[:3, :3]) <= 0:
            raise RuntimeError('Unsupported reflected collider')
        if kind == 'boundingSphere' and not require_native:
            # Retain the old conservative robot-only bound, explicitly marked.
            vertices = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(), dtype=float)
            points = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            center = (points.min(0) + points.max(0)) / 2
            radius = float(np.linalg.norm(points - center, axis=1).max())
            shapes.append({'path': str(child.GetPath()), 'approximation': kind,
                           'sphere_center_body': center.tolist(), 'sphere_radius_m': radius,
                           'native_convex': False,
                           'scope': 'Same conservative sphere as prior vertex representation; no original mesh copied'})
            continue
        if kind not in ('convexHull', 'convexDecomposition'):
            raise RuntimeError('Native target approximation not supported: ' + str(kind))
        for index, piece in enumerate(reader(stage, child.GetPath())):
            points = np.asarray(piece['points_local'], dtype=float)
            points = points @ matrix[:3, :3].T + matrix[:3, 3]
            faces = piece['faces']
            if (len(points) < 4 or not np.isfinite(points).all() or len(faces) < 4
                or any(len(f) < 3 or min(f) < 0 or max(f) >= len(points) for f in faces)):
                raise RuntimeError('Malformed native convex topology')
            shapes.append({'path': str(child.GetPath()) + f'#cooked_{index:04d}',
                           'collider_path': str(child.GetPath()), 'piece_index': index,
                           'approximation': kind, 'points_body': points.tolist(),
                           'faces': faces, 'native_convex': True})
    return shapes


def export_geometry(bridge):
    stage = bridge.context.stage
    robot = {name: export_body(stage, body.paths[0], require_native=False)
             for name, body in bridge.links.items()}
    target = export_body(stage, bridge.objects['test_object'].paths[0])
    if not target or not all(robot.get(k) for k in ('right_link26', 'right_link27', 'right_link28')):
        raise RuntimeError('Incomplete original ARX/target cooked representation')
    data = {'schema': 'arx_static_target_cooked_convex_v2', 'robot': robot, 'target': target,
            'physics_or_collision_configuration_modified': False,
            'target_all_pieces_native': all(p['native_convex'] for p in target),
            'whole_target_hull_fallback': False,
            'scope': 'Original PhysX convex pieces and topology; conservative robot-only bounding spheres where authored; geometric nonintersection, not trajectory/contact certification'}
    return {**data, 'geometry_sha256': digest(data)}


class SurfaceChecker(OriginalChecker):
    def __init__(self, geometry):
        import fcl
        data = {k: v for k, v in geometry.items() if k != 'geometry_sha256'}
        if (digest(data) != geometry['geometry_sha256'] or
            geometry['schema'] != 'arx_static_target_cooked_convex_v2' or
            not geometry['target_all_pieces_native'] or geometry['whole_target_hull_fallback']):
            raise ValueError('Native ARX collision geometry identity failed')
        self.fcl, self.geometry, self.version = fcl, geometry, fcl.__version__

        def shape(row):
            if row['approximation'] == 'boundingSphere' and not row['native_convex']:
                if 'sphere_radius_m' in row:
                    return row['path'], fcl.Sphere(row['sphere_radius_m']), np.asarray(row['sphere_center_body'])
                vertices = np.asarray(row['points_body'], dtype=float)
                center = (vertices.min(0) + vertices.max(0)) / 2
                return row['path'], fcl.Sphere(float(np.linalg.norm(vertices - center, axis=1).max())), center
            if not row['native_convex']:
                raise ValueError('Whole-hull substitution is prohibited')
            vertices = np.asarray(row['points_body'], dtype=float)
            packed = np.asarray([x for face in row['faces'] for x in [len(face), *face]], dtype=np.int32)
            return row['path'], fcl.Convex(vertices, len(row['faces']), packed), np.zeros(3)

        self.robot = {name: [shape(row) for row in rows] for name, rows in geometry['robot'].items()}
        self.target = [shape(row) for row in geometry['target']]

    def check(self, snapshot):
        result = super().check(snapshot)
        for row in result['checks']:
            if row['minimum_distance_m'] < 0:
                row['fcl_raw_distance_sentinel'] = row['minimum_distance_m']
                row['minimum_distance_m'] = 0.
        result['minimum_distance_m'] = min(r['minimum_distance_m'] for r in result['checks'])
        result['negative_FCL_sentinel_is_not_penetration_depth'] = True
        result['target_piece_count'] = len(self.target)
        return result
