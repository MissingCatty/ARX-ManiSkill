"""Batch read the same live ARX object properties; never author physics.

USD traversal, robot getters, comparison fields and thresholds are unchanged.
The original scalar capture is the live equivalence oracle on first use.
"""
import time
import importlib.util
import numpy as np
from .common import write,sha


def object_records(objects, paths, masses, inertias, positions, orientations, pose_matrix):
    """Preserve scalar getter shapes and aliases such as test_object exactly."""
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in the batched frozen-property view')
    index = {path:i for i,path in enumerate(paths)}
    expected = {obj.paths[0] for obj in objects.values()}
    if set(paths) != expected:
        raise ValueError('Batched frozen-property paths differ from original objects')
    values = [np.asarray(v) for v in (masses,inertias,positions,orientations)]
    n = len(paths)
    if [v.shape for v in values] != [(n,1),(n,9),(n,3),(n,4)]:
        raise ValueError('Unexpected batched frozen-property tensor shapes')
    masses,inertias,positions,orientations = values
    poses = {}
    runtime = {}
    for name,obj in objects.items():
        path = obj.paths[0];i = index[path]
        poses[name] = {'path':path,'world':pose_matrix(positions[i],orientations[i]).tolist()}
        runtime[path] = {'get_masses':masses[i:i+1].tolist(),
                         'get_inertias':inertias[i:i+1].tolist()}
    return poses,runtime


class BatchedObjectReader:
    def __init__(self, bridge):
        self.bridge = bridge
        self.paths = None
        self.view = None

    def __call__(self, objects):
        from pxr import UsdPhysics, PhysxSchema
        from isaacsim.core.experimental.prims import RigidPrim
        from arx_sim.geometry import pose_matrix
        paths = sorted({obj.paths[0] for obj in objects.values()})
        if self.paths != paths:
            # RigidPrim can apply missing APIs. Refuse such a construction;
            # all APIs must already exist on the unchanged, scalar-wrapped prims.
            stage = self.bridge.context.stage
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    raise RuntimeError('Read-only batched view requires pre-existing rigid APIs: '+path)
            self.view = RigidPrim(paths, resolve_paths=False, reset_xform_op_properties=False)
            if list(self.view.paths) != paths:
                raise RuntimeError('SDK changed explicit batched rigid-body path ordering')
            self.paths = paths
        masses = self.view.get_masses().numpy()
        inertias = self.view.get_inertias().numpy()
        positions,orientations = self.view.get_world_poses()
        return object_records(objects,self.paths,masses,inertias,positions.numpy(),orientations.numpy(),pose_matrix)


def capture_fast(bridge, capture, configuration):
    first = not hasattr(bridge,'_frozen_object_reader')
    old = None
    old_seconds = None
    if first and configuration.get('verify_scalar_equivalence',True):
        began = time.monotonic()
        oracle = configuration.get('scalar_oracle')
        if oracle:
            if sha(oracle['path']) != oracle['sha256']:
                raise RuntimeError('Frozen physical configuration mismatch: scalar oracle changed')
            spec = importlib.util.spec_from_file_location('arx_original_scalar_fingerprint',oracle['path'])
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            old = module.capture(bridge)
        else:
            old = capture(bridge)
        old_seconds = time.monotonic()-began
    began = time.monotonic()
    if first:
        bridge._frozen_object_reader = BatchedObjectReader(bridge)
    fast = capture(bridge,object_reader=bridge._frozen_object_reader,
                   progress_interval_s=configuration.get('progress_interval_s',2.))
    seconds = time.monotonic()-began
    receipt = {'version':'batched_live_properties_v1','optimized_capture_seconds':seconds,
        'scalar_capture_seconds':old_seconds,'initial_equivalence_checked':old is not None,
        'objects':len(fast['invariant']['runtime_objects']),
        'all_usd_and_robot_getters_retained':True,'physics_setters_called':False}
    if old is not None:
        # Retain both independent readings, including a mismatch before stopping.
        write(bridge.root/'batched_fingerprint_scalar.json',old)
        write(bridge.root/'batched_fingerprint_batch.json',fast)
        receipt.update(invariants_exactly_equal=old['invariant']==fast['invariant'],
            scene_state_exactly_equal=old['scene_state']==fast['scene_state'],
            scalar_invariant_sha256=old['invariant_sha256'],fast_invariant_sha256=fast['invariant_sha256'])
        write(bridge.root/'batched_fingerprint_equivalence.json',receipt)
        if not receipt['invariants_exactly_equal'] or not receipt['scene_state_exactly_equal']:
            raise RuntimeError('Frozen physical configuration mismatch: batched fingerprint differs from scalar ARX evidence')
    write(bridge.root/'fingerprint_timing.json',receipt)
    return fast,receipt
