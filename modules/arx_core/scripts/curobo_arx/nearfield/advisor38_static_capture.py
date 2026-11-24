"""Bind render synchronization to the original authored ARX camera mount."""
import copy
import numpy as np
from .advisor36_static_capture import StaticCapture as Previous
from .advisor37_motion import FrozenURDF


def authored_mount(urdf, reference_snapshot, camera_rows, parent_link):
    model = FrozenURDF(urdf)
    if not model.verify(reference_snapshot)['pass']:
        raise RuntimeError('Original URDF differs from recorded ARX link frames')
    q = dict(zip(reference_snapshot['joint_names'], reference_snapshot['joint_positions']))
    poses = model.poses(q, np.eye(4))
    return np.linalg.inv(poses['right_eef_link']) @ poses[parent_link] @ np.asarray(camera_rows).T


class StaticCapture(Previous):
    def __init__(self, bridge, model_reference):
        super().__init__(bridge)
        self.model_reference = model_reference

    def bind_mount(self, reference_snapshot):
        from pathlib import Path
        from pxr import UsdGeom
        from scripts.curobo_arx.data_factory.common import sha
        config = self.bridge.context.config['controllers']['camera']
        relative = config['roles']['right_wrist']
        parent = relative.rsplit('/', 1)[0]
        spec = config['definitions']['right_wrist']
        prim = self.bridge.context.stage.GetPrimAtPath(self.bridge.context.camera_prims['right_wrist'].paths[0])
        authored = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
        if not np.allclose(authored, spec['xform_rows'], atol=1e-12, rtol=0):
            raise RuntimeError('Live camera local transform differs from frozen configuration')
        urdf = Path(self.model_reference) / 'model/arx.urdf'
        mount = authored_mount(urdf, reference_snapshot, spec['xform_rows'], parent)
        historical = np.linalg.inv(reference_snapshot['link_world']['right_eef_link']) @ np.asarray(reference_snapshot['camera_world']['right_wrist'])
        reference = copy.deepcopy(reference_snapshot)
        reference['camera_world']['right_wrist'] = (np.asarray(reference['link_world']['right_eef_link']) @ mount).tolist()
        super().bind_mount(reference)
        self.render_binding.update(mount_reference_source='original authored camera-local matrix and frozen URDF fixed chain',
            original_URDF_sha256=sha(urdf), camera_local_matrix=authored.tolist(),
            historical_measured_eef_from_camera=historical.tolist(),
            authored_eef_from_camera=mount.tolist(),
            historical_measurement_difference_max=float(np.abs(historical - mount).max()),
            actual_mount_matrix_tolerance_unchanged=1e-5, camera_calibration_or_pose_modified=False)
