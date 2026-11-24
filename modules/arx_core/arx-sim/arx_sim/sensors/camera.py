from __future__ import annotations

from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class IsaacRgbdCamera:
    """Isaac 6.0 RTX camera with a synchronized, immutable last-frame cache."""

    def __init__(
        self,
        prim_path: str,
        resolution: tuple[int, int],
        publish_hz: float,
        active_gpu: int,
        source_prim: Any,
        intrinsics: dict[str, Any] | None = None,
    ) -> None:
        import warp as wp
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        positions, orientations = source_prim.get_world_poses()
        rtx_camera = RtxCamera(
            prim_path,
            tick_rate=publish_hz,
            positions=positions.numpy(),
            orientations=orientations.numpy(),
            reset_xform_op_properties=True,
        )
        self._source_prim = source_prim
        self._last_positions = np.array(positions.numpy(), copy=True)
        self._last_orientations = np.array(orientations.numpy(), copy=True)
        self._copy_optics(source_prim, rtx_camera)
        width, height = resolution
        self._sensor = CameraSensor(
            rtx_camera,
            resolution=(height, width),
            annotators=["rgb", "distance_to_image_plane"],
        )
        # CameraSensor defaults to cuda:0. Isaac's renderer can be explicitly
        # assigned to another GPU, so bind output buffers to that exact device.
        device = f"cuda:{active_gpu}"
        self._rgb_buffer = wp.empty((height, width, 3), dtype=wp.uint8, device=device)
        self._depth_buffer = wp.empty((height, width, 1), dtype=wp.float32, device=device)
        self._camera = rtx_camera.camera
        self.prim_path = prim_path
        self.resolution = (width, height)
        self._intrinsics = dict(intrinsics or {})
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self.frame_time: float | None = None

    @staticmethod
    def _copy_optics(source_prim: Any, target: Any) -> None:
        """Copy authored optics while keeping the source USD untouched."""
        from isaacsim.core.experimental.utils.stage import get_current_stage

        stage = get_current_stage(backend="usd")
        source = stage.GetPrimAtPath(source_prim.paths[0])
        target_prim = stage.GetPrimAtPath(target.paths[0])
        for name in (
            "clippingRange", "focalLength", "focusDistance", "fStop",
            "horizontalAperture", "horizontalApertureOffset", "projection",
            "verticalAperture", "verticalApertureOffset",
        ):
            value = source.GetAttribute(name).Get()
            if value is not None:
                target_prim.GetAttribute(name).Set(value)

    def sync_pose(self, *, force: bool = False) -> None:
        positions, orientations = self._source_prim.get_world_poses()
        positions_np = positions.numpy()
        orientations_np = orientations.numpy()
        if not force and np.allclose(positions_np, self._last_positions, atol=1e-7) and np.allclose(
            orientations_np, self._last_orientations, atol=1e-7
        ):
            return
        self._sensor.authoring_object.set_world_poses(positions=positions, orientations=orientations)
        self._last_positions = np.array(positions_np, copy=True)
        self._last_orientations = np.array(orientations_np, copy=True)

    def refresh(self, sim_time: float) -> bool:
        """Cache RGB and linear Z-depth only when both exist on this render tick."""
        rgb_value, _ = self._sensor.get_data("rgb", out=self._rgb_buffer)
        depth_value, _ = self._sensor.get_data("distance_to_image_plane", out=self._depth_buffer)
        rgb = _numpy(rgb_value)
        depth = _numpy(depth_value)
        if rgb is None or depth is None:
            return self._latest_rgb is not None
        depth = np.squeeze(depth)
        width, height = self.resolution
        if rgb.shape != (height, width, 3) or depth.shape != (height, width):
            return self._latest_rgb is not None
        self._latest_rgb = np.array(rgb, dtype=np.uint8, copy=True)
        self._latest_depth = np.array(depth, dtype=np.float32, copy=True)
        self.frame_time = float(sim_time)
        return True

    def get_rgba(self, device: str = "cpu") -> np.ndarray | None:
        del device
        return self._latest_rgb

    def get_depth(self, device: str = "cpu") -> np.ndarray | None:
        del device
        return self._latest_depth

    def get_intrinsics_matrix(self, device: str = "cpu") -> np.ndarray:
        del device
        if self._intrinsics:
            return np.array(
                [[float(self._intrinsics["fx"]), 0.0, float(self._intrinsics["cx"])],
                 [0.0, float(self._intrinsics["fy"]), float(self._intrinsics["cy"])],
                 [0.0, 0.0, 1.0]],
                dtype=float,
            )
        width, height = self.resolution
        focal = float(_numpy(self._camera.get_focal_lengths())[0])
        horizontal_values, vertical_values = self._camera.get_apertures()
        horizontal = float(_numpy(horizontal_values)[0])
        vertical = float(_numpy(vertical_values)[0])
        if focal <= 0 or horizontal <= 0 or vertical <= 0:
            raise RuntimeError("camera focal length/aperture is invalid")
        return np.array(
            [[focal * width / horizontal, 0.0, width / 2.0],
             [0.0, focal * height / vertical, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def destroy(self) -> None:
        destroy = getattr(self._sensor, "destroy", None)
        if destroy is not None:
            destroy()
