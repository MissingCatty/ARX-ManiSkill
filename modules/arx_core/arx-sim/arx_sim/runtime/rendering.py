from __future__ import annotations

from typing import Any


def configure_camera_antialiasing(simulation_app: Any, mode: int) -> None:
    """Apply camera AA after RTX initialization and before render products exist.

    RTX Real-Time restricts the available modes to DLSS/DLAA by default. Unlock
    the other supported modes before requesting TAA/FXAA, otherwise the renderer
    silently keeps DLSS. Some Linux driver/Kit combinations return zero RGBA
    through that path even though depth and physics continue to work.
    """
    if mode not in (0, 1, 2, 3, 4):
        raise ValueError(f"unsupported camera anti-aliasing mode: {mode}")
    if mode in (0, 1, 2):
        simulation_app.set_setting("/rtx-transient/post/aa/limitedOps", False)
    simulation_app.set_setting("/rtx/post/aa/op", mode)
