from __future__ import annotations

import threading
import time
from typing import Any

from arx_sim.backend.command_queue import QueueState


def run_simulation_loop(
    context: Any,
    stop_event: threading.Event,
    *,
    max_steps: int = 0,
    render: bool = True,
    api_server: Any | None = None,
) -> int:
    """Own every mutable Isaac operation on the process main thread."""
    steps = 0
    sim = context.config.get("simulation", {})
    realtime = bool(sim.get("realtime", False))
    wall_start = time.monotonic()
    sim_start = float(context.world.current_time)
    while (
        context.simulation_app.is_running()
        and not stop_event.is_set()
        and (max_steps <= 0 or steps < max_steps)
    ):
        if api_server is not None and not api_server.alive:
            raise RuntimeError("arx-ops uvicorn thread exited while simulation was running")
        while True:
            command = context.command_queue.get_nowait()
            if command is None:
                break
            context.controller.start(command)

        if context.command_queue.state is QueueState.PAUSED:
            context.simulation_app.update()
            wall_start = time.monotonic()
            sim_start = float(context.world.current_time)
            continue

        if realtime:
            ahead = float(context.world.current_time) - sim_start - (time.monotonic() - wall_start)
            if ahead > 0:
                if stop_event.wait(ahead):
                    break

        try:
            context.controller.update()
        except Exception as exc:
            context.controller.fail_active(f"controller update failed: {exc}")
        for camera in context.cameras.values():
            camera.sync_pose()
        # Kit advances multiple fixed physics substeps per rendered frame.
        # Configure both rates before world.reset(): changing simulation_dt
        # at runtime invalidates experimental articulation tensor views.
        context.world.step(render=render)
        steps += 1
        context.controller.publish_snapshot(context.world.current_time)
        if context.ros2_publishers is not None:
            context.ros2_publishers.publish(context)
    return steps
