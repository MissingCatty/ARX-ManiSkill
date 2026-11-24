from __future__ import annotations

import signal
import threading
from typing import Any


class JointLifecycle:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._previous: dict[int, Any] = {}

    def install_signal_handlers(self) -> None:
        def request_stop(_signum: int, _frame: Any) -> None:
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()

    def shutdown(self, context: Any, api_server: Any | None) -> None:
        self.stop_event.set()
        if context is not None:
            context.publish(lifecycle="STOPPING")
            try:
                if context.controller is not None:
                    context.controller.safe_stop()
            except Exception:
                pass
            try:
                context.command_queue.stop("simulation main loop stopped")
            except Exception:
                pass
            if api_server is not None:
                try:
                    api_server.stop()
                except Exception:
                    pass
            if context.ros2_publishers is not None:
                try:
                    context.ros2_publishers.close()
                except Exception:
                    pass
            for camera in context.cameras.values():
                try:
                    camera.destroy()
                except Exception:
                    pass
            try:
                context.publish(lifecycle="STOPPED")
            except Exception:
                pass
        elif api_server is not None:
            try:
                api_server.stop()
            except Exception:
                pass
