from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any


def _uvicorn_stdout_log_config() -> dict[str, Any]:
    import uvicorn

    config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    for handler_name in ("default", "access"):
        if handler_name in config.get("handlers", {}):
            config["handlers"][handler_name]["stream"] = "ext://sys.stdout"
    return config


class BackgroundApiServer:
    def __init__(self, app: Any, *, host: str, port: int) -> None:
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                log_config=_uvicorn_stdout_log_config(),
            )
        )
        self._thread = threading.Thread(target=self._server.run, name="arx-ops-uvicorn", daemon=True)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self, timeout_sec: float = 10.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout_sec
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            self.stop()
            raise RuntimeError("arx-ops uvicorn server did not become ready")

    def stop(self, timeout_sec: float = 10.0) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=2.0)
