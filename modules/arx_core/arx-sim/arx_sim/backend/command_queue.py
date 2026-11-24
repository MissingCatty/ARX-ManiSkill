from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from arx_sim.errors import SimNotReadyError


class QueueState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    PAUSED = "PAUSED"
    RESETTING = "RESETTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class Command:
    kind: str
    payload: Any
    generation: int
    timeout_sec: float | None
    id: str = field(default_factory=lambda: uuid4().hex)
    submitted_at: float = field(default_factory=time.monotonic)
    future: Future = field(default_factory=Future, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def deadline(self) -> float | None:
        return None if self.timeout_sec is None else self.submitted_at + self.timeout_sec


class CommandQueue:
    def __init__(self, capacity: int = 32) -> None:
        self._queue: queue.Queue[Command] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._state = QueueState.STARTING
        self._generation = 0
        self._failure = ""
        self._inflight: dict[str, Command] = {}

    @property
    def state(self) -> QueueState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def set_ready(self) -> None:
        with self._lock:
            if self._state not in {QueueState.STARTING, QueueState.RESETTING, QueueState.PAUSED}:
                raise RuntimeError(f"cannot become READY from {self._state}")
            self._state = QueueState.READY

    def set_paused(self) -> None:
        with self._lock:
            if self._state is not QueueState.READY:
                raise RuntimeError(f"cannot pause from {self._state}")
            self._state = QueueState.PAUSED

    def begin_reset(self, *, exclude: Command | None = None) -> int:
        with self._lock:
            self._state = QueueState.RESETTING
            self._generation += 1
            generation = self._generation
        self.cancel_pending("simulation reset")
        self.cancel_inflight("simulation reset", exclude=exclude)
        return generation

    def fail(self, reason: str) -> None:
        with self._lock:
            self._state = QueueState.FAILED
            self._failure = reason
        self.cancel_pending(reason)
        self.cancel_inflight(reason)

    def stop(self, reason: str = "simulation stopped") -> None:
        with self._lock:
            self._state = QueueState.STOPPING
            self._failure = reason
        self.cancel_pending(reason)
        self.cancel_inflight(reason)
        with self._lock:
            self._state = QueueState.STOPPED

    def submit(self, kind: str, payload: Any, timeout_sec: float | None) -> Any:
        with self._lock:
            allowed = self._state is QueueState.READY or (
                self._state is QueueState.PAUSED and kind in {"simulation_resume", "simulation_reset"}
            )
            if not allowed:
                detail = self._failure or f"simulation is {self._state.value}"
                raise SimNotReadyError(detail)
            command = Command(
                kind=kind,
                payload=payload,
                generation=self._generation,
                timeout_sec=None if timeout_sec is None else max(0.01, timeout_sec),
            )
        try:
            self._queue.put_nowait(command)
        except queue.Full as exc:
            raise SimNotReadyError("Isaac command queue is full") from exc
        try:
            return command.future.result(timeout=command.timeout_sec)
        except FutureTimeout as exc:
            command.cancelled.set()
            assert command.timeout_sec is not None
            raise SimNotReadyError(
                f"command {command.id} timed out after {command.timeout_sec:.2f}s"
            ) from exc

    def get_nowait(self) -> Command | None:
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return None
            if command.cancelled.is_set():
                if not command.future.done():
                    command.future.set_exception(SimNotReadyError("command cancelled before execution"))
                continue
            if command.generation != self.generation:
                command.future.set_exception(SimNotReadyError("command belongs to a stale reset generation"))
                continue
            with self._lock:
                self._inflight[command.id] = command
            return command

    def finish(self, command: Command) -> None:
        with self._lock:
            self._inflight.pop(command.id, None)

    def cancel_inflight(self, reason: str, *, exclude: Command | None = None) -> None:
        with self._lock:
            commands = list(self._inflight.values())
        for command in commands:
            if command is exclude:
                continue
            command.cancelled.set()
            if not command.future.done():
                command.future.set_exception(SimNotReadyError(reason))

    def cancel_pending(self, reason: str) -> None:
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return
            command.cancelled.set()
            if not command.future.done():
                command.future.set_exception(SimNotReadyError(reason))
