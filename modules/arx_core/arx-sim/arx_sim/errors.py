class SimBackendError(RuntimeError):
    """An Isaac-backed operation cannot currently satisfy the real API contract."""


class SimNotReadyError(SimBackendError):
    """The simulator is not ready to accept or complete a command."""
