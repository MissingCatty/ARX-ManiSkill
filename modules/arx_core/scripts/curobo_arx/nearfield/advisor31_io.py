"""Deduplicate identical successful status writes; preserve all evidence writes."""
import time
from scripts.curobo_arx.data_factory.common import digest


class StatusWriter:
    def __init__(self, durable_write):
        self.durable_write = durable_write
        self.last_pass = {}
        self.metrics = {'status_writes': 0, 'identical_pass_status_writes_avoided': 0,
                        'durable_write_seconds': 0., 'safety_checks_skipped': 0}

    def __call__(self, path, value):
        from pathlib import Path
        path = Path(path)
        status = path.name == 'capture_preflight.json'
        key = str(path.resolve())
        signature = digest(value) if status and value.get('pass') is True else None
        if signature is not None and self.last_pass.get(key) == signature:
            self.metrics['identical_pass_status_writes_avoided'] += 1
            return
        started = time.perf_counter()
        self.durable_write(path, value)
        self.metrics['durable_write_seconds'] += time.perf_counter()-started
        if status:
            self.metrics['status_writes'] += 1
            if signature is None:
                self.last_pass.pop(key, None)
            else:
                self.last_pass[key] = signature
