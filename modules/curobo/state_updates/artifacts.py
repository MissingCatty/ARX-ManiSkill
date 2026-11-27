"""Ordered background receipt writing with mandatory error propagation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import yaml


@dataclass(frozen=True)
class RenderedYAML:
    text: str


class ModelTemplate:
    """Serialize invariant model data once; render only two scalar lock values."""
    def __init__(self, robot, names):
        self.names = tuple(names)
        value = deepcopy(robot)
        if set(value['kinematics']['lock_joints']) != set(self.names):
            raise ValueError('Unexpected model lock set')
        self.tokens = {name: 'ARX_MEASURED_LOCK_VALUE_' + str(i) for i, name in enumerate(self.names)}
        value['kinematics']['lock_joints'] = dict(self.tokens)
        self.text = yaml.dump({'robot_cfg': value}, Dumper=yaml.CSafeDumper, sort_keys=False)
        if any(self.text.count(token) != 1 for token in self.tokens.values()):
            raise RuntimeError('Model lock placeholder is not unique')

    def render(self, locks):
        if set(locks) != set(self.names):
            raise ValueError('Measured lock set changed')
        result = self.text
        for name, token in self.tokens.items():
            number = float(locks[name])
            if not math.isfinite(number):
                raise ValueError('Nonfinite measured lock')
            # Use the same scalar emitter: e.g. 1.0e-07 must remain a YAML
            # float, rather than the string produced by bare 1e-07 in YAML 1.1.
            scalar = yaml.dump(number, Dumper=yaml.CSafeDumper).splitlines()[0]
            result = result.replace(token, scalar)
        return RenderedYAML(result)


def write_packet(directory, packet):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in packet.items():
        text = (value.text if isinstance(value, RenderedYAML) else
                yaml.dump(value, Dumper=yaml.CSafeDumper, sort_keys=False) if name.endswith('.yml')
                else json.dumps(value, indent=2, allow_nan=False) + '\n')
        (directory / name).write_text(text)


class ReceiptWriter:
    def __init__(self, writer=write_packet):
        self.writer = writer
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='binding-receipts')
        self.futures = []
        self.closed = False

    def submit(self, directory, packet):
        if self.closed:
            raise RuntimeError('Receipt writer already closed')
        # Packets contain newly allocated scalar dictionaries and references to
        # admitted immutable model templates. Callers must never mutate them.
        self.futures.append(self.executor.submit(self.writer, Path(directory), packet))

    def finish(self):
        if self.closed:
            raise RuntimeError('Receipt writer already closed')
        self.closed = True
        begin = time.perf_counter()
        try:
            for future in self.futures:
                future.result()
        finally:
            self.executor.shutdown(wait=True)
        return {'packets': len(self.futures), 'drain_seconds': time.perf_counter() - begin,
                'all_receipts_written_before_RPC_completion': True}
