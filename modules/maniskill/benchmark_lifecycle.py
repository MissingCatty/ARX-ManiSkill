"""Shared admission gate: every registered component is warm before any slot."""
from datetime import datetime
import os
from pathlib import Path

from resident_backend import rpc, save
import json


def load_counts(health):
    if health['kind'] == 'graspcl':
        return {k: health[k] for k in ['encoder_load_count', 'bank_load_count']}
    return {'model_load_count': health['details']['model_load_count']}


def mark_ready(root, required, config, children):
    if set(required) != set(children) or set(required) != set(config['sockets']):
        raise RuntimeError('Declared component set differs from owned services')
    components = {}
    for kind in required:
        child = children[kind]
        health = rpc(Path(config['sockets'][kind]), {'command': 'health'}, timeout=30)
        if child.poll() is not None or health['status'] != 'ready' or health['pid'] != child.pid:
            raise RuntimeError('Component is not ready: ' + kind)
        counts = load_counts(health)
        values = [v for value in counts.values() for v in (value.values() if isinstance(value, dict) else [value])]
        if any(v != 1 for v in values):
            raise RuntimeError('A component was loaded more than once: ' + kind)
        if kind == 'graspcl' and not health.get('warmup_passed'):
            raise RuntimeError('GraspCL has not completed warmup')
        components[kind] = {'pid': child.pid, 'load_counts': counts, 'ready_at': health['ready_at']}
    receipt = {'ready_at': datetime.now().astimezone().isoformat(), 'owner_pid': os.getpid(),
               'required_components': required, 'components': components, 'warmup_passed': True}
    save(root / 'all_ready.json', receipt)
    return receipt


def require_ready(root, required):
    """Check immutable loading counters and liveness before claiming a scene."""
    receipt = json.loads((root / 'all_ready.json').read_text())
    if set(receipt['required_components']) != set(required) or not receipt['warmup_passed']:
        raise RuntimeError('Required components were not preloaded')
    os.kill(receipt['owner_pid'], 0)
    for kind in required:
        expected = receipt['components'][kind]
        health = json.loads((root / (kind + '_ready.json')).read_text())
        os.kill(expected['pid'], 0)
        if health['pid'] != expected['pid'] or health['status'] != 'ready' or load_counts(health) != expected['load_counts']:
            raise RuntimeError('Resident component restarted or reloaded: ' + kind)
        if (root / (kind + '_stopped.json')).exists():
            raise RuntimeError('Resident component stopped: ' + kind)
    return receipt
