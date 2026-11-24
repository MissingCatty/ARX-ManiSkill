"""Host-only storage gates for newly registered ARX campaigns."""
import json
import shutil
from pathlib import Path


def load_policy(campaign, config):
    """Require an explicit campaign snapshot; never rewrite historical budgets."""
    path = Path(campaign) / 'storage_policy.json'
    policy = json.loads(path.read_text())
    for name in ('disk_reserve_bytes', 'min_pair_start_bytes', 'pre_trial_extra_bytes'):
        value = policy[name]
        if type(value) is not int or value < 0:
            raise ValueError('Invalid registered storage policy: ' + name)
    if policy['disk_reserve_bytes'] == 0:
        raise ValueError('Storage reserve must be positive')
    if policy['min_pair_start_bytes'] < policy['disk_reserve_bytes']:
        raise ValueError('Pair-start floor cannot be below the reserve')
    for name in ('disk_reserve_bytes', 'min_pair_start_bytes'):
        if config[name] != policy[name]:
            raise ValueError('Campaign/storage policy mismatch: ' + name)
    return policy


def require_space(path, policy, phase):
    """Read available disk bytes; this does not change robot or scene state."""
    if phase == 'pair_start':
        required = policy['min_pair_start_bytes']
    elif phase == 'before_trial':
        required = policy['disk_reserve_bytes'] + policy['pre_trial_extra_bytes']
    else:
        raise ValueError('Unknown storage phase: ' + phase)
    available = shutil.disk_usage(path).free
    if available < required:
        raise RuntimeError(
            f'Registered storage floor reached at {phase}: '
            f'{available} bytes available, {required} required; '
            f"use the authorized fallback at {policy['fallback_root']}"
        )
    return {'phase': phase, 'available_bytes': available, 'required_bytes': required}
