"""Compare read-only live ARX evidence against one immutable research baseline."""
import hashlib
import json
import time
import copy
from pathlib import Path


FORBIDDEN = {'grasp_arm_gains', 'grasp_solver_iterations', 'obstacle',
             'grasp_feedback_camera', 'grasp_enable_physics_control'}


def invariant_digest(evidence):
    invariant = copy.deepcopy(evidence['invariant'])
    # These are renderer camera proxies updated from live physical camera links.
    # Their WORLD poses are state, not the fixed sensor extrinsic calibration.
    # Retain the physical camera mounts, configured calibration and all physics.
    for path, prim in invariant['usd'].items():
        if path.startswith('/World/COOLASSensors/'):
            prim['attributes'] = {k: v for k, v in prim['attributes'].items()
                                  if not k.startswith('xformOp')}
    return hashlib.sha256(json.dumps(invariant, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def changed_files(expected):
    changed = []
    for name, digest in expected.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    return changed


def check(client, guard_file, request_id, phase):
    guard_file = Path(guard_file)
    frozen = json.loads(guard_file.read_text())
    evidence = client.request('grasp_research_fingerprint', timeout=180)
    changed = changed_files(frozen['file_sha256'])
    invariant = invariant_digest(evidence)
    same = invariant == frozen['invariant_sha256']
    output = guard_file.parent / 'guard_checks'
    output.mkdir(exist_ok=True)
    result = {'time_unix': time.time(), 'request_id': request_id, 'phase': phase,
              'invariant_sha256': invariant,
              'raw_capture_invariant_sha256': evidence['invariant_sha256'],
              'invariant_equal': same, 'changed_files': changed, 'pass': same and not changed}
    (output / (request_id + '_' + phase + '.json')).write_text(json.dumps(result, indent=2))
    if not result['pass']:
        (output / (request_id + '_' + phase + '_mismatch.json')).write_text(json.dumps(evidence, indent=2))
        raise RuntimeError('Research configuration guard failed; no further motion: ' + str(result))
    return evidence
