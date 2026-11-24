"""Read every saved witness for frame-scope, finiteness and joint-limit checks."""
from 公共 import EXPERIMENTS
from pathlib import Path
import numpy as np
from 公共 import ROOT, read, save, sha


def audit(directory):
    complete = read(directory/'完成.json')
    assert complete['passed']
    for name, digest in complete['file_sha256'].items():
        assert sha(directory/name) == digest
    m = read(directory/'地图元数据.json')
    bank = np.memmap(directory/m['witness_file'], dtype=m['dtype'], mode='r',
                     shape=(m['witnesses'], len(m['fields'])))
    bounds = np.asarray(m['joint_limits'])
    front = rear = invalid = 0
    minimum_margin = np.inf
    quaternion_error = 0.
    for start in range(0, len(bank), 131072):
        rows = np.asarray(bank[start:start+131072])
        assert np.isfinite(rows).all()
        q = rows[:, :6].astype(float)
        invalid += int(np.count_nonzero(((q < bounds[:, 0]) | (q > bounds[:, 1])).any(axis=1)))
        minimum_margin = min(minimum_margin, float(np.minimum(q-bounds[:, 0], bounds[:, 1]-q).min()))
        front += int((rows[:, 6] >= 0).sum())
        rear += int((rows[:, 6] < 0).sum())
        quaternion_error = max(quaternion_error, float(np.abs(np.linalg.norm(rows[:, 9:13].astype(float), axis=1)-1).max()))
    assert invalid == 0 and quaternion_error < 1e-6
    if m.get('front_halfspace'):
        assert rear == 0
    return {'directory': str(directory), 'witnesses': len(bank), 'front': front, 'rear': rear,
            'joint_limit_violations': invalid, 'minimum_joint_margin_rad': minimum_margin,
            'maximum_quaternion_norm_error': quaternion_error, 'file_integrity_passed': True}


def main():
    target = EXPERIMENTS / '地图流程比较_20261121_01/地图全库复核.json'
    assert not target.exists()
    result = {name: audit(ROOT/'产物'/folder) for name, folder in
              [('old', '首版_01'), ('new', '前方精细_20261121_01')]}
    result.update(passed=True, code_sha256=sha(Path(__file__)), physical_attempts=0)
    save(target, result)
    print(result)


if __name__ == '__main__':
    main()
