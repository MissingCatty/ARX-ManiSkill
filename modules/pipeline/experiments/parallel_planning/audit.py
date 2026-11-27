"""Offline checks of the exact recorded command arrays and stage joins."""
import argparse
from pathlib import Path

import numpy as np

from arx_project.layout import now, read_json, write_json
from modules.arx_core.arms import get_arm
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard


def audit(root):
    reg = guard(root)
    rows = []
    for item in reg['schedule']:
        directory = root / 'plans' / item['case_id'] / item['variant']
        plan = read_json(directory / 'plan_result.json')
        if not plan['success']: continue
        state = read_json(directory / 'state.json')
        selected = plan['selected']; spec = get_arm(selected['arm'])
        limits = np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in spec.joints])
        previous = None
        for name, target in [('pregrasp', 'q_pre'), ('approach', 'q_grasp')]:
            path = plan['paths'][name]
            q, times = np.asarray(path['q']), np.asarray(path['times'])
            if (q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or times.shape != q.shape[:1]
                    or not np.isfinite(q).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
                raise RuntimeError('Corrupt recorded command array')
            if (q < limits[:, 0] - 1e-6).any() or (q > limits[:, 1] + 1e-6).any():
                raise RuntimeError('Recorded command exceeds unchanged physical joint limits')
            endpoint_error = float(np.max(np.abs(q[-1] - selected[target])))
            join_error = float(np.max(np.abs(previous - q[0]))) if previous is not None else 0.
            if max(endpoint_error, join_error) >= .005:
                raise RuntimeError('Recorded endpoint or pregrasp/approach join differs')
            derivative = q.copy(); sample_t = times.copy(); peaks = {}
            for order, bound in [(1, 1.5), (2, 5.), (3, 100.)]:
                derivative = np.diff(derivative, axis=0) / np.diff(sample_t)[:, None]
                sample_t = (sample_t[:-1] + sample_t[1:]) / 2
                peak = float(np.max(np.abs(derivative))) if derivative.size else 0.
                if peak > bound * (1 + 1e-4):
                    raise RuntimeError('Recorded command exceeds the original derivative limit')
                peaks[str(order)] = peak
            rows.append({'request': item['id'], 'stage': name, 'samples': len(q),
                'endpoint_error_rad': endpoint_error, 'stage_join_error_rad': join_error,
                'derivative_peaks': peaks, 'passed': True})
            previous = q[-1]
    result = {'at': now(), 'passed': True, 'command_paths': len(rows), 'rows': rows,
        'planning_requests_repeated': 0, 'GPU_calls': 0}
    write_json(root / 'trajectory_review.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root.resolve())
