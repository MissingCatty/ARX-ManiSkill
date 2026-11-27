"""Lazy fixed raw-candidate windows; numerical IK and collision run in batches."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from modules.curobo.indexed_planning.branch_validation import validate_rows


def same_width_indices(candidates, index, capacity=16):
    if capacity != 16 or not 0 <= index < len(candidates):
        raise ValueError('Invalid raw-candidate window')
    stop = min((index // capacity + 1) * capacity, len(candidates))
    width = float(candidates[index]['width'])
    return [i for i in range(index, stop) if float(candidates[i]['width']) == width]


def goals(candidate):
    grasp = np.asarray(candidate['pose'], dtype=float).copy()
    grasp[:3, 3] -= .043 * grasp[:3, 2]
    pre = grasp.copy(); pre[:3, 3] -= .08 * grasp[:3, 2]
    lift = grasp.copy(); lift[2, 3] += .03
    return {'grasp': grasp, 'pregrasp': pre, 'lift': lift}


def padded_starts(starts, capacity=16):
    import torch
    if starts.ndim != 2 or starts.shape[1] != 6 or not 0 < len(starts) <= capacity:
        raise ValueError('Expected separate six-joint seeds for each real target')
    return torch.cat([starts, starts[:1].expand(capacity - len(starts), -1)]).clone()


@dataclass
class PreoptimizedPath:
    value: dict


class EndpointWindow:
    def __init__(self, pool, candidates, path_capacity):
        self.pool, self.candidates, self.path_capacity = pool, candidates, path_capacity
        self.goals = [goals(c) for c in candidates]
        self.rows, self.paths, self.planned = {}, {}, set()
        self.consumed, self.current = set(), None
        self.stats = {'batches': [], 'endpoint_groups': [], 'scalar_grasp_calls': 0,
            'original_pre_or_lift_fallbacks': 0, 'serial_extra_endpoint_evaluations': 0}

    def identify(self, context, grasp):
        locks = context.checker.kinematics.kinematics_config.lock_jointstate
        width = float(locks.position.reshape(-1)[0]) * 2
        matches = [i for i, candidate in enumerate(self.candidates)
            if abs(width - float(candidate['width'])) < 2e-7
            and np.allclose(grasp, self.goals[i]['grasp'], atol=1e-12, rtol=0)]
        if len(matches) != 1:
            raise RuntimeError('Batched endpoint lost the original candidate identity')
        return matches[0]

    def grasp(self, context, state, grasp, start, validator, original):
        index = self.identify(context, grasp)
        self.current = index; self.consumed.add(index)
        if index not in self.rows:
            indices = same_width_indices(self.candidates, index)
            if len(indices) == 1:
                self.stats['scalar_grasp_calls'] += 1
                return original(context, state, grasp, start, validator)
            self.compute(context, state, indices, start, validator)
        row = self.rows[index]
        return (row['q_grasp'].clone() if row['q_grasp'] is not None else None), dict(row['evidence'])

    def compute(self, context, state, indices, start, validator):
        import torch
        engine = self.pool.prepare_endpoint_engine(context)
        engine.bind(context)
        before = len(engine.calls)
        worlds = [self.goals[i]['grasp'] for i in indices]
        with self.pool.stage_timing.stage('batched_grasp_IK_and_collision'):
            native = engine.native(worlds, start)
            missing = [j for j, q in enumerate(native) if q is None]
            pure = engine.pure([worlds[j] for j in missing], start) if missing else []
            fallback = dict(zip(missing, pure))
            for j, index in enumerate(indices):
                q = native[j]
                self.rows[index] = {'q_grasp': q, 'q_pre': None, 'q_lift': None,
                    'start': start.clone(), 'evidence': {'kind': 'native_full_world_ik'} if q is not None else
                    {'kind': 'no_local_grasp_endpoint', 'kinematic_solutions': len(fallback[j]), 'reasons': []}}
            # Evaluate every fallback branch together in the unchanged V2
            # validator, retaining original per-target branch priority.
            flat, owner = [], []
            for j in missing:
                for q in fallback[j]: flat.append(q); owner.append(indices[j])
            if flat:
                q = torch.stack(flat)
                actual = validator.geometry.frames(q.cpu().numpy())['right_eef_link']
                expected = np.asarray([self.goals[i]['grasp'] for i in owner])
                pe = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1)
                re = Rotation.from_matrix(actual[:, :3, :3] @ expected[:, :3, :3].transpose(0, 2, 1)).magnitude()
                keep = np.flatnonzero((pe <= self.pool.contact_profile.position_tolerance_m) &
                    (re <= self.pool.contact_profile.rotation_tolerance_rad))
                passed = np.zeros(len(flat), dtype=bool); reasons = ['independent_fk_error'] * len(flat)
                if len(keep):
                    valid, details = validate_rows(validator, q[keep.tolist()])
                    passed[keep] = valid
                    for k, reason in zip(keep, details): reasons[k] = reason
                for k, index in enumerate(owner):
                    row = self.rows[index]
                    if row['q_grasp'] is not None: continue
                    if passed[k]:
                        row.update(q_grasp=q[k].clone(), evidence={'kind': 'bounded_ik_with_local_table_check',
                            'independent_position_error_m': float(pe[k]),
                            'independent_rotation_error_rad': float(re[k]),
                            'native_full_world_clear': bool(context.checker.validate(q[k:k + 1, None]).all())})
                    else:
                        row['evidence']['reasons'].append(reasons[k])
        ready = [i for i in indices if self.rows[i]['q_grasp'] is not None]
        # A lone result proceeds immediately; there is no search for more
        # endpoint-valid candidates and no additional window is opened.
        if len(ready) > 1:
            starts = torch.stack([self.rows[i]['q_grasp'] for i in ready])
            starts = padded_starts(starts, engine.capacity)
            for stage, key in [('pregrasp', 'q_pre'), ('lift', 'q_lift')]:
                with self.pool.stage_timing.stage('batched_' + stage + '_IK_and_collision'):
                    q, valid = engine._solve(engine.full_solver,
                        [self.goals[i][stage] for i in ready], starts, 4, stage + '_row_seed_IK')
                    admitted = q[valid]
                    if len(admitted):
                        valid[valid.clone()] = context.checker.validate(admitted[:, None]).reshape(-1)
                    for j, index in enumerate(ready):
                        values = q[j, valid[j]]
                        if len(values):
                            best = values[(values - starts[j]).square().sum(-1).argmin()].clone()
                            self.rows[index][key] = best
        self.stats['endpoint_groups'].append({'indices': indices, 'width_m': float(self.candidates[indices[0]]['width']),
            'raw_window_stop_exclusive': min((indices[0] // 16 + 1) * 16, len(self.candidates)),
            'grasp_endpoints_passed': len(ready), 'solver_calls': list(engine.calls[before:]),
            'pre_and_lift_rows': len(ready) if len(ready) > 1 else 0})

    def ik(self, world, start, original):
        row = self.rows.get(self.current)
        if row is None:
            return original(world, start)
        for stage, key in [('pregrasp', 'q_pre'), ('lift', 'q_lift')]:
            if np.allclose(world, self.goals[self.current][stage], atol=1e-12, rtol=0):
                if row[key] is not None: return row[key].clone()
                self.stats['original_pre_or_lift_fallbacks'] += 1
                self.pool._pending_ik = [stage + '_IK']
                result = original(world, start)
                row[key] = result.clone() if result is not None else None
                return result
        return original(world, start)

    def path(self, context, start, goal):
        import torch
        index = self.current; row = self.rows.get(index)
        if (not self.path_capacity or row is None or row['q_pre'] is None or
                not torch.equal(start, row['start']) or not torch.equal(goal, row['q_pre'])):
            return None
        if index not in self.planned:
            ready = [i for i, r in sorted(self.rows.items()) if i >= index
                and i // 16 == index // 16 and self.candidates[i]['width'] == self.candidates[index]['width']
                and r['q_pre'] is not None and r['q_lift'] is not None and i not in self.planned]
            ready = ready[:self.path_capacity]
            if len(ready) < 2: return None
            jobs = [{'index': i, 'width': float(self.candidates[i]['width']),
                'start': self.rows[i]['start'], 'q_pre': self.rows[i]['q_pre']} for i in ready]
            with self.pool.stage_timing.stage('parallel_complete_paths'):
                paths, receipt = self.pool.prepare_parallel(context, self.path_capacity).plan(context, jobs)
            self.paths.update(paths); self.planned.update(ready); self.stats['batches'].append(receipt)
        return PreoptimizedPath(self.paths[index]) if index in self.paths else None

    def summary(self):
        return {**self.stats, 'raw_candidate_capacity': 16, 'path_capacity': self.path_capacity,
            'computed_endpoint_candidates': len(self.rows), 'consumed_endpoint_candidates': len(self.consumed),
            'extra_endpoint_evaluations': len(set(self.rows) - self.consumed),
            'padding_rows_are_never_candidates': True, 'searched_for_more_valid_candidates': False}
