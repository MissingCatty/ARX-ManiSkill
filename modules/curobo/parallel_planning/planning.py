"""Collect endpoint-valid candidates, then plan known q_pre in parallel."""
from pathlib import Path

import numpy as np

from arx_project.layout import write_json
from modules.curobo.candidate_io.store import candidate_pool
from modules.graspgenx.native_postprocessing.width_planning import (
    target_state, binding_proof, COLLISION_MESSAGE)
from .groups import width_groups


def parallel_pool(base):
    class ParallelPool(base):
        def __init__(self, *args, **kwargs):
            self.parallel_capacity = 0
            self.parallel_sessions = {}
            self.parallel_statistics = []
            self._parallel_bound = None
            super().__init__(*args, **kwargs)

        def prepare_parallel(self, context, capacity):
            from .session import ParallelPathSession
            if capacity not in self.parallel_sessions:
                self.parallel_sessions[capacity] = ParallelPathSession(context, capacity)
            return self.parallel_sessions[capacity]

        def _candidate_context(self, directory, state, candidate):
            width = float(candidate['width'])
            cached = self._parallel_bound
            if cached is not None and cached[0] is state and cached[1] == width:
                _, _, context, validator, measured = cached
                return context, validator, measured, binding_proof(validator, width, context.model)
            self._parallel_bound = None
            measured = target_state(state, width)
            context, validator = self.width_context(directory, measured)
            proof = binding_proof(validator, width, context.model)
            self._parallel_bound = (state, width, context, validator, measured)
            return context, validator, measured, proof

        def select_contact(self, context, state, directory, validator):
            if not self.parallel_capacity:
                return super().select_contact(context, state, directory, validator)
            directory = Path(directory)
            self._parallel_bound = None
            work_root = directory / 'parallel_work' / f'{len(self.parallel_statistics):04d}'
            candidates = candidate_pool(directory)['candidates']
            events, rejected, statistics = [], {}, {'capacity': self.parallel_capacity,
                'batches': [], 'endpoint_evaluations': 0, 'fallback_paths': 0,
                'single_width_rows': 0}
            cursor = 0
            def reject(reason):
                rejected[reason] = rejected.get(reason, 0) + 1
            try:
                while cursor < len(candidates):
                    jobs, pending = [], []
                    # Endpoint tests remain unchanged. Only the path stage is
                    # batched in this comparison; no Cartesian connector swap.
                    while cursor < len(candidates) and len(jobs) < self.parallel_capacity:
                        index, candidate = cursor, candidates[cursor]
                        cursor += 1
                        work = work_root / 'targets' / f'{index:06d}'
                        try:
                            context, validator, measured, proof = self._candidate_context(work, state, candidate)
                        except AssertionError as error:
                            if str(error) != COLLISION_MESSAGE:
                                raise
                            pending.append({'candidate_index': index, 'kind': 'candidate_width_start_in_collision',
                                'rejection': 'candidate_width_start_in_collision'})
                            continue
                        start = context.measured(measured)
                        contact = np.asarray(candidate['pose'], dtype=float)
                        grasp = contact.copy(); grasp[:3, 3] -= .043 * contact[:3, 2]
                        pre = grasp.copy(); pre[:3, 3] -= .08 * contact[:3, 2]
                        lift = grasp.copy(); lift[2, 3] += .03
                        qg, evidence = self.grasp_endpoint(context, measured, grasp, start, validator)
                        event = {'candidate_index': index, 'candidate_width_m': float(candidate['width']), **evidence}
                        pending.append(event)
                        statistics['endpoint_evaluations'] += 1
                        if qg is None:
                            event['rejection'] = 'grasp_ik'; continue
                        qp, ql = context.ik(pre, qg), context.ik(lift, qg)
                        if qp is None or ql is None:
                            event['rejection'] = 'pre_or_lift_ik'; continue
                        jobs.append({'index': index, 'width': float(candidate['width']),
                            'start': start.clone(), 'q_grasp': qg.clone(), 'q_pre': qp.clone(), 'q_lift': ql.clone(),
                            'contact': contact, 'grasp': grasp, 'pre': pre, 'lift': lift,
                            'candidate': candidate, 'evidence': evidence})
                    paths = {}
                    for group in width_groups(jobs, self.parallel_capacity):
                        if len(group) == 1:
                            statistics['single_width_rows'] += 1
                            continue
                        first = group[0]
                        context, validator, _, _ = self._candidate_context(
                            work_root / 'batches' / f"{first['index']:06d}", state, first['candidate'])
                        with self.stage_timing.stage('parallel_complete_paths'):
                            session = self.prepare_parallel(context, self.parallel_capacity)
                            batch_paths, receipt = session.plan(context, group)
                        paths.update(batch_paths)
                        statistics['batches'].append(receipt)
                    by_index = {job['index']: job for job in jobs}
                    for event in pending:
                        index = event['candidate_index']; events.append(event)
                        if 'rejection' in event:
                            reject(event.pop('rejection')); continue
                        job = by_index[index]
                        work = work_root / 'targets' / f'{index:06d}'
                        context, validator, _, proof = self._candidate_context(work / 'path', state, job['candidate'])
                        reach = paths.get(index)
                        raw_reach = None
                        if reach is None:
                            statistics['fallback_paths'] += 1
                            raw_reach = context.path(job['start'], job['q_pre'])
                            if raw_reach is None:
                                reject('path'); continue
                        approach = None
                        if bool(context.checker.validate(job['q_grasp'][None, None]).all()):
                            raw = context.path(job['q_pre'], job['q_grasp'])
                            if raw is not None:
                                approach = context.optimize('approach', raw, 5.)
                        if approach is None:
                            approach = validator.straight_approach(job['q_pre'].cpu().numpy(), job['q_grasp'].cpu().numpy())
                        if approach is None:
                            reject('local_approach'); continue
                        if reach is None:
                            reach = context.optimize('pregrasp', raw_reach, 6.)
                        sweep = []
                        if abs(job['width'] - .088) > 1e-12:
                            self._parallel_bound = None
                            checked = self.validate_narrowing(work, state, job['width']); sweep = checked['samples']
                            if not checked['passed']:
                                event['kind'] = 'candidate_width_narrowing_collision'
                                reject(event['kind']); continue
                            context, validator, _, proof = self._candidate_context(work / 'selected', state, job['candidate'])
                        proof.update(narrowing_samples=sweep, maximum_finger_step_m=.0005,
                            narrowing_target_collision_checked=True)
                        selected = {'index': index, 'score': float(job['candidate']['score']),
                            'candidate_metadata': job['candidate'].get('metadata', {}),
                            'contact_pose_world': job['contact'].tolist(), 'grasp_world': job['grasp'].tolist(),
                            'pregrasp_world': job['pre'].tolist(), 'lift_world': job['lift'].tolist(),
                            'target_contact_world': job['contact'][:3, 3].tolist(), 'table_clearance_raise_m': 0.,
                            'q_grasp': job['q_grasp'].tolist(), 'q_pre': job['q_pre'].tolist(), 'q_lift': job['q_lift'].tolist(),
                            'contact_policy_id': self.contact_profile.profile_id, 'pose_adjustment_m': 0.,
                            'grasp_ik_evidence': job['evidence'], 'open_finger_m': job['width'] / 2,
                            'width_binding': proof}
                        write_json(work / 'width_binding.json', proof)
                        return {'success': True, 'selected': selected, 'rejected': rejected,
                            'paths': {'pregrasp': reach, 'approach': approach}, 'candidate_events': events,
                            'parallel_paths': statistics}
                return {'success': False, 'reason': 'no_feasible_native_post_or_recalled_candidate',
                    'count': len(candidates), 'candidate_events': events, 'rejected': rejected,
                    'parallel_paths': statistics}
            finally:
                self._parallel_bound = None
                statistics['extra_endpoint_evaluations'] = statistics['endpoint_evaluations'] - sum(
                    e.get('kind') != 'candidate_width_start_in_collision' for e in events)
                self.parallel_statistics.append(statistics)
                write_json(directory / 'parallel_windows' / f'{len(self.parallel_statistics):04d}.json', statistics)
    return ParallelPool
