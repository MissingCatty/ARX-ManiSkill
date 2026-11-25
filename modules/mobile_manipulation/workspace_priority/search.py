"""Target-centered, priority workspace placement; never execute robot commands."""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, write_json
from ..placement import SweepChecker, hypothetical_state, pose_matrix


def target_in_body(state, target_world):
    body = np.asarray(state['links']['body4'])
    return body[:3, :3].T@(np.asarray(target_world)-body[:3, 3])


def region_gate(state, target_world, region):
    point = target_in_body(state, target_world)
    low, high = np.asarray(region['lower_body4_m']), np.asarray(region['upper_body4_m'])
    return {'passed': bool(((point >= low-1e-8) & (point <= high+1e-8)).all()),
            'target_body4_m': point.tolist(), 'lower_body4_m': low.tolist(),
            'upper_body4_m': high.tolist()}


class HighWorkspaceSearch:
    """Shared preferred region first; then the better single-arm region.

    Native grasp planning is reached only after target placement, head view,
    destination collision and swept-path checks. Failure never falls back to
    a low-workspace minimum-travel station.
    """

    def __init__(self, model, settings, backend, workspace, policy, *, checker_factory=SweepChecker, density_data=None):
        self.model, self.settings, self.backend = model, dict(settings), backend
        self.regions, self.policy = deepcopy(workspace['regions']), dict(policy)
        self.checker_factory = checker_factory
        if self.policy.get('outside_workspace_fallback'):
            raise ValueError('Outside-workspace fallback is forbidden')
        data = np.load(ROOT/policy['density_atlas'], allow_pickle=False) if density_data is None else density_data
        cells, counts = np.asarray(data['cells']), np.asarray(data['sample_count'])
        self.step = float(data['resolution_m']) if 'resolution_m' in data else float(policy['density_resolution_m'])
        self.threshold = float(policy['minimum_sample_count'])
        thresholds = workspace.get('thresholds_per_arm_per_2cm_cell', {})
        for key, actual in [('sample_count', self.threshold),
                            ('orientation_count', policy['minimum_orientation_bins']),
                            ('down_orientation_count', policy['minimum_downward_bins'])]:
            if key in thresholds and thresholds[key] != actual:
                raise ValueError('High-workspace threshold differs from the registered recommendation: '+key)
        valid = ((counts >= self.threshold) &
                 (data['orientation_count'] >= policy['minimum_orientation_bins']) &
                 (data['down_orientation_count'] >= policy['minimum_downward_bins']) &
                 (cells[:, 0]*self.step >= policy['minimum_forward_m']-1e-9))
        self.right_counts = {tuple(c): int(n) for c, n in zip(cells[valid], counts[valid])}
        shift = np.array([0, int(round(.5/self.step)), 0])
        self.left_counts = {tuple(np.asarray(c)+shift): n for c, n in self.right_counts.items()}
        def in_region(values, scope):
            low = np.asarray(self.regions[scope]['lower_body4_m'])
            high = np.asarray(self.regions[scope]['upper_body4_m'])
            return {c for c in values if (np.asarray(c)*self.step >= low-1e-9).all()
                    and ((np.asarray(c)+1)*self.step <= high+1e-9).all()}
        self.cells = {'right': in_region(self.right_counts, 'right'),
                      'left': in_region(self.left_counts, 'left'),
                      'shared': in_region(set(self.right_counts) & set(self.left_counts), 'shared')}
        if density_data is None:
            data.close()

    def workspace_gate(self, state, target, scope):
        point = target_in_body(state, target)
        cell = tuple(np.floor(point/self.step).astype(int))
        return {'passed': cell in self.cells[scope], 'scope': scope,
                'target_body4_m': point.tolist(), 'cell': list(map(int, cell)),
                'high_density_threshold': self.threshold,
                'right_sample_count': self.right_counts.get(cell, 0),
                'left_sample_count': self.left_counts.get(cell, 0)}

    def references(self, state, scope):
        cells = np.asarray(sorted(self.cells[scope]), dtype=int).reshape(-1, 3)
        points = (cells+.5)*self.step
        if not self.policy['require_head_view'] or not len(points):
            return points
        mount = np.linalg.inv(np.asarray(state['links']['body4']))@np.asarray(state['camera_world_optical'])
        optical = (points-mount[:3, 3])@mount[:3, :3]
        pixels = optical@np.asarray(state['intrinsics']).T
        uv = pixels[:, :2]/np.maximum(optical[:, 2:3], 1e-12)
        m = self.policy['head_image_margin_pixels']
        valid = ((optical[:, 2] > .05) & (optical[:, 2] < 2.) &
                 (uv[:, 0] >= m) & (uv[:, 0] <= 640-m) & (uv[:, 1] >= m) & (uv[:, 1] <= 480-m))
        return points[valid]

    def proposals(self, state, target, scope):
        region = self.regions[scope]
        low, high = np.asarray(region['lower_body4_m']), np.asarray(region['upper_body4_m'])
        margin = self.policy['reference_margin_m']
        center = (low+high)/2
        if np.any(high-low <= 2*margin):
            raise ValueError('Workspace is smaller than its placement margin')
        base = np.asarray(state['links']['base_link'])
        lift_index = state['joint_names'].index('joint4')
        lo, hi = state['joint_limits'][lift_index]
        zero = deepcopy(state)
        zero['links'] = {'base_link': np.eye(4).tolist()}
        zero['qpos'][lift_index] = 0.
        mount = self.model.frames(zero)['body4']
        zero['qpos'][lift_index] = .1
        raised = self.model.frames(zero)['body4']
        if not np.allclose(mount[:3, :3], np.eye(3), atol=1e-7) or not np.allclose(
                raised[:3, 3]-mount[:3, 3], [0, 0, .1], atol=1e-7):
            raise ValueError('Unsupported body4/lift mounting')
        yaw0 = float(Rotation.from_matrix(base[:3, :3]).as_euler('xyz')[2])
        references = self.references(state, scope)
        candidates = []
        lifts = target[2]-base[2, 3]-mount[2, 3]-references[:, 2]
        keep = (lifts >= lo+.002) & (lifts <= hi-.002)
        references, lifts = references[keep], lifts[keep]
        centralities = np.linalg.norm((references-center)/(high-low), axis=1)
        arrays = []
        for degrees in self.policy['yaw_offsets_degrees']:
            yaw = yaw0+np.deg2rad(degrees)
            offset = Rotation.from_euler('z', yaw).apply(mount[:3, 3]+references)
            xy = np.asarray(target[:2])-offset[:, :2]
            distance = np.linalg.norm(xy-base[:2, 3], axis=1)
            movement = distance+.12*abs(np.deg2rad(degrees))+.3*np.abs(lifts-state['qpos'][lift_index])
            valid = distance <= self.settings['maximum_base_travel_m']
            indices = np.flatnonzero(valid)
            arrays.append(np.column_stack([movement[valid], centralities[valid],
                np.full(len(indices), yaw), indices, xy[valid]]))
        ranked = np.concatenate(arrays) if arrays else np.empty((0, 6))
        order = np.lexsort((ranked[:, 1], ranked[:, 0]))
        # No untested station can be reached after the destination budget is
        # exhausted. Materialize only that prefix, rather than tens of thousands
        # of repeated matrices for the finer atlas.
        for row in ranked[order[:self.policy['maximum_destination_checks_per_scope']]]:
            movement, centrality, yaw, index, x, y = row
            index = int(index)
            reference, lift = references[index], float(lifts[index])
            candidates.append({'base_world': pose_matrix(float(x), float(y), yaw, base[2, 3]).tolist(),
                'lift_m': lift, 'reference_body4_m': reference.tolist(), 'workspace_scope': scope,
                'arm': None if scope == 'shared' else scope, 'centrality': centrality,
                'cost': float(movement), 'source': 'high_workspace_target_placement'})
        if self.workspace_gate(state, target, scope)['passed']:
            candidates.append({'base_world': base.tolist(), 'lift_m': state['qpos'][lift_index],
                'workspace_scope': scope, 'arm': None if scope == 'shared' else scope,
                'source': 'already_in_high_workspace', 'centrality': -1., 'cost': 0.})
        return sorted(candidates, key=lambda p: (p['cost'], p['centrality']))

    def check_goal(self, state, target, proposal):
        prospective = hypothetical_state(self.model, state, proposal['base_world'], proposal['lift_m'])
        gate = self.workspace_gate(prospective, target, proposal['workspace_scope'])
        if not gate['passed']:
            raise ValueError('Placement generator produced an outside-workspace goal')
        if self.policy['require_head_view']:
            original_body = np.asarray(state['links']['body4'])
            camera_in_body = np.linalg.inv(original_body)@np.asarray(state['camera_world_optical'])
            camera = np.asarray(prospective['links']['body4'])@camera_in_body
            point = camera[:3, :3].T@(np.asarray(target)-camera[:3, 3])
            projected = np.asarray(state['intrinsics'])@point
            uv = projected[:2]/max(point[2], 1e-12)
            margin = self.policy['head_image_margin_pixels']
            visible = bool(.05 < point[2] < 2. and margin <= uv[0] <= 640-margin and margin <= uv[1] <= 480-margin)
            gate['head_view'] = {'passed': visible, 'uv': uv.tolist(), 'depth_m': float(point[2])}
            gate['passed'] &= visible
        return prospective, gate

    def select(self, state, pool, measured_target, directory, *, geometry_only=False, scopes=None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        target = np.asarray(measured_target)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Expected a finite target measured from perception')
        checker = self.checker_factory(self.model, state, self.settings['transit_clearance_m'])
        started = time.perf_counter()
        wanted = ['shared', 'right', 'left'] if scopes is None else list(scopes)
        generated = {scope: self.proposals(state, target, scope) for scope in wanted}
        # Arm preference is decided by its available high-region placements,
        # rather than an unconditional right-arm preference.
        singles = sorted([s for s in wanted if s != 'shared'], key=lambda s: min(
            (p['cost'] for p in generated[s]), default=float('inf')))
        scopes = ['shared', *singles] if scopes is None else list(scopes)
        events, full_calls, selected = [], 0, None
        budget = self.policy['maximum_full_planning_stations']
        summaries = {}
        for stage, scope in enumerate(scopes):
            summary = {'generated': len(generated[scope]), 'workspace_and_view_passes': 0,
                       'destination_passes': 0, 'transit_passes': 0, 'planning_calls': 0}
            summaries[scope] = summary
            # Reserve a full native planning opportunity for later arm fallbacks.
            allowance = max(1, budget-full_calls-(len(scopes)-stage-1))
            for proposal in generated[scope]:
                if (summary['workspace_and_view_passes'] >= self.policy['maximum_destination_checks_per_scope']
                        or summary['destination_passes'] >= self.policy['maximum_path_checks_per_scope']):
                    summary['search_budget_exhausted'] = True
                    break
                if not geometry_only and (full_calls >= budget or summary['planning_calls'] >= allowance):
                    break
                prospective, workspace_gate = self.check_goal(state, target, proposal)
                event = {'scope': scope, **proposal, 'workspace_gate': workspace_gate}
                events.append(event)
                if not workspace_gate['passed']:
                    continue
                summary['workspace_and_view_passes'] += 1
                destination = checker.check(proposal['base_world'], proposal['lift_m'])
                event['destination_geometry'] = destination
                if not destination['passed']:
                    continue
                summary['destination_passes'] += 1
                transit, path = checker.path(proposal, self.settings)
                event['transit_geometry'] = transit
                if not transit['passed']:
                    continue
                summary['transit_passes'] += 1
                work = directory/f'candidate_{len(events)-1:04d}'
                write_json(work/'state.json', prospective)
                write_json(work/'workspace_gate.json', workspace_gate)
                write_json(work/'transit_check.json', transit)
                if geometry_only:
                    selected = {'proposal': proposal, 'path': path, 'transit': transit,
                                'work': str(work), 'grasp_planning_verified': False}
                    break
                write_json(work/'candidates/candidate_pool.json', pool)
                full_calls += 1; summary['planning_calls'] += 1
                result = self.backend.plan(work, arm=proposal['arm'])
                event['planning_success'] = bool(result['success'])
                if not result['success']:
                    continue
                chosen_arm = result['selected']['arm']
                if chosen_arm not in (['right', 'left'] if scope == 'shared' else [scope]):
                    raise ValueError('Planner returned an arm outside the chosen high-workspace scope')
                write_json(work/'plan_result.json', result)
                selected = {'proposal': proposal, 'path': path, 'transit': transit,
                            'work': str(work), 'plan_result': result, 'grasp_planning_verified': True}
                break
            if selected is not None:
                break
        reason = 'high_workspace_geometry_ready' if selected and geometry_only else 'success' if selected else 'unable_to_plan'
        failed_stage = None if selected else 'high_workspace_or_body_transit' if not full_calls else 'native_grasp_planning'
        report = {'success': selected is not None, 'reason': reason, 'failed_stage': failed_stage,
            'priority': ['shared', 'single_arm'], 'scope_summaries': summaries,
            'selected_scope': selected['proposal']['workspace_scope'] if selected else None,
            'full_planning_calls': full_calls, 'geometry_only': geometry_only,
            'target_world_from_perception': target.tolist(), 'simulator_target_pose_used': False,
            'outside_workspace_fallback': False, 'total_seconds': time.perf_counter()-started,
            'events': events, 'physical_grasps': 0}
        write_json(directory/'search_result.json', report)
        return selected, report
