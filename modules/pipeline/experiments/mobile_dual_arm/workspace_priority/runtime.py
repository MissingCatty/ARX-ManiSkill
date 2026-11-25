"""Prepare a high-workspace grasp completely before the arm executor starts."""
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from arx_project.layout import read_json, sha256, write_json
from modules.mobile_manipulation.placement import backproject
from modules.pipeline.experiments.mobile_dual_arm.runtime import Stationing, capture


def audit_segment(directory, *, record=False):
    directory = Path(directory)
    command = read_json(directory/'command.json')
    receipt = read_json(directory/'receipt.json')
    state = read_json(directory/'settled_state.json')
    lift = state['qpos'][state['joint_names'].index('joint4')]
    rows = [__import__('json').loads(line) for line in (directory/'execution.jsonl').read_text().splitlines()]
    if (not receipt['success'] or receipt['idle_samples'] < 60 or len(rows) < 60
            or abs(lift-command['goal']['lift_m']) >= .001
            or np.max(np.abs(np.asarray(state['links']['base_link'])-command['goal']['base_world'])) > .001
            or max(np.abs(r['qvel']).max() for r in rows[-60:]) >= .01
            or read_json(directory/'contacts.json')):
        raise RuntimeError('High-workspace station motion failed independent verification')
    evidence = {'passed': True,
        'required_files': {name: sha256(directory/name) for name in
            ['command.json', 'receipt.json', 'settled_state.json', 'execution.jsonl', 'contacts.json']},
        'idle_samples': len(rows[-60:]), 'final_lift_m': lift}
    if record:
        write_json(directory/'motion_verification.json', evidence)
    return evidence


class HighWorkspaceStationing(Stationing):
    def measure(self, directory):
        response = self.services.request('sam3', directory)
        if not response['result'].get('target_found', True):
            return None, 'sam3_no_target_mask'
        state = read_json(directory/'state.json')
        try:
            points = backproject(np.asarray(Image.open(directory/'camera/mask.png')) > 0,
                np.load(directory/'camera/depth_m.npy').squeeze(),
                state['intrinsics'], state['camera_world_optical'])
        except ValueError:
            return None, 'invalid_depth'
        return np.median(points, axis=0), None

    def canonical_motion(self, station, segment):
        for name in ['command.json', 'receipt.json', 'settled_state.json', 'execution.jsonl',
                     'motion.mp4', 'head.mp4', 'contacts.json']:
            shutil.copy2(segment/name, station/name)

    def prepare(self, env, out, entry):
        from repair_contract import make_candidates
        station = out/'stationing'
        coarse = station/'coarse'
        capture(env, coarse, self.executor, entry)
        target, failure = self.measure(coarse)
        if failure:
            return {'success': False, 'reason': 'coarse_'+failure, 'motion_started': False}
        write_json(station/'measurement.json', {'source': 'SAM3_mask_and_measured_RGBD',
            'world_median': target.tolist(), 'simulator_target_pose_used': False})
        scopes = ['shared', 'right', 'left']
        current = read_json(coarse/'state.json')
        singles = sorted(scopes[1:], key=lambda side: min((p['cost'] for p in
            self.station_search.proposals(current, target, side)), default=float('inf')))
        scopes = ['shared', *singles]
        history, any_motion, last_receipt = [], False, {}
        for number, scope in enumerate(scopes):
            segment = station/'segments'/f'{number:02d}_{scope}'
            current = self.executor.snapshot(env, env.get_obs())
            selected, search = self.station_search.select(current, {'candidates': []},
                target, segment/'search', geometry_only=True, scopes=[scope])
            event = {'scope': scope, 'search_report': str(segment/'search/search_result.json'),
                     'geometry_ready': selected is not None, 'native_planning_calls': 0}
            history.append(event)
            if selected is None:
                event['failure'] = search['failed_stage']
                continue
            proposal, path = selected['proposal'], selected['path']
            write_json(segment/'command.json', {'path': path, 'goal': proposal})
            write_json(segment/'selected.json', proposal)
            moved = self.execute(env, segment, current, proposal, path)
            write_json(segment/'receipt.json', moved)
            audit_segment(segment, record=True)
            any_motion = True
            last_receipt = moved
            self.canonical_motion(station, segment)
            write_json(station/'selected.json', {**proposal,
                'transit': selected['transit'], 'grasp_planning_before_move': False})
            fresh = segment/'observation'
            capture(env, fresh, self.executor, entry)
            measured, failure = self.measure(fresh)
            if failure:
                event['failure'] = failure
                continue
            target = measured
            fresh_state = read_json(fresh/'state.json')
            high_gate = self.station_search.workspace_gate(fresh_state, target, scope)
            write_json(fresh/'high_workspace_gate.json', high_gate)
            if not high_gate['passed']:
                event['failure'] = 'fresh_observation_outside_high_workspace'
                continue
            response = self.services.request('rayst3r', fresh)
            if response['result'].get('point_count', 0) < 20:
                event['failure'] = 'rayst3r_empty_pointcloud'
                continue
            self.services.request('graspgenx', fresh, seed=self.seed)
            make_candidates(fresh, background=True)
            arm = None if scope == 'shared' else scope
            response = self.services.request('curobo_dual_arm', fresh, **({} if arm is None else {'arm': arm}))
            event['native_planning_calls'] = 1
            plan = response['result']
            if not plan['success']:
                event['failure'] = plan.get('reason', 'native_grasp_planning')
                continue
            if arm is not None and plan['selected']['arm'] != arm:
                raise RuntimeError('Native plan violated single-arm workspace binding')
            for source in fresh.iterdir():
                destination = out/source.name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            Image.open(fresh/'overview.jpg').save(out/'initial_overview.png')
            write_json(out/'initial_conditions.json', {'spawn': env.spawn,
                'base_pose': entry['base_pose'], 'initial_q': env.initial_q.tolist(),
                'grasp_start_base_world': fresh_state['links']['base_link'],
                'object_bbox': env.object_bbox})
            write_json(out/'capture.json', {'camera': 'head', 'fresh_rgbd': True,
                'mask_source': 'SAM3', 'prepared_after_stationing': True})
            names = ['state.json', 'plan_result.json', 'camera/color.png', 'camera/depth_m.npy',
                     'camera/mask.png', 'candidates/candidate_pool.json']
            receipt = {**moved, 'prepared_plan': True, 'preferred_arm': arm,
                'workspace_scope': scope, 'high_workspace_reached': True,
                'prepared_target_world': target.tolist(), 'verified_high_cell': high_gate['cell'],
                'workspace_resolution_m': self.station_search.step,
                'prepared_artifacts': {name: sha256(out/name) for name in names},
                'stationing_segments': [str(p.relative_to(out)) for p in
                    (station/'segments').glob('*') if (p/'receipt.json').exists()],
                'priority_history': history}
            write_json(station/'priority_history.json', history)
            write_json(station/'prepared_plan_handoff.json', receipt)
            return receipt
        write_json(station/'priority_history.json', history)
        return {**last_receipt, 'success': False, 'reason': 'unable_to_plan',
                'failed_stage': 'high_workspace_positioning' if not any_motion else 'grasp_preparation_in_high_workspace',
                'motion_started': any_motion, 'priority_history': history,
                'stationing_segments': [str(p.relative_to(out)) for p in
                    (station/'segments').glob('*') if (p/'receipt.json').exists()]}
