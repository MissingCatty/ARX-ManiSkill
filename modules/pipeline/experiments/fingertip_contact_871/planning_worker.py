"""Exercise the production contact planner on registered table-contact fixtures."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import read_json, write_json
from arx_project.registry import load_adapter
from arx_project.service import serve


class FixtureWorker:
    def __init__(self, args):
        self.pool = load_adapter('curobo').create(args)

    def request(self, directory, fixture=False, **flags):
        directory = Path(directory)
        if not fixture:
            return self.pool.request(directory, **flags)
        from modules.curobo.fingertip_contact.narrow_phase import ConvexFingerPolicy
        state = read_json(directory/'state.json')
        request = read_json(directory/'fixture_request.json')
        context = self.pool.bind(directory/'fixture_plan', state, include_target=True)
        validator = self.pool.contact_view(context, state)
        profile = validator.profile
        strict = ConvexFingerPolicy(validator.geometry, state['scene_cuboids']['table'], 0.,
                                   validator.policy.asset)
        start = context.measured(state)
        top = sum([state['scene_cuboids']['table']['center'][2], state['scene_cuboids']['table']['dims'][2]/2])
        events = []; selected = paths = None
        fixture_poses = []
        for degrees in [0., 45., -45., 90., -90., 135., -135., 180.]:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('z', degrees, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
            pose[:3, 3] = [*request['xy_world_m'], top+validator.policy.tip-request['desired_depth_m']]
            fixture_poses.append(pose)
        # A separate, unexecuted call exercises the normal initial candidate route.
        synthetic = directory/'initial_route_check'; synthetic.mkdir()
        write_json(synthetic/'state.json', state)
        (synthetic/'candidates').mkdir()
        candidates = []
        for i, pose in enumerate(fixture_poses):
            contact = pose.copy(); contact[:3, 3] += .043*pose[:3, 2]
            candidates.append({'pose': contact.tolist(), 'score': 1.-i*.01,
                'metadata': {'origin': 'registered_contact_fixture', 'not_a_neural_grasp': True}})
        write_json(synthetic/'candidates/candidate_pool.json', {'candidates': candidates})
        route = self.pool.request(synthetic)
        write_json(directory/'initial_route_check.json', {'success': route['success'],
            'scope': 'Production candidate/IK/path route with a synthetic contact fixture', 'physical_grasps': 0})
        # Restore the fixture world including the target after the initial-route check.
        context = self.pool.bind(directory/'fixture_plan', state, include_target=True)
        validator = self.pool.contact_view(context, state)
        for index, pose in enumerate(fixture_poses):
            solutions = self.pool.unconstrained_ik(context, pose, start)
            for branch, qg in enumerate(solutions):
                q = qg.cpu().numpy()
                actual = validator.geometry.frames(q[None])['right_eef_link'][0]
                pe = float(np.linalg.norm(actual[:3, 3]-pose[:3, 3]))
                re = float(Rotation.from_matrix(actual[:3, :3] @ pose[:3, :3].T).magnitude())
                if pe > .002 or re > .03:
                    continue
                if not validator.validate(qg[None], stage='final_approach').all():
                    continue
                actual_contact = validator.policy.evaluate_one(q, stage='final_approach', other_world_clear=True, require_exact=True)
                if not actual_contact['valid']:
                    continue
                frames = validator.geometry.frames(q[None])
                minimum_z = min(float((part @ frames[name][0, :3, :3].T + frames[name][0, :3, 3])[:, 2].min())
                    for name, parts in validator.policy.parts.items() for part in parts)
                gap = minimum_z - top
                if request['desired_depth_m'] < 0 and gap < .0005:
                    continue
                if request['desired_depth_m'] > 0 and actual_contact['maximum_real_finger_depth_m'] < .0001:
                    continue
                pre = pose.copy(); pre[:3, 3] -= .08*pose[:3, 2]
                lift = pose.copy(); lift[2, 3] += .03
                qp, ql = context.ik(pre, qg), context.ik(lift, qg)
                if qp is None or ql is None:
                    events.append({'yaw_index': index, 'branch': branch, 'reason': 'strict_pre_or_lift_ik'}); continue
                reach = context.path(start, qp)
                if reach is None:
                    events.append({'yaw_index': index, 'branch': branch, 'reason': 'strict_reach_path'}); continue
                approach = validator.straight_approach(qp.cpu().numpy(), q, duration_s=5.)
                if approach is None:
                    events.append({'yaw_index': index, 'branch': branch, 'reason': 'local_path'}); continue
                selected = {'index': index, 'score': 1.-index*.01,
                    'q_grasp': q.tolist(), 'q_pre': qp.tolist(), 'q_lift': ql.tolist(),
                    'grasp_world': pose.tolist(), 'pregrasp_world': pre.tolist(), 'lift_world': lift.tolist(),
                    'contact_policy_id': profile.profile_id, 'fixture_only': True,
                    'independent_position_error_m': pe, 'independent_rotation_error_rad': re,
                    'strict_real_mesh': strict.evaluate_one(q, stage='final_approach', other_world_clear=True, require_exact=True),
                    'local_real_mesh': actual_contact,
                    'minimum_mesh_table_gap_m': gap,
                    'native_full_scene_clear': bool(context.checker.validate(qg[None, None]).all())}
                paths = {'pregrasp': context.optimize('pregrasp', reach, 6.), 'approach': approach}
                break
            if selected is not None:
                break
        result = {'success': selected is not None, 'reason': 'success' if selected else 'no_fixture_plan',
            'selected': selected, 'paths': paths, 'events': events, 'physical_grasps': 0,
            'all_scene_objects_checked': True, 'sphere_count': len(validator.geometry.names)}
        write_json(directory/'plan_result.json', result)
        write_json(directory/'fixture_contact_receipts.json', validator.receipts)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--warmup-trial', type=Path, required=True)
    parser.add_argument('--descriptor', type=Path, required=True)
    args = parser.parse_args(); args.kind = 'curobo'
    serve(args, FixtureWorker(args))
