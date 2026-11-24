"""ARX model-only seven-case screening; never imports a simulator dispatcher."""
import argparse
import copy
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.teacher import Rejection
from .advisor38_kinematics import CPUArm
from .advisor38_teacher import Teacher
from .advisor37_geometry import clearance_bound
from .advisor37_motion import command_times, prepose
from training.nearfield_arx.contract import pose_error


def require_seven(certificate, candidate):
    if not certificate['pass'] or len(certificate.get('cases', [])) != 7:
        raise ValueError('All seven complete model cases required')
    if {c['case'] for c in certificate['cases']} != {'center', 'X_minus', 'X_plus', 'Y_minus', 'Y_plus', 'Z_minus', 'Z_plus'}:
        raise ValueError('Repeated or missing boundary case')
    if any(not c.get('pass') for c in certificate['cases']):
        raise ValueError('An incomplete model case cannot release a physical candidate')
    for case in certificate['cases']:
        for phase in ('insertion', 'lift'):
            path = case.get(phase)
            if not path or path.get('invalid_samples') != 0 or path.get('dense_samples', 0) < 2:
                raise ValueError('Missing complete path evidence: ' + phase)
    if certificate['candidate_sha256'] != digest(candidate):
        raise ValueError('Model certificate belongs to a different candidate')
    if certificate.get('physical_commands') != 0:
        raise ValueError('Model screening cannot dispatch motion')


def model_check(teacher, request, screen, root, deadline):
    """Validate CPU solutions using the same resident CUDA collision model.

    Closed-hand dimensions/contact/slip are unknown before a real closure.
    The lift screen is explicitly an open-hand, rigid-target-attachment model;
    original post-closure checks and physical seven-case testing still apply.
    """
    snapshot, scene = request['snapshot'], read(request['scene'])
    original = np.array(snapshot['link_world']['right_eef_link'])
    target = np.array(snapshot['objects']['test_object']['world'])
    names = snapshot['right_arm']
    measured = dict(zip(snapshot['joint_names'], snapshot['joint_positions']))
    cases = []
    for case in screen['cases']:
        if time.time() >= deadline:
            raise Rejection('model_candidate_time_budget_exhausted')
        folder = root / case['case']
        folder.mkdir()
        row = {'case': case['case'], 'pass': False}
        cases.append(row)
        goal = np.array(case['target_world'])
        # Check a real planned route to the prepose as well, with the frozen
        # target-aware reach model. Commands remain data; no mailbox exists.
        reach = {**copy.deepcopy(request), 'action': 'plan', 'phase': 'reach', 'label': 'prefilter_reach',
                 'goal_world': prepose(goal).tolist(), 'grasp_world': goal.tolist(),
                 'output': str(folder), 'command_id': 'model_only_no_dispatch',
                 'ik_seed': case['pregrasp_q'], 'minimum_duration': 6.}
        reply = teacher.handle(reach)
        write(folder / 'reach_reply.json', reply)
        if reply.get('rejected'):
            row['rejected'] = reply['rejected']
            break
        for phase, key in (('insertion', 'insertion_q'), ('lift', 'lift_q')):
            teacher.initialize(scene, snapshot, include_target=phase == 'insertion')
            planner, checker, updater, models = teacher.context['open']
            updater.apply(models, {n: float(measured[n]) for n in updater.names})
            q = np.array(case[key])
            # A valid path must connect the chosen reach IK branch to the
            # insertion branch; interpolate this stationary Cartesian handoff.
            if phase == 'insertion':
                reach_end = np.array(reply['command']['position'][-1])
                bridge = np.linspace(reach_end, q[0], max(2, int(np.ceil(np.max(abs(reach_end - q[0])) / .003)) + 1))
                if len(bridge) > 4096:
                    row['rejected'] = 'prepose_branch_connection_budget_exhausted'
                    break
                q = np.concatenate([bridge[:-1], q])
            tensor = teacher.torch.tensor(q, device='cuda', dtype=teacher.torch.float32)
            alpha = teacher.torch.arange(10, device='cuda') / 10
            dense = teacher.torch.cat([(tensor[:-1, None] * (1 - alpha[None, :, None]) +
                                       tensor[1:, None] * alpha[None, :, None]).reshape(-1, len(names)), tensor[-1:]])
            invalid = sum(int((~checker.validate(batch[:, None])).sum()) for batch in dense.split(64))
            if invalid:
                row['rejected'] = phase + '_original_model_collision'
                row['invalid_samples'] = invalid
                break
            try:
                times, derivatives = command_times(q, 8. if phase == 'insertion' else 12.)
            except ValueError as error:
                raise Rejection('model_path_timing_rejected', {'reason': str(error)}) from error
            visual = {**reach, 'label': phase, 'phase': 'local' if phase == 'insertion' else 'lift'}
            visual['snapshot'] = copy.deepcopy(snapshot)
            # Held-target prediction is bound to this candidate, not to the
            # unrelated historical start of the archived episode.
            if phase == 'lift':
                visual['snapshot']['link_world']['right_eef_link'] = goal.tolist()
            teacher.visual_path(visual, planner, dense, folder)
            maximum_fit = 0.
            for i, values in enumerate(q):
                if time.time() >= deadline:
                    raise Rejection('model_candidate_time_budget_exhausted')
                actual = teacher.fk(planner, teacher.torch.tensor(values, device='cuda', dtype=teacher.torch.float32), snapshot)
                if phase == 'insertion':
                    vector = actual[:3, 3] - goal[:3, 3]
                    lateral = np.linalg.norm(vector - goal[:3, 2] * (vector @ goal[:3, 2]))
                else:
                    lateral = np.linalg.norm(actual[:2, 3] - goal[:2, 3])
                maximum_fit = max(maximum_fit, float(lateral))
                if lateral > .00075 or pose_error(actual, goal)['rotation_rad'] > np.deg2rad(.35):
                    raise Rejection(phase + '_nonaxial_model_path')
            row[phase] = {'dense_samples': len(dense), 'invalid_samples': 0,
                          'maximum_lateral_error_m': maximum_fit, 'duration_s': float(times[-1]),
                          'derivative_maxima': derivatives, 'gripper_state': 'archived_actual_open'}
        if row.get('rejected'):
            break
        row['pass'] = True
    return cases


def run(top):
    top = Path(top).resolve()
    spec = read(top / 'prefilter_registration.json')
    out = top / 'prefilter'
    if out.exists():
        raise FileExistsError('No automatic model-screen restart')
    out.mkdir()
    write(out / 'STARTED.json', {'wall': time.time(), 'registration_sha256': sha(top / 'prefilter_registration.json')})
    verify_files({**spec['files'], **spec['source_files']})
    base = read(spec['base_registration'])
    base.update(stage='advisor38_model_prefilter', advisor38_seed_protocol={
        'remove_nominal_closed_preflight_gate': True, 'training_authorized': False},
        visual_geometry_policy={**base['visual_geometry_policy'], 'minimum_surface_distance_m': .010})
    teacher, results, geometry_cache = None, [], {}
    try:
        jobs = [(obj, row) for rank in range(60) for obj, item in spec['objects'].items()
                for row in read(item['candidate_pool'])['candidates'][rank:rank + 1]]
        for obj, candidate in jobs:
            if (top / 'STOP').exists() or time.time() >= spec['deadline_unix']:
                raise RuntimeError('Original campaign stop/deadline')
            control = read(top / 'prefilter_control.json') if (top / 'prefilter_control.json').exists() else {}
            if obj in control.get('released_or_budget_exhausted', []):
                continue
            item = spec['objects'][obj]
            request = read(item['request'])
            folder = out / obj / str(candidate['index'])
            folder.mkdir(parents=True)
            started = time.time()
            result = {'object_id': obj, 'candidate_sha256': digest(candidate), 'candidate_index': candidate['index'],
                      'pass': False, 'physical_commands': 0, 'physical_success_claim': False,
                      'registered_snapshot_only': True, 'cases': [], 'started_unix': started}
            write(folder / 'STARTED.json', result)
            write(out / 'state.json', {'status': 'cpu_kinematics', 'object_id': obj, 'candidate_index': candidate['index'],
                                      'completed_candidates': len(results), 'wall': time.time()})
            try:
                world = np.array(item['source_object_world']) @ np.array(candidate['Gcanonical_object'])
                from .advisor36_contract import boundary_poses
                from scripts.curobo_arx.data_factory.visual_geometry import TargetGeometry
                if obj not in geometry_cache:
                    geometry_cache[obj] = TargetGeometry.from_asset(request['visual_context']['asset'])
                visual = request['visual_context']
                camera_checks = []
                for boundary in boundary_poses(world, 'coupled_xyz'):
                    goal = np.array(boundary['goal_world'])
                    for label, frame in (('pregrasp', prepose(goal)), ('grasp', goal)):
                        check = geometry_cache[obj].inspect(request['snapshot']['objects']['test_object']['world'],
                            frame @ np.array(visual['eef_from_camera']), visual['intrinsics'], visual['shape'],
                            visual['clipping_range_m'], base['visual_geometry_policy'])
                        camera_checks.append({'case': boundary['case'], 'label': label, **check})
                write(folder / 'camera_endpoint_screen.json', {'checks': camera_checks})
                if any(not c['pass'] or clearance_bound(c) < .010 for c in camera_checks):
                    raise Rejection('uniform_camera_endpoint_clearance_or_visibility')
                arm = CPUArm(Path(base['model_reference']) / 'model/arx.urdf', request['snapshot'])
                cpu = arm.screen(world, spec['cpu_candidate_seconds'])
                write(folder / 'cpu_screen.json', cpu)
                if not cpu['pass']:
                    result['rejected'] = cpu['rejected']
                else:
                    if teacher is None:
                        teacher = Teacher(base, out / 'resident_model')
                    write(out / 'state.json', {'status': 'original_model_path_checks', 'object_id': obj,
                          'candidate_index': candidate['index'], 'completed_candidates': len(results), 'wall': time.time()})
                    cases = model_check(teacher, request, cpu, folder,
                                        min(spec['deadline_unix'], time.time() + spec['model_candidate_seconds']))
                    result.update(cases=cases, pass_=len(cases) == 7 and all(c['pass'] for c in cases))
                    result['pass'] = result.pop('pass_')
                    if not result['pass']:
                        result['rejected'] = cases[-1].get('rejected', 'incomplete_model_cases')
            except Rejection as error:
                result['rejected'] = str(error)
            result.update(finished_unix=time.time(), seconds=time.time() - started,
                          evidence_files={str(p): sha(p) for p in folder.rglob('*.json') if p.name != 'STARTED.json'},
                          original_model_and_margins_unchanged=True,
                          scope='CPU seven-case IK plus original CUDA model path/visual checks; unknown measured closed grip and physics still need real tests')
            if result['pass']:
                require_seven(result, candidate)
            write(folder / 'result.json', result)
            results.append({k: v for k, v in result.items() if k not in ('cases', 'evidence_files')})
            write(out / 'results.json', {'records': results, 'qualified': sum(r['pass'] for r in results)})
            print('MODEL_SCREEN', obj, candidate['index'], result['pass'], result.get('rejected'), flush=True)
        write(out / 'state.json', {'status': 'complete', 'completed_candidates': len(results),
                                 'qualified': sum(r['pass'] for r in results), 'physical_commands': 0})
    except BaseException:
        write(out / 'FAULT.json', {'error': traceback.format_exc(), 'wall': time.time(), 'physical_commands': 0})
        write(out / 'state.json', {'status': 'fault', 'physical_commands': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    run(parser.parse_args().root)
