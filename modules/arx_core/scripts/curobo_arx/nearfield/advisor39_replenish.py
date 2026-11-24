"""Read-only adaptive candidate generation and original-model screening."""
import argparse
import copy
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.data_factory.teacher import Rejection
from scripts.curobo_arx.data_factory.visual_geometry import TargetGeometry
from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
from .advisor36_contract import boundary_poses
from .advisor37_geometry import clearance_bound
from .advisor37_motion import prepose
from .advisor38_kinematics import CPUArm, ScreenRejected
from .advisor38_prefilter import model_check, require_seven
from .advisor38_teacher import Teacher
from .advisor39_geometry import dense_candidates, triggered


def basic_screen(candidate, item, request, base, geometry):
    start = time.monotonic()
    goal = np.array(item['source_object_world']) @ np.array(candidate['Gcanonical_object'])
    visual = request['visual_context']
    cameras = []
    for case in boundary_poses(goal, 'coupled_xyz'):
        for label, frame in (('grasp', np.array(case['goal_world'])),
                             ('pregrasp', prepose(case['goal_world']))):
            check = geometry.inspect(request['snapshot']['objects']['test_object']['world'],
                frame @ np.array(visual['eef_from_camera']), visual['intrinsics'], visual['shape'],
                visual['clipping_range_m'], base['visual_geometry_policy'])
            cameras.append({'case': case['case'], 'label': label, **check})
    if any(not c['pass'] or clearance_bound(c) < .010 for c in cameras):
        return {'pass': False, 'rejected': 'uniform_camera_endpoint_clearance_or_visibility',
                'camera_checks': cameras, 'seconds': time.monotonic() - start}
    arm = CPUArm(Path(base['model_reference']) / 'model/arx.urdf', request['snapshot'])
    deadline, solutions = time.monotonic() + 30., []
    top = goal.copy(); top[2, 3] += .120
    try:
        q = arm.start
        for frame in (prepose(goal), goal, top):
            q = arm.solve(frame, q, deadline, multistart=True)
            solutions.append(q.tolist())
        return {'pass': True, 'camera_checks': cameras, 'center_endpoint_q': solutions,
                'seconds': time.monotonic() - start, 'full_seven_path_screen_pending': True}
    except ScreenRejected as error:
        return {'pass': False, 'rejected': str(error), 'camera_checks': cameras,
                'seconds': time.monotonic() - start, 'global_unreachability_proven': False}


def run(top):
    top = Path(top).resolve()
    reg = read(top / 'supplement_registration.json')
    out = top / 'supplement'
    if out.exists():
        raise FileExistsError('No model-screen restart or started-candidate replay')
    out.mkdir()
    write(out / 'STARTED.json', {'wall': time.time(), 'registration_sha256': sha(top / 'supplement_registration.json')})
    oldtop = Path(reg['predecessor'])
    spec = read(oldtop / 'prefilter_registration.json')
    base = read(spec['base_registration'])
    base.update(stage='advisor38_model_prefilter', advisor38_seed_protocol={
        'remove_nominal_closed_preflight_gate': True, 'training_authorized': False},
        visual_geometry_policy={**base['visual_geometry_policy'], 'minimum_surface_distance_m': .010})
    teacher, jobs, records, activated = None, {}, [], {}
    inventory = {r['object_id']: r for r in read(top / 'source_inventory.json')}
    try:
        verify_files({**reg['files'], **reg['source_files']})
        while time.time() < reg['deadline_unix']:
            if (top / 'STOP').exists() or (oldtop / 'STOP').exists():
                raise RuntimeError('Campaign STOP')
            initial = read(oldtop / 'prefilter/results.json')['records']
            prior_state = read(oldtop / 'prefilter/state.json')
            if prior_state['status'] == 'fault' or (oldtop / 'prefilter/FAULT.json').exists():
                raise RuntimeError('Initial prefilter infrastructure fault')
            control = read(top / 'supplement_control.json') if (top / 'supplement_control.json').exists() else {}
            release = control.get('released_objects', [])
            counts = control.get('registered_candidates', {})
            used = set(control.get('used_candidate_sha256', []))
            for obj, item in spec['objects'].items():
                if obj in jobs:
                    continue
                pool = read(item['candidate_pool'])['candidates']
                rows = [r for r in initial if r['object_id'] == obj]
                if not triggered(rows, pool, counts.get(obj, item['prior_registered_candidates']), obj in release):
                    continue
                mesh, mesh_proof = composed_mesh(inventory[obj]['asset']['usd_path'])
                prior = [np.array(c['Gcanonical_object']) for c in pool]
                carry = read(oldtop / 'carry_forward_registry.json')
                prior += [np.array(r['geometry']['selected']['Gcanonical_object']) for r in carry['candidate_rounds'] if r['object_id'] == obj]
                request = read(item['request'])
                jobs[obj] = {'item': item, 'mesh': mesh, 'mesh_proof': mesh_proof,
                             'geometry': TargetGeometry.from_asset(request['visual_context']['asset']),
                             'request': request, 'original': read(item['geometry']), 'previous': prior,
                             'level': 0, 'generator': None, 'terminal': False}
                activated[obj] = {'initial_screened': len(rows), 'initial_qualified': 0,
                                  'wall': time.time(), 'initial_result_files': {
                     str(oldtop / 'prefilter' / obj / str(r['candidate_index']) / 'result.json'):
                     sha(oldtop / 'prefilter' / obj / str(r['candidate_index']) / 'result.json') for r in rows}}
                write(out / obj / 'activation.json', activated[obj])
            advanced = False
            for obj, job in jobs.items():
                if job['terminal']:
                    continue
                if obj in release or counts.get(obj, 5) >= 15:
                    job['terminal'] = True
                    continue
                waiting = [r for r in records if r['object_id'] == obj and r['pass'] and r['candidate_sha256'] not in used]
                if len(waiting) >= 2:
                    continue
                if job['generator'] is None:
                    job['level'] += 1
                    if job['level'] > reg['maximum_density_levels']:
                        job['terminal'] = True
                        continue
                    wave = out / obj / f'level_{job["level"]:02d}'
                    wave.mkdir(parents=True)
                    wave_reg = {'object_id': obj, 'density_level': job['level'],
                         'maximum_geometry_candidates': reg['geometry_candidates_per_level'],
                         'deadline_unix': reg['deadline_unix'], 'activation': str(out / obj / 'activation.json'),
                         'activation_sha256': sha(out / obj / 'activation.json'),
                         'source_registration_sha256': sha(top / 'supplement_registration.json'),
                         'mesh_proof': job['mesh_proof'], 'physical_commands': 0}
                    write(wave / 'registration.json', wave_reg)
                    job['wave'] = wave
                    job['generator'] = dense_candidates(inventory[obj], job['original'], job['mesh'],
                        job['previous'], job['level'], reg['geometry_candidates_per_level'], reg['deadline_unix'])
                try:
                    candidate = next(job['generator'])
                except StopIteration:
                    job['generator'] = None
                    continue
                advanced = True
                job['previous'].append(np.array(candidate['Gcanonical_object']))
                folder = job['wave'] / str(candidate['index']); folder.mkdir()
                candidate_file = folder / 'candidate.json'
                write(candidate_file, candidate)
                started = time.time()
                result = {'object_id': obj, 'candidate_index': candidate['index'],
                          'candidate_sha256': digest(candidate), 'candidate_file': str(candidate_file),
                          'candidate_file_sha256': sha(candidate_file), 'result_file': str(folder / 'result.json'),
                          'cases': [], 'pass': False, 'physical_commands': 0, 'started_unix': started,
                          'source_registration': str(top / 'supplement_registration.json')}
                write(folder / 'STARTED.json', result)
                write(out / 'state.json', {'status': 'generator_CPU_filter', 'object_id': obj,
                      'candidate_index': candidate['index'], 'completed': len(records), 'wall': time.time()})
                try:
                    basic = basic_screen(candidate, job['item'], job['request'], base, job['geometry'])
                    write(folder / 'basic_screen.json', basic)
                    result['basic_cpu_pass'] = basic['pass']
                    if not basic['pass']:
                        raise Rejection(basic['rejected'])
                    goal = np.array(job['item']['source_object_world']) @ np.array(candidate['Gcanonical_object'])
                    arm = CPUArm(Path(base['model_reference']) / 'model/arx.urdf', job['request']['snapshot'])
                    cpu = arm.screen(goal, spec['cpu_candidate_seconds'])
                    write(folder / 'cpu_screen.json', cpu)
                    if not cpu['pass']:
                        raise Rejection(cpu['rejected'])
                    while teacher is None and next(d['free_mb'] for d in devices() if d['uuid'] == reg['compute_uuid']) < 6500:
                        if (top / 'STOP').exists() or time.time() >= reg['deadline_unix']:
                            raise RuntimeError('Resource wait reached campaign STOP/deadline')
                        write(out / 'state.json', {'status': 'waiting_original_model_GPU_memory', 'object_id': obj, 'wall': time.time()})
                        time.sleep(20)
                    if teacher is None:
                        teacher = Teacher(base, out / 'resident_model')
                    write(out / 'state.json', {'status': 'original_model_path_checks', 'object_id': obj,
                          'candidate_index': candidate['index'], 'completed': len(records), 'wall': time.time()})
                    result['cases'] = model_check(teacher, job['request'], cpu, folder,
                        min(reg['deadline_unix'], time.time() + spec['model_candidate_seconds']))
                    result['pass'] = len(result['cases']) == 7 and all(c['pass'] for c in result['cases'])
                    if result['pass']:
                        require_seven(result, candidate)
                    else:
                        result['rejected'] = result['cases'][-1].get('rejected', 'incomplete_model_cases')
                except Rejection as error:
                    result['rejected'] = str(error)
                result.update(finished_unix=time.time(), seconds=time.time() - started,
                    evidence_files={**{str(p): sha(p) for p in folder.rglob('*.json') if p.name != 'STARTED.json'},
                         str(job['wave'] / 'registration.json'): sha(job['wave'] / 'registration.json'),
                         str(top / 'supplement_registration.json'): sha(top / 'supplement_registration.json'),
                         str(top / 'advisor39.md'): sha(top / 'advisor39.md')},
                    original_model_and_margins_unchanged=True,
                    physical_success_claim=False)
                write(folder / 'result.json', result)
                records.append({k: v for k, v in result.items() if k not in ('cases', 'evidence_files')})
                write(out / 'results.json', {'records': records, 'qualified': sum(r['pass'] for r in records)})
                print('SUPPLEMENT_SCREEN', obj, candidate['index'], result['pass'], result.get('rejected'), flush=True)
            exhausted = [o for o, j in jobs.items() if j['terminal'] and o not in release and counts.get(o, 5) < 15]
            write(out / 'progress.json', {'activated_objects': list(activated), 'completed': len(records),
                  'basic_cpu_pass': sum(r.get('basic_cpu_pass', False) for r in records),
                  'qualified': sum(r['pass'] for r in records), 'exhausted_objects': exhausted,
                  'active_density_levels': {o: j['level'] for o, j in jobs.items()}, 'wall': time.time()})
            if prior_state['status'] == 'complete' and all(j['terminal'] for j in jobs.values()):
                write(out / 'state.json', {'status': 'complete', 'completed': len(records),
                      'qualified': sum(r['pass'] for r in records), 'exhausted_objects': exhausted})
                return
            if not advanced:
                write(out / 'state.json', {'status': 'waiting_initial_pool_or_physical_consumption',
                      'activated_objects': list(activated), 'completed': len(records), 'wall': time.time()})
                time.sleep(20)
        write(out / 'state.json', {'status': 'complete', 'reason': 'original_deadline', 'completed': len(records)})
    except BaseException:
        write(out / 'FAULT.json', {'error': traceback.format_exc(), 'wall': time.time(), 'physical_commands': 0})
        write(out / 'state.json', {'status': 'fault', 'physical_commands': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    run(parser.parse_args().root)
