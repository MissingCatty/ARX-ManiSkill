"""Read completed V3 records, independently verify arm FK and decode videos."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import imageio_ffmpeg

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel


def collect(root):
    reg = read_json(root/'registration.json')
    progress = read_json(root/'progress.json')
    if progress['phase'] != 'complete' or (root/'STOP').exists():
        raise ValueError('Integration is not complete')
    if not read_json(root/'services_stopped.json')['all_owned_exited']:
        raise ValueError('Owned services are still running')
    model = ArmModel()
    rows = []
    for entry in reg['schedule']:
        trial = root/'trials'/entry['id']
        result = read_json(trial/'result.json')
        if read_json(root/'claims'/(entry['id']+'.json'))['state'] != 'complete':
            raise ValueError('Unfinished claim')
        evidence = read_json(trial/'independent_evaluation.json')
        if not evidence['passed'] or evidence.get('independently_successful', False) != result['success']:
            raise ValueError('Independent evaluation mismatch')
        videos = {}
        fk = None
        if result.get('planning_success'):
            selected = read_json(trial/'plan_result.json')['selected']
            arm = get_arm(selected['arm'])
            state = deepcopy(read_json(trial/'state.json'))
            state['links'] = {key: state['links'][key] for key in ['base_link', 'body4']}
            for name, q in zip(arm.joints, selected['q_grasp']):
                i = state['joint_names'].index(name)
                lo, hi = state['joint_limits'][i]
                if not lo-1e-6 <= q <= hi+1e-6:
                    raise ValueError('Selected physical arm violates authored limits')
                state['qpos'][i] = q
            actual = model.frames(state)[arm.eef]
            goal = np.asarray(selected['grasp_world'])
            fk = {'position_error_m': float(np.linalg.norm(actual[:3, 3]-goal[:3, 3])),
                  'rotation_error_rad': float(Rotation.from_matrix(actual[:3, :3]@goal[:3, :3].T).magnitude())}
            if fk['position_error_m'] > .002 or fk['rotation_error_rad'] > .03:
                raise ValueError('Independent physical-arm FK failed')
            if not read_json(trial/'arms'/arm.name/'opening_sweep.json')['passed']:
                raise ValueError('Selected arm has no checked opening sweep')
        if result.get('physical_attempt'):
            entities = read_json(trial/'contact_entities.json')
            arm = get_arm(result['selected_arm'])
            if set(entities['fingers'].values()) != set(arm.finger_links):
                raise ValueError('Wrong physical fingers in contact evidence')
            for name in ['execution.mp4', 'head.mp4']:
                file = trial/name
                completed = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(file), '-f', 'null', '-'],
                                           capture_output=True, text=True, timeout=90)
                if completed.returncode != 0 or completed.stderr:
                    raise ValueError('Video decode failed: '+name+' '+completed.stderr)
                videos[name] = {'decoded': True, 'sha256': sha256(file), 'bytes': file.stat().st_size}
        rows.append({'id': entry['id'], 'target': entry.get('target_text', entry['model_id']),
            'forced_arm': entry.get('planner_arm'), 'selected_arm': result.get('selected_arm'),
            'planning_success': result['planning_success'], 'physical_attempt': result.get('physical_attempt', False),
            'success': result['success'], 'reason': result['reason'],
            'arm_results': read_json(trial/'plan_result.json').get('arm_results', {}) if (trial/'plan_result.json').exists() else {},
            'minimum_hold_lift_m': result.get('minimum_hold_lift_m'),
            'hold_seconds': result.get('hold_seconds'), 'opening': result.get('stages', {}).get('open'),
            'independent_physical_arm_fk': fk, 'videos': videos,
            'result_sha256': sha256(trial/'result.json')})
    report = {'at': now(), 'passed': True, 'version': 'V3', 'registered': len(reg['schedule']),
              'completed': len(rows), 'not_executed': len(reg['schedule'])-len(rows),
              'planning_passes': sum(row['planning_success'] for row in rows),
              'physical_attempts': sum(row['physical_attempt'] for row in rows),
              'grasp_successes': sum(row['success'] for row in rows),
              'all_record_checks_passed': True, 'sphere_comparison': model.sphere_proof,
              'registration_sha256': sha256(root/'registration.json'),
              'acceptance_source_sha256': sha256(Path(__file__)), 'results': rows}
    write_json(root/'acceptance.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = collect(ensure_run_path(args.root))
    print('Validated', result['completed'], 'records;', result['grasp_successes'], 'successful grasps')
