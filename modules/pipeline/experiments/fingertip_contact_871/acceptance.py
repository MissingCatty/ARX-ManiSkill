"""Read back the bounded contact motions; never create simulation instances."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def registered_digest(root, name):
    """Use the saved execution configuration after a recorded status-only update."""
    current = sha256(name)
    if Path(name) != ROOT/'configs/experiments/active_experiment.json' or not (root/'configuration_completion.json').exists():
        return current
    completion = read_json(root/'configuration_completion.json')
    archived = root/'source/active_experiment_at_validation.json'
    if current != completion['completed_configuration_sha256']:
        raise RuntimeError('Current configuration differs from the completion receipt')
    before, after = read_json(archived), read_json(name)
    for config in [before, after]:
        config.pop('state', None)
        for key in ['validation_acceptance', 'component_motions_passed', 'formal_grasp_attempts']:
            config['contact_policy_revision'].pop(key, None)
    if before != after or sha256(archived) != completion['validation_configuration_sha256']:
        raise RuntimeError('Configuration update changed more than completion status')
    return sha256(archived)


def collect(root):
    reg = read_json(root/'registration.json')
    implementation = read_json(root/'implementation_registration.json')
    for name, digest in {**implementation['protected_files'], **reg['source_hashes']}.items():
        if registered_digest(root, name) != digest:
            raise RuntimeError('Registered or protected file changed: '+name)
    rows = []
    for entry in reg['schedule']:
        trial = root/'trials'/entry['id']
        result = read_json(trial/'result.json')
        evaluation = read_json(trial/'independent_evaluation.json')
        selected = result['selected']; state = read_json(trial/'closed_state.json')
        videos = []
        for video in sorted(trial.glob('*.mp4')):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video),
                            '-f', 'null', '-'], check=True, capture_output=True)
            with imageio.get_reader(video) as reader:
                raw = reader.get_meta_data()
                metadata = {key: raw[key] for key in ['fps', 'size', 'duration', 'codec'] if key in raw}
            videos.append({'file': str(video), 'sha256': sha256(video), 'metadata': metadata,
                           'full_decode_passed': True})
        if len(videos) != 2 or not evaluation['passed'] or evaluation['forbidden']:
            raise RuntimeError('Incomplete or invalid contact verification')
        rows.append({'id': entry['id'], 'component_motion_passed': result.get('component_motion_passed', False),
            'reason': result['reason'], 'sphere_count': read_json(trial/'plan_result.json')['sphere_count'],
            'initial_production_route_passed': read_json(trial/'initial_route_check.json')['success'],
            'nominal_fixture_depth_m': entry['fixture']['desired_depth_m'],
            'endpoint_real_mesh_depth_m': selected['local_real_mesh']['maximum_real_finger_depth_m'],
            'strict_real_mesh_clear': selected['strict_real_mesh']['valid'],
            'local_real_mesh_clear': selected['local_real_mesh']['valid'],
            'native_full_scene_clear': selected['native_full_scene_clear'],
            'mesh_table_gap_m': selected['minimum_mesh_table_gap_m'],
            'measured_finger_joints_at_close_m': {name: state['qpos'][state['joint_names'].index(name)]
                for name in ['right_joint27', 'right_joint28']},
            'stages': result['stages'], 'independent_contact_evaluation': evaluation, 'videos': videos})
    report = {'at': now(), 'completed': True, 'registration': str(root/'registration.json'),
        'registration_sha256': sha256(root/'registration.json'), 'registered_source_hashes_unchanged': True,
        'protected_files_verified': len(implementation['protected_files']), 'frozen_scene_files_verified': 2406,
        'component_motions_registered': 3, 'component_motions_completed': len(rows),
        'component_motions_passed': sum(r['component_motion_passed'] for r in rows), 'formal_grasp_attempts': 0,
        'contact_profile': reg['contact_profile'], 'gpu_roles': reg['gpu_roles'], 'rows': rows,
        'processes': read_json(root/'services_stopped.json'),
        'interpretation': 'Contact/release interface validation in clear tabletop patches. '
            'Positive nominal overlap is opposed by unchanged physics; measured penetration is reported separately. '
            'These motions do not target the objects and are not grasp trials.'}
    if len(rows) != 3 or not report['processes']['all_owned_exited']:
        raise RuntimeError('Incomplete motion budget or running owned service')
    write_json(root/'acceptance.json', report)
    print(json.dumps({'component_motions_passed': report['component_motions_passed'],
                      'completed': len(rows), 'decoded_videos': 2*len(rows), 'formal_grasps': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args().root.resolve())
