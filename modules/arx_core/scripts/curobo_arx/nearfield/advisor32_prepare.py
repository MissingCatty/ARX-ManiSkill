"""Prepare inspectable CPU-only advisor32 proposals; deliberately no launcher."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha
from training.nearfield_arx.projection import Projector, model_views
from .advisor32_diagnose import OLD, PARENT, episode
from .advisor32_spec import make_trials, validate_trials, model_images


def donors(out, reg):
    projector = Projector(reg['learned_policy']['robot_usd']); result = {}
    for obj in ('household_hope_milk_carton', 'household_hot3d_bbq_sauce_bottle'):
        trial = next(t for t in reg['seed_trials'] if t['object_id']==obj and t['method']=='D' and t['condition']=='normal')
        ep = episode(OLD, trial['trial_id'])
        entry = read(ep/'physical_entry_observation.json')
        samples = [json.loads(line) for line in (ep/'samples.jsonl').read_text().splitlines() if line.strip()]
        matched = [row for row in samples if abs(row['frame_time']-entry['frame_time']) < 1e-9]
        if len(matched) != 1:
            raise ValueError('No unique archived entry frame')
        calls = read(ep/'policy_requests.json')
        if len(calls) < 2:
            raise ValueError('No historical second-plane observation')
        req = read(calls[1]['request']); second = read(req['observation'])
        asset = next(a for a in reg['catalog']['objects'] if a['object_id']==obj)
        images_by_plane = {}; records = []
        for plane, row, rgb, depth in (
            (0, matched[0], ep/matched[0]['rgb_relative'], ep/matched[0]['depth_relative']),
            (1, second, Path(req['rgb']), Path(req['depth']))):
            z = np.load(depth, allow_pickle=False)
            masks, projection, _ = projector.project(asset['usd_path'], row, z)
            views = model_views(Image.open(rgb), z, masks, reg['learned_policy']['visual_contract']['depth_scale'])
            paths = [out/f'{obj}_plane{plane}_{name}.png' for name in ('rgb', 'depth')]
            for im, path in zip(views, paths): im.save(path)
            images_by_plane[str(plane)] = list(map(str, paths))
            records.append({'plane': plane, 'stage': 'historical H before blind insertion' if plane==0 else 'historical stopped first visual refinement',
                'raw_rgb': str(rgb), 'raw_depth': str(depth), 'frame_time': row['frame_time'],
                'EE_world': row['snapshot']['link_world']['right_eef_link'],
                'camera_world_optical': row['camera_world_optical'], 'intrinsics': row['intrinsics'],
                'projection': projection, 'model_images': list(map(str, paths)),
                'files': {str(p): sha(p) for p in [rgb, depth, *paths]},
                'new_actual_pose_matching_verified': False})
        for index in (0, 1):
            check = model_images('full_stale', index, images_by_plane['0'], images_by_plane)
            if not check['replaced']:
                raise AssertionError('Every stale neural decision must be replaced')
        result[obj] = {'source_trial_id': trial['trial_id'], 'source_episode': str(ep),
            'source_target_shift_m': [0., 0., 0.], 'images_by_plane': images_by_plane,
            'records': records, 'fallback_to_fresh_allowed': False,
            'scope': 'Full stale neural images. New execution will independently record current safety observations.',
            'confound_to_report': 'Historical second-plane camera/hand pose need not equal the new policy pose; phase matching alone is not pose matching'}
    return result


def prepare(root):
    root = Path(root).resolve(); diagnosis = read(root/'diagnosis_complete/review.json')
    if (root/'proposal_manifest.json').exists():
        raise FileExistsError('Never overwrite a prior prepared proposal')
    reg = read(OLD/'registration.json')
    canonical = next(g['canonical_proposal'] for g in diagnosis['geometry'] if g['object_id']=='household_ycb_spam_can')
    trials = make_trials(reg, canonical)
    review = validate_trials(trials)
    write(root/'proposal_manifest.json', {'trials': trials, 'review': review,
        'physical_execution_authorized_by_this_file': False,
        'no_registration_or_launcher_created': True,
        'pending': ['Resolve camera-distance versus hand-relative depth contract',
                    'Integrate and independently audit runtime/closure/worker isolation',
                    'Validate canonical Spam IK and collision under the frozen model',
                    'Register resource UUIDs and fresh frozen-source/model/asset preflight']})
    target = root/'stale_donors'; target.mkdir(exist_ok=False)
    write(target/'manifest.json', donors(target, reg))
    command = [sys.executable, '-m', 'unittest', 'scripts.curobo_arx.nearfield.test_advisor32', '-v']
    check = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    (root/'offline_checks.log').write_text(check.stdout+check.stderr)
    write(root/'offline_checks.json', {'pass': check.returncode==0, 'command': command,
        'log_sha256': sha(root/'offline_checks.log'), 'physical_trials': 0,
        'scope': 'New standalone local rules/spec only; no end-to-end executor certification'})
    if check.returncode:
        raise RuntimeError('New offline rules failed checks')
    sources = list((ROOT/'scripts/curobo_arx/nearfield').glob('advisor32_*.py'))
    sources += [ROOT/'scripts/curobo_arx/nearfield/test_advisor32.py']
    write(root/'preparation_source_files.json', {str(p): sha(p) for p in sources})
    write(root/'historical_preservation.json', {'files': {str(p): sha(p) for p in (
        PARENT/'registration.json', PARENT/'STOP', PARENT/'final_analysis.json',
        OLD/'registration.json', ROOT/'research_plan/chat/gemini回复/gemini回复32.md')},
        'prior_completed_results_modified': False, 'prior_interrupted_retried': False,
        'prior_unexecuted_slots_restarted': False})
    write(root/'state.json', {'status': 'offline_preparation_waiting_for_depth_contract',
        'physical_trials': 0, 'new_model_inferences': 0, 'services_started': 0,
        'new_proposed_slots': 30, 'formal_registered_slots': 0,
        'completed_physical_slots': 0, 'new_physical_failures': 0,
        'runtime_integrated': False, 'wall': time.time()})
    return {'root': str(root), 'proposed_slots': 30, 'physical_trials': 0, 'offline_checks_pass': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root')
    print(json.dumps(prepare(parser.parse_args().root), ensure_ascii=False))
