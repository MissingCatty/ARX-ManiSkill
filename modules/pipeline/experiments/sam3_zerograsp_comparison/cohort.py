"""Freeze every eligible paired case without selecting by either grasp outcome."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT, read_json, sha256, write_json
from website.tools.grasp_metrics import sam3_failure_phase

INPUTS = ['state.json', 'entry.json', 'camera/color.png', 'camera/depth.png',
          'camera/depth_m.npy', 'camera/mask.png', 'camera/params.json', 'sam3.json']


def eligible_rows(source):
    """Complete claims, rather than existence of result.json, define terminals."""
    source = Path(source)
    registration = read_json(source/'registration.json')
    pairs, excluded = [], []
    for index, entry in enumerate(registration['schedule']):
        trial = source/'trials'/entry['id']
        claim = source/'claims'/(entry['id']+'.json')
        if not claim.is_file():
            excluded.append({'id': entry['id'], 'kind': 'not_started'})
            continue
        receipt = read_json(claim)
        if receipt['state'] != 'complete':
            excluded.append({'id': entry['id'], 'kind': 'interrupted'})
            continue
        if sha256(trial/'result.json') != receipt['result_sha256']:
            raise RuntimeError('Baseline result claim hash differs: '+entry['id'])
        result = read_json(trial/'result.json')
        failure = sam3_failure_phase(result)
        if failure:
            excluded.append({'id': entry['id'], 'kind': 'sam3_failure', 'phase': failure})
            continue
        pair = {'id': entry['id'], 'entry': entry, 'schedule_index': index,
                'baseline_trial': str(trial.relative_to(ROOT)),
                'baseline_result_sha256': sha256(trial/'result.json'),
                'baseline_claim_sha256': sha256(claim),
                'baseline_planning_success': bool(result.get('planning_success')),
                'baseline_success': bool(result.get('success')), 'baseline_reason': result['reason']}
        if result.get('planning_success'):
            observation = trial
            preparation = read_json(trial/'stationing/prepared_plan_handoff.json')
            pair['workspace'] = {key: preparation[key] for key in
                ['preferred_arm', 'workspace_scope', 'prepared_target_world',
                 'verified_high_cell', 'workspace_resolution_m']}
        else:
            observations = [p.parent for p in sorted(trial.glob(
                'stationing/segments/*/observation/high_workspace_gate.json'))
                if read_json(p)['passed'] and read_json(p.parent/'sam3.json')['target_found']]
            if observations:
                observation = observations[-1]
                gate = read_json(observation/'high_workspace_gate.json')
                from modules.mobile_manipulation.placement import backproject
                import numpy as np
                from PIL import Image
                state = read_json(observation/'state.json')
                points = backproject(np.asarray(Image.open(observation/'camera/mask.png')) > 0,
                    np.load(observation/'camera/depth_m.npy').squeeze(),
                    state['intrinsics'], state['camera_world_optical'])
                scope = observation.parent.name.split('_', 1)[1]
                pair['workspace'] = {'preferred_arm': None if scope == 'shared' else scope,
                    'workspace_scope': scope, 'prepared_target_world': np.median(points, axis=0).tolist(),
                    'verified_high_cell': gate['cell'], 'workspace_resolution_m': .02}
            elif result.get('stationing', {}).get('failed_stage') == 'high_workspace_positioning':
                # The unchanged stationing prerequisite failed before either generator.
                observation = trial/'stationing/coarse'
                pair['common_precondition_failure'] = True
            else:
                raise RuntimeError('Eligible case has no comparable SAM observation: '+entry['id'])
        pair['observation'] = str(observation.relative_to(ROOT))
        pair['input_hashes'] = {name: sha256(observation/name) for name in INPUTS}
        for name in ['overview.jpg', 'initial_overview.png', 'high_workspace_gate.json']:
            if (observation/name).is_file():
                pair['input_hashes'][name] = sha256(observation/name)
        if not read_json(observation/'sam3.json')['target_found']:
            raise RuntimeError('Eligible observation has a failed SAM3 mask')
        pairs.append(pair)
    return pairs, excluded


def freeze(source, destination):
    source, destination = Path(source), Path(destination)
    acceptance = read_json(source/'acceptance.json')
    if not acceptance['passed'] or not acceptance['record_integrity']:
        raise RuntimeError('Baseline independent acceptance failed')
    if not read_json(source/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Baseline owned model services are still running')
    if read_json(source/'progress.json')['phase'] != 'complete':
        raise RuntimeError('Wait for the registered baseline continuation to finish')
    pairs, excluded = eligible_rows(source)
    if any(row['kind'] == 'not_started' for row in excluded):
        raise RuntimeError('Baseline has unstarted slots; cohort must wait')
    for pair in pairs:
        original = ROOT/pair['observation']
        frozen = destination/pair['id']
        for name, digest in pair['input_hashes'].items():
            target = frozen/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original/name, target)
            if sha256(target) != digest:
                raise RuntimeError('Paired input copy changed')
        trial = ROOT/pair['baseline_trial']
        for name in ['result.json', 'plan_result.json', 'independent_evaluation.json']:
            if (trial/name).exists():
                shutil.copy2(trial/name, frozen/('baseline_'+name))
        pair['frozen_observation'] = str(frozen.relative_to(ROOT))
    return {'pairs': pairs, 'excluded': excluded,
        'baseline_run': str(source.relative_to(ROOT)),
        'registration_sha256': sha256(source/'registration.json'),
        'acceptance_sha256': sha256(source/'acceptance.json'),
        'pair_count': len(pairs), 'original_count': len(pairs)+len(excluded)}


def check_pair(pair):
    directory = ROOT/pair['frozen_observation']
    for name, digest in pair['input_hashes'].items():
        if sha256(directory/name) != digest:
            raise RuntimeError('Frozen SAM comparison input changed: '+pair['id']+'/'+name)
    trial = ROOT/pair['baseline_trial']
    if sha256(trial/'result.json') != pair['baseline_result_sha256']:
        raise RuntimeError('Paired baseline outcome changed')
    return directory
