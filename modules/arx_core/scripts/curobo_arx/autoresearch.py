"""ARX experiment boundary for an active AutoResearch agent; no embedded LLM.

evaluate is read-only. pilot executes one preregistered paired hypothesis,
archives both outcomes and restores the original scene/view. No git reset.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = [
    'scripts/curobo_arx/grasp_execute.py', 'scripts/curobo_arx/grasp_input.py',
    'scripts/curobo_arx/optimize_path.py', 'scripts/curobo_arx/settling.py',
    'scripts/curobo_arx/build_model.py', 'scripts/curobo_arx/gripper_update.py',
    'scripts/curobo_arx/run_grasp.sh', 'scripts/curobo_arx/autoresearch.py',
    'arx-sim/config/runtime.yaml', 'arx-sim/config/controllers.yaml',
    'arx-sim/arx_sim/diagnostics/grasp_actions.py',
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}


def evaluate(folder):
    """Independently require executed stages, bilateral closure and stable lift.

    Planning certificates and measured external contacts remain separate.
    Invalid/missing artifacts cannot produce a positive score.
    """
    folder = Path(folder)
    result = read(folder / 'result.json')
    snap = read(folder / 'capture_snapshot.json')
    scenario = read(folder / 'scenario.json')
    target = scenario.get('target', 'test_object')
    target_path = snap['objects'][target]['path']
    initial_z = snap['objects'][target]['world'][2][3]
    required = ['pregrasp', 'approach', 'close', 'lift', 'hold']
    executed, converged, violations, missing = [], [], [], []
    planned_max, dense_invalid, planned_stages = 0., 0, []
    fingers, heights = set(), []
    hold_duration = 0.
    for stage in required:
        path = folder / f'{stage}_execution.json'
        if not path.is_file():
            missing.append(stage)
            continue
        execution = read(path)
        executed.append(stage)
        settling = read(folder / f'{stage}_settling.json')
        if settling.get('accepted') is True:
            converged.append(stage)
        for event in execution['contacts']:
            if event['num_contacts'] <= 0:
                continue
            pair = event['a'] + event['b']
            if '/World/Robot/' in event['a'] and '/World/Robot/' in event['b']:
                continue
            allowed = target_path in pair and any(n in pair for n in ['right_link27', 'right_link28'])
            if not allowed:
                violations.append({'stage': stage, **event})
            if stage == 'close' and target_path in pair:
                fingers.update(n for n in ['right_link27', 'right_link28'] if n in pair)
        command = read(folder / f'{stage}_command.json')
        if command['joint_names'] != snap['right_arm']:
            violations.append({'stage': stage, 'error': 'Wrong ARX joint order'})
        limits = [snap['joint_limits'][snap['joint_names'].index(n)] for n in command['joint_names']]
        for q in command['position']:
            if len(q) != len(limits) or any(not math.isfinite(v) or v < lo-1e-6 or v > hi+1e-6 for v, (lo, hi) in zip(q, limits)):
                violations.append({'stage': stage, 'error': 'Planned joint bound violation'})
                break
        if stage in ['pregrasp', 'approach', 'lift']:
            cert = read(folder / f'{stage}_optimization.json')
            planned_stages.append(stage)
            value = float(cert['max_limit_exceed_rad'])
            if not math.isfinite(value) or not cert['success']:
                violations.append({'stage': stage, 'error': 'Invalid optimization certificate'})
            planned_max = max(planned_max, value)
            dense_invalid += cert['invalid_samples']
        if stage == 'hold':
            heights = [r['objects'][target][2][3] - initial_z for r in execution['records']]
            hold_duration = execution['sim_duration_s']
            times = [r['time'] for r in execution['records']]
            if len(times) < 10 or times[-1] - times[0] < 2.5:
                violations.append({'stage': stage, 'error': 'Incomplete recorded hold coverage'})
    complete = len(executed) == len(required)
    guard = not violations and dense_invalid == 0 and planned_max <= 1e-6
    stable = bool(heights) and all(math.isfinite(z) and z > .06 for z in heights) and hold_duration >= 2.99
    success = complete and len(converged) == len(required) and len(fingers) == 2 and stable and guard
    return {'pass': bool(success), 'score': float(success), 'metric_value': float(success),
            'executed_stages': executed, 'missing_stages': missing, 'converged_stages': converged,
            'bilateral_closure': len(fingers) == 2, 'hold_min_rise_m': min(heights) if heights else None,
            'recorded_hold_duration_s': hold_duration, 'guard_pass': guard,
            'guard_violations': violations, 'planning': {'optimized_stages': planned_stages,
                'max_limit_exceed_rad': planned_max, 'dense_invalid_samples': dense_invalid},
            'reported_success': result.get('success'), 'reported_outcome': result.get('outcome'),
            'scope': 'Actual ARX execution; minimum rise over recorded final hold samples'}


def pilot(campaign):
    from client import Client
    os.chdir(ROOT)
    campaign = Path(campaign).resolve()
    campaign.relative_to(ROOT / 'runs')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    if branch != 'zhaoyechong':
        raise RuntimeError('ARX research requires zhaoyechong branch')
    config = read(campaign / 'campaign.json')
    if (campaign / 'state.json').exists():
        raise RuntimeError('Campaign already has state; inspect it before creating a new bounded run')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Insufficient free space for archived ARX recordings')
    lock_path = ROOT / 'runs/arx_curobo/autoresearch.lock'
    lock = lock_path.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = Client(ROOT / 'runs/arx_curobo/mailbox')
    before_hashes = hashes()
    write(campaign / 'protected_hashes.json', before_hashes)
    write(campaign / 'before_snapshot.json', client.request('snapshot'))
    view = client.request('grasp_viewport_state')
    write(campaign / 'before_viewport.json', view)
    control = client.request('grasp_control_probe')
    write(campaign / 'before_control.json', control)
    trials = campaign / 'trials'
    scenario = dict(config['scenario'])
    source = (ROOT / config['proposal_source']).resolve()
    scenario.update(name='00_control', paired_baseline=str(source))
    variant = {**scenario, 'name': '01_depth_minus_10mm', 'paired_baseline': '00_control',
               'research_depth_shift_m': -.010}
    # Immutable preregistration written before either physical trial.
    write(campaign / 'scenarios.json', [scenario, variant])
    state = {'status': 'running', 'hypothesis': 'H001', 'max_hypotheses': 1,
             'max_physical_trials': 2, 'baseline_default_changed': False,
             'started_unix': time.time(), 'agent_required_for_next_hypothesis': True}
    write(campaign / 'state.json', state)
    try:
        for name, policy in [('00_control', 'baseline'), ('01_depth_minus_10mm', 'research_depth')]:
            if hashes() != before_hashes:
                raise RuntimeError('Protected source changed during campaign')
            state['active_trial'] = name
            write(campaign / 'state.json', state)
            command = [sys.executable, 'scripts/curobo_arx/failure_scenarios.py',
                       '--scenarios', str(campaign / 'scenarios.json'), '--output', str(trials),
                       '--names', name, '--policy', policy, '--gpu', '1', '--timeout', '720']
            with (campaign / f'{name}.log').open('w') as log:
                # Inner runner owns timeout and robot-child cleanup. Do not run
                # another trial after an infrastructure failure or interruption.
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            score = evaluate(trials / name)
            write(campaign / f'{name}_evaluation.json', score)
            print(json.dumps({'trial': name, **score}), flush=True)
            if not score['guard_pass']:
                raise RuntimeError('ARX guard failed; no further trials')
        baseline = read(campaign / '00_control_evaluation.json')
        variant_score = read(campaign / '01_depth_minus_10mm_evaluation.json')
        gain = variant_score['score'] - baseline['score']
        decision = 'confirmation_required' if gain > 0 else 'reverted'
        state.update(status='pilot_budget_exhausted', decision=decision, score_delta=gain,
                     promotion='none; no held-out/repeat confirmation in this pilot')
        fields = ['iteration','metric_value','delta','delta_pct','status','description','evaluator_source','timestamp']
        with (campaign / 'autoresearch-results.tsv').open('w') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(fields)
            for i, (s, status, desc) in enumerate([(baseline, 'baseline', 'Fixed-proposal ARX control'),
                 (variant_score, 'reference' if gain > 0 else 'reverted', 'H001 same proposal, world Z -10 mm')]):
                writer.writerow([i,s['score'], '-' if i == 0 else gain, '-',status,desc,'mechanical',time.strftime('%Y-%m-%dT%H:%M:%S%z')])
        with (campaign / 'research_log.md').open('a') as stream:
            stream.write('\n## H001 completed\n'+json.dumps({'baseline':baseline,'variant':variant_score,'decision':decision},indent=2)+'\n')
        with (campaign / 'research.md').open('a') as stream:
            stream.write(f"\n| 0 | paired control | {baseline['score']} | baseline |\n| 1 | Z -10 mm | {variant_score['score']} | {decision} |\n")
    except BaseException as exc:
        state.update(status='stopped_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['protected_hashes_unchanged'] = hashes() == before_hashes
        write(campaign / 'state.json', state)
        # The runner finishes every trajectory before returning; never reset
        # while its mailbox still has an outstanding request.
        try:
            if (client.root / 'request.json').exists():
                raise RuntimeError('Mailbox busy; automatic restoration deferred')
            shutil.copy2(ROOT / 'runs/tabletop_trials/initial_setup.json', ROOT / 'runs/arx_grasp/capture_snapshot.json')
            client.request('grasp_reset_setup')
            for name, pos in [('blue_cylinder',[4.33,1.79,.76]),('green_sphere',[4.46,1.96,.76]),('red_cube',[4.22,1.75,.73])]:
                client.request('place_object', name=name, position=pos)
            restored = config['restore_asset']
            client.request('grasp_asset_load', **restored)
            # Keep the next RGB-D crop's size metadata aligned with the restored asset.
            asset = next(a for a in read(ROOT / 'runs/arx_grasp/tabletop_catalog.json')
                         if a['container_path'] == restored['asset_path'])
            write(ROOT / 'runs/arx_grasp/active_asset.json', asset)
            client.request('grasp_viewport_state', restore=view)
            write(campaign / 'after_snapshot.json', client.request('snapshot'))
            write(campaign / 'after_viewport.json', client.request('grasp_viewport_state'))
            write(campaign / 'after_control.json', client.request('grasp_control_probe'))
            state['scene_restored'] = True
        except Exception as exc:
            state['restore_error'] = str(exc)
        state['finished_unix'] = time.time()
        write(campaign / 'state.json', state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('evaluate').add_argument('--folder', required=True)
    sub.add_parser('pilot').add_argument('--campaign', required=True)
    args = parser.parse_args()
    if args.command == 'evaluate':
        print(json.dumps(evaluate(args.folder), allow_nan=False))
    else:
        pilot(args.campaign)
