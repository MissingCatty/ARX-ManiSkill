"""One fresh-goal ARX transfer probe, with an optional separate teacher diagnosis.

This is a research run outside the two-case qualified handoff profile. It does
not expand that profile, change the default controller, or resume EGAD49.
"""
from arx_project.layout import canonical_path
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

# Resolve-only must work with system Python and no numerical/GPU/RPC packages.
if __name__ == '__main__' and '--resolve-only' in sys.argv:
    from interaction_contract import resolve_cli
    resolve_cli(sys.argv[1:], 'policy')
    raise SystemExit(0)

import httpx

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / 'scripts/curobo_arx'), str(HERE)]
from interaction_contract import load as load_contract, reject_overrides, resolve as resolve_interaction, prepare_environment, adapter_compatibility, check_health
from client import Client
from compare_chunks import run_process
from egad49 import compare_configuration, INITIAL_SHA
from egad_evaluate import evaluate
from grasp_handoff import NEAR_RULES
from summarize_chunks import encode, read, sha, write
from gpu_policy import resolve, compute_env, verify_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object-name', required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--policy-gpu', required=True)
    parser.add_argument('--planner-gpu', required=True)
    parser.add_argument('--neural-gpu', required=True)
    parser.add_argument('--mailbox', type=Path, default=ROOT / 'runs/lilo_egad49_20261026_eval01/runtime/mailbox')
    parser.add_argument('--ops-url', default='http://127.0.0.1:8101')
    parser.add_argument('--port', type=int, default=8009)
    parser.add_argument('--diagnose-teacher', action='store_true',
                        help='If actual policy inference fails after an accepted entry, execute that exact goal with the teacher once')
    parser.add_argument('--reach-target-veto', action='store_true',
                        help='Extra reach-only target query, with one 80mm waypoint alternative and final command veto')
    parser.add_argument('--goal-source', type=Path,
                        help='Explicit prior planned selection from this same object, for a fixed-goal route comparison; fresh inference is still retained')
    parser.add_argument('--teacher-only', action='store_true', help='Run one separately recorded same-goal teacher diagnosis without starting a VLA service')
    parser.add_argument('--resolve-only', action='store_true', help='Resolve common semantics without GPU/RPC; use --resolve-only --help for its minimal arguments')
    parser.add_argument('--interaction-contract', type=Path, help='Explicit common ARX interaction configuration; legacy experiments remain unchanged without it')
    args = parser.parse_args()
    contract = load_contract(args.interaction_contract) if args.interaction_contract else None
    if contract:
        reject_overrides(os.environ)

    if args.teacher_only and (not args.goal_source or args.diagnose_teacher):
        parser.error('--teacher-only requires --goal-source and excludes --diagnose-teacher')
    compatibility = adapter_compatibility(args.adapter, contract) if contract else None
    if contract:
        args.reach_target_veto = True
    for name in ('policy_gpu', 'planner_gpu', 'neural_gpu'):
        setattr(args, name, resolve(getattr(args, name))['uuid'])
    os.environ.update(compute_env(os.environ, args.planner_gpu))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    adapter = args.adapter.resolve()
    manifest = read(adapter / 'adapter_manifest.json')
    if manifest['state_convention'] != 'grasp_goal_world_error_v1':
        raise ValueError('This probe requires the pinned goal-conditioned head')
    if args.object_name in set(manifest['train_objects'] + manifest['validation_objects']):
        raise ValueError('Choose an object outside this head training and validation sets')
    historical = ROOT / 'runs/lilo_egad49_20261026_eval01'
    catalog_file = historical / 'runtime/historical_reference/catalog.json'
    catalog = {x['name']: x for x in read(catalog_file)}
    asset = catalog[args.object_name]
    if 'egad' in asset['container_path'].lower():
        raise ValueError('The previously viewed EGAD49 pool remains stopped')
    scenario = {'name': args.object_name, 'target': 'test_object', 'asset': args.object_name,
                'placement': {'position': [4.048, 2.08, .679 + asset['size_m'][2] / 2 + .012]}}
    from interaction_contract import instruction_for
    instruction = instruction_for(args.object_name) if contract else 'pick up the ' + args.object_name.replace('_', ' ')
    initial_file = ROOT / 'runs/tabletop_trials/initial_setup.json'
    if sha(initial_file) != INITIAL_SHA:
        raise ValueError('Historical robot reset differs')
    reference_setup = historical / 'runtime/historical_reference/setup.json'
    reference_view = historical / 'original_viewport.json'
    rule_file = out / 'handoff_rule.json'
    rule = contract['handoff_rule'] if contract else NEAR_RULES['ten_mm']
    write(rule_file, rule)
    inputs = out / 'registered_inputs'
    inputs.mkdir()
    for path in (initial_file, catalog_file, reference_setup, reference_view):
        shutil.copy2(path, inputs / path.name)
    write(inputs / 'scenario.json', scenario)
    goal_source = args.goal_source.resolve() if args.goal_source else None
    if goal_source:
        if read(goal_source.parent / 'scenario.json')['asset'] != args.object_name:
            raise ValueError('Comparison goal belongs to another object')
        shutil.copy2(goal_source, inputs / 'planned_grasp_source.json')
    sources = [Path(__file__), *[HERE / n for n in (
        'rollout.py', 'observations.py', 'grasp_handoff.py', 'egad49.py', 'egad_evaluate.py',
        'recheck_entry_range.py', 'summarize_p2.py', 'summarize_chunks.py', 'compare_chunks.py',
        'check_reach_target.py', 'tracking.py', 'interaction_contract.py')],
        *[ROOT / 'scripts/curobo_arx' / n for n in (
        'static_pick/reproduce_record.py', 'static_pick/reproduce_setup.py', 'grasp_execute.py',
        'grasp_input.py', 'prepare.py', 'model_archive.py', 'gripper_update.py',
        'optimize_path.py', 'settling.py', 'client.py', 'research_guard.py')],
        ROOT / 'servers/lilo_vla/backend.py', ROOT / 'servers/lilo_vla/app.py']
    sources += [ROOT / 'scripts/run_lilo_vla_server.sh']
    sources += [ROOT / 'scripts/curobo_arx/gpu_policy.py']
    if contract:sources += [ROOT / 'configs/lilo_vla/model_compatibility.json']
    physical = [ROOT / p for p in (
        'arx-sim/config/sim.yaml', 'arx-sim/config/runtime.yaml', 'arx-sim/config/controllers.yaml',
        'configs/calibration.yaml', 'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    physical += [canonical_path(ROOT / 'asset' / asset['container_path'].removeprefix('/workspace/asset/'))]
    physical += [p for directory in ('model', 'model_closed')
                 for p in (ROOT / 'scripts/curobo_arx/static_pick/reference' / directory).glob('arx.*')]
    archive = out / 'registered_source_snapshot'
    for path in sources:
        destination = archive / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    registration = {
        'scope': 'Single nontraining/nonvalidation legacy asset, ' + ('explicit fixed-goal route comparison' if goal_source else 'fresh ZeroGrasp goal') + '; development extension, not a held-out benchmark',
        'scenario': scenario, 'instruction': instruction, 'adapter': str(adapter),
        'state_convention': manifest['state_convention'], 'rule': rule,
        'interaction_contract': contract, 'model_compatibility': compatibility,
        'qualified_profile_used': False,
        'qualification': 'Only geometric/actual-entry gates apply; this new object and grasp goal have no prior learned-policy qualification',
        'max_policy_attempts': 0 if args.teacher_only else 1, 'teacher_only': args.teacher_only, 'teacher_diagnosis_on_failure': args.diagnose_teacher,
        'max_total_attempts': 2 if args.diagnose_teacher else 1,
        'max_actions': 90, 'execute_steps': 1, 'max_local_sim_seconds': 60, 'trial_wall_seconds': 900,
        'candidate_handoff_filter': True, 'handoff_candidate_budget': 12,
        'reach_target_veto': args.reach_target_veto,
        'reach_algorithm': 'target_veto_via_080mm_stop_v2' if args.reach_target_veto else 'original_direct_PRM',
        'goal_source': str(goal_source) if goal_source else 'Fresh ZeroGrasp selection',
        'goal_source_sha256': sha(goal_source) if goal_source else None,
        'object_placement_rule': 'Existing legacy clear-table rule, unchanged asset physics and historical ARX reset',
        'initial_setup_sha256': INITIAL_SHA,
        'policy_artifact_sha256': {str(adapter / n): sha(adapter / n) for n in
                                  ('action_head.pt', 'dataset_statistics.json', 'adapter_manifest.json')},
        'physical_source_sha256': {str(p): sha(p) for p in physical},
        'algorithm_sha256': {str(p): sha(p) for p in sources},
        'input_sha256': {str(p): sha(p) for p in [rule_file, *inputs.glob('*.json')]},
        'policy_gpu': args.policy_gpu, 'planner_gpu': args.planner_gpu, 'neural_gpu': args.neural_gpu,
        'registered_unix': time.time(), 'old_EGAD49_state': read(historical / 'state.json')}
    write(out / 'registration.json', registration)
    storage = ROOT / 'runs/research_contract_20261019/storage_policy.json'
    if storage.exists():
        shutil.copy2(storage, out / 'storage_policy.json')
    else:
        write(out / 'storage_policy.json', {'local_first': True, 'reserve_bytes': 1073741824,
                                          'historical_file': 'Unavailable; not assumed verified'})
    state = {'status': 'starting', 'completed': 0, 'successes': 0}
    write(out / 'state.json', state)
    hashes = {}
    for key in ('policy_artifact_sha256', 'physical_source_sha256', 'algorithm_sha256', 'input_sha256'):
        hashes.update(registration[key])

    def check():
        if shutil.disk_usage(out).free <= 1073741824:
            raise RuntimeError('Local 1GiB reserve reached')
        for path, digest in hashes.items():
            if sha(path) != digest:
                raise RuntimeError('Registered source changed: ' + path)
        if read(historical / 'state.json') != registration['old_EGAD49_state']:
            raise RuntimeError('Stopped EGAD49 state changed')

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('ARX_LILO_', 'GRASP_')) or key in ('ARX_CUROBO_VALIDATION_DIR', 'ARX_RESEARCH_GUARD_FILE'):
            env.pop(key)
    os.environ.pop('ARX_CUROBO_VALIDATION_DIR', None)
    os.environ.pop('ARX_RESEARCH_GUARD_FILE', None)
    client = Client(args.mailbox.resolve())
    neural = canonical_path(ROOT.parent / 'tmp/arx-fullchain/env/bin/python')
    planner = ROOT.parent / 'curobo/.venv/bin/python'
    server = None
    results = []

    def execute(controller, fixed_selection=None):
        check()
        trial = out / controller
        trial.mkdir()
        (trial / 'environment_evidence').mkdir()
        write(trial / 'scenario.json', scenario)
        state.update(status='running', active=controller)
        write(out / 'state.json', state)
        start = time.monotonic()
        error = None
        try:
            setup = client.request('dynamic_setup', scenario=scenario, timeout=240)
            write(trial / 'setup.json', setup)
            mismatch = compare_configuration(read(reference_setup)['fingerprint'], setup['fingerprint'])
            expected = read(initial_file)
            clipped = [min(max(q, lo), hi) for q, (lo, hi) in
                       zip(expected['joint_positions'], setup['snapshot']['joint_limits'])]
            if setup['initial_conditions']['joint_positions'] != clipped or setup['initial_conditions']['joint_names'] != expected['joint_names']:
                mismatch.append('historical robot reset')
            actual = setup['initial_conditions']['objects']['asset:' + args.object_name]
            if actual['position'] != scenario['placement']['position'] or actual['quaternion_wxyz'] != [1., 0., 0., 0.]:
                mismatch.append('declared object placement')
            if setup['physics_steps'] != 360 or abs(setup['captured_sim_time'] - setup['reset_sim_time'] - 3.) > .001:
                mismatch.append('historical three-second settling')
            write(trial / 'setup_verification.json', {'pass': not mismatch, 'mismatches': mismatch,
                                                     'historical_physical_properties_equal': not mismatch})
            if mismatch:
                raise RuntimeError('Frozen conditions differ: ' + str(mismatch[:12]))
            write(trial / 'active_asset.json', setup['asset'])
            client.request('grasp_viewport_state', restore=read(reference_view))
            write(trial / 'viewport_initial.json', client.request('grasp_viewport_state'))
            write(trial / 'guard.json', {'invariant_sha256': 'Initial fingerprint archived', 'file_sha256': {}})
            child_env = env.copy()
            child_env.update(ARX_CUROBO_VALIDATION_DIR=str(client.root), ARX_OPS_URL=args.ops_url,
                             ARX_LILO_HANDOFF_RULE_FILE=str(rule_file), ARX_OBJECT_LABEL=args.object_name,
                             ARX_LILO_FILTER_CANDIDATE_HANDOFF='1', ARX_LILO_HANDOFF_CANDIDATE_BUDGET='12',
                             ARX_RESEARCH_AUDIT='0', PYTHONUNBUFFERED='1')
            if args.reach_target_veto:
                child_env['ARX_LILO_REACH_TARGET_VETO'] = '1'
            if controller == 'policy':
                child_env.update(ARX_LILO_POLICY_URL=f'http://127.0.0.1:{args.port}',
                                 ARX_LILO_MAX_STEPS='90', ARX_LILO_MAX_SIM_SECONDS='60',
                                 ARX_LILO_EXECUTE_STEPS='1', ARX_LILO_VERIFY_CLOSURE='1',
                                 ARX_LILO_INSTRUCTION=instruction)
            else:
                child_env['ARX_LILO_TEACHER_DIR'] = str(trial / 'teacher_raw')
            command = [neural, ROOT / 'scripts/curobo_arx/static_pick/reproduce_record.py',
                       '--output', trial, '--reference-run', ROOT / 'scripts/curobo_arx/static_pick/reference',
                       '--mailbox', client.root, '--curobo-python', planner,
                       '--neural-gpu', args.neural_gpu, '--planner-gpu', args.planner_gpu,
                       '--trajopt-policy', 'interactive', '--no-audit', '--skip-encoding']
            if fixed_selection:
                selection = trial / 'policy_selected_goal.json'
                shutil.copy2(fixed_selection, selection)
                write(trial / 'diagnosis_goal_source.json', {'source': str(fixed_selection), 'sha256': sha(selection),
                                                           'same_planned_goal_as_failed_policy': True})
                command += ['--fixed-selection', selection]
            if contract:
                resolved = resolve_interaction(contract, args.object_name,
                    'policy' if controller == 'policy' else 'teacher',
                    goal_source=selection if fixed_selection else None,
                    adapter=adapter if controller == 'policy' else None)
                child_env = prepare_environment(child_env, trial, resolved)
            print('TRANSFER_START', controller, args.object_name, flush=True)
            code = run_process(command, child_env, trial / 'pipeline.log', start + 900)
            if code:
                raise RuntimeError('Pipeline exited ' + str(code))
        except Exception:
            error = traceback.format_exc()
            write(trial / 'runner_error.json', {'error': error})
        finally:
            write(trial / 'boundary_snapshot.json', client.request('snapshot', timeout=300))
            if (trial / 'frames').exists() and not (trial / 'recording_stop.json').exists():
                write(trial / 'recording_stop.json', client.request('grasp_video_stop', timeout=120))
        score = evaluate(trial)
        report = read(trial / 'grasp_report.json') if (trial / 'grasp_report.json').exists() else {}
        result = {'scene': args.object_name, 'controller': controller, 'pass': score['pass'],
                  'elapsed_seconds': time.monotonic() - start, 'evaluation': score,
                  'policy_calls': report.get('local_policy_calls', 0),
                  'executed_actions': report.get('local_action_count', 0), 'runner_error': error}
        write(trial / 'result.json', result)
        results.append(result)
        write(out / 'results.json', results)
        state.update(completed=len(results), successes=sum(r['pass'] for r in results))
        write(out / 'state.json', state)
        if (trial / 'frames/frames.json').exists():
            result['video'] = encode(trial)
        if controller == 'policy':
            subprocess.run([str(planner), str(HERE / 'recheck_entry_range.py'), '--trial', str(trial),
                            '--output', str(trial / 'provenance_runtime_check.json')], check=True, cwd=ROOT, env=env)
        write(trial / 'result.json', result)
        write(out / 'results.json', results)
        check()
        print('TRANSFER_RESULT', json.dumps(result), flush=True)
        return result

    try:
        check()
        with (client.root / 'p0_execution.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if client.request('snapshot', timeout=60).get('command_id') is not None:
                raise RuntimeError('Simulator is not idle')
            if args.teacher_only:
                result = execute('teacher_diagnosis', inputs / 'planned_grasp_source.json')
            else:
                with socket.socket() as probe:
                    probe.settimeout(2)
                    if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                        raise RuntimeError('Temporary policy port occupied; preserving its owner')
                server_env = env.copy()
                server_env.update(LILO_GPU=args.policy_gpu, LILO_ADAPTER=str(adapter), LILO_PORT=str(args.port),
                                  LILO_PYTHON=str(ROOT.parent / 'lilo-vla/.venv/bin/python'),
                                  LILO_CHECKPOINT=str(ROOT.parent / 'lilo-vla/checkpoints/lilo-vla-openvla-oft'))
                with (out / 'policy_server.log').open('x') as stream:
                    server = subprocess.Popen(['bash', str(ROOT / 'scripts/run_lilo_vla_server.sh')],
                                              cwd=ROOT, env=server_env, stdout=stream, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                write(out / 'policy_server_process.json', {'pid': server.pid, 'port': args.port, 'gpu': args.policy_gpu})
                deadline = time.monotonic() + 600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Temporary policy server exited')
                    try:
                        response = httpx.get(f'http://127.0.0.1:{args.port}/health', timeout=2, trust_env=False)
                        if response.status_code == 200:
                            health = response.json()
                            if Path(health['adapter']).resolve() != adapter or health['state_convention'] != manifest['state_convention'] or not health['checkpoint_verified']:
                                raise RuntimeError('Policy identity/device mismatch')
                            verify_health(health, args.policy_gpu)
                            if health['gpu_process_pid'] != server.pid:
                                raise RuntimeError('Health endpoint belongs to a different policy process')
                            if health['policy_artifact_sha256'] != {Path(p).name: digest for p, digest in registration['policy_artifact_sha256'].items()}:
                                raise RuntimeError('Loaded policy artifact hashes differ')
                            if contract:
                                check_health(resolve_interaction(contract, args.object_name, 'policy', adapter=adapter), health)
                            write(out / 'policy_health.json', health)
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Policy startup exceeded 600s')
                    time.sleep(2)
                result = execute('policy', inputs / 'planned_grasp_source.json' if goal_source else None)
                selected = out / 'policy/selected_grasp.json'
                entry_file = out / 'policy/handoff_actual_entry.json'
                entry_accepted = entry_file.exists() and read(entry_file)['pass']
                if args.diagnose_teacher and not result['pass'] and selected.exists() and entry_accepted and result['policy_calls'] > 0:
                    execute('teacher_diagnosis', selected)
            check()
            write(out / 'completion_integrity.json', {'registered_sources_and_policy_unchanged': True,
                  'source_archive_matches': all(sha(archive / Path(p).relative_to(ROOT)) == digest
                                                for p, digest in registration['algorithm_sha256'].items()),
                  'stopped_EGAD49_unchanged': True})
            write(out / 'summary.json', {'scope': registration['scope'], 'results': results,
                  'policy_attempts': 0 if args.teacher_only else 1, 'policy_successes': 0 if args.teacher_only else int(result['pass']),
                  'teacher_diagnosis_attempts': len(results) if args.teacher_only else len(results) - 1, 'qualified_profile_used': False,
                  'model_retrained': False, 'global_defaults_changed': False, 'EGAD49_resumed': False,
                  'limitation': 'Single exploratory object and fresh goal; no generalization rate or capture-basin qualification'})
            state.update(status='completed', active=None)
            write(out / 'state.json', state)
    except BaseException:
        state.update(status='stopped_on_error', error=traceback.format_exc())
        write(out / 'state.json', state)
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == '__main__':
    main()
