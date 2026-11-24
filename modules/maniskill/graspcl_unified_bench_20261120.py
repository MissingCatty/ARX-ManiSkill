"""Revalidate all three 100-scene arms with uniform, checked refinement."""
from arx_project.layout import canonical_path
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from unittest.mock import patch

import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_pipeline as common
import graspcl_unseen_bench as bench
import graspcl_unseen_resident_resume_20261119 as resident
import graspcl_unseen_shared_backends_20261119 as shared
import graspcl_unified_protocol_20261120 as protocol
from arx_service_readiness import wait_ready
from graspcl_execute import guard
from resident_backend import rpc

CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
PRIOR = CODE/'outputs/arx_graspcl_unseen_50_100_20261119_01'
EPOCH = 'runtime_01'
ARMS = ['A_zerograsp', 'B_graspcl_no_experience', 'C_graspcl_experience50']
SIMULATOR = CODE/'graspcl_unified_simulator_20261120.py'


def aggregate(root, phase=None):
    old = common.read(root/'progress.json') if (root/'progress.json').exists() else {}
    states = {a: common.read(root/a/'progress.json') for a in ARMS if (root/a/'progress.json').exists()}
    value = {'phase': phase or old.get('phase', 'running'), 'registered': 300,
             'completed': sum(v['completed'] for v in states.values()), 'groups': states,
             'updated_at': bench.core.now(), 'runtime_epoch': EPOCH,
             'protocol': protocol.PROTOCOL, 'source_experience_reexecuted': 0}
    if resident.POOL is not None:
        value['simulation_workers'] = resident.POOL.status()
    if shared.MANAGER is not None:
        value['model_services'] = {k: {'pid': r['process'].pid, 'alive': r['process'].poll() is None,
                                      'kind': r['kind']} for k, r in shared.MANAGER.all_processes.items()}
    common.save(root/'progress.json', value)


def preflight(root):
    directory = root/'preflight'
    directory.mkdir(exist_ok=False)
    protocol.install()
    rows = []
    for profile in ['zerograsp_200', 'graspcl_200']:
        for named in [True, False]:
            group = directory/(profile+('_named' if named else '_hashed'))
            trial = group/'arm/trials/mock';trial.mkdir(parents=True)
            reg = {'suite': str(group), 'pipeline_profile': profile}
            registration = group/'registration.json';common.save(registration, reg)
            entry = group/'entry.json';common.save(entry, {'campaign_registration': str(registration)})
            sockets = {k: str(group/((k if named else 'arbitrary_'+str(i))+'.sock'))
                       for i, k in enumerate(['zerograsp', 'graspcl', 'curobo'])}
            common.save(group/'resident/backend_config.json', {'sockets': sockets})
            calls = []
            def native(path):
                calls.append('native');return {'success': True}
            def transport(path, message, **kwargs):
                calls.append(protocol.component(path, {'sockets': sockets}));return {'status': 'ok'}
            def refine(path, config):
                calls.append('refinement')
            def execute(path, entry_path):
                if profile == 'zerograsp_200':
                    protocol.timing.native_candidates(path)
                else:
                    protocol.timing.real_rpc(Path(sockets['graspcl']), {'directory': str(path)})
                protocol.timing.real_rpc(Path(sockets['curobo']), {'directory': str(path)})
            with patch.object(protocol.timing, 'native_candidates', native), \
                 patch.object(protocol.timing, 'real_rpc', transport), \
                 patch.object(protocol.timing, 'execute', execute), \
                 patch.object(protocol.truncation, 'refine_client', refine), \
                 patch.object(protocol, 'validate_pool', lambda *a: {'passed': True}):
                protocol.execute(trial, entry)
            receipt = common.read(trial/'unified_protocol_receipt.json')
            assert calls.count('refinement') == receipt['refinement_calls'] == 1
            assert receipt['initial_planning_gate_checks'] == 1
            assert protocol.component_for_trial(Path(sockets['graspcl']), trial) == 'graspcl'
            rows.append({'profile': profile, 'named_sockets': named, 'calls': calls, 'receipt': receipt})
    # The gate must reject planning before any refinement, even if RPC works.
    group = directory/'blocked';trial = group/'arm/trials/mock';trial.mkdir(parents=True)
    reg = {'suite': str(group), 'pipeline_profile': 'graspcl_200'}
    common.save(group/'registration.json', reg)
    common.save(group/'entry.json', {'campaign_registration': str(group/'registration.json')})
    sockets = {k: str(group/(k+'.sock')) for k in ['graspcl', 'curobo']}
    common.save(group/'resident/backend_config.json', {'sockets': sockets})
    def premature(path, entry):
        protocol.timing.real_rpc(Path(sockets['curobo']), {'directory': str(path)})
    with patch.object(protocol.timing, 'execute', premature):
        try:
            protocol.execute(trial, group/'entry.json')
        except AssertionError as error:
            assert 'refinement was not executed' in str(error)
        else:
            raise AssertionError('An unrefined pool reached planning')
    # Validate genuine saved geometry, not only mocked route counters.
    sample = 'T0024_S4_clutter_029_plate'
    a = PRIOR/'A_zerograsp/fixed_zerograsp'
    geometry = protocol.validate_pool(a/'trials'/sample, common.read(a/'registration.json'))
    assert geometry['retained_candidates'] == 17
    b = PRIOR/'B_graspcl_no_experience/fixed_graspcl'
    try:
        protocol.validate_pool(b/'trials'/sample, common.read(b/'registration.json'))
    except (FileNotFoundError, AssertionError):
        pass
    else:
        raise AssertionError('The historical broken GraspCL output passed the new gate')
    result = {'passed': True, 'routing_cases': rows, 'premature_planning_rejected': True,
              'actual_archived_truncated_geometry_validated': geometry,
              'historical_unrefined_GraspCL_rejected': True,
              'physical_grasps': 0, 'models_loaded': 0,
              'timing_dispatch_uses_explicit_component_config': True}
    common.save(directory/'verification.json', result)
    return result


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir();(root/EPOCH).mkdir()
    verification = preflight(root)
    assert verification['passed']
    prior = common.read(PRIOR/'registration.json')
    inputs = Path(prior['inputs'])
    assert common.read(PRIOR/'bank_provenance_audit.json')['passed']
    b = common.read(PRIOR/'banks/no_experience/bank_complete.json')
    c = common.read(PRIOR/'banks/experience50/bank_complete.json')
    assert b['online_points'] == 0 and c['positive_demonstrations'] == 39 and c['negative_feedback'] == 2
    manifest = common.read(inputs/'test/manifest.json')
    assert len(manifest['schedule']) == 100 and manifest['admission_failed'] == 0
    assert len(list((PRIOR/'shared_test_inputs').glob('*/manifest.json'))) == 100
    sources = dict(prior['source_hashes'])
    sources.update(common.read(PRIOR/'A_zerograsp/registration.json')['source_hashes'])
    for p in [SELF, Path(protocol.__file__), SIMULATOR, Path(shared.__file__), Path(resident.__file__),
              CODE/'graspcl_unseen_services_quiet_20261119.py', CODE/'graspcl_unseen_shared_services_20261119.py',
              CODE/'graspcl_unseen_simulator_pinned_20261119.py', CODE/'graspcl_unseen_simulator_20261119.py',
              CODE/'arx_quiet_diagnostics_20261119.py', CODE/'graspcl_pipeline.py',
              root/'preflight/verification.json', PRIOR/'bank_provenance_audit.json']:
        sources[str(p)] = common.sha(p)
    for e in manifest['schedule']:
        p = PRIOR/'shared_test_inputs'/e['id']/'manifest.json'
        sources[str(p)] = common.sha(p)
    reg = deepcopy(prior)
    reg.update(created_at=bench.core.now(), source_hashes=sources, total_slots=300,
               prior_output=str(PRIOR), prior_registration_sha256=common.sha(PRIOR/'registration.json'),
               experience_collection_reexecutions=0, test_arms=ARMS,
               test_observation_cache=str(PRIOR/'shared_test_inputs'),
               test_design_scope='Corrected same-scene revalidation, not a newly sampled blind test; no outcomes are training inputs.',
               protocol=protocol.PROTOCOL,
               changes=['Explicit configured RPC identity in timing and refinement routing',
                        'Mandatory insertion truncation for A/B/C',
                        'Mandatory refined-pose rescoring for B/C',
                        'Mandatory recall TCP40 to TCP43 conversion',
                        'Per-case processing and geometry checks before planning'],
               runtime_backend_gpu=shared.PLANNER_GPU,
               runtime_scene_gpus=[shared.PLANNER_GPU, shared.PLANNER_GPU],
               source_experience_bank=str(PRIOR/'banks/experience50'),
               source_experience_labels={'positive': 39, 'negative': 2},
               test_feedback_updates=False, required_components_by_arm={
                   ARMS[0]: ['zerograsp', 'curobo'], ARMS[1]: ['zerograsp', 'graspcl', 'curobo'],
                   ARMS[2]: ['zerograsp', 'graspcl', 'curobo']})
    common.save(root/'registration.json', reg)
    guard(reg)
    for name, bank in [(ARMS[0], None), (ARMS[1], PRIOR/'banks/no_experience'), (ARMS[2], PRIOR/'banks/experience50')]:
        group, arm = bench.prepare_group(root, name, inputs/'test', bank)
        for directory in [group, arm]:
            path = directory/'registration.json';r = common.read(path)
            r.update(shared_input_role='consumer', shared_input_cache=str(PRIOR/'shared_test_inputs'),
                     uniform_processing_protocol=protocol.PROTOCOL, component_startup_wait_seconds=1800,
                     model_gpu_by_component={'zerograsp': shared.NEURAL_GPU, 'curobo': shared.PLANNER_GPU, 'graspcl': 'CPU'},
                     required_refinement_receipt='unified_protocol_receipt.json',
                     test_design_scope=reg['test_design_scope'])
            common.save(path, r)
    for n in ['graspcl_unified_bench_20261120.py', 'graspcl_unified_protocol_20261120.py', 'graspcl_unified_simulator_20261120.py']:
        shutil.copy2(CODE/n, root/'source'/n)
    common.save(root/EPOCH/'registration.json', {'created_at': bench.core.now(), 'registered_slots': 300,
                'original_registration_sha256': common.sha(root/'registration.json'),
                'user_authorization': '统一一下，重新验证', 'source_experience_reexecuted': 0,
                'model_lifecycle': 'One warmed neural/planner pair across all arms; frozen B/C scorers load once per arm.',
                'unknown_outcome_policy': 'No replay of allocated attempts. Preserve interrupted evidence and resume only unclaimed slots.'})
    aggregate(root, 'prepared')
    print('PREPARED', root, '300 fresh execution slots, same frozen 100 inputs, 39 positive + 2 negative source labels', flush=True)


class Manager(shared.Manager):
    def launch(self, kind, group, reg):
        root = group/'resident'
        key = hashlib.sha256((str(group)+self.epoch+kind).encode()).hexdigest()[:14]
        sock = CODE/'tmp'/('unified20_'+key+'.sock')
        assert not sock.exists()
        python = canonical_path(CODE.parent/{'zerograsp': 'tmp/arx-fullchain/env/bin/python',
                              'curobo': 'curobo/.venv/bin/python',
                              'graspcl': 'tmp/arx-graspcl-env/bin/python'}[kind])
        gpu = shared.NEURAL_GPU if kind == 'zerograsp' else shared.PLANNER_GPU
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, ARX_NEURAL_GPU=shared.NEURAL_GPU,
                   ARX_PLANNER_GPU=shared.PLANNER_GPU, ARX_SIM_GPU=shared.PLANNER_GPU, LILO_GPU=shared.PLANNER_GPU,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH', None)
        command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'unified-frozen-service',
                   str(CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'), str(python),
                   str(CODE/'graspcl_unseen_services_quiet_20261119.py'), '--kind', kind,
                   '--root', str(root), '--socket', str(sock), '--warmup-trial', reg['warmup_trial']]
        if kind == 'graspcl':
            env['CUDA_VISIBLE_DEVICES'] = ''
            command.extend(['--bank', reg['bank'], '--policy-registration', str(group/'registration.json')])
        stream = (root/(kind+'.log')).open('x');self.streams.append(stream)
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {'process': process, 'socket': sock, 'owner': root, 'kind': kind, 'group': group.name}
        self.all_processes[group.name+':'+kind] = record
        health = wait_ready(sock, root/(kind+'_ready.json'), process, kind, timeout=1800)
        print('UNIFIED RESIDENT READY', kind, process.pid, health['startup_seconds'], flush=True)
        return record


class Simulators(resident.Simulators):
    def start(self):
        for i, gpu in enumerate(bench.GPUS):
            key = hashlib.sha256((str(self.root)+EPOCH+str(i)).encode()).hexdigest()[:14]
            sock = CODE/'tmp'/('unifiedsim20_'+key+'.sock');assert not sock.exists()
            directory = self.root/EPOCH/'simulation'/str(i)
            stream = (self.root/EPOCH/('simulator_'+str(i)+'.log')).open('x');self.streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MANISKILL_GPU_UUID=gpu, ARX_SIM_GPU=gpu,
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            self.children[i] = subprocess.Popen(['bash', str(CODE/'run.sh'), str(SIMULATOR),
                '--root', str(directory), '--socket', str(sock), '--owner-pid', str(os.getpid())],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.sockets[i] = sock
        health = {str(i): wait_ready(self.sockets[i], self.root/EPOCH/'simulation'/str(i)/'ready.json',
                                    p, 'maniskill_simulation', timeout=1800) for i, p in self.children.items()}
        assert all(v['runtime_import_count'] == 1 for v in health.values())
        common.save(self.root/EPOCH/'simulation_all_ready.json', {'workers': health,
                    'all_libraries_loaded_before_first_grasp': True, 'grasp_slots_claimed': 0})


def check_group(group):
    reg = common.read(group/'registration.json');arm = group/reg['arms'][0][0]
    outcomes = [];processing = Counter();origins = Counter()
    for e in reg['schedule']:
        trial = arm/'trials'/e['id'];result = common.read(trial/'result.json')
        assert common.read(trial/'independent_evaluation.json')['passed']
        receipt = protocol.validate_pool(trial, {**reg, 'pipeline_profile': 'zerograsp_200' if group.name == ARMS[0] else 'graspcl_200', 'orientation_mode': 'fixed_orientation'})
        recorded = common.read(trial/'unified_protocol_receipt.json')
        assert recorded['passed'] and recorded['refinement_calls'] == 1
        assert recorded['candidate_pool_sha256'] == receipt['candidate_pool_sha256']
        assert (trial/'zg_outputs/grasp_top10.json').exists()
        plan = common.read(trial/'plan_result.json')
        if plan.get('selected'):
            assert recorded['initial_planning_gate_checks'] >= 1
            origins[plan['selected']['candidate_metadata']['origin']] += 1
        for key in ['input_candidates', 'retained_candidates', 'retracted_candidates', 'retained_recall_candidates', 'recall_tcp43_shift_verified']:
            processing[key] += receipt[key]
        outcomes.append(result)
    result = {'passed': True, 'registered': 100, 'completed': len(outcomes),
              'successes': sum(bool(r['success']) for r in outcomes),
              'planning_passes': sum(bool(r['planning_success']) for r in outcomes),
              'failure_reasons': dict(Counter(r['reason'] for r in outcomes if not r['success'])),
              'refinement_cases': len(outcomes), 'refined_rescore_cases': 0 if group.name == ARMS[0] else len(outcomes),
              'processing': dict(processing), 'selected_origins': dict(origins)}
    common.save(group/'uniform_processing_verification.json', result)
    return result


def run(root):
    reg = common.read(root/'registration.json');guard(reg)
    assert common.read(root/'preflight/verification.json')['passed']
    assert common.sha(root/'registration.json') == common.read(root/EPOCH/'registration.json')['original_registration_sha256']
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root/EPOCH/'started.json').open('x') as stream:
            stream.write(bench.core.now()+'\n')
        bench.GPUS = [shared.PLANNER_GPU, shared.PLANNER_GPU];bench.GPU0 = shared.PLANNER_GPU
        resident.EPOCH = EPOCH;resident.aggregate = aggregate;bench.aggregate = aggregate
        bench.Backends = shared.Backends;bench.Campaign = resident.Campaign
        resident.POOL = Simulators(root);shared.MANAGER = Manager(root, EPOCH)
        started = time.perf_counter();success = False;groups = {}
        try:
            aggregate(root, 'preloading_simulation');resident.POOL.start()
            for name in ARMS:
                aggregate(root, 'testing_'+name)
                group = root/name;arm = group/common.read(group/'registration.json')['arms'][0][0]
                bench.run_group(group, arm)
                groups[name] = check_group(group)
            bench.analyze(root)
            guard(reg)
            for name in ARMS:
                guard(common.read(root/name/'registration.json'))
            common.save(root/'final_verification.json', {'passed': True, 'registered_and_completed': 300,
                        'uniform_processing_verified': True, 'all_arms_truncated': True,
                        'graspcl_arms_refined_rescored': True, 'recall_tcp_conversion_verified': True,
                        'experience_bank_frozen': True, 'test_feedback_updates': 0,
                        'source_experience_reexecuted': 0})
            success = True
        except BaseException:
            common.save(root/EPOCH/'halt.json', {'time': bench.core.now(), 'error': traceback.format_exc(), 'replay_allowed': False})
            aggregate(root, 'halted')
            raise
        finally:
            aggregate(root, 'finalizing_services' if success else 'halting_services')
            resident.POOL.stop();shared.MANAGER.stop()
            simulation = common.read(root/EPOCH/'simulation_stopped.json')
            models = common.read(root/EPOCH/'model_services_stopped.json')
            common.save(root/'suite_timing.json', {'wall_seconds': time.perf_counter()-started,
                        'all_owned_services_exited': models['all_owned_model_services_exited'],
                        'all_owned_simulators_exited': simulation['all_owned_simulators_exited']})
            if success:
                final = common.read(root/'final_verification.json')
                final.update(all_services_exited=models['all_owned_model_services_exited'],
                             all_simulators_exited=simulation['all_owned_simulators_exited'])
                common.save(root/'final_verification.json', final)
                common.save(root/'final_summary.json', {'groups': groups, 'analysis': common.read(root/'analysis.json'),
                            'wall_seconds': time.perf_counter()-started,
                            'uniform_processing_verified': True, 'same_scene_revalidation': True})
                aggregate(root, 'complete')
            else:
                aggregate(root, 'halted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)(args.output.resolve())
