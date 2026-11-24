"""Native ZeroGrasp postprocessing phase within the original 200 ARX slots."""
from arx_project.layout import canonical_path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import graspcl_batch as batch
import graspcl_pipeline as common
import resident_bench as resident
import postprocess_pipeline as post
from graspcl_execute import guard
from resident_backend import rpc
from benchmark_lifecycle import mark_ready, require_ready

PROTOCOL = 'zerograsp_native_post_no_custom_geometry_v1'


def install(render=False):
    common.infer = post.infer
    common.write_inference = post.write_inference
    batch.__file__ = str(Path(__file__).resolve())
    if render:
        import grasp_visuals
        if not hasattr(grasp_visuals, '_native_post_original_render'):
            grasp_visuals._native_post_original_render = grasp_visuals.render
        grasp_visuals.render = post.render


def infer_resident(trial, campaign):
    """Standalone per-case entry obeys the same gate; never cold-load GraspCL."""
    reg = common.read(campaign / 'registration.json')
    guard(reg)
    require_ready(campaign / 'resident', reg['required_components'])
    config = common.read(campaign / 'resident/backend_config.json')
    try:
        response = rpc(Path(config['sockets']['graspcl']),
            {'directory': str(trial), 'policy': reg['candidate_policy']}, timeout=600)
        common.save(trial / 'graspcl_inference.log', response)
        if response['status'] != 'ok':
            raise RuntimeError('resident_backend_failed: ' + response.get('error', 'unknown'))
        return response['returncode']
    except Exception as error:
        common.save(trial / 'resident_request_error.json', {'error': repr(error), 'kind': 'graspcl'})
        raise


def protocol_summary(out, reg):
    old_ids = set(reg['prior_protocol_completed_ids'])
    result = {'scope': 'Mixed protocols: prior raw/custom-geometry phase and current native-post/no-custom-geometry phase are reported separately',
              'original_budget': 200, 'current_protocol': PROTOCOL,
              'phases': {name: {'registered': count, 'completed': 0, 'physical_attempts': 0, 'successes': 0}
                         for name, count in [('prior_raw_custom_geometry', len(old_ids)), (PROTOCOL, 200-len(old_ids))]}}
    for e in reg['schedule']:
        claim = out / 'claims' / (e['id'] + '.json')
        if not claim.exists() or common.read(claim)['state'] != 'complete':
            continue
        trial = out / 'trials' / e['id']
        if not common.read(trial / 'independent_evaluation.json')['passed']:
            raise RuntimeError('Independent evaluation disagrees')
        row = common.read(trial / 'result.json')
        phase = 'prior_raw_custom_geometry' if e['id'] in old_ids else PROTOCOL
        result['phases'][phase]['completed'] += 1
        result['phases'][phase]['physical_attempts'] += int(row['physical_attempt'])
        result['phases'][phase]['successes'] += int(row['success'])
    common.save(out / 'protocol_summary.json', result)
    return result


def prepare(out, previous, validation):
    check = common.read(validation / 'verification.json')
    if not check['passed']:
        raise RuntimeError('Native-postprocessing validation is incomplete')
    for path, digest in check['validated_source_sha256'].items():
        if common.sha(path) != digest:
            raise RuntimeError('Validated source changed: ' + path)
    old = common.read(previous / 'registration.json')
    guard(old)
    if common.read(previous / 'progress.json')['phase'] != 'stopped' or not (previous / 'postprocessing_review_request.json').exists():
        raise RuntimeError('Previous phase must finish its active slots under the protocol review request')
    state = subprocess.check_output(['systemctl', '--user', 'show', 'maniskill-arx-graspcl-200-resident-20261117-02.service', '--property=ActiveState', '--value'], text=True).strip()
    if state != 'inactive':
        raise RuntimeError('Old phase service is still active')
    for kind in ['curobo', 'zerograsp']:
        pid = common.read(previous / 'resident' / (kind + '_ready.json'))['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Old resident process remains: ' + str(pid))
    completed = {}
    for path in (previous / 'claims').glob('*.json'):
        claim = common.read(path)
        trial = previous / 'trials' / path.stem
        if claim['state'] != 'complete' or not common.read(trial / 'independent_evaluation.json')['passed']:
            raise RuntimeError('Started/unknown slot must not be replayed: ' + path.stem)
        if common.sha(trial / 'result.json') != claim['result_sha256']:
            raise RuntimeError('Old result changed')
        completed[path.stem] = claim
    if any(t.name not in completed for t in (previous / 'trials').iterdir()):
        raise RuntimeError('Old phase contains an unknown trial')
    out.mkdir(parents=True, exist_ok=False)
    for name in ['trials', 'claims', 'entries', 'workers', 'backend_requests', 'reference', 'source']:
        (out / name).mkdir()
    for slot, claim in completed.items():
        (out / 'trials' / slot).symlink_to(previous / 'trials' / slot, target_is_directory=True)
        common.save(out / 'claims' / (slot + '.json'), claim)
    shutil.copy2(previous / 'reference/common_initial_conditions.json', out / 'reference/common_initial_conditions.json')
    shutil.copy2(previous / 'storage_policy.json', out / 'storage_policy.json')
    reg = deepcopy(old)
    reg.update(previous_output=str(previous), previous_registration_sha256=common.sha(previous / 'registration.json'),
        candidate_protocol=PROTOCOL, prior_protocol_completed_ids=list(completed),
        current_protocol_slot_ids=[e['id'] for e in old['schedule'] if e['id'] not in completed],
        postprocess_validation=str(validation),
        chain='right-wrist RGB-D -> native ZeroGrasp width/depth correction, collision filtering and NMS -> all processed proposals + demonstration recall -> GraspCL -> frozen ARX cuRobo -> ARX execution',
        previous_candidate_policy=old['candidate_policy'],
        candidate_policy={'zero_input': 'network/raw_outputs/*.grasp.npy', 'zero_top_k': None,
            'add_finger_symmetry': False, 'custom_geometry_filters': [],
            'hardware_opening_limit': 'sum of frozen right finger joint upper limits from state.json',
            'recall_max_matches_per_template': old['candidate_policy']['recall_max_matches_per_template'],
            'patch_time_ms': old['candidate_policy']['patch_time_ms']},
        selection_change='Remove all handwritten approach, target-center, target-cloud-distance and height-shift criteria. Use every native postprocessed ZeroGrasp pose with corrected width/depth, no raw top256 cap or added symmetry. Only input validity, immutable hardware opening, and original cuRobo IK/collision/path/optimizer gates remain.',
        scope='Original cumulative 200-slot allocation, with a user-authorized algorithm change after prior completed slots. Never interpret cumulative successes as 200 tests of the new protocol; use protocol_summary.json and comparison_by_protocol.json.',
        required_components=['zerograsp', 'graspcl', 'curobo'],
        runtime_change='All required components load and warm up before slot admission; ZeroGrasp, GraspCL encoder/frozen bank and open/closed cuRobo remain resident. Static robot configuration is not reconstructed per request.',
        backend_lifecycle='Load, warm up and health-check every required component before the first claim; reuse until batch ends; shut down all owned services in finally. No cold fallback or replay on service failure.',
        native_postprocessing='Exact existing demo.py behavior, including its original all-colliding fallback and NMS settings; generic postprocessing gripper is followed by actual frozen ARX model validation')
    for name in ['postprocess_pipeline.py', 'postprocess_planner.py', 'postprocess_backend.py', 'postprocess_bench.py',
                 'native_resident_services.py', 'fast_resident_planner.py', 'benchmark_lifecycle.py']:
        path = CODE / name
        reg['source_hashes'][str(path.resolve())] = common.sha(path)
        shutil.copy2(path, out / 'source' / name)
    reg['source_hashes'][str(out / 'reference/common_initial_conditions.json')] = common.sha(out / 'reference/common_initial_conditions.json')
    common.save(out / 'registration.json', reg)
    common.save(out / 'continuation.json', {'created_at': datetime.now().astimezone().isoformat(),
        'original_budget': 200, 'prior_protocol_completed': len(completed), 'remaining_new_protocol': 200-len(completed),
        'old_phase': str(previous), 'old_STOP_preserved': True, 'unknown_slots': [], 'algorithm_change': True,
        'carried_result_sha256': {k: v['result_sha256'] for k, v in completed.items()}})
    _, baseline = batch.original_modules()
    baseline.summary(out, reg['schedule'])
    protocol_summary(out, reg)
    common.save(out / 'progress.json', {'phase': 'prepared', 'completed': len(completed), 'total': 200,
        'current_protocol': PROTOCOL, 'current_protocol_completed': 0, 'current_protocol_registered': 200-len(completed)})
    print('NATIVE POST PHASE PREPARED', len(completed), 'prior slots;', 200-len(completed), 'new-protocol slots', flush=True)


class Backends(resident.Backends):
    def start(self):
        root = self.campaign / 'resident'
        root.mkdir(exist_ok=False)
        key = hashlib.sha256(str(self.campaign).encode()).hexdigest()[:12]
        sockets = CODE / 'tmp' / ('post_' + key)
        sockets.mkdir(mode=0o700, exist_ok=False)
        warmup = CODE / 'outputs/arx_zerograsp_graspcl_physical_20261117_01/trials/B0069_S3_sparse_014_lemon'
        self.config = {'gpu': batch.BACKEND, 'owner_pid': os.getpid(), 'candidate_protocol': PROTOCOL,
            'required_components': self.reg['required_components'],
            'sockets': {k: str(sockets / (k + '.sock')) for k in self.reg['required_components']}}
        common.save(root / 'backend_config.json', self.config)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=batch.BACKEND, ARX_NEURAL_GPU=batch.BACKEND,
            ARX_PLANNER_GPU=batch.BACKEND, ARX_SIM_GPU=batch.BACKEND, LILO_GPU=batch.BACKEND,
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH', None)
        for kind in self.reg['required_components']:
            python = canonical_path(batch.ROOT / {'curobo': 'curobo/.venv/bin/python',
                'zerograsp': 'tmp/arx-fullchain/env/bin/python', 'graspcl': 'tmp/arx-graspcl-env/bin/python'}[kind])
            command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'native-post-worker',
                str(batch.ROOT / 'arx_core/scripts/curobo_arx/environment.sh'), str(python),
                str(CODE / 'native_resident_services.py'), '--kind', kind, '--socket', self.config['sockets'][kind],
                '--root', str(root), '--warmup-trial', str(warmup)]
            child_env = dict(env)
            if kind == 'graspcl':
                child_env['CUDA_VISIBLE_DEVICES'] = ''
                command.extend(['--bank', self.reg['bank'], '--policy-registration', str(self.campaign / 'registration.json')])
            stream = (root / (kind + '.log')).open('x')
            self.streams.append(stream)
            process = subprocess.Popen(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
            self.children[kind] = process
            deadline = time.monotonic() + 240
            while not Path(self.config['sockets'][kind]).exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Native-post resident startup failed: ' + kind)
                time.sleep(.2)
            health = rpc(Path(self.config['sockets'][kind]), {'command': 'health'})
            if health['status'] != 'ready':
                raise RuntimeError('Native-post resident is not ready')
            print('RESIDENT READY', kind, health['pid'], health['startup_seconds'], flush=True)
        mark_ready(root, self.reg['required_components'], self.config, self.children)


class Campaign(resident.ResidentCampaign):
    def next_entry(self, worker, gpu):
        require_ready(self.out / 'resident', self.reg['required_components'])
        entry, claim = super().next_entry(worker, gpu)
        if entry is not None:
            if entry['id'] not in self.reg['current_protocol_slot_ids']:
                raise RuntimeError('Prior protocol slot must never be replayed')
            entry['candidate_protocol'] = PROTOCOL
            receipt = self.out / 'entries' / (entry['id'] + '.json')
            common.save(receipt, entry)
            path = self.out / 'entries' / (entry['id'] + '_binding.json')
            binding = common.read(path)
            binding.update(entry_sha256=common.sha(receipt), candidate_protocol=PROTOCOL)
            common.save(path, binding)
        return entry, claim

    def progress(self, worker, **state):
        with self.mutex:
            self.states[worker] = {**self.states.get(worker, {}), **state, 'updated_at': batch.now()}
            common.save(self.out / 'workers' / (worker + '.json'), self.states[worker])
            _, baseline = batch.original_modules()
            summary = baseline.summary(self.out, self.reg['schedule'])
            phases = protocol_summary(self.out, self.reg)
            summary['protocol_warning'] = phases['scope']
            summary['by_protocol'] = phases['phases']
            common.save(self.out / 'summary.json', summary)
            common.save(self.out / 'progress.json', {'phase': 'draining_after_error' if self.failed.is_set() else 'running',
                'completed': summary['completed'], 'successes': summary['successes'], 'total': 200,
                'current_protocol': PROTOCOL, 'current_protocol_completed': phases['phases'][PROTOCOL]['completed'],
                'current_protocol_registered': len(self.reg['current_protocol_slot_ids']),
                'scene_workers': 2, 'backend_concurrency': 1, 'workers': self.states, 'updated_at': batch.now()})


def execute(trial, entry_path):
    install(render=True)
    run, baseline = batch.original_modules()
    entry = common.read(entry_path)
    reg_path = Path(entry['campaign_registration'])
    campaign = reg_path.parent
    reg = common.read(reg_path)
    guard(reg)
    require_ready(campaign / 'resident', reg['required_components'])
    config = common.read(campaign / 'resident/backend_config.json')
    binding = common.read(campaign / 'entries' / (entry['id'] + '_binding.json'))
    if common.sha(entry_path) != binding['entry_sha256']:
        raise RuntimeError('Bound trial entry changed')
    for path, digest in binding['source_evidence_sha256'].items():
        if common.sha(path) != digest:
            raise RuntimeError('Historical scene evidence changed: ' + path)
    initial = common.read(campaign / 'reference/common_initial_conditions.json')
    historical = Path(entry['historical_trial'])
    historical_initial = common.read(historical / 'initial_conditions.json') if (historical / 'initial_conditions.json').is_file() else None

    def request(kind, directory, message, log, timeout):
        try:
            response = rpc(Path(config['sockets'][kind]), message, timeout=timeout)
            common.save(log, response)
            if response['status'] != 'ok':
                raise RuntimeError('resident_backend_failed: ' + response.get('error', 'unknown'))
            return response['returncode']
        except Exception as error:
            common.save(directory / 'resident_request_error.json', {'error': repr(error), 'kind': kind})
            raise

    def worker(script, directory, gpu, log, **flags):
        guard_receipt = common.read(directory / 'source_guard.json')
        if common.sha(guard_receipt['registration']) != guard_receipt['sha256']:
            raise RuntimeError('Campaign registration changed')
        guard(reg)
        require_ready(campaign / 'resident', reg['required_components'])
        if flags.get('observe'):
            state = common.read(directory / 'state.json')
            expected = historical_initial or initial
            error = float(np.max(np.abs(np.array(state['qpos'])-np.array(expected['initial_q']))))
            if error >= .005:
                raise RuntimeError('Declared historical robot reset differs')
            common.save(directory / 'reset_preflight.json', {'joint_error_rad': error,
                'historical_runtime_available': historical_initial is not None,
                'reference': str(historical / 'initial_conditions.json') if historical_initial else reg['common_reset_source']})
        kind = 'zerograsp' if script == 'neural.py' else 'curobo'
        message = {'directory': str(directory)}
        if kind == 'curobo':
            message.update(flags, selector='graspcl')
        receipt_path = campaign / 'backend_requests' / (uuid.uuid4().hex + '.json')
        receipt = {'slot': entry['id'], 'script': script, 'kind': kind, 'flags': flags,
            'gpu': gpu, 'state': 'queued', 'queued_at': batch.now(), 'pid': os.getpid()}
        common.save(receipt_path, receipt)
        with batch.locked(reg['backend_lock']):
            receipt.update(state='running', started_at=batch.now())
            common.save(receipt_path, receipt)
            try:
                status = request(kind, directory, message, log, 900)
                receipt.update(state='complete', returncode=status)
            except BaseException:
                receipt.update(state='failed', error=traceback.format_exc())
                raise
            finally:
                receipt['finished_at'] = batch.now()
                common.save(receipt_path, receipt)
        if kind == 'zerograsp' and status == 0:
            request('graspcl', directory, {'directory': str(directory), 'policy': reg['candidate_policy']},
                directory / 'graspcl_inference.log', 600)
            from grasp_visuals import render
            render(directory / 'graspcl', directory / 'zg_outputs')
        return status

    run.worker = worker
    run.trial(entry, trial, batch.BACKEND, batch.BACKEND, 'right_wrist')
    if (trial / 'resident_request_error.json').exists():
        raise RuntimeError('Resident backend error; preserve slot and stop without replay')
    result = common.read(trial / 'result.json')
    infrastructure_words = ['out of memory', 'vk_error', 'resource descriptor',
                            'frozen source', 'declared historical robot reset differs']
    if result['reason'] == 'zerograsp_failed' or any(word in result.get('error', '').lower() for word in infrastructure_words):
        raise RuntimeError('Infrastructure/integrity failure preserved: ' + entry['id'])
    if (trial / 'graspcl_inference.log').exists() and not (trial / 'zg_outputs/grasp_top10.json').exists():
        raise RuntimeError('GraspCL inference/visualization integration failed; preserve slot')
    if (trial / 'initial_conditions.json').exists():
        current = common.read(trial / 'initial_conditions.json')
        reference = historical_initial or initial
        comparison = {'historical_runtime_available': historical_initial is not None,
            'base_pose_equal': reference['base_pose'] == current['base_pose'],
            'initial_q_max_difference': float(np.max(np.abs(np.array(reference['initial_q'])-np.array(current['initial_q'])))),
            'spawn_equal': historical_initial['spawn'] == current['spawn'] if historical_initial else None,
            'observation_equal_to_source_admission': entry['observation'] == common.read(entry['source_admission'])['entry']['observation']}
        common.save(trial / 'reset_comparison.json', comparison)
        if not comparison['base_pose_equal'] or comparison['initial_q_max_difference'] != 0. or comparison['spawn_equal'] is False or not comparison['observation_equal_to_source_admission']:
            raise RuntimeError('Matched reset check failed')
    baseline.verify_trial(trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'execute', 'infer'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    install()
    if args.stage == 'prepare':
        prepare(out, args.previous.resolve(), args.validation.resolve())
    elif args.stage == 'infer':
        infer_resident(out, args.campaign.resolve())
    elif args.stage == 'execute':
        execute(out, args.entry.resolve())
    else:
        guard(common.read(out / 'registration.json'))
        backends = Backends(out)
        try:
            backends.start()
            Campaign(out).start()
        finally:
            backends.stop()
            registration = common.read(out / 'registration.json')
            phases = protocol_summary(out, registration)
            if (out / 'summary.json').exists():
                summary = common.read(out / 'summary.json')
                summary.update(protocol_warning=phases['scope'], by_protocol=phases['phases'])
                common.save(out / 'summary.json', summary)
            if (out / 'progress.json').exists():
                progress = common.read(out / 'progress.json')
                progress.update(current_protocol=PROTOCOL,
                    current_protocol_completed=phases['phases'][PROTOCOL]['completed'],
                    current_protocol_registered=len(registration['current_protocol_slot_ids']))
                common.save(out / 'progress.json', progress)
