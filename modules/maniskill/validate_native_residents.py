"""Zero physical attempts: lifecycle, measured-state and inference RPC checks."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np

import graspcl_pipeline as common
from graspcl_execute import guard
from postprocess_bench import Backends
from benchmark_lifecycle import require_ready, load_counts
from resident_backend import rpc
from postprocess_pipeline import processed_rows, canonical_candidates

CODE = Path(__file__).resolve().parent


def run(out, reference):
    reg = common.read(out / 'registration.json')
    guard(reg)
    services = Backends(out)
    budget_used = 0
    checks, timing = [], []

    def query(kind, directory, **flags):
        nonlocal budget_used
        budget_used += 1
        assert budget_used <= reg['model_request_budget']
        message = {'directory': str(directory), **flags}
        if kind == 'graspcl':
            message['policy'] = reg['candidate_policy']
        started = time.perf_counter()
        reply = rpc(Path(services.config['sockets'][kind]), message, timeout=600)
        elapsed = time.perf_counter()-started
        common.save(directory / (kind + ('_'+next(iter(flags)) if flags else '') + '_rpc.json'), reply)
        assert reply['status'] == 'ok', reply
        return elapsed

    def copy_case(source, directory, network=True):
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source / 'camera', directory / 'camera')
        shutil.copy2(source / 'state.json', directory / 'state.json')
        if network:
            shutil.copytree(source / 'network/raw_outputs', directory / 'network/raw_outputs')
            shutil.copy2(source / 'raw_grasps.npy', directory / 'raw_grasps.npy')

    def verify_candidates(directory):
        rows, _ = processed_rows(directory)
        pool = common.read(directory / 'graspcl/candidate_pool.json')
        candidates = [c for c in pool['candidates'] if c['metadata']['origin'] == 'zerograsp']
        assert len(candidates) == len(rows)
        assert pool['custom_geometry_filters'] == [] and pool['zero_top_k_cap'] is None
        assert pool['additional_finger_symmetry_candidates'] is False
        expected = canonical_candidates(rows, np.array(pool['camera']))
        assert sorted(c['metadata']['postprocessed_row_index'] for c in candidates) == list(range(len(rows)))
        for candidate in candidates:
            other = expected[candidate['metadata']['postprocessed_row_index']]
            np.testing.assert_array_equal(candidate['pose'], other['pose'])
            assert candidate['width'] == other['width']
        bind = common.read(directory / 'plan/resident_rebind.json')
        assert bind['runtime_robot_configuration_rebuilds'] == 0
        assert bind['model_load_count'] == 1
        checks.append({'case': directory.name, 'group': directory.parent.name,
            'native_rows_all_consumed_once': len(rows), 'runtime_robot_rebuilds': 0,
            'fast_bind_seconds': bind['fast_bind_wall_seconds']})

    try:
        try:
            require_ready(out / 'resident', reg['required_components'])
        except FileNotFoundError:
            checks.append({'no_admission_before_startup': True})
        else:
            raise AssertionError('Admission passed before startup')
        services.start()
        require_ready(out / 'resident', reg['required_components'])
        try:
            require_ready(out / 'resident', reg['required_components'] + ['undeclared_test_component'])
        except RuntimeError:
            checks.append({'missing_component_blocks_admission': True})
        else:
            raise AssertionError('Incomplete component set admitted')
        ids = ['B0031_S1_open_008_pudding_box', 'B0069_S3_sparse_014_lemon', 'B0083_S5_occluded_016_pear']
        for group, fresh in [('fixed_input', False), ('fresh_inference', True)]:
            for slot in ids:
                source = reference / 'fresh_v2' / slot
                directory = out / group / slot
                copy_case(source, directory, network=not fresh)
                z = query('zerograsp', directory) if fresh else None
                cl = query('graspcl', directory)
                cu = query('curobo', directory, selector='graspcl')
                verify_candidates(directory)
                plan = common.read(directory / 'plan_result.json')
                prior = common.read(source / 'plan_result.json')
                if not fresh:
                    assert plan['success'] == prior['success']
                timing.append({'id': slot, 'group': group, 'zerograsp_seconds': z,
                    'graspcl_seconds': cl, 'curobo_seconds': cu,
                    'model_chain_seconds': z+cl+cu if fresh else None,
                    'planning_success': plan['success'], 'physical_attempts': 0})
                require_ready(out / 'resident', reg['required_components'])

        # A-B-A: a positive bank source must not leak, or cumulatively delete memory.
        bank_root = CODE / 'outputs/arx_baseline_20261116_parallel_01/trials'
        positive = 'B0075_S3_sparse_015_peach'
        heldout = ids[1]
        exclusions = []
        for number, slot in enumerate([positive, heldout, positive]):
            directory = out / ('exclusion_' + str(number)) / slot
            source = bank_root / slot if slot == positive else reference / 'fresh_v2' / slot
            copy_case(source, directory)
            query('graspcl', directory)
            exclusions.append(common.read(directory / 'self_case_exclusion.json'))
        assert exclusions[0] == exclusions[2]
        assert all(x['before']['feedback'] == 25 and x['before']['object'] == 12 for x in exclusions)
        assert exclusions[0]['after']['feedback'] == 24 and exclusions[0]['after']['object'] < 12
        assert exclusions[1]['after'] == exclusions[1]['before']
        checks.append({'positive_source_exclusion_and_A_B_A_restore': True, 'receipts': exclusions})

        # Exercise observation, measured approach, measured closure/lift roles too.
        source = CODE / 'outputs/arx_zerograsp_graspcl_physical_20261117_01/trials' / heldout
        directory = out / 'measured_modes' / heldout
        directory.mkdir(parents=True, exist_ok=False)
        for name in ['state.json', 'pregrasp_state.json', 'closed_state.json', 'plan_result.json']:
            shutil.copy2(source / name, directory / name)
        state = common.read(directory / 'state.json')
        common.save(directory / 'observation_request.json', {'q': [state['qpos'][state['joint_names'].index('right_joint'+str(i))] for i in range(21, 27)]})
        for flag, name in [('observe', 'observation_plan'), ('approach', 'approach_plan'), ('lift', 'lift_plan')]:
            elapsed = query('curobo', directory, **{flag: True})
            result = common.read(directory / (name+'_result.json'))
            bind = common.read(directory / name / 'resident_rebind.json')
            assert bind['runtime_robot_configuration_rebuilds'] == 0
            checks.append({'mode': flag, 'rpc_seconds': elapsed, 'planning_success': result['success'],
                'measured_state_refreshed': bind['measured_finger_joints'], 'runtime_robot_rebuilds': 0})

        guard(reg)
        health = {k: rpc(Path(services.config['sockets'][k]), {'command': 'health'}) for k in reg['required_components']}
        common.save(out / 'final_health.json', health)
        common.save(out / 'timing.json', {'scope': 'Saved matched RGB-D, model-only RPC durations; no queue or simulation/physical execution. Startup and warmup recorded separately.', 'results': timing})
        common.save(out / 'verification.json', {'passed': True, 'physical_attempts': 0,
            'model_requests': budget_used, 'checks': checks,
            'validated_source_sha256': {str(CODE / name): common.sha(CODE / name) for name in [
                'postprocess_pipeline.py', 'postprocess_planner.py', 'postprocess_backend.py',
                'fast_resident_planner.py', 'native_resident_services.py', 'benchmark_lifecycle.py', 'postprocess_bench.py']},
            'inherited_native_output_checks': str(reference / 'verification.json')})
    except BaseException:
        import traceback
        common.save(out / 'validation_error.json', {'error': traceback.format_exc(), 'model_requests': budget_used})
        raise
    finally:
        services.stop()
        common.save(out / 'cleanup.json', {'all_owned_services_exited': all(p.poll() is not None for p in services.children.values()),
            'exit_codes': {k: p.returncode for k, p in services.children.items()}})
    print('VALIDATION PASSED', budget_used, timing, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.reference.resolve())
