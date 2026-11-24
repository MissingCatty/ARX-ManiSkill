"""Explicit component routing and mandatory refinement gates for new runs only."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import types
from unittest.mock import patch

import numpy as np
import graspcl_pipeline as common
import native_pair_timing as timing
import table_truncation_bench as truncation

PROTOCOL = 'arx_unified_refinement_tcp43_20261120_v1'
INSTALLED = False


def component(path, config):
    target = Path(path).resolve()
    matches = [kind for kind, endpoint in config['sockets'].items()
               if Path(endpoint).resolve() == target]
    if len(matches) != 1:
        raise RuntimeError('RPC endpoint must identify exactly one configured component: ' + str(path))
    return matches[0]


def component_for_trial(path, trial):
    config = common.read(Path(trial).parent.parent.parent / 'resident/backend_config.json')
    return component(path, config)


def validate_pool(directory, reg):
    directory = Path(directory)
    pool = common.read(directory / 'graspcl/candidate_pool.json')
    before = common.read(directory / 'graspcl/unrefined_candidate_pool.json')
    refinement = common.read(directory / 'truncation.json')
    assert pool['refinement'] == refinement, 'Missing or inconsistent refinement receipt'
    assert refinement['mode'] == reg['orientation_mode'] == 'fixed_orientation'
    assert refinement['input_candidates'] == len(before['candidates'])
    assert refinement['retained'] == len(pool['candidates'])
    assert len(pool['candidates']) + len(refinement['rejected']) == len(before['candidates'])
    assert refinement['table_closed_margin_m'] == 0.
    assert refinement['nominal_closed_precheck'] is False
    assert refinement['collision_policy'] == reg['collision_policy']
    scored = reg['pipeline_profile'] == 'graspcl_200'
    if scored:
        rescore = common.read(directory / 'refined_rescore.json')
        assert pool['refined_candidates_rescored'] is True
        assert rescore['success'] and rescore['rescored_candidates'] == len(pool['candidates'])
        assert rescore['bank_updates'] == 0
        memory = common.read(directory / 'frozen_memory_rescore.json')
        assert memory['before_sha256'] == memory['after_sha256']
        assert memory['test_feedback_added'] == 0
    else:
        assert not (directory / 'refined_rescore.json').exists()
        assert all(c['metadata']['origin'] == 'zerograsp' for c in before['candidates'])
    recalled = shifted = retracted = 0
    for candidate in pool['candidates']:
        meta = candidate['metadata']
        source = before['candidates'][meta['source_pool_index']]
        old = np.asarray(source['pose'], dtype=float)
        got = np.asarray(candidate['pose'], dtype=float)
        is_recall = meta['origin'].startswith('demonstration_')
        shift = .003 if is_recall else 0.
        distance = meta['insertion_retraction_m']
        depth = source['metadata']['source_network_depth_m']
        assert -.000000001 <= distance <= depth + 1e-9
        assert np.allclose(got[:3, :3], old[:3, :3], atol=1e-9, rtol=0)
        assert np.allclose(got[:3, 3], old[:3, 3] + (shift-distance)*old[:3, 2], atol=1e-9, rtol=0)
        assert meta['action_tcp_offset_eef_z_m'] == .043
        assert meta['truncation_protocol'] == reg['collision_policy']
        assert meta['observed_target_points_in_closing_sweep'] > 0
        assert meta['gripper_table_clearance_after_m'] >= 0
        if is_recall:
            recalled += 1
            assert meta['demonstration_tcp40_to_tcp43_translation_m'] == .003
            shifted += 1
        retracted += distance > 0
    return {'passed': True, 'protocol': PROTOCOL, 'profile': reg['pipeline_profile'],
            'orientation_mode': reg['orientation_mode'], 'action_tcp_eef_z_m': .043,
            'input_candidates': len(before['candidates']), 'retained_candidates': len(pool['candidates']),
            'retracted_candidates': retracted, 'retained_recall_candidates': recalled,
            'recall_tcp43_shift_verified': shifted,
            'refined_candidates_rescored': scored, 'test_feedback_added': 0,
            'refinement_sha256': common.sha(directory/'truncation.json'),
            'candidate_pool_sha256': common.sha(directory/'graspcl/candidate_pool.json')}


def execute(directory, entry_path):
    entry = common.read(entry_path)
    reg = common.read(entry['campaign_registration'])
    config = common.read(Path(reg['suite']) / 'resident/backend_config.json')
    original_native, original_rpc = timing.native_candidates, timing.real_rpc
    state = {'refinement_calls': 0, 'initial_planning_gate_checks': 0}

    def refine(path):
        if state['refinement_calls']:
            raise RuntimeError('Candidate refinement may execute only once per episode')
        truncation.refine_client(path, reg)
        state['refinement_calls'] += 1
        receipt = validate_pool(path, reg)
        common.save(Path(path)/'unified_protocol_receipt.json', {**receipt, **state})

    def native(path):
        result = original_native(path)
        refine(path)
        return result

    def routed(path, message, timeout=1800):
        kind = component(path, config)
        if kind == 'curobo' and not any(message.get(k) for k in ['observe', 'approach', 'lift', 'refine', 'command']):
            assert state['refinement_calls'] == 1, 'Planning blocked: refinement was not executed'
            receipt = validate_pool(directory, reg)
            state['initial_planning_gate_checks'] += 1
            common.save(Path(directory)/'unified_protocol_receipt.json', {**receipt, **state})
        response = original_rpc(path, message, timeout=timeout)
        if kind == 'graspcl' and not message.get('policy', {}).get('rescore_only') and response['status'] == 'ok':
            refine(Path(message['directory']))
        return response

    with patch.object(timing, 'native_candidates', native), patch.object(timing, 'real_rpc', routed):
        timing.execute(directory, entry_path)
    if (Path(directory)/'graspcl/candidate_pool.json').exists():
        assert state['refinement_calls'] == 1
        receipt = validate_pool(directory, reg)
        common.save(Path(directory)/'unified_protocol_receipt.json', {**receipt, **state})


def install():
    global INSTALLED
    if INSTALLED:
        return
    # Keep the original timing/execution body; replace its one filename-based
    # dispatch expression with lookup in the actual registered backend config.
    tree = ast.parse(textwrap.dedent(inspect.getsource(timing.execute)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.unparse(node) == 'kind = Path(path).stem':
            node.value = ast.parse('_unified_component(path, trial)', mode='eval').body
            replacements += 1
    assert replacements == 1
    scope = dict(timing.__dict__)
    exec(compile(ast.fix_missing_locations(tree), __file__+':configured_timing_dispatch', 'exec'), scope)
    timing._unified_component = component_for_trial
    timing.execute = types.FunctionType(scope['execute'].__code__, timing.__dict__, 'execute')
    truncation.execute = execute
    INSTALLED = True

