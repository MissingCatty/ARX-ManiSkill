"""Five paired engineering cases: fixed-pregrasp MPC versus one continuous route."""
import argparse
import ast
from copy import deepcopy
import fcntl
import inspect
from pathlib import Path
import shutil
import time
import traceback
import types
from unittest.mock import patch

import arx_runtime_imports
arx_runtime_imports.install()
import numpy as np
import graspcl_unified_bench_20261120 as parent
import arx_direct_grasp_20261120 as direct

common, bench, resident, shared = parent.common, parent.bench, parent.resident, parent.shared
CODE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
PRIOR = CODE/'outputs/arx_graspcl_unseen_unified_20261120_01'
GROUPS = ['B_fixed_pregrasp', 'C_continuous_grasp']
EPOCH = 'runtime_01'
BACKEND_GPU = 'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169'
SIM_GPU = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'


def aggregate(root, phase=None):
    old = common.read(root/'progress.json') if (root/'progress.json').exists() else {}
    states = {n: common.read(root/n/'progress.json') for n in GROUPS if (root/n/'progress.json').exists()}
    common.save(root/'progress.json', {'phase': phase or old.get('phase', 'running'),
                'completed': sum(v['completed'] for v in states.values()), 'registered': 10,
                'groups': states, 'updated_at': bench.core.now()})


def preflight(root):
    goal = np.eye(4);rotation = np.repeat(np.eye(3)[None], 101, axis=0)
    xyz = np.zeros((101, 3));xyz[:, 2] = np.linspace(-.06, 0., 101)
    axial = direct.terminal_geometry(xyz, rotation, goal)
    assert axial['passed']
    side = xyz.copy();side[:, 0] = side[:, 2];side[:, 2] = 0
    assert not direct.terminal_geometry(side, rotation, goal)['passed']
    wrong = rotation.copy();wrong[:, 0, 0] = -1.;wrong[:, 1, 1] = -1.
    assert not direct.terminal_geometry(xyz, wrong, goal)['passed']
    # Compile both real execution variants without creating a simulator.
    source = (CODE/'arx_pipeline/run.py').read_text()
    tree = ast.parse(source);node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trial')
    text = ast.get_source_segment(source, node).replace('result["minimum_hold_lift_m"] > .06', 'result["minimum_hold_lift_m"] >= .01')
    ns = {'PERCEPTION_CAMERA': 'right_wrist'}
    exec(compile(text, str(SELF)+':execution_contract', 'exec'), ns)
    original = ns['trial'];original.registered_short_lift_source = text
    original.short_lift_protocol = bench.lift.PROTOCOL
    proofs = {}
    for mode in [False, True]:
        run = types.SimpleNamespace(trial=original)
        with direct.execution_hook(run, mode):
            constants = run.trial.__code__.co_consts
            assert ('pregrasp' in constants) == (not mode)
            assert ('pregrasp_world' in constants) == (not mode)
            proofs[str(mode)] = {'pregrasp_segment_present': 'pregrasp' in constants,
                                 'fixed_pregrasp_goal_access_present': 'pregrasp_world' in constants}
    receipt = {'passed': True, 'axial_approach_accepted': axial, 'lateral_approach_rejected': True,
               'wrong_orientation_rejected': True, 'compiled_execution_contracts': proofs,
               'physical_executions': 0, 'models_loaded': 0}
    common.save(root/'preflight.json', receipt)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False);(root/'source').mkdir();(root/EPOCH).mkdir()
    preflight(root)
    old = common.read(PRIOR/'registration.json')
    prior_group = common.read(PRIOR/'C_graspcl_experience50/registration.json')
    manifest = common.read(Path(old['inputs'])/'test/manifest.json')
    schedule = manifest['schedule'];selected = []
    for layout in ['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter']:
        selected.append(next(e['id'] for e in schedule if e['layout'] == layout))
    selected.append([e['id'] for e in schedule if e['layout'] == 'S2_edge'][1])
    assert len(set(selected)) == 5
    pools = {i: {'path': str(PRIOR/'C_graspcl_experience50/fixed_graspcl/trials'/i/'graspcl/unrefined_candidate_pool.json'),
                 'sha256': common.sha(PRIOR/'C_graspcl_experience50/fixed_graspcl/trials'/i/'graspcl/unrefined_candidate_pool.json')}
             for i in selected}
    sources = dict(old['source_hashes'])
    historical = common.read(CODE/'outputs/arx_graspcl_mpc_200_20261119_03/registration.json')
    for filename in historical['source_hashes']:
        p = Path(filename)
        if p.suffix == '.py' and p.is_file():sources[str(p)] = common.sha(p)
    for p in [SELF, Path(direct.__file__), CODE/'arx_direct_services_20261120.py',
              CODE/'arx_direct_simulator_20261120.py', root/'preflight.json']:
        sources[str(p)] = common.sha(p)
    for record in pools.values():sources[record['path']] = record['sha256']
    reg = deepcopy(old)
    reg.update(created_at=bench.core.now(), source_hashes=sources, total_slots=10,
               test_arms=GROUPS, selected_case_ids=selected,
               selected_case_rule='First registered case of each of four layouts, plus the second registered edge case; no selection by outcome.',
               cached_graspcl_pools=pools, prior_output=str(PRIOR),
               scope='Five paired engineering examples, not an estimate of the 100-scene success rate.',
               runtime_backend_gpu=BACKEND_GPU, runtime_scene_gpus=[SIM_GPU],
               source_experience_reexecuted=0, mpc=True,
               common_changes=['Same frozen pre-refinement predictions and same 39-positive/2-negative bank',
                               'Same fixed orientation, TCP43, truncation, rescoring and 3 cm lift',
                               'Both arms check target collision outside the final 26 mm contact region'],
               direct_policy={'fixed_pregrasp_goal': False, 'additional_pregrasp_ik': False,
                   'global_reference': 'One continuous path from measured capture state to grasp',
                   'terminal_region_radius_m': .020, 'terminal_lateral_tolerance_m': .004,
                   'terminal_orientation_tolerance_rad': .10,
                   'near_goal_visual_feedback_hz': 5, 'visual_translation_limit_m': .006,
                   'mandatory_intermediate_stop': False,
                   'seed_policy': 'Global endpoint route; if needed, differential-kinematics terminal seeds of up to 20/10/5 mm. Seeds are not intermediate goals or stops.'})
    common.save(root/'registration.json', reg)
    parent.guard(reg)
    for name in GROUPS:
        group, arm = bench.prepare_group(root, name, Path(old['inputs'])/'test', Path(prior_group['bank']))
        for directory in [group, arm]:
            path = directory/'registration.json';value = common.read(path)
            value.update(schedule=[e for e in value['schedule'] if e['id'] in selected],
                         total_slots=5, attempt_budget=5, scene_workers=1,
                         direct_grasp=name == GROUPS[1], cached_graspcl_pools=pools,
                         shared_input_role='consumer', shared_input_cache=prior_group['shared_input_cache'],
                         runtime_backend_gpu=BACKEND_GPU, runtime_scene_gpus=[SIM_GPU],
                         component_startup_wait_seconds=1800)
            common.save(path, value)
        for directory in [group, arm]:common.save(directory/'progress.json', {'phase': 'prepared', 'completed': 0, 'successes': 0, 'total': 5})
    for p in [SELF, Path(direct.__file__), CODE/'arx_direct_services_20261120.py', CODE/'arx_direct_simulator_20261120.py']:
        shutil.copy2(p, root/'source'/p.name)
    common.save(root/EPOCH/'registration.json', {'created_at': bench.core.now(), 'registered_slots': 10,
                'root_registration_sha256': common.sha(root/'registration.json'), 'physical_executions': 'Five new attempts per arm; no replay of an allocated attempt.'})
    aggregate(root, 'prepared');print('PREPARED', root, selected, flush=True)


def manager_class():
    source = inspect.getsource(parent.Manager.launch)
    import textwrap
    source = textwrap.dedent(source)
    assert source.count('graspcl_unseen_services_quiet_20261119.py') == 1
    scope = dict(parent.Manager.launch.__globals__)
    exec(compile(source.replace('graspcl_unseen_services_quiet_20261119.py', 'arx_direct_services_20261120.py'),
                 str(SELF)+':service_entry', 'exec'), scope)
    class Manager(parent.Manager):
        def component(self, kind, group, reg):
            if kind != 'graspcl':return super().component(kind, group, reg)
            if kind not in self.shared:self.shared[kind] = self.launch(kind, group, reg)
            record = self.shared[kind]
            if record['owner'] != group/'resident':
                (group/'resident/graspcl_ready.json').symlink_to(record['owner']/'graspcl_ready.json')
                (group/'resident/graspcl_model').symlink_to(record['owner']/'graspcl_model', target_is_directory=True)
            return record
    Manager.launch = types.FunctionType(scope['launch'].__code__, parent.Manager.launch.__globals__, 'launch')
    return Manager


class Backends(shared.Backends):
    def start(self):
        super().start()
        assert common.read(self.campaign/'resident/planner_warmup/mpc_ready.json')['passed']
        if self.campaign.name == GROUPS[0]:
            self.training_plan_check()

    def stop(self):
        # The identical scorer, neural model, planner and MPC all belong to the suite.
        pass

    def training_plan_check(self):
        warm = Path(self.reg['warmup_trial'])
        old_plan = common.read(warm/'plan_result.json')['selected']
        grasp = np.asarray(old_plan['grasp_world']);contact = grasp.copy();contact[:3, 3] += .043*grasp[:3, 2]
        proofs = []
        for mode in [False, True]:
            directory = self.campaign/'startup_validation'/('direct' if mode else 'control')/warm.name
            directory.mkdir(parents=True)
            common.save(directory/'state.json', common.read(warm/'state.json'))
            registration = directory.parent/'training_registration.json'
            common.save(registration, {**self.reg, 'direct_grasp': mode})
            common.save(directory/'source_guard.json', {'registration': str(registration), 'sha256': common.sha(registration)})
            common.save(directory/'graspcl/candidate_pool.json', {'candidates': [{'pose': contact.tolist(), 'width': .05,
                'score': 1., 'metadata': {'origin': 'zerograsp', 'training_only_validation': True}}]})
            reply = parent.rpc(Path(self.config['sockets']['curobo']), {'directory': str(directory)}, timeout=900)
            common.save(directory/'planning_response.json', reply)
            assert reply['status'] == 'ok', reply
            plan = common.read(directory/'plan_result.json')
            assert plan['success'], ('Training planning smoke test failed', mode, plan)
            if mode:
                assert set(plan['paths']) == {'approach'} and 'q_pre' not in plan['selected']
                assert common.read(directory/'continuous_grasp_plan.json')['passed']
            proofs.append({'direct': mode, 'passed': True, 'directory': str(directory), 'robot_motions': 0})
        common.save(self.campaign.parent/'training_planning_verification.json', {'passed': True, 'tests': proofs,
                    'test_scene_inputs_used': False, 'models_reloaded': 0, 'physical_executions': 0})


def verify(root):
    all_results = {}
    for name in GROUPS:
        group = root/name;reg = common.read(group/'registration.json');arm = group/'fixed_graspcl';rows = []
        for e in reg['schedule']:
            trial = arm/'trials'/e['id'];result = common.read(trial/'result.json')
            assert common.read(trial/'independent_evaluation.json')['passed']
            parent.protocol.validate_pool(trial, {**reg, 'pipeline_profile': 'graspcl_200', 'orientation_mode': 'fixed_orientation'})
            plan = common.read(trial/'plan_result.json')
            if plan.get('selected'):
                if reg['direct_grasp']:
                    assert set(plan['paths']) == {'approach'} and 'q_pre' not in plan['selected']
                    assert 'pregrasp_world' not in plan['selected'] and 'pregrasp' not in result.get('stages', {})
                    assert not (trial/'route_mpc_pregrasp.json').exists()
                    assert common.read(trial/'continuous_grasp_plan.json')['passed']
                if result['success']:
                    mpc = common.read(trial/'mpc_approach.json')
                    assert mpc['settled'] and mpc['mpc_solves'] >= 2 and mpc['tracked_frames'] > 0
            rows.append({'id': e['id'], 'success': result['success'], 'planning_success': result['planning_success'],
                         'reason': result['reason'], 'trial': str(trial), 'wall_seconds': result['wall_seconds']})
        assert len(rows) == 5
        all_results[name] = rows
    # Pre-refinement predictions must be identical in the paired execution.
    for ident in common.read(root/'registration.json')['selected_case_ids']:
        pools = [common.read(root/n/'fixed_graspcl/trials'/ident/'graspcl/unrefined_candidate_pool.json') for n in GROUPS]
        assert pools[0] == pools[1]
    common.save(root/'pilot_results.json', {'verified': True, 'scope': 'Five engineering examples per arm',
        'groups': {k: {'completed': len(v), 'planned': sum(x['planning_success'] for x in v),
                      'successes': sum(x['success'] for x in v), 'cases': v} for k, v in all_results.items()}})
    return all_results


def run(root):
    registration = common.read(root/'registration.json');parent.guard(registration)
    assert common.read(root/'preflight.json')['passed']
    shared.PLANNER_GPU = BACKEND_GPU
    bench.GPU0 = BACKEND_GPU;bench.GPUS = [SIM_GPU]
    parent.SIMULATOR = CODE/'arx_direct_simulator_20261120.py'
    resident.EPOCH = EPOCH;resident.aggregate = aggregate;bench.aggregate = aggregate
    bench.Backends = Backends;bench.Campaign = resident.Campaign
    resident.POOL = parent.Simulators(root);shared.MANAGER = manager_class()(root, EPOCH)
    success = False;started = time.perf_counter()
    with (root/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root/EPOCH/'started.json').open('x') as stream:stream.write(bench.core.now()+'\n')
        try:
            aggregate(root, 'preloading_simulation');resident.POOL.start()
            for name in GROUPS:
                aggregate(root, 'testing_'+name)
                bench.run_group(root/name, root/name/'fixed_graspcl')
            verify(root);parent.guard(registration);success = True
        except BaseException:
            common.save(root/EPOCH/'halt.json', {'at': bench.core.now(), 'error': traceback.format_exc(), 'replay_allowed': False})
            raise
        finally:
            resident.POOL.stop();shared.MANAGER.stop()
            models = common.read(root/EPOCH/'model_services_stopped.json')
            sim = common.read(root/EPOCH/'simulation_stopped.json')
            common.save(root/'suite_timing.json', {'wall_seconds': time.perf_counter()-started,
                'all_owned_services_exited': models['all_owned_model_services_exited'],
                'all_owned_simulators_exited': sim['all_owned_simulators_exited']})
            if success:
                common.save(root/'final_verification.json', {'passed': True, 'completed': 10,
                    'frozen_feedback_bank': True, 'test_feedback_updates': 0,
                    'all_owned_services_exited': models['all_owned_model_services_exited'],
                    'all_owned_simulators_exited': sim['all_owned_simulators_exited']})
            aggregate(root, 'complete' if success else 'halted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)(args.output.resolve())
