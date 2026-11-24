"""Bounded V2 planning comparison with strict reach and checked local approach."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from audit import HERE, PROJECT, WORKSPACE, ASSET, SOURCE, now, read, save, sha
from mesh_probe import ROOT as MESH
from runtime_adapter import LocalContactValidator

ROOT = HERE / 'runs/initial_plan_20261123_01'
TOLERANCES = [0., .002]


def prepare():
    assert read(MESH / 'completion_verification.json')['passed']
    assert not ROOT.exists(), 'Never overwrite a planning registration'
    rows = [read(p) for p in sorted((MESH / 'cases').glob('*/result.json'))]
    eligible = [r for r in rows if r['thresholds'][-1]['valid_candidates'] > 0]
    # Cover different objects first, then any remaining distinct positions.
    order = sorted(eligible, key=lambda r: (r['cohort'] != 'original53', r['id']))
    selected, classes = [], set()
    for row in order:
        semantic = read(Path(row['source_case']) / 'entry.json')['semantic_class']
        if semantic in classes: continue
        selected.append(row); classes.add(semantic)
        if len(selected) == 8: break
    for row in order:
        if len(selected) == 8: break
        if row['id'] not in {r['id'] for r in selected}: selected.append(row)
    schedule = [{'id': r['id'], 'source_case': r['source_case'], 'cohort': r['cohort'],
                 'candidate_witnesses': r['witnesses']} for r in selected]
    source = read(SOURCE / 'registration.json')
    save(ROOT / 'registration.json', {'at': now(), 'version': 'V2', 'schedule': schedule,
         'positions': len(schedule), 'planning_slots': len(schedule)*2, 'tolerances_m': TOLERANCES,
         'candidate_policy': 'At most 8 native-order candidates per registered position/configuration, from independently checked exact-mesh witnesses',
         'initial_reach': 'Unchanged native cuRobo PRM, optimizer and checker',
         'local_approach': 'Dense joint-space straight connector, <=0.5mrad per joint and original non-table checks; quintic timing at 40Hz',
         'lift_endpoint': 'Unchanged native full-world checker, open geometry as in V1 initial planning; measured-closure lift is later physical work',
         'physical_attempt_budget': 0, 'automatic_retries': 0, 'warmup_trial': source['planner_warmup_trial'],
         'source_sha256': {str(p): sha(p) for p in [Path(__file__), HERE/'runtime_adapter.py', HERE/'narrow_phase.py', HERE/'geometry.py']},
         'input_sha256': {str(MESH/'summary.json'): sha(MESH/'summary.json'),
                          **{str(MESH/'cases'/r['id']/'result.json'): sha(MESH/'cases'/r['id']/'result.json') for r in selected}}})
    save(ROOT / 'progress.json', {'phase': 'registered', 'completed': 0, 'registered': len(schedule)*2})


def run():
    reg = read(ROOT / 'registration.json')
    assert not (ROOT / 'STOP').exists() and not (ROOT / 'started.json').exists()
    for p, h in {**reg['source_sha256'], **reg['input_sha256']}.items(): assert sha(p) == h
    save(ROOT / 'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES')})
    sys.path[:0] = [str(PROJECT / 'experiments/rayst3r_graspgenx'), str(WORKSPACE/'maniskill'), str(WORKSPACE/'maniskill/arx_pipeline')]
    from services import planner
    from planning_rejection_audit import make_pure
    from ik_workspace_audit import solve
    from resident_planner import all_kinematics_params, collision_scene_config
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    import torch
    pool = planner(SimpleNamespace(root=ROOT/'resident', warmup_trial=Path(reg['warmup_trial'])))
    model = pool.contexts['open'].model
    pure = make_pure(model)
    aux = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
        scene_model={'cuboid': {}}, n_meshes=8, n_cuboids=8, collision_activation_distance=0.))
    aux.self_collision_cost = None
    params = all_kinematics_params(aux)
    save(ROOT / 'warmup_complete.json', {'at': now(), 'loads': pool.loads, 'physical_attempts': 0})
    rows = []
    for row in reg['schedule']:
        state = read(Path(row['source_case'])/'state.json')
        for tolerance in TOLERANCES:
            assert not (ROOT / 'STOP').exists()
            begin = time.perf_counter()
            slot = row['id']+('_mesh0' if tolerance == 0 else '_mesh2mm')
            out = ROOT / 'cases' / slot; out.mkdir(parents=True, exist_ok=False)
            context = pool.bind(out/'plan', state)
            for p in params: p.copy_(context.checker.kinematics.kinematics_config)
            scene = read(out/'plan/scene.json')
            scene['cuboid'].pop('table')
            aux.clear_scene_cache(); aux.update_world(collision_scene_config(scene))
            validator = LocalContactValidator(context, state, aux, ASSET, tolerance)
            start = context.measured(state)
            events = []; selected = None; paths = None
            source_entry = read(Path(row['source_case'])/'translation.json')
            candidates = read(Path(source_entry['source_trial'])/'candidates/candidate_pool.json')['candidates']
            witnesses = [w for w in row['candidate_witnesses'] if any(t['valid'] and t['tolerance_m'] <= tolerance for t in w['thresholds'])]
            witnesses.sort(key=lambda w: (w['candidate_index'], w['branch_index']))
            seen = set()
            for witness in witnesses:
                index = witness['candidate_index']
                if index in seen: continue
                seen.add(index)
                if len(seen) > 8: break
                qg = torch.tensor(witness['q'], dtype=torch.float32, device='cuda')
                if not validator.validate(qg[None], stage='final_approach').all():
                    events.append({'candidate': index, 'reason': 'endpoint_policy_failed'}); continue
                grasp = np.asarray(candidates[index]['pose']).copy()
                grasp[:3, 3] += np.asarray(source_entry['translation_m'])
                grasp[:3, 3] -= .043*grasp[:3, 2]
                pre, lift = grasp.copy(), grasp.copy()
                pre[:3, 3] -= .08*pre[:3, 2]; lift[2, 3] += .03
                def endpoint(world):
                    q, ok, pe, re = solve(pure, context, [world], qg)
                    mask = ok[0] & (pe[0] <= .002001) & (re[0] <= .030001)
                    choices = q[0][mask]
                    if not len(choices): return None
                    choices = choices[context.checker.validate(choices[:, None]).reshape(-1)]
                    if not len(choices): return None
                    return choices[(choices-qg).square().sum(-1).argmin()].clone()
                qp, ql = endpoint(pre), endpoint(lift)
                if qp is None or ql is None:
                    events.append({'candidate': index, 'reason': 'pregrasp_or_lift_endpoint_failed', 'pregrasp': qp is not None, 'lift': ql is not None}); continue
                reach = context.path(start, qp)
                if reach is None:
                    events.append({'candidate': index, 'reason': 'strict_initial_reach_failed'}); continue
                approach = validator.straight_approach(qp.cpu().numpy(), qg.cpu().numpy())
                if approach is None:
                    events.append({'candidate': index, 'reason': 'local_approach_collision'}); continue
                reach_path = context.optimize('pregrasp', reach, 6.)
                selected = {'candidate_index': index, 'q_grasp': qg.cpu().numpy(), 'q_pre': qp.cpu().numpy(),
                            'q_lift': ql.cpu().numpy(), 'grasp_world': grasp, 'pregrasp_world': pre, 'lift_world': lift}
                paths = {'pregrasp': reach_path, 'approach': approach}
                break
            save(out/'contact_receipts.json', validator.receipts)
            result = {'id': slot, 'source_id': row['id'], 'cohort': row['cohort'], 'tolerance_m': tolerance,
                      'initial_planning_success': selected is not None, 'selected': selected, 'paths': paths,
                      'events': events, 'tested_candidate_count': len(seen), 'physical_attempts': 0,
                      'wall_s': time.perf_counter()-begin}
            save(out/'result.json', result); rows.append(result)
            save(ROOT/'progress.json', {'phase': 'running', 'completed': len(rows), 'registered': reg['planning_slots'], 'last': slot})
            print('PLAN', len(rows), slot, selected is not None, events, flush=True)
    for p, h in {**reg['source_sha256'], **reg['input_sha256']}.items(): assert sha(p) == h
    save(ROOT/'summary.json', {'completed': len(rows), 'registered': reg['planning_slots'],
        'positions': reg['positions'], 'variants': [{'tolerance_m': t,
            'initial_plans_passed': sum(r['initial_planning_success'] for r in rows if r['tolerance_m'] == t),
            'positions': reg['positions']} for t in TOLERANCES], 'physical_attempts': 0})
    save(ROOT/'completion_verification.json', {'passed': True, 'completed': len(rows), 'source_inputs_unchanged': True, 'physical_attempts': 0})
    save(ROOT/'progress.json', {'phase': 'complete', 'completed': len(rows), 'registered': reg['planning_slots']})


if __name__ == '__main__':
    try:
        prepare(); run()
    except BaseException:
        import traceback
        if ROOT.exists(): (ROOT/'STOP').write_text(traceback.format_exc())
        raise
