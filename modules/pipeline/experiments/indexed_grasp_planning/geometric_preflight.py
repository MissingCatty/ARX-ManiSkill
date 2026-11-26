"""Register and independently verify CPU geometric IK on physical ARX targets."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from arx_project.assets import resource_path
from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.arx_core.arms import get_arm
from modules.curobo.dual_arm.model import ArmModel
from modules.curobo.indexed_planning.geometric.kinematics import GeometricSolver
from .replay import verify_hashes


def prepare(root):
    if root.exists(): raise ValueError('Preserve earlier geometric verification; use a fresh registration')
    config = read_json(ROOT/'configs/experiments/indexed_grasp_planning.json')
    source = ROOT/config['native_batch_implementation']['full_run']
    if not read_json(source/'independent_review.json')['passed']:
        raise ValueError('The reference native replay must be independently accepted')
    cases = read_json(source/'registration.json')['cases']
    inputs = {}
    for case in cases:
        for filename in ['state.json', 'plan_result.json']:
            path = source/'plans'/case['id']/'control'/filename
            inputs[str(path)] = sha256(path)
    sources = [Path(__file__), ROOT/'modules/curobo/dual_arm/model.py',
        ROOT/'modules/curobo/fingertip_contact/geometry.py', resource_path('arx_robot')/'robot.urdf']
    sources += list((ROOT/'modules/curobo/indexed_planning/geometric').glob('*.py'))
    import eaik.pybindings.EAIK as EAIK
    sources.append(Path(EAIK.__file__))
    root.mkdir(parents=True)
    for artifact in sources:
        target=root/'source_registered'/artifact.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(artifact,target)
    write_json(root/'registration.json', {'at': now(), 'experiment': 'D_geometric_CPU_contract',
        'source_run': str(source), 'cases': [c['id'] for c in cases],
        'random_states_per_arm': 128, 'random_seed': 20261126,
        'frozen_targets': len(cases)*3, 'new_planning_requests': 0,
        'physical_grasps': 0, 'simulation_instances': 0, 'GPU_calls': 0,
        'environment': 'environments/geometric_ik',
        'versions': {name: importlib.metadata.version(name) for name in ['eaik','numpy','scipy','PyYAML']},
        'input_hashes': inputs, 'source_hashes': {str(p): sha256(p) for p in sources}})


def physical_pose(model, template, side, q):
    arm = get_arm(side)
    state = deepcopy(template)
    state['links'] = {'base_link': template['links']['base_link']}
    for name, angle in zip(arm.joints, q):
        state['qpos'][state['joint_names'].index(name)] = float(angle)
    return model.frames(state)[arm.eef]


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Do not repeat an already started geometric verification')
    reg = read_json(root/'registration.json'); verify_hashes(reg)
    write_json(root/'started.json', {'at': now()})
    model = ArmModel()
    solvers = {side: GeometricSolver(side) for side in ['right','left']}
    source = Path(reg['source_run'])
    template = read_json(source/'plans'/reg['cases'][0]/'control/state.json')
    rng = np.random.default_rng(reg['random_seed'])
    random_rows, frozen_rows = [], []
    for side, solver in solvers.items():
        arm = get_arm(side)
        for index, q in enumerate(rng.uniform(solver.chain.limits[:,0], solver.chain.limits[:,1],
                                             (reg['random_states_per_arm'],6))):
            target_world = physical_pose(model,template,side,q)
            target = np.linalg.inv(np.asarray(template['links']['body4']))@target_world
            original_fk = solver.chain.fk(q)[0]
            fk_error = float(np.abs(original_fk-target).max())
            native_fk = solver.robot.fwdkin(q*solver.axis_signs)
            native_fk_error = float(np.abs(native_fk-target).max())
            answer = solver.solve(target,q)
            if fk_error > 2e-6 or native_fk_error > 2e-6 or not answer['admitted_branches']:
                raise RuntimeError('Geometric chain failed a known physical target: '+str((side,index,fk_error,native_fk_error)))
            max_position, max_rotation = 0., 0.
            for value in answer['q']:
                actual = physical_pose(model,template,side,value)
                max_position = max(max_position,float(np.linalg.norm(actual[:3,3]-target_world[:3,3])))
                max_rotation = max(max_rotation,float(Rotation.from_matrix(actual[:3,:3]@target_world[:3,:3].T).magnitude()))
            if max_position > .002 or max_rotation > .03:
                raise RuntimeError('Returned geometric branch failed independent full-robot FK')
            random_rows.append({'arm':side,'index':index,'chain_FK_error':fk_error,
                'EAIK_FK_error':native_fk_error,'returned_branches':answer['admitted_branches'],
                'maximum_position_error_m':max_position,'maximum_rotation_error_rad':max_rotation})
    for ordinal,case in enumerate(reg['cases']):
        directory = source/'plans'/case/'control'
        state, result = read_json(directory/'state.json'), read_json(directory/'plan_result.json')
        selected = result['selected']; side = selected['arm']; solver = solvers[side]; arm = get_arm(side)
        start = np.array([state['qpos'][state['joint_names'].index(n)] for n in arm.joints])
        for stage,key in [('grasp','grasp_world'),('pre','pregrasp_world'),('lift','lift_world')]:
            target_world = np.asarray(selected[key])
            target = np.linalg.inv(np.asarray(state['links']['body4']))@target_world
            answer = solver.solve(target,start)
            error = 0.
            for q in answer['q']:
                actual = physical_pose(model,state,side,q)
                pe = float(np.linalg.norm(actual[:3,3]-target_world[:3,3]))
                re = float(Rotation.from_matrix(actual[:3,:3]@target_world[:3,:3].T).magnitude())
                error = max(error,pe)
                if pe > .002 or re > .03:
                    raise RuntimeError('Frozen target branch failed independent physical-arm FK')
            frozen_rows.append({'id':case,'arm':side,'stage':stage,
                'admitted_branches':answer['admitted_branches'],'raw_branches':answer['raw_branches'],
                'native_solver_seconds':answer['native_solver_seconds'],'wall_seconds':answer['wall_seconds'],
                'maximum_independent_position_error_m':error,
                'no_admitted_branch_action':'original_IK_fallback' if not answer['admitted_branches'] else None})
        write_json(root/'progress.json', {'at':now(),'phase':'physical_target_contract',
            'completed_cases':ordinal+1,'total_cases':len(reg['cases'])})
    verify_hashes(reg)
    result = {'at':now(),'passed':True,'random_physical_states':len(random_rows),
        'frozen_targets':len(frozen_rows),'frozen_targets_with_geometric_branches':sum(r['admitted_branches']>0 for r in frozen_rows),
        'raw_solver_median_us':float(np.median([r['native_solver_seconds'] for r in frozen_rows])*1e6),
        'query_and_filter_median_ms':float(np.median([r['wall_seconds'] for r in frozen_rows])*1e3),
        'families':{s:v.family for s,v in solvers.items()},'axis_signs':{s:v.axis_signs.tolist() for s,v in solvers.items()},
        'all_returned_branches_independently_verified':True,'record_integrity':True,
        'new_planning_requests':0,'physical_grasps':0,'GPU_calls':0,'simulation_instances':0,
        'random_results':random_rows,'frozen_results':frozen_rows}
    write_json(root/'acceptance.json',result)
    write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_cases':len(reg['cases'])})
    print(json.dumps({k:v for k,v in result.items() if k not in ['random_results','frozen_results']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()});raise
