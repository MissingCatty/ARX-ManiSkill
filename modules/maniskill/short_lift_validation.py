"""CPU contract checks for the short lift; no model loading or physical trials."""
import argparse
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import short_lift_protocol as policy

CODE = Path(__file__).resolve().parent


def selector_contract(out):
    original_tree = ast.parse((CODE / 'postprocess_planner.py').read_text())
    original = next(n for c in original_tree.body if isinstance(c, ast.ClassDef)
                    for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'select')
    tree = ast.parse((CODE / 'short_lift_planner.py').read_text())
    actual = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                  for n in c.body if isinstance(n, ast.FunctionDef))

    class OnlyRegisteredChanges(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == .040:
                return ast.copy_location(ast.Constant(.043), node)
            if node.value == .12:
                return ast.copy_location(ast.Name('LIFT_DISTANCE_M', ast.Load()), node)
            if node.value == 'zerograsp_native_post_no_custom_geometry_v1':
                return ast.copy_location(ast.Name('PROTOCOL', ast.Load()), node)
            return node

    expected = OnlyRegisteredChanges().visit(deepcopy(original))
    expected.name = actual.name
    assert ast.dump(expected) == ast.dump(actual), 'Unexpected planning gate change'
    scope = {'np': np, 'json': json, 'LIFT_DISTANCE_M': .03, 'PROTOCOL': policy.PROTOCOL}
    exec(compile(ast.Module(body=[actual], type_ignores=[]), __file__, 'exec'), scope)
    directory = out / 'selector'
    (directory / 'graspcl').mkdir(parents=True)
    (directory / 'planner').mkdir()
    (directory / 'graspcl/candidate_pool.json').write_text(json.dumps({'candidates':[
        {'pose': [[1,0,0,0],[0,1,0,0],[0,0,1,.243],[0,0,0,1]], 'score': 1., 'metadata': {}}]}))
    goals = []

    def ik(pose, seed):
        goals.append(np.asarray(pose).copy())
        return np.zeros(6) if pose[2,3] <= .25 else None

    fixture = SimpleNamespace(out=directory/'planner', measured=lambda state: np.zeros(6), ik=ik,
        closed_checker=SimpleNamespace(validate=lambda q: np.array([True])),
        path=lambda a,b: np.stack([a,b]), optimize=lambda label,path,seconds: {'q':path.tolist(),'times':[0,seconds]})
    result = scope[actual.name](fixture, {}, [], None)
    assert result['success']
    policy.validate_selected(result['selected'])
    assert np.allclose(goals[-1][:3,3], [0,0,.23])
    old_scope = dict(scope, LIFT_DISTANCE_M=.12)
    exec(compile(ast.Module(body=[actual], type_ignores=[]), __file__, 'exec'), old_scope)
    assert not old_scope[actual.name](fixture, {}, [], None)['success']
    return {'selector_ast_only_registered_changes': True, 'three_cm_endpoint_passes_where_twelve_cm_fails': True}


def verifier_contract(out):
    import graspcl_batch
    run, baseline = graspcl_batch.original_modules()
    original_trial_source = textwrap.dedent(inspect.getsource(run.trial))
    policy.install_execution_policy(run, baseline)
    assert run.trial.registered_short_lift_source == original_trial_source.replace(
        'result["minimum_hold_lift_m"] > .06', 'result["minimum_hold_lift_m"] >= .01')
    assert run.trial.__globals__ is run.__dict__
    checks = []
    cases = [
        ('stable_15mm', .015, 360, True, True, True, True, False),
        ('exact_10mm', .01, 360, True, True, True, True, False),
        ('only_9mm', .009, 360, False, True, True, True, False),
        ('falls_during_hold', .015, 360, False, True, True, True, True),
        ('short_hold', .015, 359, False, True, True, True, False),
        ('forbidden_contact', .015, 360, False, False, True, True, False),
        ('one_finger_only', .015, 360, False, True, False, True, False),
        ('unsettled_action', .015, 360, False, True, True, False, False),
    ]
    for name, height, count, success, allowed, bilateral, settled, falls in cases:
        trial = out / name
        trial.mkdir()
        result = {'success': success, 'stages': {k:{'settled':settled} for k in ['pregrasp','approach','close','lift','hold']}}
        (trial/'result.json').write_text(json.dumps(result))
        (trial/'execution_origin.json').write_text(json.dumps({'object_initial_z_m':.05}))
        rows = [{'stage':'close','object_pose':[0,0,.05], 'contacts':[
            {'a':'target','b':'right_link27','allowed':allowed},
            *([{'a':'target','b':'right_link28','allowed':True}] if bilateral else [])]}]
        rows += [{'stage':'hold','object_pose':[0,0,.05+(0 if falls and i==count-1 else height)], 'contacts':[]} for i in range(count)]
        (trial/'execution.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        baseline.verify_trial(trial)
        verified = json.loads((trial/'independent_evaluation.json').read_text())
        assert verified['passed'] and verified['independently_successful'] == success
        assert verified['evaluation_protocol'] == policy.PROTOCOL
        if not success:
            result['success'] = True
            (trial/'result.json').write_text(json.dumps(result))
            try:
                baseline.verify_trial(trial)
            except RuntimeError:
                pass
            else:
                raise AssertionError('False success was not rejected: '+name)
            result['success'] = False
            (trial/'result.json').write_text(json.dumps(result))
            baseline.verify_trial(trial)
        checks.append({'case':name,'expected_success':success,'passed':True})
    return {'live_execution_globals_preserved':True, 'trace_cases':checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    result = {**selector_contract(out), **verifier_contract(out), 'physical_attempts':0, 'passed':True}
    (out/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result), flush=True)
