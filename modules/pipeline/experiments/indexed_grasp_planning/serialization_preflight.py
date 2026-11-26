"""CPU-only preparation equivalence on all saved arm states; no cuRobo import."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json
from modules.curobo.indexed_planning.binding_serialization import BindingCodec


def raw(path): return json.loads(Path(path).read_text())


def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def preparation_function(cache):
    source = ROOT/'modules/maniskill/resident_planner.py'
    policy_path = ROOT/'modules/maniskill/compat_tools/physical_table_pair_20261118_01.py'
    spec = importlib.util.spec_from_file_location('arx_cpu_serialization_policy', policy_path)
    policy = importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bind_inputs')
    tree = policy.transform_tree(ast.unparse(function), 'bind')
    planner = ast.parse((ROOT/'modules/maniskill/arx_pipeline/planner.py').read_text())
    pose = next(n for n in planner.body if isinstance(n, ast.FunctionDef) and n.name == 'pose7')
    scope = {'np': np, 'Rotation': Rotation}
    exec(compile(ast.Module(body=[pose], type_ignores=[]), '<original_pose7>', 'exec'), scope)
    from modules.pipeline.experiments.current_configuration.configuration import require_current_profile
    def cached_python(text): return deepcopy(cache[text]) if text in cache else yaml.safe_load(text)
    scope.update(yaml=SimpleNamespace(safe_load=cached_python, safe_dump=yaml.safe_dump),
        frozen=SimpleNamespace(ARX=ROOT/'modules/arx_core', FINGERS=['right_joint27', 'right_joint28'], pose7=scope['pose7']),
        save=lambda path, value: Path(path).write_text(json.dumps(value, indent=2)+'\n'),
        require_current_profile=require_current_profile, deepcopy=deepcopy, hashlib=hashlib, Path=Path)
    exec(compile(tree, str(source)+':current_policy_cpu', 'exec'), scope)
    return scope['bind_inputs']


def run(root):
    if root.exists(): raise ValueError('Preserve earlier serialization preflight registrations')
    root.mkdir(parents=True)
    source = ROOT/'tmp/indexed_grasp_planning/validation69_20261126_01'
    if not raw(source/'independent_review.json')['passed']:
        raise ValueError('The source planning cohort was not accepted')
    reg = raw(source/'registration.json');cache = {}
    for role in ['model', 'model_closed']:
        text = (ROOT/'modules/arx_core/scripts/curobo_arx/static_pick/reference'/role/'arx.yml').read_text()
        cache[text] = yaml.safe_load(text)
    original = preparation_function(cache);codec = BindingCodec(cache);optimized = codec.function(original)
    sources = [Path(__file__), ROOT/'modules/curobo/indexed_planning/binding_serialization.py',
        ROOT/'modules/maniskill/resident_planner.py', ROOT/'modules/maniskill/fast_resident_planner.py',
        ROOT/'modules/maniskill/compat_tools/physical_table_pair_20261118_01.py']
    inputs = {};tasks = []
    for case in reg['cases']:
        for side in ['right', 'left']:
            directory = source/'plans'/case['id']/'control_1/arms'/side
            state = directory/'state.json';inputs[str(state)] = sha256(state)
            model = directory/'plan/arx.yml';scene = directory/'plan/scene.json'
            for path in [model, scene]: inputs[str(path)] = sha256(path)
            tasks.append((case['id'], side, state, False, False, model, scene))
    for side in ['right', 'left']:
        first = next(t for t in tasks if t[1] == side)
        for closed, target in [(False, True), (True, False), (True, True)]:
            tasks.append((first[0], side, first[2], closed, target, None, None))
    registration = {'at': now(), 'cases': 69, 'pairs': len(tasks), 'preparation_calls': len(tasks)*2,
        'source_sha256': {str(p): sha256(p) for p in sources}, 'input_sha256': inputs,
        'new_IK_calls': 0, 'planning_requests': 0, 'physical_grasps': 0, 'simulation_instances': 0}
    write_json(root/'registration.json', registration)
    rows = []
    for index, (case, side, state_path, closed, target, saved_model, saved_scene) in enumerate(tasks):
        state = raw(state_path);pair = {}
        modes = [('control', original), ('fast_binding', optimized)]
        if index%2: modes.reverse()
        for mode, function in modes:
            out = root/'preparations'/f'{index:03d}_{case}_{side}'/mode
            context = SimpleNamespace();began = time.perf_counter()
            result = function(context, out, state, closed, target)
            seconds = time.perf_counter()-began
            if canonical(yaml.safe_load(context.model.read_text())) != canonical(result[0]):
                raise ValueError('Saved robot YAML differs from the actual returned configuration')
            if raw(out/'scene.json') != result[1] or result[2] is not None:
                raise ValueError('Current actual-table policy or scene receipt changed')
            pair[mode] = {'seconds': seconds, 'configuration': result[0], 'scene': result[1],
                          'profile': raw(out/'collision_profile.json')}
        for key in ['configuration', 'scene', 'profile']:
            if canonical(pair['control'][key]) != canonical(pair['fast_binding'][key]):
                raise ValueError('Input preparation semantics differ: '+key)
        if saved_model is not None:
            if canonical(yaml.safe_load(saved_model.read_text())) != canonical(pair['fast_binding']['configuration']):
                raise ValueError('Prepared configuration differs from the historical actual model')
            if canonical(raw(saved_scene)) != canonical(pair['fast_binding']['scene']):
                raise ValueError('Prepared collision world differs from the historical actual scene')
        rows.append({'case': case, 'arm': side, 'closed': closed, 'include_target': target,
            'seconds': {mode: value['seconds'] for mode, value in pair.items()},
            'same_all_configuration_values': True, 'same_scene_and_profile': True,
            'matches_original_actual_receipts': saved_model is not None})
        write_json(root/'progress.json', {'at': now(), 'phase': 'checking', 'completed_pairs': len(rows), 'total_pairs': len(tasks)})
    for name, digest in {**registration['source_sha256'], **inputs}.items():
        if sha256(name) != digest: raise ValueError('Serialization preflight source/input changed')
    summaries = {}
    for mode in ['control', 'fast_binding']:
        values = [row['seconds'][mode] for row in rows]
        summaries[mode] = {'calls': len(values), 'median_seconds': float(np.median(values)),
                          'p90_seconds': float(np.percentile(values, 90)), 'total_seconds': sum(values)}
    result = {'at': now(), 'passed': True, 'pairs': len(rows), 'historical_arm_models_and_scenes': 138,
        'summaries': summaries, 'cases': rows, 'codec': codec.report(),
        'new_IK_calls': 0, 'planning_requests': 0, 'physical_grasps': 0, 'simulation_instances': 0}
    write_json(root/'acceptance.json', result)
    write_json(root/'progress.json', {'at': now(), 'phase': 'complete', 'completed_pairs': len(rows), 'total_pairs': len(tasks)})
    print({k: v for k, v in result.items() if k not in ['cases', 'codec']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--root', type=Path, required=True)
    root = ensure_run_path(parser.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
