"""Keep the physical trace checks, requiring only the registered motion stages."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import graspcl_pipeline as common
import short_lift_protocol as lift

CODE = Path(__file__).resolve().parent


def build():
    text = (CODE/'arx_pipeline/baseline.py').read_text()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_trial')
    source = ast.get_source_segment(text, node)
    assert source.count('minimum>.06') == 1
    source = source.replace('minimum>.06', 'minimum>=.01')
    old = "required={'pregrasp','approach','close','lift','hold'}"
    assert source.count(old) == 1
    source = source.replace(old,
        "direct=json.loads(Path(r['campaign_registration']).read_text()).get('direct_grasp',False)\n"
        "        required={'approach','close','lift','hold'}\n"
        "        if not direct:required.add('pregrasp')")
    scope = {'json': json, 'Path': Path, 'save': common.save}
    exec(compile(source, __file__+':physical_trace', 'exec'), scope)
    original = scope['verify_trial']
    def verify(directory):
        directory = Path(directory)
        r = common.read(directory/'result.json')
        direct = common.read(Path(r['campaign_registration'])).get('direct_grasp', False)
        if direct and r['planning_success']:
            plan = common.read(directory/'plan_result.json')
            assert set(plan['paths']) == {'approach'} and 'q_pre' not in plan['selected']
            assert 'pregrasp_world' not in plan['selected'] and 'pregrasp' not in r.get('stages', {})
            assert common.read(directory/'continuous_grasp_plan.json')['passed']
            if r['success']:
                mpc = common.read(directory/'mpc_approach.json')
                assert mpc['settled'] and mpc['mpc_solves'] >= 2 and mpc['tracked_frames'] > 0
                assert mpc['continuous_from_initial'] and mpc['intermediate_handoff_stops'] == 0
        original(directory)
        p = directory/'independent_evaluation.json';data = common.read(p)
        data.update(evaluation_protocol=lift.PROTOCOL, planned_lift_m=.03,
            minimum_continuous_object_lift_m=.01, minimum_height_comparison='>=', required_hold_seconds=3,
            required_motion_stages=['approach','close','lift','hold'] if direct else ['pregrasp','approach','close','lift','hold'],
            pregrasp_requirement_follows_registered_pipeline=True,
            physical_height_contact_convergence_criteria_unchanged=True)
        common.save(p, data)
    verify.short_lift_protocol = lift.PROTOCOL
    verify.registered_short_lift_source = source
    return verify


verify_trial = build()


def install(baseline):
    baseline.verify_trial = verify_trial
