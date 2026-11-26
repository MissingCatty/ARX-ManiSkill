"""Reuse frozen input registration, then bind the explicitly authorized raw order."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, sha256, write_json
from ..physical.registration import guard, raw

CONFIG = ROOT/'configs/experiments/indexed_grasp_raw_order.json'


def add_sources(root, reg):
    sources = {CONFIG, *Path(__file__).parent.glob('*.py')}
    sources.update((ROOT/'modules/curobo/indexed_planning/kernel_layout').glob('*.py'))
    sources.add(ROOT/'modules/curobo/indexed_planning/mesh_query/backend.py')
    sources.add(ROOT/'modules/curobo/indexed_planning/execution_service.py')
    # These are the already validated geometry/backend sources, not new kernels.
    previous = raw(ROOT/'tmp/indexed_grasp_planning/layout69_20261126_01/registration.json')
    sources.update(Path(p) for p in previous['source_sha256']
                   if '/curobo/_src/geom/' in p or p.endswith('/warp/native/mesh.h'))
    for path in sources:
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        reg['source_sha256'][str(path)] = sha256(path)
    write_json(root/'registration.json', reg)


def prepare_planning(root):
    from ..validation_replay import prepare
    prepare(root, CONFIG)
    reg = raw(root/'registration.json')
    reg.update(campaign_stage='planning', original_candidate_order_required=True,
        NMS_enabled=False, automatic_default_activation=False,
        followup_physical_directory=raw(CONFIG)['followup_physical_directory'])
    for case in reg['cases']:
        case['input_directory'] = str(Path(case['source_directory']).relative_to(ROOT))
    add_sources(root, reg)
    from .launch import monitor_policy
    write_json(root/'monitor_config.json', monitor_policy(root))
    reg['input_sha256'][str(root/'monitor_config.json')] = sha256(root/'monitor_config.json')
    write_json(root/'registration.json', reg); write_json(root/'registration_pending.json', reg)
    write_json(root/'research.md', '# Original-order V8 computation comparison\n\n'
        '## Goal\nPreserve original choices and physical successes while reducing planning latency.\n\n'
        '## Success Metric\n69 matched planning pairs; if retained,10 matched physical pairs.\n\n'
        '## Constraints\n- max_iterations:1\n- pause_every:never\n- keep_policy:score_improvement\n'
        '- Evaluator: raw_order.review plus original independent physical review\n'
        '- guard: original order,grasp,arm,physics,871,TCP43,V2 and resource thresholds unchanged\n'
        '- budgets:138 planning requests,20 physical slots,one attempt,no automatic retries\n'
        '- campaign/RPC1800s deadlines supersede shell setup timeout\n'
        '- no automatic default activation or200-scene extension\n\n'
        '## Current Approach\nBaseline scalar; optimized B16+C YAML+obstacle-major queries; both raw order.\n\n'
        '## Search Space\nOne registered combination, no parameter search.\n\n'
        '## History\n| Iteration | Work | Result | Status |\n| --- | --- | --- | --- |\n')
    write_json(root/'progress.json', {'at':now(),'phase':'waiting_for_gpu','completed_requests':0,
        'total_requests':138,'results':[]})
    return reg


def physical_gate(proof):
    return all(proof.get(k) is True for k in ['passed', 'record_integrity',
        'control_successes_retained','original_order_verified','original_grasps_and_arms_retained',
        'joint_endpoints_verified','timing_gate_passed']) and proof.get('checked_requests') == 138


def prepare_physical(root, planning):
    from ..physical.registration import prepare
    proof = raw(planning/'independent_review.json')
    if not physical_gate(proof) or (planning/'STOP').exists():
        raise ValueError('Planning review did not authorize the physical follow-up')
    prepare(root)
    reg = raw(root/'registration.json'); cfg = raw(CONFIG)
    reg.update(at=now(),experiment='raw_order_computation_physical_pilot',
        authorization=cfg['authorization'],campaign_stage='physical',
        optimized='original_order_B16_C_YAML_obstacle_major_queries',NMS_enabled=False,
        original_candidate_order_required=True, planning_gate_source=str(planning.relative_to(ROOT)),
        automatic_default_activation=False)
    for name in ['registration.json','acceptance.json','independent_review.json']:
        reg['input_sha256'][str(planning/name)] = sha256(planning/name)
    add_sources(root, reg)
    from .launch import monitor_policy
    write_json(root/'monitor_config.json', monitor_policy(root))
    reg['input_sha256'][str(root/'monitor_config.json')] = sha256(root/'monitor_config.json')
    write_json(root/'registration.json', reg)
    write_json(root/'progress.json', {'at':now(),'phase':'waiting_for_gpu','completed_requests':0,
        'total_requests':20,'results':[]})
    return reg
