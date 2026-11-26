"""Freeze ten inputs and twenty distinct physical slots before device selection."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, now, sha256, write_json
from .contract import CASES, changed_cases, schedule

CONFIG = ROOT/'configs/experiments/indexed_grasp_physical.json'


def raw(path): return json.loads(Path(path).read_text())


def guard(root, *, full=True):
    if (root/'STOP').exists(): raise RuntimeError('Physical comparison is stopped; preserve its records')
    reg = raw(root/'registration.json')
    values = {**reg['source_sha256'], **reg['input_sha256']} if full else reg['source_sha256']
    for path, expected in values.items():
        if sha256(path) != expected: raise ValueError('Frozen physical source/input changed: '+path)
    return reg


def prepare(root):
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    from modules.pipeline.experiments.sam3_zerograsp_comparison.cohort import check_pair
    from modules.graspgenx.postprocessing.nms import native_backend
    if root.exists(): raise ValueError('Use a fresh physical registration; keep all old results')
    cfg = raw(CONFIG);current = load_current()
    source = ROOT/cfg['source_run'];accepted = raw(source/'acceptance.json')
    reviewed = raw(source/'planning_review.json')
    if not accepted['task_complete'] or not reviewed['passed']: raise ValueError('V7 source not accepted')
    changed = changed_cases(accepted)
    if len(changed) != 6 or not set(changed).issubset(CASES):
        raise ValueError('All six changed-grasp fixtures must be included')
    if {c.split('_')[1] for c in CASES} != {'S1', 'S2', 'S3', 'S4', 'S5'}:
        raise ValueError('The fixed pilot must cover all five layouts')
    layout = ROOT/cfg['accepted_layout_run'];proof = raw(layout/'independent_review.json')
    if not all(proof[k] for k in ['passed', 'record_integrity', 'timing_gate_passed',
                                  'all_original_selections_and_branches_retained']):
        raise ValueError('Full collision-layout comparison not accepted')
    previous = raw(layout/'registration.json')
    for path, digest in previous['source_sha256'].items():
        archived = layout/'source_registered'/Path(path).relative_to(ROOT)
        if sha256(archived) != digest: raise ValueError('Accepted algorithm archive changed')
        rel = str(Path(path).relative_to(ROOT))
        if rel.startswith(('modules/curobo/', 'modules/maniskill/', 'arx_project/',
                           'modules/graspgenx/postprocessing/')) and sha256(path) != digest:
            raise ValueError('The validated planning implementation changed: '+path)
    parent = ROOT/cfg['candidate_source'];origin = raw(parent/'registration.json')
    cohort = {p['id']: p for p in origin['cohort']['pairs']}
    directories = {c['id']: Path(c['source_directory']) for c in raw(source/'registration.json')['cases']}
    root.mkdir(parents=True)
    for folder in ['inputs', 'claims', 'trials', 'preflight', 'warmup', 'source_registered']:
        (root/folder).mkdir()
    hashes = {};cases = []
    for case in CASES:
        pair = cohort[case];check_pair(pair);directory = directories[case]
        if pair.get('common_precondition_failure'): raise ValueError('A physical fixture has no valid handoff')
        if sha256(directory/'state.json') != pair['input_hashes']['state.json']:
            raise ValueError('Candidate generation and frozen handoff do not match')
        frozen = root/'inputs'/case
        for name in [*pair['input_hashes'], 'candidates/candidate_pool.json']:
            path = directory/name;target = frozen/name
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
            hashes[str(path)] = sha256(path);hashes[str(target)] = sha256(target)
        item = {**pair, 'input_directory': str(frozen.relative_to(ROOT)),
                'candidate_source_directory': str(directory.relative_to(ROOT)),
                'selection_changed_by_NMS': case in changed}
        cases.append(item)
    for name in ['object_catalog.json', 'common_initial_conditions.json', 'gripper_descriptor.json']:
        shutil.copy2(parent/name, root/name)
        hashes[str(parent/name)] = sha256(parent/name);hashes[str(root/name)] = sha256(root/name)
    warm = ROOT/'tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01/methods/zerograsp/preflight/head_warmup'
    for name in ['state.json', 'closed_state.json']:
        shutil.copy2(warm/name, root/'warmup'/name)
        hashes[str(warm/name)] = sha256(warm/name);hashes[str(root/'warmup'/name)] = sha256(root/'warmup'/name)
    for folder, names in [(source, ['registration.json', 'acceptance.json', 'planning_review.json']),
                          (layout, ['registration.json', 'acceptance.json', 'independent_review.json'])]:
        for name in names: hashes[str(folder/name)] = sha256(folder/name)
    diagnostic = ROOT/cfg['diagnostic_source']
    for path, digest in previous['input_sha256'].items():
        if Path(path).is_relative_to(diagnostic): hashes[path] = digest
    sources = {CONFIG, native_backend()[1], ROOT/'configs/modules.json',
        ROOT/'configs/experiments/active_experiment.json', ROOT/'configs/policies/fingertip_table.json',
        ROOT/'modules/maniskill/short_lift_protocol.py', ROOT/'modules/maniskill/resident_planner.py',
        ROOT/'modules/maniskill/fast_resident_planner.py', ROOT/'modules/maniskill/checked_native_services.py',
        ROOT/'modules/curobo/arx_adapter.py', ROOT/'modules/curobo/scene_cache.py',
        ROOT/'modules/pipeline/monitoring/gpu_auto_start.py', ROOT/'modules/arx_core/arms.py'}
    sources.update(Path(p) for p in current['configuration_hashes'])
    for folder in ['arx_project', 'modules/curobo/dual_arm', 'modules/curobo/indexed_planning',
        'modules/curobo/fingertip_contact', 'modules/curobo/candidate_io',
        'modules/graspgenx/postprocessing', 'modules/maniskill/arx_pipeline',
        'modules/maniskill/planning_common', 'modules/pipeline/experiments/repaired_grasp_benchmark',
        'modules/pipeline/experiments/diverse_grasp_benchmark',
        'modules/pipeline/experiments/diverse_object_scenes',
        'modules/pipeline/experiments/sam3_zerograsp_comparison',
        'modules/pipeline/experiments/indexed_grasp_planning/physical']:
        sources.update((ROOT/folder).glob('*.py'))
    sources.update((ROOT/'modules/curobo/indexed_planning/kernel_layout').glob('*.py'))
    sources.add(ROOT/'modules/curobo/indexed_planning/mesh_query/backend.py')
    sources.add(ROOT/'modules/pipeline/experiments/indexed_grasp_planning/validation_worker.py')
    sources.update(Path(p) for p in previous['source_sha256'] if '/curobo/_src/geom/' in p or p.endswith('/warp/native/mesh.h'))
    for path in sources:
        target = root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, target)
    reg = {'at': now(), **cfg, 'cases': cases, 'schedule': schedule(CASES),
        'physical_slot_budget': 20, 'maximum_attempts_per_case_per_variant': 1,
        'automatic_retries': False, 'automatic_full_benchmark': False, 'automatic_default_activation': False,
        'configuration_id': current['profile_id'], 'changed_grasp_cases': changed,
        'device_assignment_pending': True, 'gpu': None,
        'source_sha256': {str(p): sha256(p) for p in sorted(sources)}, 'input_sha256': hashes}
    write_json(root/'registration.json', reg)
    write_json(root/'progress.json', {'at': now(), 'phase': 'waiting_for_gpu',
        'completed_requests': 0, 'total_requests': 20, 'results': []})
    from .launch import monitor_policy
    write_json(root/'monitor_config.json', monitor_policy(root))
    return reg
