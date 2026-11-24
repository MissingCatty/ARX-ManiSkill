"""Versioned ARX interaction semantics; importing/resolving never loads CUDA or RPC.

The contract records existing algorithm choices, not adjustable robot/physics
settings. Episode identity is separate from common semantics and runtime role.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'arx_interaction_v1'
PREPROCESSING = 'released_jpeg_lanczos3_crop90_arx_v1'
ACTION = 'arx_world_delta_pose_v1'
STATE = 'grasp_goal_world_error_v1'
BASE_REVISION = 'f54c56709004f6680d2ce8bc9e06d8cc96567ad8'
RULE_BASE = dict(cartesian_step_m=.005, rotation_step_rad=.05,
                 image_margin_pixels=32, reject_pregrasp_target_contact=True)
RULES = {
    'grasp_backoff_040mm_s02_v1': dict(RULE_BASE, version='grasp_backoff_040mm_s02_v1', ordered_backoff_m=[.04]),
    'grasp_backoff_010mm_v2': dict(RULE_BASE, version='grasp_backoff_010mm_v2', ordered_backoff_m=[.01]),
    'grasp_backoff_v1': dict(RULE_BASE, version='grasp_backoff_v1', ordered_backoff_m=[.04, .06, .08]),
}
SEMANTICS = dict(state_convention=STATE, action_convention=ACTION,
                 preprocessing_version=PREPROCESSING, camera_role='right_wrist',
                 action_shape=[8, 7], position_scale_m=.05, rotation_scale_rad=.5)
FIXED = dict(version=VERSION, reach_algorithm='target_veto_via_080mm_stop_v2',
             action_alignment_version='command_interval_v2', sample_period_seconds=.2,
             execute_steps=1, instruction_template='pick up the <display_name>',
             max_actions=90, max_local_sim_seconds=60, trial_wall_seconds=900,
             verify_closure=True, exclude_gated_dwell_starts=True,
             filter_handoff_candidates=True, handoff_candidate_budget=12, **SEMANTICS)
CONTROL_KEYS = (
    'ARX_LILO_HANDOFF_RULE_FILE', 'ARX_LILO_ENTRY_PERTURBATION_FILE',
    'ARX_LILO_REACH_TARGET_VETO', 'ARX_LILO_EXECUTE_STEPS', 'ARX_LILO_INSTRUCTION',
    'ARX_LILO_MAX_STEPS', 'ARX_LILO_MAX_SIM_SECONDS', 'ARX_LILO_VERIFY_CLOSURE',
    'ARX_LILO_HANDOFF_OBSERVATION', 'ARX_LILO_HANDOFF_PROFILE', 'ARX_LILO_INTERACTION_FILE',
    'ARX_LILO_FILTER_CANDIDATE_HANDOFF', 'ARX_LILO_HANDOFF_CANDIDATE_BUDGET',
)
DISPLAY_NAMES = {'rubiks_cube': "rubik's cube", 'lemon_01': 'lemon',
                 '005_tomato_soup_can': 'tomato soup can'}


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: _invalid('Nonfinite JSON: '+x))


def _invalid(message):
    raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load(path):
    contract = read(path) if not isinstance(path, dict) else copy.deepcopy(path)
    if set(contract) != set(FIXED) | {'handoff_rule'}:
        raise ValueError('Interaction contract has missing/unknown fields')
    for key, value in FIXED.items():
        if canonical(contract[key]) != canonical(value):
            raise ValueError('Unsupported interaction field '+key+': '+repr(contract[key]))
    if canonical(contract['handoff_rule']) not in [canonical(r) for r in RULES.values()]:
        raise ValueError('Unsupported handoff rule or modified frozen checks')
    canonical(contract)
    return contract


def instruction_for(object_name):
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError('An explicit object identity is required')
    return ('pick up the object' if object_name.startswith('egad_') else
            'pick up the '+DISPLAY_NAMES.get(object_name, object_name.replace('_', ' ')))


def validate_delta(delta):
    if not isinstance(delta, list) or len(delta) != 6 or any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in delta):
        raise ValueError('Entry perturbation requires six finite world m/rad values')
    if max(map(abs, delta[:3])) > .01 or math.sqrt(sum(x*x for x in delta[3:])) > .15:
        raise ValueError('Entry perturbation exceeds existing 1cm/0.15rad bounds')


def validate_goal(goal):
    if not isinstance(goal, list) or len(goal) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in goal):
        raise ValueError('Planned grasp must be a finite rigid 4x4 transform')
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for r in goal for x in r):
        raise ValueError('Planned grasp has nonfinite values')
    if any(abs(a-b) > 1e-8 for a,b in zip(goal[3], [0,0,0,1])):
        raise ValueError('Planned grasp has invalid homogeneous row')
    R = goal[:3]
    if any(abs(sum(R[k][i]*R[k][j] for k in range(3))-(i == j)) > 1e-5 for i in range(3) for j in range(3)):
        raise ValueError('Planned grasp rotation is not rigid')
    det = sum(R[0][i]*(R[1][(i+1)%3]*R[2][(i+2)%3]-R[1][(i+2)%3]*R[2][(i+1)%3]) for i in range(3))
    if abs(det-1) > 1e-5:
        raise ValueError('Planned grasp rotation is reflected')


def model_semantics(state_convention=STATE):
    return {**copy.deepcopy(SEMANTICS), 'state_convention': state_convention}


def adapter_compatibility(adapter, contract):
    contract = load(contract)
    adapter = Path(adapter)
    files = {n: file_sha(adapter/n) for n in ('action_head.pt','dataset_statistics.json','adapter_manifest.json')}
    manifest = read(adapter/'adapter_manifest.json')
    if manifest.get('state_convention') != contract['state_convention'] or manifest.get('base_checkpoint_revision') != BASE_REVISION:
        raise ValueError('Model state/base checkpoint does not match interaction contract')
    explicit = manifest.get('interaction_semantics')
    if explicit is not None:
        if explicit != model_semantics():
            raise ValueError('Model interaction semantics mismatch')
        source = 'explicit_adapter_manifest'
    else:
        known = read(ROOT/'configs/lilo_vla/model_compatibility.json')
        if files != known['policy_artifact_sha256'] or known['interaction_semantics'] != model_semantics():
            raise ValueError('Legacy model lacks verified interaction compatibility; no automatic defaults')
        source = 'hash_bound_legacy_record'
    if manifest.get('action_alignment_version') != contract['action_alignment_version']:
        raise ValueError('Model action alignment differs')
    return dict(compatible=True, source=source, policy_artifact_sha256=files,
                interaction_semantics=model_semantics(), physical_replay='Not performed by this compatibility check')


def check_health(resolution, health):
    if health.get('interaction_semantics') != model_semantics():
        raise ValueError('Policy service input/action/preprocessing semantics mismatch')
    if health.get('state_convention') != STATE or health.get('action_shape') != [8,7]:
        raise ValueError('Policy service state/action shape mismatch')
    binding = resolution.get('model_compatibility')
    if binding and health.get('policy_artifact_sha256') != binding['policy_artifact_sha256']:
        raise ValueError('Loaded policy artifacts differ from resolved adapter')


def reject_overrides(environment):
    present = [key for key in CONTROL_KEYS if key in environment]
    if present:
        raise ValueError('External interaction overrides conflict with --interaction-contract: '+', '.join(present))


def resolve(contract, object_name, role, *, delta=None, goal_source=None, adapter=None):
    contract = load(contract)
    if role not in ('teacher', 'policy'):
        raise ValueError('Unknown interaction role')
    delta = [0.]*6 if delta is None else list(delta)
    validate_delta(delta)
    goal = None
    source = {'kind': 'fresh_zerograsp', 'status': 'pending_selection'}
    if goal_source:
        p = Path(goal_source).resolve()
        scenario = read(p.parent/'scenario.json')
        if scenario['asset'] != object_name:
            raise ValueError('Fixed planned goal belongs to another object')
        goal = read(p)['grasp_world']; validate_goal(goal)
        source = dict(kind='fixed_planned_grasp', path=str(p), sha256=file_sha(p), status='selected_before_motion')
    return dict(version=VERSION, contract=contract, contract_sha256=digest(contract),
                object_name=object_name, instruction=instruction_for(object_name), role=role,
                entry_delta_world=delta, goal_source=source, grasp_world=goal,
                references={}, qualification='unverified; geometric checks do not qualify a learned policy',
                model_compatibility=adapter_compatibility(adapter, contract) if adapter else None)


def validate_resolution(value, require_goal=False):
    contract = load(value['contract'])
    if value['version'] != VERSION or value['contract_sha256'] != digest(contract):
        raise ValueError('Resolved interaction contract identity changed')
    if value['role'] not in ('teacher','policy') or value['instruction'] != instruction_for(value['object_name']):
        raise ValueError('Resolved interaction role/instruction mismatch')
    validate_delta(value['entry_delta_world'])
    source = value['goal_source']
    if source['kind'] not in ('fresh_zerograsp','fixed_planned_grasp'):
        raise ValueError('Unknown planned goal source')
    if value['grasp_world'] is not None:
        validate_goal(value['grasp_world'])
        if source['status'] != 'selected_before_motion':
            raise ValueError('Selected goal has no pre-motion provenance')
    elif require_goal:
        raise ValueError('Planned goal has not been frozen before motion')
    if source['kind'] == 'fixed_planned_grasp':
        if file_sha(source['path']) != source['sha256'] or read(source['path'])['grasp_world'] != value['grasp_world']:
            raise ValueError('Fixed planned goal source changed')
    return value


def environment_values(value, directory):
    c = value['contract']; directory = Path(directory).resolve()
    return dict(ARX_LILO_INTERACTION_FILE=str(directory/'resolved_interaction.json'),
                ARX_LILO_HANDOFF_RULE_FILE=str(directory/'interaction/handoff_rule.json'),
                ARX_LILO_ENTRY_PERTURBATION_FILE=str(directory/'interaction/entry_perturbation.json'),
                ARX_LILO_REACH_TARGET_VETO='1', ARX_LILO_EXECUTE_STEPS=str(c['execute_steps']),
                ARX_LILO_INSTRUCTION=value['instruction'], ARX_LILO_MAX_STEPS=str(c['max_actions']),
                ARX_LILO_MAX_SIM_SECONDS=str(c['max_local_sim_seconds']), ARX_LILO_VERIFY_CLOSURE='1',
                ARX_LILO_FILTER_CANDIDATE_HANDOFF='1', ARX_LILO_HANDOFF_CANDIDATE_BUDGET=str(c['handoff_candidate_budget']))


def prepare_environment(environment, directory, value):
    validate_resolution(value)
    directory = Path(directory).resolve(); (directory/'interaction').mkdir(exist_ok=False)
    write(directory/'resolved_interaction.json', value)
    write(directory/'interaction/handoff_rule.json', value['contract']['handoff_rule'])
    write(directory/'interaction/entry_perturbation.json', {'delta_world': value['entry_delta_world']})
    return {**environment, **environment_values(value, directory)}


def runtime(environment=None):
    environment = os.environ if environment is None else environment
    path = environment.get('ARX_LILO_INTERACTION_FILE')
    if not path:
        return None
    value = validate_resolution(read(path))
    if value['role'] == 'policy' and not (value.get('model_compatibility') or {}).get('compatible'):
        raise ValueError('Policy interaction needs an explicitly verified adapter binding')
    if value['role'] == 'teacher' and environment.get('ARX_LILO_POLICY_URL'):
        raise ValueError('Teacher interaction cannot invoke a policy controller')
    if value['role'] == 'policy' and environment.get('ARX_LILO_TEACHER_DIR'):
        raise ValueError('Policy interaction cannot invoke a teacher controller')
    for key, expected in environment_values(value, Path(path).parent).items():
        if environment.get(key) != expected:
            raise ValueError('Runtime interaction override: '+key)
    for key in ('ARX_LILO_HANDOFF_OBSERVATION','ARX_LILO_HANDOFF_PROFILE'):
        if key in environment:
            raise ValueError('Legacy handoff source conflicts with resolved interaction: '+key)
    if read(environment['ARX_LILO_HANDOFF_RULE_FILE']) != value['contract']['handoff_rule']:
        raise ValueError('Runtime handoff rule changed')
    if read(environment['ARX_LILO_ENTRY_PERTURBATION_FILE']) != {'delta_world':value['entry_delta_world']}:
        raise ValueError('Runtime entry perturbation changed')
    return value


def freeze_goal(directory, value, selected):
    """Called after perception, before the first motion; no future EE pose is read."""
    validate_resolution(value)
    goal = selected['grasp_world']; validate_goal(goal)
    if selected.get('handoff_rule') != value['contract']['handoff_rule']['version']:
        raise ValueError('Selected grasp uses another handoff rule')
    if value['grasp_world'] is not None and value['grasp_world'] != goal:
        raise ValueError('Selected goal differs from resolved planned goal')
    value = copy.deepcopy(value); value['grasp_world'] = goal
    value['goal_source']['status'] = 'selected_before_motion'
    value['selected_grasp_sha256'] = digest(selected)
    directory = Path(directory)
    if read(directory/'scenario.json')['asset'] != value['object_name']:
        raise ValueError('Resolved object differs from actual scenario')
    if not read(directory/'setup_verification.json')['pass']:
        raise ValueError('Actual initial conditions were not verified')
    value['references'] = {name:file_sha(directory/name) for name in
        ('scenario.json','setup.json','setup_verification.json','active_asset.json','viewport_initial.json')}
    value['episode_sha256'] = digest({k:value[k] for k in
        ('contract_sha256','object_name','instruction','entry_delta_world','grasp_world','references')})
    validate_resolution(value, require_goal=True)
    write(directory/'resolved_interaction.json', value)
    return value


def validate_episode(source, contract):
    source = Path(source)
    p = source/'resolved_interaction.json'
    if not p.is_file():
        raise ValueError('Episode has no recorded interaction contract; legacy data cannot be relabeled: '+str(source))
    value = validate_resolution(read(p), require_goal=True)
    if value['contract_sha256'] != digest(load(contract)):
        raise ValueError('Episode interaction contract differs: '+str(source))
    if value['role'] != 'teacher' or read(source/'scenario.json')['asset'] != value['object_name']:
        raise ValueError('Episode role/object mismatch')
    selected = read(source/'selected_grasp.json')
    if digest(selected) != value.get('selected_grasp_sha256'):
        raise ValueError('Episode selected grasp metadata changed')
    if selected['grasp_world'] != value['grasp_world']:
        raise ValueError('Episode planned goal changed')
    for name, expected in value['references'].items():
        if file_sha(source/name) != expected:
            raise ValueError('Episode reference changed: '+name)
    identity = digest({k:value[k] for k in ('contract_sha256','object_name','instruction','entry_delta_world','grasp_world','references')})
    if value.get('episode_sha256') != identity:
        raise ValueError('Episode identity changed')
    return value


def resolve_cli(argv, role):
    parser = argparse.ArgumentParser(description='Resolve ARX semantics only; no GPU, model, RPC, output directory or instance lock')
    parser.add_argument('--resolve-only', action='store_true', required=True)
    parser.add_argument('--interaction-contract', type=Path, required=True)
    parser.add_argument('--object-name', required=True)
    parser.add_argument('--entry-delta-world', type=json.loads, default=[0.]*6)
    parser.add_argument('--goal-source', type=Path)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args(argv)
    reject_overrides(os.environ)
    value = resolve(args.interaction_contract, args.object_name, role,
                    delta=args.entry_delta_world, goal_source=args.goal_source, adapter=args.adapter)
    print(json.dumps(value, indent=2, allow_nan=False))
