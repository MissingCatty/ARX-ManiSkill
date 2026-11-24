"""Register exactly 30 native-mask slots after the paired D544 decision."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices
from training.nearfield_arx.lateral_campaign import ledgers
from training.nearfield_arx import native_mask
from .advisor31_campaign import exited, runtime_dependencies
from .advisor32_spec import validate_trials
from .closedloop30_policy import validate_checkpoints
from .visual_grounding_spec import expected_reset

PREVIOUS = ROOT/'runs/nearfield_advisor33_20261111_04'
AUTHORITY = ROOT/'research_plan/chat/gemini回复/gemini回复33-补充1.md'


def selected_policy(diagnostic):
    diagnostic = Path(diagnostic)
    verification = read(diagnostic/'verification.json')
    if (not verification['complete'] or not verification['record_integrity_pass'] or
            verification['validation_observations'] != 89 or verification['scored_inferences'] != 180):
        raise ValueError('Paired inference incomplete or damaged')
    top = read(diagnostic/'registration.json')
    verify_files({**top['source_files'], **top['dependencies'], **verification['evidence']})
    from training.nearfield_arx.mask_repair_eval import decide
    selection = read(diagnostic/'selection.json')
    if selection != verification['selection'] or selection != decide(verification['by_arm'],top['selection_rule']):
        raise ValueError('Mask selection differs from preregistered rule')
    policy = copy.deepcopy(top['learned_policy'])
    changed = not selection['reuse_D544']
    evidence = [diagnostic/'registration.json', diagnostic/'verification.json', diagnostic/'selection.json']
    if changed:
        train = diagnostic/'training'
        reg = read(train/'registration.json'); receipt = read(train/'D/receipt.json')
        if (not receipt['pass'] or receipt['optimizer_steps'] != 544 or
                receipt['sample_presentations'] != 4350 or
                read(train/'D/state.json')['status'] != 'complete'):
            raise ValueError('Registered D retraining not complete')
        for key in ('source_files', 'data_files', 'image_files', 'model_files'):
            verify_files(reg[key])
        previous_initial = read(Path(top['baseline_training'])/'D/model_check.json')['initial_trainable_sha256']
        if read(train/'D/model_check.json')['initial_trainable_sha256'] != previous_initial:
            raise ValueError('Original D initialization changed')
        if sha(train/'D/training.jsonl') != receipt['training_log_sha256']:
            raise ValueError('Training log changed')
        policy['checkpoints']['D'] = str(train/'D/checkpoint-544')
        policy['training_bindings']['D'] = {'registration': str(train/'registration.json'),
                                             'sha256': sha(train/'registration.json')}
        evidence += [train/'registration.json', train/'D/receipt.json', train/'D/model_check.json',
                     train/'configuration_comparison.json']
    return policy, changed, evidence


def prepare(out, diagnostic, policy_index, compute_index, sim_index, prior_preparation=None):
    out = Path(out).resolve(); diagnostic = Path(diagnostic).resolve()
    if out.exists(): raise FileExistsError('Keep all prior registrations')
    policy, trained, selection_files = selected_policy(diagnostic)
    old = read(PREVIOUS/'registration.json'); previous = read(PREVIOUS/'advisor33_verification.json')
    if not previous['record_integrity_pass'] or previous['unknown']:
        raise ValueError('Unresolved old record chain')
    verify_files(previous['evidence'])
    verify_files({**old['execution_source_files'], **old['dependencies']})
    if not (PREVIOUS/'STOP').is_file(): raise ValueError('Historical STOP must remain')
    predecessor = exited(PREVIOUS)
    geometry = read(diagnostic/'geometry_checks/geometry_preflight.json')
    geometry_verification = read(diagnostic/'geometry_checks/verification.json')
    if not geometry_verification['pass'] or not read(diagnostic/'offline_checks.json')['pass']:
        raise ValueError('Offline checks failed')
    if (not geometry['pass'] or geometry['observations'] != 20 or
            geometry['successful_preclose_states'] != 4 or geometry['false_rejections']):
        raise ValueError('Historical successful closures rejected')
    # Archived native-view manifests may contain paths relative to the real
    # repository. Bind their exact bytes using absolute paths in the container.
    def absolute(path):
        p=Path(path)
        return str(p if p.is_absolute() else (ROOT/p).resolve())
    def file_map(files):return {absolute(p):h for p,h in files.items()}
    geometry_files=file_map(geometry_verification['files'])
    verify_files(geometry_files)
    donors=read(diagnostic/'geometry_checks/stale_donors/manifest.json')
    relocations={p:absolute(p) for p in geometry_verification['files'] if not Path(p).is_absolute()}
    for donor in donors.values():
        donor['images_by_plane']={k:[absolute(p) for p in values] for k,values in donor['images_by_plane'].items()}
        for record in donor['records']:
            record['model_images']=[absolute(p) for p in record['model_images']]
            record['files']=file_map(record['files'])
    top = read(diagnostic/'registration.json')
    out.mkdir(); space(out)
    prior = Path(top['source_snapshot']); source = out/'source_snapshot/COOLAS-ARX'
    shutil.copytree(prior, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tmp', 'curobo'): (source.parent/name).symlink_to(ROOT.parent/name, target_is_directory=True)
    execution = {str(source/Path(p).relative_to(prior)) if Path(p).is_relative_to(prior) else p: h
                 for p, h in top['source_files'].items()}
    for name in ('mask_repair_physical.py', 'native_mask_preflight.py'):
        original = ROOT/'scripts/curobo_arx/nearfield'/name
        dest = source/original.relative_to(ROOT); shutil.copy2(original, dest); execution[str(dest)] = sha(dest)
    reg = copy.deepcopy(old); trials = copy.deepcopy(old['seed_trials'])
    for trial in trials:
        trial.update(trial_id=trial['trial_id'].replace('a33_', 'a33m_', 1),
                     policy='advisor33_native_instance_mask')
    validate_trials(trials)
    for a, b in zip(old['seed_trials'], trials):
        assert {k:v for k,v in a.items() if k not in ('trial_id', 'policy')} == {
            k:v for k,v in b.items() if k not in ('trial_id', 'policy')}
    inventory = devices(); by = {d['index']: d for d in inventory}
    roles = {r+'_uuid': by[i]['uuid'] for r, i in
             (('policy', policy_index), ('compute', compute_index), ('sim', sim_index))}
    if roles['sim_uuid'] in (roles['compute_uuid'], roles['policy_uuid']):
        raise ValueError('Dedicated simulator GPU required')
    for key in ('carried_results', 'prior_results', 'prerequisite_results', 'carried_terminal_ids', 'ancestor_campaigns'):
        reg[key] = []
    now = time.time()
    reg.update(stage='advisor33_native_mask_30', continuation_of=str(PREVIOUS),
        source_snapshot=str(source), execution_source_files=execution, source_identity=digest(execution),
        seed_trials=trials, remaining_trial_ids=[t['trial_id'] for t in trials], learned_policy=policy,
        full_stale_donors=donors,
        created_unix=now, campaign_started_unix=now, deadline_unix=now+12*3600, wall_hours=12,
        requested_devices=roles, global_physical_budget=30, max_physical_trials=30,
        training_authorized_here=False, new_independent_mask_campaign=True,
        mask_diagnostic=str(diagnostic), D_retrained_for_native_mask=trained,
        execution_authority={'file':str(AUTHORITY), 'sha256':sha(AUTHORITY), 'full_30_authorized':True})
    reg['learned_policy']['gpu_uuid'] = roles['policy_uuid']
    reg['advisor33_policy']['native_mask'] = native_mask.VERSION
    validate_checkpoints(reg)
    deps = runtime_dependencies(reg, reg['seeds'])
    deps.update(geometry_files)
    for record in reg['full_stale_donors'].values():
        for item in record['records']: deps.update(item['files'])
    # Keep all original canonical model/path certificates and exact reset evidence.
    for p,h in old['dependencies'].items():
        if '/spam_model_preflight/' in p or '/spam_path_preflight/' in p: deps[p] = h
    for p in [*selection_files, AUTHORITY, PREVIOUS/'registration.json', PREVIOUS/'STOP',
              PREVIOUS/'advisor33_verification.json', PREVIOUS/'mask_repair_hold_complete.json',
              diagnostic/'geometry_checks/verification.json', diagnostic/'offline_checks.json']:
        deps[str(p)] = sha(p)
    if prior_preparation:
        failed=Path(prior_preparation).resolve()
        if (failed/'STARTED.json').exists() or not (failed/'STOP').is_file():
            raise ValueError('Preparation handoff requires zero physical starts and retained STOP')
        if read(failed/'container_preflight.json')['physical_trials']!=0:
            raise ValueError('Prior preparation is not zero-physical')
        evidence={str(p):sha(p) for p in (failed/'STOP',failed/'container_preflight.json',failed/'state.json')}
        deps.update(evidence)
        write(out/'preparation_handoff.json',{'prior_preparation':str(failed),'evidence':evidence,
            'previous_physical_trials':0,'same_original_30_slots':True,'no_grasp_retry':True})
    write(out/'absolute_path_binding.json',{'relocations':relocations,'image_bytes_changed':False,
        'original_manifests_preserved':True,'resolved_files':{v:geometry_files[v] for v in relocations.values()}})
    plan = ROOT/'research_plan/docs/nearfield_mask_repair_20261111.md'
    for name in ('PLAN.md', 'research.md'): shutil.copy2(plan, out/name)
    shutil.copy2(PREVIOUS/'storage_policy.json', out/'storage_policy.json')
    write(out/'predecessor_exit_receipt.json', predecessor)
    write(out/'retired_old_version.json', {'old_completed_retained':previous['completed'],
        'old_successes':previous['successes'], 'old_unknown':previous['unknown'],
        'cancelled_old_unexecuted':previous['unexecuted'], 'old_results_pooled':False,
        'new_slots':30, 'retries':0, 'old_STOP_preserved':True})
    write(out/'case_manifest.json', {'slots':trials, 'maximum_slots':30, 'retries':0,
        'old_slot_parameters_identical_except_new_id_and_mask_policy':True})
    write(out/'expected_resets.json', {t['trial_id']:expected_reset(reg['seeds'][t['seed_id']],t) for t in trials})
    write(out/'resource_plan.json', {'inventory':inventory, 'roles':roles, 'other_users_jobs_untouched':True})
    write(out/'source_change_receipt.json', {'native_mask':native_mask.VERSION,
        'full_stale_images_reencoded':4, 'D_retrained':trained, 'C_weights_and_prompt_unchanged':True,
        'robot_scene_camera_physics_initial_conditions_success_criteria_unchanged':True,
        'first_plane_close_gate_stop_logic_unchanged':True, 'independent_image_and_geometry_reconstruction':True})
    write(out/'sources.json', {'advisor':reg['execution_authority'], 'mask_diagnostic':str(diagnostic),
        'checkpoint_selection':read(diagnostic/'selection.json'), 'engineering_method_not_paper_reproduction':True})
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.json','.md'): deps[str(p)] = sha(p)
    command = ['podman','run','--pull=never','--rm','-i','--user','0','--security-opt=label=disable',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
        '-v','/coolas/data/arx_teacher_zhaoyechong:/coolas/data/arx_teacher_zhaoyechong:ro',
        '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1','-c',
        'import json,sys,pathlib;import scripts.curobo_arx.nearfield.runtime_advisor33; '
        'import training.nearfield_arx.native_mask; p=json.load(sys.stdin); '
        'missing=[x for x in p if not pathlib.Path(x).is_file()]; '
        'print(json.dumps({"missing":missing,"checked":len(p)}));sys.exit(bool(missing))']
    probe = subprocess.run(command, input=json.dumps(list({*deps,*execution})), capture_output=True, text=True, timeout=120)
    write(out/'container_preflight.json', {'pass':probe.returncode==0,'stdout':probe.stdout,
        'stderr':probe.stderr,'physical_trials':0,'image_download':False})
    if probe.returncode: raise RuntimeError('Container path/import check failed')
    deps[str(out/'container_preflight.json')] = sha(out/'container_preflight.json')
    reg['dependencies'] = deps
    ledgers(out,reg); write(out/'catalog.json',reg['catalog']); write(out/'registration.json',reg)
    verify_files({**execution,**deps})
    write(out/'full_preflight.json', {'pass':True,'checked':len({**execution,**deps}),'physical_trials':0})
    (out/'research_log.md').write_text('Native mask and selected D registered before 30 physical slots; original10 retained separately.\n')
    write(out/'state.json', {'status':'prepared','scheduled_slots':30,'physical_trials':0})
    return str(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('diagnostic')
    for r in ('policy','compute','sim'):p.add_argument('--'+r+'-index',type=int,required=True)
    p.add_argument('--prior-preparation')
    a=p.parse_args();print(prepare(a.root,a.diagnostic,a.policy_index,a.compute_index,a.sim_index,a.prior_preparation))
