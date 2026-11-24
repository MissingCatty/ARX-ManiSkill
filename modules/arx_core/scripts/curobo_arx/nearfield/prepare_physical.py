"""Bound the physical-data return to six train and three held-out ARX sources."""
import argparse
import copy
from pathlib import Path
import shutil
import time
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest, verify_files, space
from scripts.curobo_arx.data_factory.ledger import Ledger
from .prepare_v2 import predecessors
from .physical_spec import VERSION, physical_trials, MAX_TEACHER_STEPS

TRAIN = ROOT/'runs/nearfield_stage1_20261105_04/recovery_24'
HELD = ROOT/'runs/nearfield_validation_20261106_03'
DATA = ROOT/'runs/nearfield_dataset_20261106_03'
AUTH = ROOT/'research_plan/chat/gemini回复/gemini回复20.md'


def source_inventory():
    train, held = read(TRAIN/'registration.json'), read(HELD/'registration.json')
    if len(train['seeds']) != 6 or len(held['seeds']) != 3:
        raise RuntimeError('Original admitted source pool changed')
    sources = []; seeds = {}; families = {}; geometries = {}
    for split, reg in (('train', train), ('validation', held)):
        for seed_id, seed in sorted(reg['seeds'].items()):
            verify_files(seed['files'])
            trial = next(t for t in reg['seed_trials'] if t['seed_id'] == seed_id)
            asset = next(a for a in reg['catalog']['objects'] if a['object_id'] == seed['object_id'])
            family = seed.get('source_family', seed_id)
            geometry = asset['geometry_group']
            if asset['split'] != split or families.get(family, split) != split or geometries.get(geometry, split) != split:
                raise RuntimeError('Source family/geometry split mismatch')
            families[family] = split; geometries[geometry] = split
            if seed_id in seeds:raise RuntimeError('Cross-split seed collision')
            seeds[seed_id] = seed
            sources.append({'seed_id':seed_id, 'source_family':family,
                'source_split':split, 'geometry_group':geometry,
                'object_id':seed['object_id'], 'renderer':seed['renderer'],
                'placement':seed['placement'], 'source_Gstar_world':trial['source_Gstar_world']})
    return sources, seeds, families, geometries


def prepare(out, compute_uuid, sim_uuid, offline_checks):
    out = Path(out).resolve(); offline_checks = Path(offline_checks).resolve()
    if out.exists():raise FileExistsError('Preserve all preceding registrations')
    checks = read(offline_checks)
    if not checks['pass']:raise RuntimeError('Offline checks must pass first')
    verify_files(checks['sources'])
    old = read(TRAIN/'registration.json'); held = read(HELD/'registration.json')
    proof = predecessors([TRAIN, HELD])
    for reg in (old, held):verify_files({**reg['execution_source_files'], **reg['dependencies']})
    sources, seeds, families, geometries = source_inventory(); trials = physical_trials(sources)
    assert len(trials) == 36
    out.mkdir(parents=True); space(out)
    write(out/'state.json', {'status':'preparing', 'physical_trials':0})
    previous = Path(old['source_snapshot']); source = out/'source_snapshot/COOLAS-ARX'
    source.parent.mkdir()
    shutil.copytree(previous, source, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo', target_is_directory=True)
    new_files = ('physical_spec.py', 'runtime_physical.py', 'prepare_physical.py', 'test_physical.py')
    for name in new_files:shutil.copy2(Path(__file__).with_name(name), source/'scripts/curobo_arx/nearfield'/name)
    execution = {str(source/Path(p).relative_to(previous) if Path(p).is_relative_to(previous) else p):h
                 for p,h in old['execution_source_files'].items()}
    for name in new_files:
        f = source/'scripts/curobo_arx/nearfield'/name; execution[str(f)] = sha(f)
    reg = copy.deepcopy(old); deps = {**old['dependencies'], **held['dependencies']}
    for key in ('bulk_storage','resume','lineage','controlled_resource_pause'):reg.pop(key, None)
    for seed in seeds.values():deps.update(seed['files'])
    for name, path in old['reference_files'].items():
        dest = out/'references'/(name+'.json'); dest.parent.mkdir(exist_ok=True); shutil.copy2(path,dest)
        reg['reference_files'][name] = str(dest); deps[str(dest)] = sha(dest)
    for origin, name in ((TRAIN/'storage_policy.json','storage_policy.json'),(AUTH,'advisor20.md'),(offline_checks,'offline_checks.json')):
        shutil.copy2(origin,out/name); deps[str(out/name)] = sha(out/name)
    for root in (TRAIN, HELD):
        for name in ('registration.json','state.json','summary.json','postrun_verification.json'):
            f = root/name
            if f.exists():deps[str(f)] = sha(f)
    deps[str(DATA/'manifest.json')] = sha(DATA/'manifest.json')
    for trial in trials:
        catalog = copy.deepcopy(old['catalog'])
        for asset in catalog['objects']:
            asset['collection_enabled'] = asset['object_id'] == trial['object_id']
            if asset['collection_enabled']:asset['collection_placements'] = [trial['placement']]
        path = out/'trial_ledgers'/(trial['trial_id']+'.sqlite3'); path.parent.mkdir(exist_ok=True)
        ledger = Ledger(path)
        ledger.initialize(catalog,{**old['quotas'],'first_object':trial['object_id'],
            'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
            'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1})
        ledger.close()
    reg.update(stage=VERSION, created_unix=time.time(), campaign_started_unix=time.time(),
        deadline_unix=time.time()+12*3600, wall_hours=12, production=False,
        seeds=seeds, seed_trials=trials, source_snapshot=str(source),
        execution_source_files=execution, source_identity=digest(execution), dependencies=deps,
        runtime_module='scripts.curobo_arx.nearfield.runtime_physical',
        worker_module='scripts.curobo_arx.nearfield.worker', envs_per_gpu=1,
        max_physical_trials=36, max_replay_trials=0, replay_slots_total=0,
        recovery_slots_total=36, carried_results=[], prior_results=[], prerequisite_results=[],
        no_automatic_retry=True, training_authorized_here=False, formal_48_trials_authorized_here=False,
        requested_devices={'compute_uuid':compute_uuid,'sim_uuid':sim_uuid},
        source_family_splits=families, source_geometry_splits=geometries,
        physical_data_contract={'train_slots':24,'validation_slots':12,'max_teacher_steps':MAX_TEACHER_STEPS,
            'one_actual_prior_per_observation':True,'virtual_variants':0,
            'positive_segment':'Fresh physical_entry_stopped and subsequent accepted teacher motion only',
            'teacher_limits':{'position_m':.020,'angle_degrees':4.5},
            'source_Gstar_usage':'Offline teacher only; export labels from this successful execution preclosure measurement',
            'model_inputs':'Unchanged RGB plus depth/three-role visibility projection; instance truth audit only',
            'interface':'Original C STEP / D fixed-G0 REFINE / explicit GRASP',
            'frozen_robot_scene_physics_and_success_gates':True})
    write(out/'predecessor_exit_receipt.json',proof)
    write(out/'source_change_receipt.json',{'parent':str(TRAIN),'base_source_identity':old['source_identity'],
        'new_execution_files':new_files,'inherited_runtime_planner_capture_evaluator_bytes_unchanged':True,
        'changes':'New registered motion goals and bounded offline teacher; no robot/scene/camera/physics or success change'})
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    write(out/'sources.json',{'advisor_file':str(out/'advisor20.md'),'source_pool':sources,
        'hypothesis':'Physical entry pose variation provides visual geometry absent from fixed-image prior-only variants',
        'method_scope':'ARX teacher data collection; no learned-policy success claimed'})
    (out/'research.md').write_text('# Real physical entry data\n\n36 fixed slots: six train sources x4 and three held-out sources x4. Each reaches H=G0@Tz(-d) through original cuRobo checks and records new synchronized views. Biases5/10/15degrees and lateral10/15/20mm; near recovery5/10/20mm. Only one actual registered G0 per observation. Original scene/reset, robot, projection and success gates frozen. No retries or replacement sources. Ordinary failures remain; infrastructure/configuration/record/idle failures stop. Subsequent training and learned closed-loop validation require separate bounded registrations using the admitted frozen export.\n')
    (out/'research_log.md').write_text('Registered before any new physical command. Advisor20 supersedes the cancelled virtual-prior state-token comparison.\n')
    verify_files({**execution,**deps})
    write(out/'state.json',{'status':'prepared','scheduled_slots':36,'physical_trials':0})
    return str(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--compute-uuid',required=True)
    p.add_argument('--sim-uuid',required=True);p.add_argument('--offline-checks',required=True)
    a=p.parse_args();print(prepare(a.out,a.compute_uuid,a.sim_uuid,a.offline_checks))
