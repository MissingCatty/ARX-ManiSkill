"""Fixed-goal S02 registration and runtime evidence gates; no device initialization."""
import copy
import json
from pathlib import Path

from interaction_contract import ROOT, RULES, read, file_sha, digest, load, resolve, write


def registration(design_file):
    design=read(design_file)
    if design['version'] != 'arx_s02_design_v1':
        raise ValueError('Unsupported S02 design version')
    if design['rules']!={'R10':RULES['grasp_backoff_010mm_v2'],'R40':RULES['grasp_backoff_040mm_s02_v1']}:
        raise ValueError('S02 requires pure 10mm and pure 40mm rules')
    if not design['fixed_goal_only'] or design['goal_fallback_allowed'] or design['automatic_entry_fallback_allowed']:
        raise ValueError('S02 forbids fallback or fresh goal substitution')
    adapter=ROOT/design['baseline_adapter']
    base=load(ROOT/'configs/lilo_vla/handoff_10mm.json')
    cases={}; dependencies={}
    for case in design['cases']:
        case=copy.deepcopy(case)
        for key,sha_key in (('scenario_source','scenario_sha256'),('setup_source','setup_source_sha256'),
                            ('fixed_goal_source','fixed_goal_file_sha256')):
            path=ROOT/case[key]
            if file_sha(path)!=case[sha_key]:raise ValueError('S02 source changed: '+str(path))
            dependencies[str(path)]=case[sha_key]
        scenario=read(ROOT/case['scenario_source']);setup=read(ROOT/case['setup_source'])
        if scenario['asset']!=case['asset']:raise ValueError('S02 object identity differs')
        if digest(read(ROOT/case['fixed_goal_source'])['grasp_world'])!=case['fixed_goal_matrix_sha256']:
            raise ValueError('S02 goal matrix changed')
        case['resolved']={arm:resolve({**base,'handoff_rule':rule},case['asset'],'policy',
            delta=case['entry_perturbation_world'],goal_source=ROOT/case['fixed_goal_source'],adapter=adapter)
            for arm,rule in design['rules'].items()}
        case['renderer']=setup['fingerprint']['invariant']['configured']['simulation']['renderer']
        case['simulator_seed']=setup['fingerprint']['invariant']['configured']['simulation']['seed']
        cases[case['case_id']]=case
    pilot=[]
    for index,case in enumerate(c for c in cases.values() if c['cohort']=='core'):
        for arm in (('R10','R40') if index%2==0 else ('R40','R10')):
            pilot.append(dict(trial_id='pilot__'+case['case_id']+'__'+arm,case_id=case['case_id'],arm=arm,phase='pilot'))
    schedule=pilot+[{**row,'phase':'main'} for row in design['planned_main_schedule']]
    if len(pilot)!=4 or len(schedule)!=36:raise ValueError('S02 trial budget differs')
    if len({row['trial_id'] for row in schedule})!=36 or any('/' in row['trial_id'] or '..' in row['trial_id'] for row in schedule):
        raise ValueError('S02 trial identities must be unique and safe')
    for case in cases.values():
        for arm in ('R10','R40'):
            count=sum(r['case_id']==case['case_id'] and r['arm']==arm for r in design['planned_main_schedule'])
            if count!=case['paired_repeats']:raise ValueError('Unbalanced S02 schedule')
    if any(c['entry_perturbation_world'] != [0.]*6 for c in cases.values()):raise ValueError('S02 forbids entry jitter')
    for row in read(ROOT/'runs/research_s01_20261029_05/configuration_evidence.json'):
        if file_sha(row['path'])!=row['registered_sha256']:raise ValueError('Protected source differs: '+row['path'])
        dependencies[row['path']]=row['registered_sha256']
    for name in ('action_head.pt','dataset_statistics.json','adapter_manifest.json'):
        dependencies[str(adapter/name)]=file_sha(adapter/name)
    for case in cases.values():
        view=ROOT/case['source_trial']/'viewport_initial.json'
        dependencies[str(view)]=file_sha(view)
    for path in (Path(design_file).resolve(),ROOT/'configs/lilo_vla/handoff_10mm.json',ROOT/'configs/lilo_vla/handoff_40mm_s02.json'):
        dependencies[str(path)]=file_sha(path)
    algorithms={str(p):file_sha(p) for folder in (ROOT/'scripts/curobo_arx',ROOT/'arx-sim/arx_sim/diagnostics',
                  ROOT/'servers/lilo_vla',ROOT.parent/'curobo/source/curobo') for p in folder.rglob('*.py')}
    return dict(version='arx_s02_execution_v3',gate_version='arx_s02_gates_v3',
        pilot_coverage_policy='Each critical event observed and verified in each arm and each core case; N/A is never coverage',design_sha256=file_sha(design_file),cases=cases,schedule=schedule,
        baseline_adapter=str(adapter),dependencies=dependencies,algorithms=algorithms,
        budget=design['budget'],qualification='Entry diagnosis only; does not extend P3 admission profile',
        rng=dict(planner_python=0,planner_numpy=0,planner_torch=0,source='Existing grasp_execute.py; planner.reset_seed() retained',
                 scope='Same supported seeds within pairs; GPU numerical determinism is not asserted'))


def verify_sources(value):
    for path,expected in {**value['dependencies'],**value['algorithms']}.items():
        if file_sha(path)!=expected:raise ValueError('Registered source changed: '+path)


def check_runtime(before,after):
    from egad49 import compare_configuration
    mismatches=compare_configuration(before,after)
    # Reuse the historical comparator; mounted camera world transforms move
    # with the arm, whereas calibrated transforms are protected in configured.
    for label,fp in (('before',before),('after',after)):
        invariant=fp['invariant']
        if invariant.get('runtime_api_errors'):mismatches.append(label+'/runtime_api_errors')
        if len(invariant.get('runtime_robot',{}))!=12:mismatches.append(label+'/runtime_robot_coverage')
        if not invariant.get('runtime_objects'):mismatches.append(label+'/runtime_objects_missing')
    a=before['invariant'];b=after['invariant']
    if a['configured']['simulation']['camera_roles']!=b['configured']['simulation']['camera_roles']:
        mismatches.append('camera_roles')
    return dict(pass_check=not mismatches,mismatches=mismatches,
        scope='Composed physical properties and all 12 available runtime getters; not inferred from file hashes')


def check_setup(historical,current):
    result=check_runtime(historical['fingerprint'],current['fingerprint'])
    if historical['initial_conditions']!=current['initial_conditions']:
        result['mismatches'].append('full_declared_initial_conditions')
    if current['physics_steps']!=360 or abs(current['captured_sim_time']-current['reset_sim_time']-3)>.001:
        result['mismatches'].append('historical_settle')
    result['pass_check']=not result['mismatches']
    return result
