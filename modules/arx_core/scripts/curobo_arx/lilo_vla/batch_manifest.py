"""Explicit ARX fixture/schedule resolution; standard library, no device access."""
import copy
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

from interaction_contract import digest, file_sha, load, read, validate_delta, validate_episode, write

ROOT=Path(__file__).resolve().parents[3]
GPU3='GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
INITIAL_SHA='aa52dc17e576422d7aa9f679d1085464623cf4aed05835dab3831ca6b0801768'
CATALOG_SHA='2c818cf1bd8d65def0d364a067a27b83aef023378083d5b121be4ebbedbb2594'
SETUP_SHA='3da6733b9e7456004e37b5ef67b3c7ab825ba964f0f3dc75b81f1cd990d2dfe1'
VIEWPORT_SHA='14d9b524d17198b632557e0085173afc2d7960e8347c33b37aba2da82ef09cad'
FLOOR=1073741824


def resolve_manifest(path):
    path=Path(path).resolve(); value=read(path);base=path.parent
    if value['version']!='arx_teacher_batch_v1':raise ValueError('Unsupported batch manifest')
    value=copy.deepcopy(value);fixture=value['fixture'];resources=value['resources']
    def absolute(p):return str((base/p).resolve())
    for key in ('bundle','mailbox','runtime','model_reference','asset_root'):fixture[key]=absolute(fixture[key])
    for key in ('neural_python','planner_python'):
        # Resolving a venv's python symlink selects the system interpreter and
        # discards the environment's site-packages. Normalize without realpath.
        resources[key]=os.path.abspath(base/resources[key])
    for key in ('zerograsp_checkpoint','curobo_source'):resources[key]=absolute(resources[key])
    if Path(resources['curobo_source'])!=(ROOT.parent/'curobo/source').resolve():
        raise ValueError('Current teacher pipeline requires the explicit sibling cuRobo source checkout')
    fallback=Path(resources['storage_fallback_root'])
    if not fallback.is_absolute() or fallback.parent!=Path('/coolas/data'):raise ValueError('Use a dedicated fallback directory directly under /coolas/data')
    if not resources['gpu_uuid'].startswith('GPU-') or ',' in resources['gpu_uuid']:raise ValueError('Register one explicit GPU UUID')
    fixture['gpu_uuid']=resources['gpu_uuid']
    if value['budget']!={**value['budget'],'attempt_seconds':900,'cleanup_seconds':60,'encoding_seconds':120}:
        raise ValueError('Fixed attempt/cleanup/encoding budgets changed')
    if type(value['budget']['max_attempts']) is not int or value['budget']['max_attempts']<=0:raise ValueError('Invalid cumulative attempt budget')
    bundle=Path(fixture['bundle']);provenance=read(bundle/'provenance.json')
    expected={'initial_setup.json':INITIAL_SHA,'catalog.json':CATALOG_SHA,'setup.json.gz':SETUP_SHA,'viewport_initial.json':VIEWPORT_SHA}
    for name,sha in expected.items():
        raw=(bundle/name).read_bytes()
        if file_sha(bundle/name)!=provenance[name]['sha256']:raise ValueError('Fixture package changed: '+name)
        if name.endswith('.gz'):raw=gzip.decompress(raw)
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=sha or provenance[name]['original_sha256']!=sha:raise ValueError('Historical fixture identity mismatch: '+name)
    setup=json.loads(gzip.decompress((bundle/'setup.json.gz').read_bytes()))
    fixture['input_sha256']={'initial_setup':INITIAL_SHA,'catalog':CATALOG_SHA,'setup':SETUP_SHA}
    fixture['renderer']=setup['fingerprint']['invariant']['configured']['simulation']['renderer']
    if fixture['renderer']!='RaytracedLighting':raise ValueError('Unexpected fixture renderer')
    catalog=read(bundle/'catalog.json');names=[a['name'] for a in catalog]
    if len(names)!=67 or len(set(names))!=67:raise ValueError('Full ordered 67-item catalog required')
    contracts={key:load(base/p) for key,p in value['contracts'].items()}
    value['contracts']=contracts
    if not value['roles'] or any(role not in ('train','validation','development') for role in value['roles'].values()):raise ValueError('Invalid object roles')
    if set(value['roles'])-set(names):raise ValueError('Object missing from catalog')
    ids=set();groups={};external_dependencies=[]
    for case in value['cases']:
        ident=case['case_id'];name=case['object_name'];group=case.get('goal_group_id')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',ident) or ident in ids:raise ValueError('Duplicate/unsafe case_id: '+ident)
        ids.add(ident)
        if case.get('retry_of') is not None and (case['retry_of'] not in ids or case['retry_of']==ident):
            raise ValueError('Explicit retry must reference an earlier scheduled case')
        if name not in value['roles'] or case['contract'] not in contracts:raise ValueError('Case lacks object role/contract')
        validate_delta(case['entry_delta_world'])
        pos=case['placement']['position'];quat=case['placement']['orientation']
        if len(pos)!=3 or len(quat)!=4 or any(not math.isfinite(x) for x in pos+quat) or abs(sum(x*x for x in quat)-1)>1e-6:raise ValueError('Invalid registered pose')
        if case['goal_mode']=='external':
            binding=case['external_goal']
            selection=Path(absolute(binding['selection']));source=selection.parent
            registered=Path(absolute(binding['campaign_registration']))
            old=read(registered)
            if old['fixture']['input_sha256']!=fixture['input_sha256'] or old['fixture']['renderer']!=fixture['renderer']:
                raise ValueError('External planned goal belongs to another fixture')
            if file_sha(selection)!=binding['selection_sha256']:raise ValueError('External goal identity changed')
            resolved=read(source/'resolved_interaction.json');validate_episode(source,resolved['contract'])
            scenario=read(source/'scenario.json')
            if scenario['asset']!=name or scenario['placement']!=case['placement']:raise ValueError('External goal scene/object mismatch')
            binding.update(selection=str(selection),campaign_registration=str(registered))
            external_dependencies += [selection,registered,source/'resolved_interaction.json']+[source/n for n in resolved['references']]
        if case['goal_mode'] in ('fresh','external'):
            if group:
                if group in groups:raise ValueError('Only one fresh goal per group')
                groups[group]=(name,case['placement'])
        elif case['goal_mode']=='group':
            if not group or groups.get(group)!=(name,case['placement']):raise ValueError('Goal group requires preceding fresh case with same scene/object')
        else:raise ValueError('Use fresh, group or explicitly bound external planned goals')
        case['split']=value['roles'][name]
    if not ids:raise ValueError('Empty collection schedule')
    # All loader assets are dependencies, including parked objects. Do not prune.
    dependencies=[bundle/name for name in (*expected,'provenance.json','storage_policy.json')]
    dependencies+=external_dependencies
    dependencies += [base/p for p in read(path)['contracts'].values()]
    dependencies += [Path(resources[k]) for k in ('neural_python','planner_python','zerograsp_checkpoint')]
    model=Path(fixture['model_reference'])
    dependencies += [model/d/n for d in ('model','model_closed') for n in ('arx.yml','arx.urdf','model_manifest.json')]
    asset_root=Path(fixture['asset_root'])
    dependencies += [asset_root/a['container_path'].removeprefix('/workspace/asset/') for a in catalog]
    sim=setup['fingerprint']['invariant']['configured']['simulation']
    dependencies += [asset_root/sim[k].removeprefix('/workspace/asset/') for k in ('robot','environment','task_overlay')]
    dependencies += [ROOT/p for p in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml','arx-sim/config/controllers.yaml','configs/calibration.yaml')]
    missing=[str(p) for p in dependencies if not p.is_file()]
    if missing:raise FileNotFoundError('Missing explicit dependencies:\n'+'\n'.join(missing))
    if urlparse(fixture['ops_url']).scheme!='http' or not urlparse(fixture['ops_url']).port:raise ValueError('Explicit HTTP ops port required')
    value['dependency_sha256']={str(p.resolve()):file_sha(p) for p in dependencies}
    folders=(ROOT/'scripts/curobo_arx',ROOT/'arx-sim/arx_sim/diagnostics',ROOT/'servers/zerograsp',ROOT/'shared',Path(resources['curobo_source'])/'curobo')
    value['algorithm_sha256']={str(p):file_sha(p) for folder in folders for p in sorted(folder.rglob('*'))
        if p.is_file() and p.suffix in ('.py','.yml','.yaml','.sh')}
    value['manifest_sha256']=digest(value)
    return value


def verify_registration(value):
    for name,sha in {**value['dependency_sha256'],**value['algorithm_sha256']}.items():
        if file_sha(name)!=sha:raise ValueError('Registered input/algorithm changed: '+name)


def materialize_fixture(value):
    fixture=value['fixture'];runtime=Path(fixture['runtime']);bundle=Path(fixture['bundle'])
    reference=runtime/'historical_reference'
    if reference.exists():
        for source,destination in (('initial_setup.json','initial_setup.json'),('catalog.json','catalog.json'),('viewport_initial.json','viewport_initial.json'),('setup.json.gz','setup.json')):
            expected=(bundle/source).read_bytes()
            if source.endswith('.gz'):expected=gzip.decompress(expected)
            if (reference/destination).read_bytes()!=expected:raise ValueError('Existing fixture input differs: '+destination)
        return reference
    reference.mkdir(parents=True,exist_ok=False)
    for source,destination in (('initial_setup.json','initial_setup.json'),('catalog.json','catalog.json'),('viewport_initial.json','viewport_initial.json')):
        shutil.copy2(bundle/source,reference/destination)
    (reference/'setup.json').write_bytes(gzip.decompress((bundle/'setup.json.gz').read_bytes()))
    return reference


def startup(value):
    f=value['fixture'];s=f['startup'];runtime=Path(f['runtime']);reference=runtime/'historical_reference'
    env={'ARX_SIM_DEVICE':value['resources']['gpu_uuid'],'ARX_SIM_GPU':'0',
         'ARX_SIM_CONTAINER':s['container'],'ARX_SIM_IMAGE':s['image'],
         'ARX_SIM_EXTRA_RUN_ROOT':value['resources']['storage_fallback_root'],
         'ARX_SIM_ENTRYPOINT':'scripts/curobo_arx/lilo_vla/egad_start_sim.py',
         'ARX_CUROBO_VALIDATION_DIR':f['mailbox'],'ARX_TABLETOP_CATALOG':str(reference/'catalog.json'),
         'ARX_HISTORICAL_SETUP':str(reference/'initial_setup.json'),'ARX_REPRO_REFERENCE':str(reference),
         'ARX_EGAD_EFFECTIVE_CONFIG':str(runtime/'effective_config.json'),
         'ARX_REPRO_OPS_PORT':str(urlparse(f['ops_url']).port),'ARX_REPRO_ROS_DOMAIN':str(s['ros_domain']),
         'ARX_SIM_CAMERA_ROLES':'head,right_wrist'}
    command=['bash',str(ROOT/'scripts/curobo_arx/start_sim_container.sh'),'--detach','--task','grasp',
             '--webrtc-port',str(s['webrtc_port']),'--run-dir',str(runtime/'simulator_runtime')]
    return {'environment':env,'command':command}


def pipeline_command(value,trial):
    r=value['resources'];f=value['fixture']
    return [r['neural_python'],str(ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py'),
            '--output',str(trial),'--reference-run',f['model_reference'],'--mailbox',f['mailbox'],
            '--curobo-python',r['planner_python'],'--neural-gpu',r['gpu_uuid'],'--planner-gpu',r['gpu_uuid'],
            '--trajopt-policy','interactive','--no-audit','--skip-encoding']


def rehearsal(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'registration.json',value)
    write(output/'rehearsal.json',{'mode':'no-motion; no RPC, CUDA, process launch, or physical attempts',
        'physical_attempts':0,'startup':startup(value),'cases':[
            {**case,'planned_directory':str(output/'attempts'/(case['case_id']+'__<unique-attempt-id>')),
             'attempt_seconds':900,'command':pipeline_command(value,output/'attempts'/(case['case_id']+'__<unique-attempt-id>'))}
            for case in value['cases']]})
