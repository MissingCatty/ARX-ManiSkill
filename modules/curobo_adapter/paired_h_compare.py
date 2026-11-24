"""Register and run matched old/new offline H-entry planning, serial per goal set."""
from arx_project.layout import canonical_path
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
MS=HERE.parent/'maniskill';ARX=HERE.parent/'arx_core'
sys.path.insert(0,str(MS))
from resident_backend import rpc,save
from benchmark_lifecycle import mark_ready,require_ready


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now().astimezone().isoformat()


def prepare(out):
    old=read(HERE/'research/20261117_02/scope.json')
    inherited=read(HERE/'research/20261117_06/native_registration.json')['hashes']
    for path,h in inherited.items():assert sha(path)==h,path
    assert not subprocess.check_output(['git','-C',str(HERE/'source'),'diff','--name-only'],text=True).strip()
    out.mkdir(parents=True,exist_ok=False)
    for name in ['resident','results','claims','logs','sources']:(out/name).mkdir()
    from planner_cli import build_frozen_models
    warm=Path(read(MS/'benchmark_inputs/arx_native_post_20261117_01/manifest.json')['warmup_trial'])
    build_frozen_models(read(warm/'state.json'),out/'warmup_input')
    files=[HERE/name for name in ['paired_h_worker.py','paired_h_compare.py','entry_selector.py','environment.sh','run_isolated.sh']]
    files+=[MS/'resident_backend.py',MS/'benchmark_lifecycle.py',MS/'planning_common/grasp_selector.py',ARX/'scripts/curobo_arx/gripper_update.py',ARX/'scripts/curobo_arx/optimize_path.py']
    hashes={**inherited,**{str(p):sha(p) for p in files}}
    for p in (out/'warmup_input').rglob('*'):
        if p.is_file():hashes[str(p)]=sha(p)
    for p in files:
        target=out/'sources'/p.relative_to(HERE.parent);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    legacy=HERE.parent/'curobo/source'
    diff=subprocess.check_output(['git','-C',str(legacy),'diff','--binary'])
    (out/'legacy_patch_snapshot.diff').write_bytes(diff)
    key=hashlib.sha256(str(out).encode()).hexdigest()[:10];socket_dir=MS/'tmp'/('Hpair_'+key)
    reg={'registered_at':now(),'source_campaign':old['source_campaign'],'cases':old['cases'],
         'versions':['old','new'],'required_components':['old','new'],'source_hashes':hashes,
         'runtime_hashes':{str(p):sha(p) for p in files},
         'warmup_input':str(out/'warmup_input'),
         'version_sources':{'old':str(legacy),'new':str(HERE/'source')},
         'version_expected':{'old':'0.8.0.post1.dev42','new':'0.9.0'},
         'interpreters':{'old':str(HERE.parent/'curobo/.venv/bin/python'),'new':str(HERE/'.venv/bin/python')},
         'gpu_uuid':'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169','gpu_index_at_registration':0,
         'free_mib_floor':8192,'maximum_queries_per_version':300,'maximum_queries_total':600,
         'maximum_PRM_attempts_per_case':8,'maximum_wall_seconds':2400,
         'source_configuration':'Original archived robot/state/scene files, stable native score order and original G-to-H conversion',
         'matched_settings':{'IK_seeds':32,'seed':123,'position_m':.002,'orientation_rad':.03,
                             'collision_cache_mesh':8,'collision_cache_primitive':8,'self_collision':False,
                             'optimizer_collision_activation_m':.003,'candidate_filters':'none',
                             'PRM_samples':201,'PRM_dense_factor':5,'optimized_dense_factor':10,
                             'pregrasp_duration_s':6,'action_tcp_used_by_this_comparison':'No local VLM actions; retain historical EEF/H transforms, no 43mm candidate switch'},
         'version_differences':'Version-native L-BFGS task files, LM implementation and internal numerical defaults remain part of the version comparison; not an isolated single-commit ablation',
         'per_candidate_policy':'All 256 failed-scene H plus original selected H in two positive controls; measured start in both arms; shared H-only selector; never gate on G, nominal closure or lift',
         'pair_order':'old-new in even-numbered cases, new-old in odd-numbered cases, sequential requests on the same device',
         'lifecycle':'One resident open model per version, preloaded and warmed before any case; measured finger states/world updated without rebuilding robot configuration',
         'sockets':{v:str(socket_dir/(v+'.sock')) for v in ['old','new']},
         'legacy_commit':subprocess.check_output(['git','-C',str(legacy),'rev-parse','HEAD'],text=True).strip(),
         'legacy_uncommitted_diff_sha256':hashlib.sha256(diff).hexdigest(),
         'official_new_commit':subprocess.check_output(['git','-C',str(HERE/'source'),'rev-parse','HEAD'],text=True).strip(),
         'physical_trials':0,'simulator_instances':0,'cloud_requests':0,'automatic_retries':0,
         'stop_on':['resource/configuration/source/record faults','budget exhaustion'],
         'outcome_scope':'H IK acceptance and validated transit path, not physical grasp success'}
    save(out/'registration.json',reg);save(out/'progress.json',{'phase':'registered','completed_pairs':0,'total_pairs':len(reg['cases'])})
    (out/'research.md').write_text('# Matched H-entry version comparison\n\nSame archived native H, state, model and selector; compare original old environment with isolated v0.9.0. All failed-scene candidates and two positive goals are included. Offline planning only, no default promotion.\n')
    save(out/'sources.json',{'registration':'registration.json','prior_H_run':'../20261117_02/ACCEPTANCE.md','prior_G_version_run':'../20261117_01/ACCEPTANCE.md'})
    print('Registered',sum(len(c['indices']) for c in reg['cases']),'H targets per version',flush=True)


def validate(reg):
    for path,h in reg['source_hashes'].items():assert sha(path)==h,path
    diff=subprocess.check_output(['git','-C',reg['version_sources']['old'],'diff','--binary'])
    assert hashlib.sha256(diff).hexdigest()==reg['legacy_uncommitted_diff_sha256']
    assert not subprocess.check_output(['git','-C',reg['version_sources']['new'],'diff','--name-only'],text=True).strip()
    assert shutil.disk_usage(HERE).free>1073741824
    free=int(subprocess.check_output(['nvidia-smi','-i',reg['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    assert free>=reg['free_mib_floor'],f'Resource gate: {free}MiB free'


def progress(out,phase,active=None):
    reg=read(out/'registration.json');pairs=[]
    for case in reg['cases']:
        files={v:out/'results'/v/case['id']/'result.json' for v in ['old','new']}
        if all(p.exists() for p in files.values()):pairs.append({v:read(p) for v,p in files.items()})
    summary={'phase':phase,'active':active,'completed_pairs':len(pairs),'total_pairs':len(reg['cases']),
             'old_cases_with_H_path':sum(p['old']['entry_path_passed'] for p in pairs),
             'new_cases_with_H_path':sum(p['new']['entry_path_passed'] for p in pairs),'updated_at':now(),'pairs':pairs}
    save(out/'summary.json',summary);save(out/'progress.json',{k:v for k,v in summary.items() if k!='pairs'});return summary


def run(out):
    reg=read(out/'registration.json')
    if (out/'STOP').exists() or list((out/'claims').glob('*.json')):raise RuntimeError('No replay of started or stopped comparison')
    validate(reg);began=time.monotonic();children={};streams=[];phase='stopped'
    try:
        progress(out,'loading_resident_planners')
        for version in reg['versions']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=reg['gpu_uuid'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                     CUROBO_KERNEL_BACKEND='cuda_core',CUROBO_SWEPT_MESH_BOUNDS='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',GRASP_TRAJOPT_POLICY='interactive')
            env.pop('PYTHONPATH',None)
            cache=HERE/('cache_legacy' if version=='old' else 'cache')
            env.update(XDG_CACHE_HOME=str(cache),WARP_CACHE_PATH=str(cache/'warp'),CUDA_CACHE_PATH=str(cache/'cuda'),
                       TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),TMPDIR=str(HERE/'tmp'),
                       CUDA_HOME=str(canonical_path(HERE.parent/'tmp/arx-fullchain/cuda')),CUDA_PATH=str(canonical_path(HERE.parent/'tmp/arx-fullchain/cuda')),
                       CC='/usr/bin/gcc',CXX='/usr/bin/g++',CUDAHOSTCXX='/usr/bin/g++',TORCH_CUDA_ARCH_LIST='8.9')
            env['PATH']=str(canonical_path(HERE.parent/'tmp/arx-fullchain/cuda/bin'))+':'+env['PATH']
            argv=[reg['interpreters'][version],str(HERE/'paired_h_worker.py'),'--campaign',str(out),'--backend',version]
            stream=(out/'logs'/(version+'.log')).open('x');streams.append(stream)
            child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children[version]=child
            save(out/'resident'/(version+'_owner.json'),{'pid':child.pid,'argv':argv,'gpu_uuid':reg['gpu_uuid']})
            deadline=time.monotonic()+600
            while not Path(reg['sockets'][version]).exists():
                if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Resident startup failed: '+version)
                time.sleep(.2)
            print('READY',version,child.pid,flush=True)
        mark_ready(out/'resident',reg['required_components'],{'sockets':reg['sockets']},children)
        save(out/'resident/startup_timing.json',{'wall_seconds':time.monotonic()-began})
        for index,case in enumerate(reg['cases']):
            assert time.monotonic()-began<reg['maximum_wall_seconds']
            validate(reg);require_ready(out/'resident',reg['required_components'])
            if (out/'STOP').exists():raise RuntimeError('Stop marker before next pair')
            order=['old','new'] if index%2==0 else ['new','old']
            save(out/'claims'/(case['id']+'.json'),{'case':case['id'],'order':order,'claimed_at':now()})
            for version in order:
                progress(out,'running',{'case':case['id'],'version':version})
                print('START',version,case['id'],flush=True)
                answer=rpc(Path(reg['sockets'][version]),{'command':'case','case':case},timeout=600)
                save(out/'claims'/(case['id']+'_'+version+'_receipt.json'),answer)
                if answer['status']!='ok':raise RuntimeError(answer['error'])
                print('DONE',version,case['id'],'H',len(answer['result']['accepted_H_indices']),'path',answer['result']['entry_path_passed'],flush=True)
            progress(out,'running')
        phase='completed';validate(reg)
    except Exception as error:
        if not (out/'STOP').exists():save(out/'STOP',{'reason':str(error),'time':now(),'automatic_retries':False})
        raise
    finally:
        for version,child in children.items():
            try:
                if child.poll() is None and Path(reg['sockets'][version]).exists():rpc(Path(reg['sockets'][version]),{'command':'shutdown'},timeout=15)
                child.wait(timeout=20)
            except Exception:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            save(out/'resident'/(version+'_exit.json'),{'pid':child.pid,'returncode':child.returncode,'exit_confirmed':child.poll() is not None})
        for stream in streams:stream.close()
        progress(out,phase);save(out/'timing.json',{'phase':phase,'wall_seconds':time.monotonic()-began})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.output.resolve())
    else:run(args.output.resolve())
