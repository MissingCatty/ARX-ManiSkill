"""Paired ARX head adaptation and bounded physical checks, with explicit receipts.

Run after teacher collection. A1 stays outside both dataset splits. Physical
evaluation uses the unchanged ARX fixtures and never restarts the EGAD49 suite.
"""
from arx_project.layout import canonical_path
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

import httpx
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from gpu_policy import resolve,compute_env
OLD=ROOT/'runs/lilo_pilot_20261026_1835/adapter'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def compare_datasets(first,second):
    """Independent evidence that RGB/action/split are paired across codecs."""
    a,b=read(first/'dataset.json'),read(second/'dataset.json')
    for key in ('action_alignment_version','exclude_controller_closure_dwell_and_hold_chunk_starts'):
        if a[key]!=b[key]:raise ValueError('Action-label protocol differs: '+key)
    if len(a['episodes'])!=len(b['episodes']):raise ValueError('Episode count differs')
    checked=[]
    for ea,eb in zip(a['episodes'],b['episodes']):
        for key in ('source','split','object_name','instruction','chunk_starts','source_recording_sha256'):
            if ea[key]!=eb[key]:raise ValueError('Comparison data mismatch: '+key)
        da=np.load(first/ea['folder']/'trajectory.npz');db=np.load(second/eb['folder']/'trajectory.npz')
        np.testing.assert_array_equal(da['action'],db['action'])
        if len(ea['items'])!=len(eb['items']):raise ValueError('Frame count differs')
        if ea['items']!=eb['items']:raise ValueError('Paired frame metadata differs')
        for ia,ib in zip(ea['items'],eb['items']):
            for key in ('source','frame_time','stage','raw_sample_index','teacher_command_id'):
                if ia[key]!=ib[key]:raise ValueError('Paired frame mismatch: '+key)
            if sha(first/ia['image'])!=sha(second/ib['image']):raise ValueError('Paired wrist images differ')
        checked.append({'source':ea['source'],'split':ea['split'],'object_name':ea['object_name'],
            'transitions':len(da['action']),'chunks':len(ea['chunk_starts']),
            'same_rgb_actions_chunks_language':True,'state_arrays_differ':not np.array_equal(da['state'],db['state'])})
    sa=next(iter(read(first/'dataset_statistics.json').values()))
    sb=next(iter(read(second/'dataset_statistics.json').values()))
    if sa['action']!=sb['action']:raise ValueError('Action normalization differs')
    return {'pass':True,'episodes':checked,'same_action_normalization':True,
        'only_policy_state_and_state_normalization_differ':True,
        'note':'Both datasets use identical observed actions; no inference action override or scripted NN correction.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-gpu',required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--bottle-mailbox',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--port',type=int,default=8009)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.training_gpu=resolve(args.training_gpu)['uuid'];args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.training_gpu))
    collection=args.collection.resolve();out=args.output.resolve()
    if read(collection/'state.json')['status']!='completed':raise ValueError('Teacher collection must finish first')
    sources=read(collection/'dataset_sources.json')
    counts={split:Counter(read(Path(p)/'scenario.json')['asset'] for p in sources[split]) for split in ('train','validation')}
    if set(counts['train'])!=set(sources['expected_train_objects']) or set(counts['validation'])!=set(sources['expected_validation_objects']):
        raise ValueError('Predeclared object split lacks successful data: '+str(counts))
    if min([*counts['train'].values(),*counts['validation'].values()],default=0)<2:
        raise ValueError('Need at least two physically verified episodes for every train/validation object')
    if set(counts['train'])&set(counts['validation']):raise ValueError('Object leakage')
    if any(Path(p).resolve()==Path(sources['A1_excluded']).resolve() for split in ('train','validation') for p in sources[split]):
        raise ValueError('A1 diagnostic leaked into data')
    out.mkdir(parents=True,exist_ok=args.resume)
    checkpoint=ROOT.parent/'lilo-vla/checkpoints/lilo-vla-openvla-oft'
    neural=canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python')
    model_python=ROOT.parent/'lilo-vla/.venv/bin/python'
    files=[Path(__file__),*[HERE/n for n in ('build_dataset.py','finetune_head.py','evaluate_p2_policy.py',
        'observations.py','rollout.py','grasp_handoff.py','egad_evaluate.py')],
        ROOT/'servers/lilo_vla/backend.py',ROOT/'servers/lilo_vla/app.py',ROOT/'scripts/curobo_arx/grasp_execute.py']
    planned={'scope':'Paired head-only ARX development comparison; no full EGAD49 restart or generalization claim',
        'source_collection':str(collection),'source_manifest_sha256':sha(collection/'dataset_sources.json'),
        'train':sources['train'],'validation':sources['validation'],'A1_excluded':sources['A1_excluded'],
        'old_adapter':str(OLD),'old_artifacts':{str(OLD/n):sha(OLD/n) for n in ('action_head.pt','dataset_statistics.json','adapter_manifest.json')},
        'algorithm_sha256':{str(p):sha(p) for p in files},'training_gpu':args.training_gpu,
        'planner_gpu':args.planner_gpu,'steps':args.steps,'batch_size':args.batch_size,
        'seed':20261026,'new_policy_port':args.port,'physical_cases':['A1','bottle'],
        'selection':'Minimum held-out object action L1, including initial head; actual grasp is evaluated separately',
        'state_versions':['object_relative_v1','grasp_goal_world_error_v1']}
    if (out/'registration.json').exists():
        registration=read(out/'registration.json')
        for key in planned:
            if registration[key]!=planned[key]:raise ValueError('Resume differs from registered input: '+key)
    else:
        registration={**planned,'registered_unix':time.time()};write(out/'registration.json',registration)
        archive=out/'registered_source_snapshot';archive.mkdir()
        for p in files:shutil.copy2(p,archive/(str(p.relative_to(ROOT)).replace('/','__')))
        shutil.copy2(collection/'storage_policy.json',out/'storage_policy.json')
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_VISIBLE_DEVICES=args.training_gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='2')
    owned_server=None;state={'status':'running','completed_stages':[]}
    if args.resume and (out/'state.json').exists():state['completed_stages']=read(out/'state.json').get('completed_stages',[])
    def check():
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
        for path,digest in {**registration['algorithm_sha256'],**registration['old_artifacts']}.items():
            if sha(path)!=digest:raise RuntimeError('Registered source changed: '+path)
    def stage(name,command,expected):
        check()
        if name in state['completed_stages']:
            if not expected.exists():raise ValueError('Completed stage missing output: '+name)
            return
        state.update(stage=name);write(out/'state.json',state);print('P2_STAGE',name,flush=True)
        log=out/(name+'.log')
        with log.open('x') as stream:
            child=subprocess.Popen([str(x) for x in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write(out/(name+'_process.json'),{'pid':child.pid,'command':[str(x) for x in command],'started_unix':time.time()})
            if child.wait()!=0:raise RuntimeError(name+' failed; see '+str(log))
        if not expected.exists():raise RuntimeError(name+' exited without its complete receipt')
        state['completed_stages'].append(name);write(out/'state.json',state)
    def stop_owned():
        nonlocal owned_server
        if owned_server is not None:
            if owned_server.poll() is None:
                os.killpg(owned_server.pid,signal.SIGTERM)
                try:owned_server.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(owned_server.pid,signal.SIGKILL);owned_server.wait()
            owned_server=None
    def serve(adapter,label):
        nonlocal owned_server
        probe=socket.socket();probe.settimeout(2)
        try:busy=probe.connect_ex(('127.0.0.1',args.port))==0
        finally:probe.close()
        if busy:raise RuntimeError('Temporary policy port already in use; preserving existing process')
        server_env=env|{'LILO_GPU':args.training_gpu,'LILO_PORT':str(args.port),
            'LILO_CHECKPOINT':str(checkpoint),'LILO_ADAPTER':str(adapter),'LILO_PYTHON':str(model_python)}
        log=(out/(label+'_server.log')).open('x')
        try:owned_server=subprocess.Popen(['bash',str(ROOT/'scripts/run_lilo_vla_server.sh')],cwd=ROOT,env=server_env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:log.close()
        write(out/(label+'_server_process.json'),{'pid':owned_server.pid,'adapter':str(adapter),'port':args.port,'gpu':args.training_gpu})
        url=f'http://127.0.0.1:{args.port}'
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if owned_server.poll() is not None:raise RuntimeError('Temporary model service failed; see its log')
            try:
                response=httpx.get(url+'/health',timeout=2,trust_env=False)
                if response.status_code==200:
                    health=response.json()
                    if Path(health['adapter']).resolve()!=adapter.resolve():raise ValueError('Wrong temporary adapter')
                    write(out/(label+'_health.json'),health);return url
            except httpx.HTTPError:pass
            time.sleep(2)
        raise TimeoutError('Temporary model service did not become ready in 600 seconds')
    try:
        for label,convention in (('object','object_relative_v1'),('goal','grasp_goal_world_error_v1')):
            dataset=out/(label+'_dataset')
            command=[neural,HERE/'build_dataset.py','--output',dataset,'--state-convention',convention,
                '--dataset-key','arx_shared_'+label+'_v1','--object-split','--require-entry',
                '--include-command-starts','--exclude-gated-dwell-starts']
            for p in sources['train']:command+=['--source',p]
            for p in sources['validation']:command+=['--validation-source',p]
            stage(label+'_dataset',command,dataset/'quality_report.json')
        write(out/'paired_dataset_verification.json',compare_datasets(out/'object_dataset',out/'goal_dataset'))
        for label in ('object','goal'):
            adapter=out/(label+'_adapter')
            stage(label+'_training',[model_python,HERE/'finetune_head.py','--checkpoint',checkpoint,
                '--dataset',out/(label+'_dataset'),'--output',adapter,'--initial-adapter',OLD,
                '--steps',args.steps,'--batch-size',args.batch_size],adapter/'adapter_manifest.json')
        variants=[('old','http://127.0.0.1:8008',OLD),('object',None,out/'object_adapter'),('goal',None,out/'goal_adapter')]
        for label,url,adapter in variants:
            name=label+'_physical';evaluation=out/name
            if name in state['completed_stages']:continue
            try:
                if url is None:url=serve(adapter,label)
                stage(name,[neural,HERE/'evaluate_p2_policy.py','--output',evaluation,'--policy-url',url,
                    '--expected-adapter',adapter,'--planner-gpu',args.planner_gpu,
                    '--bottle-mailbox',args.bottle_mailbox.resolve()],evaluation/'final_integrity.json')
            finally:stop_owned()
        check();state.update(status='completed',stage=None);write(out/'state.json',state)
    except BaseException:
        state.update(status='stopped_on_error',error=traceback.format_exc());write(out/'state.json',state);raise
    finally:stop_owned()
    print('P2_COMPARISON_COMPLETED',out,flush=True)


if __name__=='__main__':main()
