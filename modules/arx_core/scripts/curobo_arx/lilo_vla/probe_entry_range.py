"""P3: test two nearer entries with a fixed goal-conditioned ARX policy.

Both distances are tested on A1 and the historical bottle positive control.
Only a point passing both cases gets two more repeats. This measures discrete
development points; it does not establish a continuous or cross-object basin.
"""
from arx_project.layout import canonical_path
import argparse
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

from grasp_handoff import NEAR_RULES

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from gpu_policy import resolve,compute_env,verify_health


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--policy-gpu',required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--bottle-mailbox',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8009)
    parser.add_argument('--handoff-profile',type=Path,help='Replay the qualified point once with runtime admission; no entry search or model selection')
    parser.add_argument('--scene',choices=('A1','bottle'),help='Single case, available only for qualified-profile replay')
    args=parser.parse_args()
    args.policy_gpu=resolve(args.policy_gpu)['uuid'];args.planner_gpu=resolve(args.planner_gpu)['uuid']
    os.environ.update(compute_env(os.environ,args.planner_gpu))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.scene and not args.handoff_profile:raise ValueError('Single-case mode requires a qualified profile')
    profile=read(args.handoff_profile) if args.handoff_profile else None
    adapter=args.adapter.resolve();assert read(adapter/'adapter_manifest.json')['state_convention']=='grasp_goal_world_error_v1'
    registered=[Path(__file__),HERE/'evaluate_p2_policy.py',HERE/'grasp_handoff.py',HERE/'rollout.py',
        HERE/'observations.py',HERE/'egad_evaluate.py',ROOT/'scripts/curobo_arx/grasp_execute.py',
        ROOT/'servers/lilo_vla/backend.py',ROOT/'servers/lilo_vla/app.py']
    registered.extend([HERE/'recheck_entry_range.py',HERE/'summarize_p2.py'])
    registered.append(ROOT/'scripts/curobo_arx/gpu_policy.py')
    if profile:registered.extend([HERE/'handoff_profile.py',HERE/'egad49.py',
        ROOT/'scripts/curobo_arx/static_pick/reproduce_setup.py'])
    archive=out/'registered_source_snapshot';archive.mkdir()
    for path in registered:shutil.copy2(path,archive/path.name)
    rules={}
    for name,rule in NEAR_RULES.items():
        rules[name]=out/(name+'_rule.json');write(rules[name],rule)
    registration={'scope':('Qualified-profile runtime replay' if profile else 'Discrete A1/bottle development entry probe')+'; no EGAD49 batch restart',
        'adapter':str(adapter),'adapter_sha256':{str(adapter/n):sha(adapter/n) for n in ('action_head.pt','dataset_statistics.json','adapter_manifest.json')},
        'algorithm_sha256':{str(p):sha(p) for p in registered},'rules':NEAR_RULES,
        'rule_sha256':{str(p):sha(p) for p in rules.values()},'initial_order':['five_mm','ten_mm'],
        'repeat_rule':'Test both points once per scene; choose largest passing distance, then repeat that point twice per scene',
        'qualification':'Exactly three actual successes in each of the two development cases, unchanged physical gates',
        'max_cases':(1 if args.scene else 2) if profile else 8,
        'qualified_profile_sha256':{str(args.handoff_profile.resolve()):sha(args.handoff_profile)} if profile else {},
        'base_40mm_comparison':str(ROOT/'runs/lilo_P2_comparison_20261028_02/goal_physical'),
        'policy_gpu':args.policy_gpu,'planner_gpu':args.planner_gpu,'port':args.port,'registered_unix':time.time()}
    write(out/'registration.json',registration)
    write(out/'storage_policy.json',{'local_first':True,'reserve_bytes':1073741824,
        'historical_file':'Unavailable on this host; not assumed verified'})
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',LILO_GPU=args.policy_gpu,LILO_ADAPTER=str(adapter),
        LILO_PORT=str(args.port),LILO_PYTHON=str(ROOT.parent/'lilo-vla/.venv/bin/python'),
        LILO_CHECKPOINT=str(ROOT.parent/'lilo-vla/checkpoints/lilo-vla-openvla-oft'))
    server=None;rows=[];state={'status':'running','active':None,'completed_cases':0,'successes':0}
    write(out/'state.json',state)
    def check():
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
        for p,digest in {**registration['algorithm_sha256'],**registration['adapter_sha256'],**registration['rule_sha256'],**registration['qualified_profile_sha256']}.items():
            if sha(p)!=digest:raise RuntimeError('Registered P3 source changed: '+p)
    def execute(name,repeat):
        check();label=f'{name}_r{repeat}';folder=out/label
        state.update(active=label);write(out/'state.json',state);print('P3_START',label,flush=True)
        command=[str(canonical_path(ROOT.parent/'tmp/arx-fullchain/env/bin/python')),'-u',str(HERE/'evaluate_p2_policy.py'),
            '--output',str(folder),'--policy-url',f'http://127.0.0.1:{args.port}',
            '--expected-adapter',str(adapter),'--planner-gpu',args.planner_gpu,
            '--bottle-mailbox',str(args.bottle_mailbox.resolve()),'--handoff-rule',str(rules[name])]
        if profile:command+=['--handoff-profile',str(args.handoff_profile.resolve())]
        if args.scene:command+=['--scene',args.scene]
        with (out/(label+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if child.wait()!=0:raise RuntimeError('P3 evaluation runner failed: '+label)
        result=read(folder/'results.json');assert len(result)==(1 if args.scene else 2)
        for item in result:
            trial=folder/item['scene'];verified=trial/'provenance_runtime_check.json'
            subprocess.run([str(ROOT.parent/'curobo/.venv/bin/python'),str(HERE/'recheck_entry_range.py'),
                '--trial',str(trial),'--output',str(verified)],cwd=ROOT,env=env,check=True)
            row=read(verified)
            if profile:
                admission=trial/'handoff_policy_admission.json'
                row['runtime_admission']=read(admission) if admission.exists() else {'pass':False,'error':'No admission occurred'}
            rows.append({**row,'entry_group':name,'backoff_m':NEAR_RULES[name]['ordered_backoff_m'][0],
                'repeat':repeat,'trial':str(trial)})
        state.update(completed_cases=len(rows),successes=sum(r['pass'] for r in rows));write(out/'state.json',state)
        write(out/'results.json',rows);check()
        print('P3_RESULT',label,json.dumps({r['scene']:r['pass'] for r in result}),flush=True)
        return all(r['pass'] for r in result)
    try:
        check();probe=socket.socket();probe.settimeout(2)
        try:busy=probe.connect_ex(('127.0.0.1',args.port))==0
        finally:probe.close()
        if busy:raise RuntimeError('Temporary policy port in use; preserving existing process')
        with (out/'policy_server.log').open('x') as stream:
            server=subprocess.Popen(['bash',str(ROOT/'scripts/run_lilo_vla_server.sh')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'policy_server_process.json',{'pid':server.pid,'adapter':str(adapter),'gpu':args.policy_gpu,'port':args.port})
        deadline=time.monotonic()+600
        while True:
            if server.poll() is not None:raise RuntimeError('Temporary P3 model service exited')
            try:
                response=httpx.get(f'http://127.0.0.1:{args.port}/health',timeout=2,trust_env=False)
                if response.status_code==200:
                    health=response.json();assert Path(health['adapter']).resolve()==adapter
                    assert health['state_convention']=='grasp_goal_world_error_v1'
                    verify_health(health,args.policy_gpu)
                    assert health['gpu_process_pid']==server.pid
                    write(out/'policy_health.json',health);break
            except httpx.HTTPError:pass
            if time.monotonic()>deadline:raise TimeoutError('P3 model startup exceeded 600s')
            time.sleep(2)
        if profile:
            selected=next(name for name,rule in NEAR_RULES.items() if rule==profile['rule'])
            passed=[selected] if execute(selected,1) else []
        else:
            passed=[name for name in ('five_mm','ten_mm') if execute(name,1)]
            selected=max(passed,key=lambda n:NEAR_RULES[n]['ordered_backoff_m'][0]) if passed else None
        write(out/'point_selection.json',{'passing_first_trials':passed,'selected_for_repeats':selected,
            'selection':('Pinned qualified profile; no new selection or repeats' if profile else 'Largest passing distance, predeclared; first-trial failures remain in results')})
        if selected and not profile:
            for repeat in (2,3):execute(selected,repeat)
        qualified=[]
        for name in NEAR_RULES:
            if all(len([r for r in rows if r['entry_group']==name and r['scene']==scene])==3
                and all(r['pass'] for r in rows if r['entry_group']==name and r['scene']==scene) for scene in ('A1','bottle')):
                qualified.append(name)
        summary={'scope':registration['scope'],'rows':rows,'attempts':len(rows),'successes':sum(r['pass'] for r in rows),
            'qualified_development_points':qualified,'selected_for_repeats':selected,
            'all_action_provenance_pass':all(r['action_provenance_pass'] for r in rows),
            'qualified_profile_replay':bool(profile),
            'qualification_source':str(args.handoff_profile.resolve()) if profile else None,
            'runtime_admission_and_grasp_verified':all(r['pass'] and r.get('runtime_admission',{}).get('pass') for r in rows) if profile else None,
            'global_default_changed':False,'default_execute_steps':1,
            'limitation':'Discrete points and fixed development grasp goals only; no continuous range or unseen-object success guarantee',
            'original_40mm_reference':registration['base_40mm_comparison']}
        check();write(out/'summary.json',summary)
        write(out/'completion_integrity.json',{'registered_sources_and_policy_unchanged':True,
            'source_archive_matches':all(sha(archive/Path(p).name)==digest for p,digest in registration['algorithm_sha256'].items())})
        state.update(status='completed',active=None);write(out/'state.json',state)
    except BaseException:
        state.update(status='stopped_on_error',error=traceback.format_exc());write(out/'state.json',state);raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
    print('P3_COMPLETED',out,flush=True)


if __name__=='__main__':main()
