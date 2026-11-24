"""Read-only experiment follower. Writes only its own monitoring directory."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def snapshot(root,unit,reg=None):
    root=Path(root).resolve();reg=read(root/'registration.json') if reg is None else reg
    state=read(root/'state.json');rows=[];carried=reg.get('carried_details',[])
    for r in carried:
        rows.append({'trial_id':r['trial_id'],'method':r['method'],'object_id':r['object_id'],'condition':r['condition'],
            'success':r['physical_success'],'outcome':r['outcome'],'physical':r['physical_attempt'],
            'task_result':r.get('task_result','observed'),'native_terminal':r.get('native_runtime_completed',True),'carried':True})
    for path in sorted((root/'groups').glob('*/trials/*/trial_result.json')):
        r=read(path);rows.append({'trial_id':r['trial_id'],'method':r['method'],'object_id':r['object_id'],'condition':r['condition'],
            'success':r['success'],'outcome':r['outcome'],'physical':bool(r['physical_dispatches']),
            'task_result':'observed','native_terminal':True,'carried':False})
    done={r['trial_id'] for r in rows};claims=[]
    for path in sorted((root/'trial_ledgers').glob('*.sqlite3')):
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:claims.extend({'trial_id':path.stem,**dict(r)} for r in db.execute('select id,state,physical,episode from attempts'))
        finally:db.close()
    pending_ids={t['trial_id'] for t in reg['seed_trials']}
    begun={r['trial_id'] for r in claims};scheduled_remaining=pending_ids-done-begun
    deferred={t['trial_id'] for t in reg.get('deferred_trials',[])};remaining=scheduled_remaining|deferred
    unit_text=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],text=True,timeout=10)
    service=dict(line.split('=',1) for line in unit_text.splitlines() if '=' in line)
    result={'wall':time.time(),'time_local':time.strftime('%F %T'),'campaign':str(root),'unit':service,
        'state':state,'registered':48,'accounted':len(rows),'observed_task_outcomes':sum(r['task_result']=='observed' for r in rows),
        'successes':sum(r['success'] for r in rows),
        'grasp_failures':sum(not r['success'] and r['task_result']=='observed' for r in rows),
        'infrastructure_interruptions':sum(r['task_result']=='infrastructure_interrupted' for r in rows),
        'physical_attempts':sum(r['physical_attempt'] for r in carried)+sum(r['physical'] for r in claims),
        'native_terminal_gaps':[r['trial_id'] for r in rows if not r['native_terminal']],
        'unexecuted_count':len(remaining),'unexecuted':sorted(remaining),
        'scheduled_unexecuted_count':len(scheduled_remaining),'deferred_unexecuted':sorted(deferred),
        'active_or_incomplete':[r for r in claims if r['trial_id'] not in done],
        'by_method':{m:{'accounted':sum(r['method']==m for r in rows),'success':sum(r['method']==m and r['success'] for r in rows),
            'grasp_failure':sum(r['method']==m and not r['success'] and r['task_result']=='observed' for r in rows),
            'interruption':sum(r['method']==m and r['task_result']=='infrastructure_interrupted' for r in rows)} for m in 'ABCD'},
        'last_results':rows[-4:],'STOP_exists':(root/'STOP').exists()}
    hearts=list((root/'groups').glob('*/trials/*/heartbeat.json'))
    if hearts:
        path=max(hearts,key=lambda p:p.stat().st_mtime);result['heartbeat']=read(path)
        result['heartbeat_age_seconds']=round(time.time()-path.stat().st_mtime,1)
        episodes=list((path.parent/'episodes').glob('*'))
        if episodes:
            episode=max(episodes,key=lambda p:p.stat().st_mtime);result['episode']=str(episode)
            if (episode/'requests.json').exists():result['last_commands']=[r['label'] for r in read(episode/'requests.json')][-3:]
            for name in ('policy','geometry'):
                file=episode/(name+'_requests.json')
                if file.exists():result[name+'_calls']=len(read(file))
    return result


def follow(root,unit):
    root=Path(root).resolve();reg=read(root/'registration.json');directory=root/'follow';directory.mkdir(exist_ok=True)
    if (directory/'started.json').exists():raise FileExistsError('Keep prior follower evidence')
    atomic(directory/'started.json',{'pid':os.getpid(),'wall':time.time(),'scope':'Read-only follow; no trial/service mutation or restart','unit':unit})
    previous=None
    while True:
        try:
            value=snapshot(root,unit,reg);atomic(directory/'progress.json',value)
            signature=json.dumps({k:value.get(k) for k in ('accounted','last_results','STOP_exists','unit')},sort_keys=True)
            if signature!=previous:
                with (directory/'events.jsonl').open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                previous=signature
            if value['unit']['ActiveState'] not in ('active','activating','deactivating') and value['state']['status'] in ('complete','fault'):
                atomic(directory/'finished.json',{'wall':time.time(),'final_state':value['state']['status'],
                    'accounted':value['accounted'],'unexecuted':value['unexecuted_count'],'no_restart_performed':True});return
        except Exception as exc:
            atomic(directory/'read_error.json',{'wall':time.time(),'error':str(exc),'no_runtime_mutation':True})
        if time.time()>reg['deadline_unix']+1800:
            atomic(directory/'finished.json',{'wall':time.time(),'reason':'Observer deadline reached; no runtime mutation'});return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--unit',required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    if a.watch:follow(a.root,a.unit)
    else:print(json.dumps(snapshot(a.root,a.unit),ensure_ascii=False))
