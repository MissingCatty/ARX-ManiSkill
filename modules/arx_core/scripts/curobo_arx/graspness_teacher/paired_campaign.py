"""Prepare/run a new bounded ARX comparison after a registered model screen."""
from arx_project.layout import canonical_path
import argparse
import copy
from collections import defaultdict
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,digest,verify_files,space
from scripts.curobo_arx.data_factory.ledger import Ledger


def prepare(research,hypothesis):
    research=Path(research).resolve();manifest=read(research/'registration.json')
    screen=read(research/hypothesis/'results.json')
    if screen['status']!='complete':raise ValueError('Complete the registered model screen first')
    pairs=defaultdict(dict)
    for row in screen['rows']:pairs[row['object_id']][row['mode']]=row
    recovered=sorted(name for name,pair in pairs.items() if pair['baseline']['split']=='development'
        and pair['baseline']['qualified']==0 and pair['variant']['qualified']>0)[:2]
    if not recovered:raise ValueError('Physical activation rule not met: no model recovery')
    names=recovered+['bbq_sauce_bottle','household_hope_milk_carton']
    out=research/('physical_'+hypothesis)
    if out.exists():raise FileExistsError('Never overwrite a physical comparison registration')
    old=read(research/'offline_config.json');parent=read(Path(manifest['parent'])/'registration.json')
    space(ROOT/'runs');out.mkdir()
    source=out/'source_snapshot/COOLAS-ARX';source.parent.mkdir()
    shutil.copytree(old['source_snapshot'],source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
    reg=copy.deepcopy(old);files=reg['source_files'];execution={};parent_source=Path(parent['source_snapshot'])
    for path,value in reg['execution_source_files'].items():
        p=Path(path);execution[str(source/p.relative_to(parent_source)) if p.is_relative_to(parent_source) else path]=value
    for row in read(research/'source_registration.json')['changes']:
        target=source/row['path']
        if sha(target)!=row['sha256']:raise RuntimeError('Research source changed after model registration')
        execution[str(target)]=row['sha256'];files[str(ROOT/row['path'])]=row['sha256']
    reg['execution_source_files']=execution
    reg.update(created_unix=time.time(),source_snapshot=str(source),source_identity=digest(files),
        stage='bounded_graspness_success_comparison',production=False,envs_per_gpu=1,object_block_attempts=1,
        deadline_unix=manifest['created_unix']+manifest['max_wall_hours']*3600,
        research_parent=str(research),hypothesis=hypothesis)
    for key in ['resume','lineage','bulk_storage']:reg.pop(key,None)
    hypothesis_cfg=next(h for h in manifest['hypotheses'] if h['id']==hypothesis)
    reg['graspness']['multi_view']['views_per_point']=hypothesis_cfg['views_per_point']
    reg['graspness']['multi_angle']['candidate_limit']=hypothesis_cfg['candidate_limit']
    trials=[];cases={c['object_id']:c for c in manifest['cases']}
    catalog={a['object_id']:a for a in reg['catalog']['objects']}
    for name in names:
        archived=read(cases[name]['path']);block=Path(archived['scene']).parent
        placement=read(block/'collection_placement.json')
        for repeat,policies in enumerate([['baseline','variant'],['variant','baseline']]):
            for policy in policies:
                ident=f'{len(trials):02d}__{name}__r{repeat}__{policy}'
                trials.append({'trial_id':ident,'object_id':name,'policy':policy,'repeat':repeat,
                    'renderer':catalog[name]['renderer'],'placement':placement,
                    'role':'recovery' if name in recovered else 'preservation'})
    if len(trials)>manifest['max_physical_trials']:raise ValueError('Physical trial budget exceeded')
    reg['paired_trials']=trials
    for key,path in list(reg['reference_files'].items()):
        target=out/'references'/(key+'.json');target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
        reg['reference_files'][key]=str(target);reg['dependencies'][str(target)]=sha(target)
    for relative in ['scripts/curobo_arx/graspness_teacher/paired_campaign.py','scripts/curobo_arx/data_factory/launch.py',
            'scripts/curobo_arx/data_factory/common.py','scripts/curobo_arx/data_factory/storage.py']:
        reg['dependencies'][str(ROOT/relative)]=sha(ROOT/relative)
    quotas={**reg['quotas'],'max_physical_attempts':1,'max_total_dispatches':1,'physical_per_bin':1,
        'planning_rejections_per_bin':1,'goals_per_object':1,'successes_per_bin':1}
    for trial in trials:
        scoped=copy.deepcopy(reg['catalog'])
        for asset in scoped['objects']:
            asset['collection_enabled']=asset['object_id']==trial['object_id']
            if asset['collection_enabled']:asset['collection_placements']=[trial['placement']]
        directory=out/'trial_ledgers';directory.mkdir(exist_ok=True)
        book=Ledger(directory/(trial['trial_id']+'.sqlite3'))
        book.initialize(scoped,{**quotas,'first_object':trial['object_id']});book.close()
    write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
    shutil.copy2(Path(manifest['parent'])/'storage_policy.json',out/'storage_policy.json')
    write(out/'state.json',{'status':'prepared','physical_trials':0,'scheduled_trials':len(trials)})
    verify_files({**execution,**reg['dependencies']})
    return out


def run(campaign):
    from scripts.curobo_arx.data_factory.launch import devices,allocate,start_worker,seed_kit_cache
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json')
    if (campaign/'launch.json').exists():raise ValueError('Do not rerun an existing physical launch')
    verify_files({**reg['execution_source_files'],**reg['dependencies']});space(campaign)
    compute,sim=allocate(devices(),1,'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834',
        ['GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e'])
    source=Path(reg['source_snapshot']);stop=reg['deadline_unix'];processes=[];active=None;error=None
    write(campaign/'launch.json',{'wall':time.time(),'compute':compute,'simulators':sim,
        'registration_sha256':sha(campaign/'registration.json'),'stop_unix':stop})
    try:
        for kind in ['planner','graspness']:
            process,directory=start_worker(campaign,kind,compute,source);processes.append((process,directory))
        deadline=time.monotonic()+180
        while not all((d/'ready.json').exists() for p,d in processes):
            if any(p.poll() is not None for p,d in processes) or time.monotonic()>deadline:
                raise RuntimeError('Resident comparison workers failed to become ready')
            time.sleep(.25)
        grouped=defaultdict(list)
        for trial in reg['paired_trials']:grouped[trial['renderer']].append(trial['trial_id'])
        for renderer,ids in grouped.items():
            if (campaign/'STOP').exists() or time.time()>=stop:break
            verify_files({**reg['execution_source_files'],**reg['dependencies']});space(campaign)
            gpu=sim[0]['uuid'];free=next(d['free_mb'] for d in devices() if d['uuid']==gpu)
            if free<7500:raise RuntimeError('Simulation GPU no longer has its registered memory budget')
            ident='paired_'+renderer.lower();name='arx-graspness-success-'+ident
            launch={'group_id':ident,'gpu_uuid':gpu,'env_count':1,'renderer':renderer,'trial_ids':ids,
                'planner_directory':str(processes[0][1]),'candidate_directory':str(processes[1][1]),
                'candidate_backend':'graspness','stop_unix':stop}
            path=campaign/'group_registrations'/(ident+'.json');write(path,launch)
            cache=ROOT/'runs/data_factory_runtime_cache';kit=seed_kit_cache(cache/gpu)
            command=['podman','run','--rm','--name',name,'--user','0','--device','nvidia.com/gpu='+gpu,
                '--security-opt=label=disable','--network=host','--shm-size=2g',
                '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=N','-e','OMNI_KIT_ALLOW_ROOT=1',
                '-e','CUDA_VISIBLE_DEVICES='+gpu,'-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
                '-e','NVIDIA_VISIBLE_DEVICES=void','-e','PYTHONUNBUFFERED=1','-e','PYTHONDONTWRITEBYTECODE=1',
                '-v',str(cache/'ov')+':/root/.cache/ov:rw','-v',str(cache/'nvidia')+':/root/.cache/nvidia:rw',
                '-v',str(kit)+':/isaac-sim/kit/cache:rw','-v',str(ROOT.parent)+':'+str(ROOT.parent)+':rw',
                '-v',str(canonical_path(ROOT/'asset'))+':/workspace/asset:ro','-v',str(ROOT)+':/workspace/COOLAS-ARX:ro',
                '-w',str(source),'--entrypoint','/isaac-sim/python.sh','localhost/arx-s02-isaac:6.0.1',
                '-m','scripts.curobo_arx.graspness_teacher.paired_runtime','--campaign',str(campaign),'--launch',str(path)]
            with path.with_suffix('.log').open('x') as log:active=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(path.with_suffix('.process.json'),{'pid':active.pid,'container':name,'command':command})
            while active.poll() is None:
                write(campaign/'state.json',{'status':'running','renderer':renderer,'updated_unix':time.time()})
                if any(p.poll() is not None for p,d in processes):
                    write(campaign/'STOP',{'reason':'owned comparison worker exited','wall':time.time()})
                time.sleep(2)
            if active.returncode!=0:raise RuntimeError('ARX comparison group failed: '+renderer)
            shutdown=read(campaign/'groups'/ident/'shutdown_receipt.json')
            if not shutdown['all_trajectories_idle']:raise RuntimeError('Comparison robot idle not confirmed')
            if subprocess.run(['podman','container','exists',name],capture_output=True).returncode!=1:
                raise RuntimeError('Comparison World removal not confirmed')
            active=None
        verify_files({**reg['execution_source_files'],**reg['dependencies']})
    except BaseException as exc:
        error=str(exc);write(campaign/'FAULT.json',{'wall':time.time(),'error':traceback.format_exc()});raise
    finally:
        if active is None or active.poll() is not None:
            for process,directory in processes:write(directory/'STOP',{'wall':time.time(),'reason':'bounded comparison ended'})
            for process,directory in processes:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
        result=summarize(campaign)
        write(campaign/'state.json',{'status':'fault' if error else ('complete' if result['complete'] else 'stopped'),
            'error':error,'finished_unix':time.time(),'completed_trials':result['completed_trials'],
            'unexecuted_trials':len(result['unexecuted']),'fault_or_unknown_trials':len(result['fault_or_unknown'])})


def summarize(campaign):
    campaign=Path(campaign);reg=read(campaign/'registration.json');completed={}
    for path in (campaign/'groups').glob('*/paired_results.json'):
        for row in read(path)['trials']:
            if row['trial_id'] in completed:raise RuntimeError('Duplicate registered trial result')
            completed[row['trial_id']]=row
    unexecuted=[];unknown=[];summary=[]
    for trial in reg['paired_trials']:
        ident=trial['trial_id']
        if ident not in completed:
            from sqlite3 import connect
            book=connect('file:'+str(campaign/'trial_ledgers'/(ident+'.sqlite3'))+'?mode=ro',uri=True)
            count=book.execute('select count(*) from attempts').fetchone()[0];book.close()
            started=any((campaign/'groups').glob('*/trials/'+ident))
            (unknown if count or started else unexecuted).append(ident)
    for name in dict.fromkeys(t['object_id'] for t in reg['paired_trials']):
        row={'object_id':name}
        for mode in ['baseline','variant']:
            scheduled=[t for t in reg['paired_trials'] if t['object_id']==name and t['policy']==mode]
            values=[completed[t['trial_id']] for t in scheduled if t['trial_id'] in completed]
            row[mode]={'scheduled':len(scheduled),'completed':len(values),
                'success':sum(v['success'] for v in values),'admitted':sum(v['admitted'] for v in values),
                'physical_dispatches':sum(v['physical_dispatches'] for v in values),
                'failure_reasons':dict(Counter(v['outcome'] for v in values if not v['success']))}
        summary.append(row)
    result={'wall':time.time(),'complete':len(completed)==len(reg['paired_trials']),
        'completed_trials':len(completed),'scheduled_trials':len(reg['paired_trials']),
        'unexecuted':unexecuted,'fault_or_unknown':unknown,'objects':summary,
        'physical_dispatches':sum(v['physical_dispatches'] for v in completed.values()),
        'successes':sum(v['success'] for v in completed.values()),'admitted':sum(v['admitted'] for v in completed.values())}
    write(campaign/'summary.json',result)
    lines=['# ARX 抓取对照结果','',f'完成 {len(completed)}/{len(reg["paired_trials"])} 个登记试验；未执行 {len(unexecuted)}，故障/未知 {len(unknown)}。',
        '','| 物体 | 原方法成功/完成 | 候选方法成功/完成 | 两组实际运动数 | 失败原因 |','|---|---:|---:|---:|---|']
    for row in summary:
        a,b=row['baseline'],row['variant']
        lines.append(f'|{row["object_id"]}|{a["success"]}/{a["completed"]}|{b["success"]}/{b["completed"]}|{a["physical_dispatches"]}/{b["physical_dispatches"]}|{a["failure_reasons"]}; {b["failure_reasons"]}|')
    lines.extend(['','无候选和规划拒绝为已完成试验的失败，未下发运动另列；未执行和基础设施中断不伪装为普通抓取失败。成功仍由原五阶段、接触和持续抬升评估判定，图像准入单列。'])
    (campaign/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('research');p.add_argument('hypothesis')
    p=sub.add_parser('run');p.add_argument('campaign')
    args=parser.parse_args()
    if args.action=='prepare':print(prepare(args.research,args.hypothesis))
    else:run(args.campaign)
