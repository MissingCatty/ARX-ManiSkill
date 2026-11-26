"""Resume the unexecuted E seed probe once a CUDA-verified GPU is available."""
import argparse
from pathlib import Path
import shutil
import subprocess
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from .replay import verify_hashes


def validate_predecessor(previous):
    reg = read_json(previous/'registration.json')
    stop = read_json(previous/'stop_summary.json')
    if (not (previous/'STOP').is_file() or (previous/'started.json').exists()
            or (previous/'acceptance.json').exists()
            or reg.get('experiment') != 'index_seed_probe'
            or stop.get('completed_IK_calls') != 0
            or stop.get('phase') != 'stopped_before_model_loading'
            or not stop.get('record_integrity') or reg.get('IK_calls') != 3132
            or len(reg.get('fixtures', [])) != 261):
        raise ValueError('Resume only the preserved, wholly unexecuted E registration')
    return reg


def prepare(root, previous, monitor_config):
    if root.exists():raise ValueError('Preserve prior registrations and monitor state')
    reg = validate_predecessor(previous); verify_hashes(reg)
    policy = read_json(monitor_config)
    if policy['run'] != str(root.relative_to(ROOT)):
        raise ValueError('Monitor target and new registration differ')
    inputs = dict(reg['input_hashes'])
    for name in ['registration.json','STOP','stop_summary.json','controller.json']:
        inputs[str(previous/name)] = sha256(previous/name)
    sources = dict(reg['source_hashes'])
    for path in [Path(__file__).resolve(), ROOT/'modules/pipeline/monitoring/gpu_auto_start.py', monitor_config]:
        sources[str(path)] = sha256(path)
    reg.update(at=now(), authorization='2026-11-26用户要求使用监测脚本，卡满足原门槛后继续E，保留旧停止记录。',
        previous_unexecuted_run=str(previous), gpu='pending_resource_selection',
        device_assignment_pending=True, monitor_config=str(monitor_config),
        source_hashes=sources, input_hashes=inputs)
    root.mkdir(parents=True)
    for name in sources:
        path=Path(name); target=root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    write_json(root/'registration.json',reg)
    write_json(root/'registration_pending.json',reg)
    write_json(root/'progress.json',{'at':now(),'phase':'waiting_for_gpu',
        'completed_IK_calls':0,'total_IK_calls':reg['IK_calls'],'completed_requests':0,'total_requests':0})
    write_json(root/'continuation.json',{'at':now(),'previous':str(previous),
        'previous_STOP_sha256':sha256(previous/'STOP'),'completed_IK_calls_inherited':0,
        'new_IK_calls':reg['IK_calls'],'fixtures_unchanged':True,'physical_grasps':0})
    path=ROOT/'configs/experiments/indexed_grasp_planning.json'; cfg=read_json(path)
    cfg['execution'].update(active_run=str(root.relative_to(ROOT)),current_phase='index_seed_probe_waiting_for_gpu',
        next_stage='start_once_after_CUDA_verified_original_resource_gate')
    cfg['implementation_status']='index_seed_probe_resource_monitor_waiting'
    cfg['index_reuse_implementation'].update(seed_probe_run=str(root.relative_to(ROOT)),
        previous_unexecuted_seed_probe=str(previous.relative_to(ROOT)),seed_probe_state='waiting_for_gpu',
        seed_probe_monitor_config=str(monitor_config.relative_to(ROOT)),seed_probe_completed_IK_calls=0)
    write_json(path,cfg)


def launch(root,gpu):
    reg=read_json(root/'registration.json')
    if (not reg.get('device_assignment_pending') or (root/'STOP').exists()
            or (root/'started.json').exists() or (root/'controller.json').exists()):
        raise ValueError('Do not dispatch an already assigned or stopped E registration')
    policy=read_json(reg['monitor_config'])
    # The exclusive claim survives monitor restarts and prevents a second
    # dispatch even if systemd startup is interrupted before started.json.
    with (root/'dispatch_claim.json').open('x') as stream:
        import json
        json.dump({'at':now(),'gpu':gpu,'registration_before_sha256':sha256(root/'registration.json')},stream)
    if sha256(reg['previous_unexecuted_run']+'/STOP') != read_json(root/'continuation.json')['previous_STOP_sha256']:
        raise ValueError('The previous stop record changed')
    reg.update(gpu=gpu,device_assignment_pending=False)
    write_json(root/'registration.json',reg)
    previous=Path(reg['previous_unexecuted_run'])
    environment=read_json(previous/'controller.json')['cuda_environment']
    unit=policy['controller_unit']
    command=['systemd-run','--user','--unit='+unit,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(root/'controller.log'),
        '--property=StandardError=inherit']
    command+=['--setenv='+key+'='+value for key,value in environment.items()]
    command+=[str(ROOT/'environments/curobo/bin/python'),'-m',
        'modules.pipeline.experiments.indexed_grasp_planning.seed_watch','execute','--root',str(root)]
    write_json(root/'controller.json',{'at':now(),'unit':unit,'gpu':gpu,'cuda_environment':environment,
        'IK_calls':reg['IK_calls'],'planning_requests':0,'physical_grasps':0})
    write_json(root/'progress.json',{'at':now(),'phase':'warming',
        'completed_IK_calls':0,'total_IK_calls':reg['IK_calls'],'completed_requests':0,'total_requests':0})
    path=ROOT/'configs/experiments/indexed_grasp_planning.json';cfg=read_json(path)
    cfg['execution']['current_phase']='index_seed_probe'
    cfg['implementation_status']='index_seed_probe_in_progress'
    cfg['index_reuse_implementation']['seed_probe_state']='running'
    write_json(path,cfg)
    subprocess.run(command,check=True,timeout=20)


def execute(root):
    from .seed_probe import run
    failed=False
    try:run(root)
    except BaseException:
        failed=True
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False})
    path=ROOT/'configs/experiments/indexed_grasp_planning.json';cfg=read_json(path)
    if cfg['execution']['active_run']==str(root.relative_to(ROOT)):
        phase='index_seed_probe_stopped' if failed else 'index_seed_probe_complete'
        cfg['execution'].update(current_phase=phase,next_stage='analyze_original_target_IK_results_before_full_planning')
        cfg['implementation_status']=phase
        cfg['index_reuse_implementation']['seed_probe_state']='stopped' if failed else 'complete'
        progress=read_json(root/'progress.json')
        cfg['index_reuse_implementation']['seed_probe_completed_IK_calls']=progress.get('completed_IK_calls',0)
        if not failed:
            accepted=read_json(root/'acceptance.json')
            cfg['index_reuse_implementation']['seed_probe_summaries']=accepted['summaries']
            cfg['index_reuse_implementation']['seed_probe_control_successes_retained']=accepted['control_successes_retained']
        write_json(path,cfg)
    from .followup import try_publish
    try_publish(root)
    if failed:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','launch','execute'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu')
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,ensure_run_path(args.previous),args.config.resolve())
    elif args.stage=='execute':execute(root)
    else:
        try:launch(root,args.gpu)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
