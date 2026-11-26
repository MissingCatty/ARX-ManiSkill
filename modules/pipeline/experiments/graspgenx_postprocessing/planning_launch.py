"""CUDA-verified one-shot dispatch; resource waiting never retries a trial."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from arx_project.layout import ROOT,ensure_run_path,now,write_json
from .planning import raw,verify


def controller_unit(root):return 'arx-v7-'+root.name.replace('_','-')+'.service'


def monitor_policy(root):
    return {'run':str(root.relative_to(ROOT)),'controller_unit':controller_unit(root),
        'status_file':str((root/'monitor/status.json').relative_to(ROOT)),'readiness_marker':'startup.json',
        'dispatch_marker_files':['dispatch_claim.json'],'poll_interval_seconds':1,
        'admission_plans_mib':[{'planner':10000}],
        'cuda_probe_before_dispatch':{'python':'environments/curobo/bin/python','minimum_free_mib':10000,
            'timeout_seconds':30,'failure_cooldown_seconds':600},
        'start_commands':[[str(ROOT/'environments/geometric_ik/bin/python'),'-m',
            'modules.pipeline.experiments.graspgenx_postprocessing.planning_launch','dispatch',
            '--root',str(root),'--gpu','{planner_uuid}']],
        'automatic_experiment_retries':False,'physical_grasp_budget':0}


def watch(root):
    if (root/'STOP').exists() or (root/'started.json').exists():raise ValueError('Do not restart a claimed V7 run')
    verify(raw(root/'registration.json'))
    config=root/'monitor_config.json'
    if raw(config)!=monitor_policy(root):raise ValueError('Registered resource monitor policy changed')
    unit='arx-v7-watch-'+root.name.replace('_','-')+'.service'
    command=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(root/'monitor.log'),'--property=StandardError=inherit',
        str(ROOT/'environments/geometric_ik/bin/python'),'-m','modules.pipeline.monitoring.gpu_auto_start',
        '--config',str(config)]
    subprocess.run(command,check=True)
    write_json(root/'monitor_controller.json',{'at':now(),'unit':unit,'policy':str(config)})


def dispatch(root,gpu):
    reg=raw(root/'registration.json');verify(reg)
    if not reg['device_assignment_pending'] or (root/'STOP').exists() or (root/'started.json').exists():
        raise ValueError('The V7 registration was already dispatched/stopped')
    with (root/'dispatch_claim.json').open('x') as stream:json.dump({'at':now(),'gpu':gpu},stream)
    reg.update(gpu=gpu,device_assignment_pending=False);write_json(root/'registration.json',reg)
    environment={'CUDA_HOME':str(ROOT/'environments/runtime_bundle/cuda'),
        'CUDA_PATH':str(ROOT/'environments/runtime_bundle/cuda'),
        'TORCH_EXTENSIONS_DIR':str(ROOT/'tmp/curobo/torch_extensions_sm89'),
        'TORCH_CUDA_ARCH_LIST':'8.9','CUROBO_KERNEL_BACKEND':'cuda_core','GRASP_TRAJOPT_POLICY':'interactive',
        'PYTHONUNBUFFERED':'1','XDG_CACHE_HOME':str(ROOT/'tmp/curobo/arx-curobo/cache'),'MAX_JOBS':'2',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PATH':str(ROOT/'environments/runtime_bundle/cuda/bin')+':'+os.environ['PATH']}
    unit=controller_unit(root)
    command=['systemd-run','--user','--unit='+unit,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=StandardOutput=append:'+str(root/'controller.log'),
        '--property=StandardError=inherit']
    command+=['--setenv='+key+'='+value for key,value in environment.items()]
    command+=[str(ROOT/'environments/geometric_ik/bin/python'),'-m',
        'modules.pipeline.experiments.graspgenx_postprocessing.planning','run','--root',str(root)]
    write_json(root/'controller.json',{'at':now(),'unit':unit,'gpu':gpu,'cuda_environment':environment})
    subprocess.run(command,check=True,timeout=20)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['watch','dispatch'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu')
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='watch':watch(root)
    else:dispatch(root,args.gpu)
