"""Register one fresh planning comparison and keep it alive outside the terminal."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from .replay import prepare


def launch(root, gpu, selected=None):
    config_path = ROOT/'configs/experiments/indexed_grasp_planning.json'
    config = read_json(config_path)
    previous = config['execution'].get('active_run')
    if previous:
        directory = ROOT/previous
        controller = directory/'controller.json'
        if controller.is_file():
            status = subprocess.check_output(['systemctl','--user','show',read_json(controller)['unit'],
                '-p','MainPID','-p','ActiveState'],text=True)
            fields = dict(line.split('=',1) for line in status.strip().splitlines())
            if fields.get('MainPID','0') != '0' or fields.get('ActiveState') in ['active','activating']:
                raise RuntimeError('The preceding V8 planning process is still running')
    source = ROOT/config['baseline_run']
    if not selected:
        pilot = ROOT/config['native_batch_implementation']['pilot_run']
        accepted = read_json(pilot/'acceptance.json')
        reviewed = read_json(pilot/'independent_review.json')
        if (not accepted['task_complete'] or not reviewed['passed']
                or not all(accepted['control_successes_retained'].values())):
            raise RuntimeError('The native batch pilot must retain control successes and pass independent FK before the full replay')
    prepare(root,source,gpu,selected)
    write_json(root/'configuration_before.json',config)
    config['execution'].update(active_run=str(root.relative_to(ROOT)),current_phase='native_batch_pilot' if selected else 'native_batch_full_replay')
    config['native_batch_implementation']['offline_checks'] = read_json(
        ROOT/'tmp/indexed_grasp_planning/implementation_20261126_01/offline_checks.json')['tests']
    if selected:
        config['native_batch_implementation']['pilot_run'] = str(root.relative_to(ROOT))
    write_json(config_path,config)
    unit = 'arx-v8-'+root.name.replace('_','-')+'.service'
    environment = {'CUDA_HOME':str(ROOT/'environments/runtime_bundle/cuda'),
        'CUDA_PATH':str(ROOT/'environments/runtime_bundle/cuda'),
        'TORCH_EXTENSIONS_DIR':str(ROOT/'tmp/curobo/torch_extensions_sm89'),
        'TORCH_CUDA_ARCH_LIST':'8.9','CUROBO_KERNEL_BACKEND':'cuda_core',
        'GRASP_TRAJOPT_POLICY':'interactive','PYTHONUNBUFFERED':'1',
        'XDG_CACHE_HOME':str(ROOT/'tmp/curobo/arx-curobo/cache'),'MAX_JOBS':'2',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'PATH':str(ROOT/'environments/runtime_bundle/cuda/bin')+':'+os.environ['PATH']}
    command = ['systemd-run','--user','--unit='+unit,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),
        '--property=StandardOutput=append:'+str(root/'controller.log'),'--property=StandardError=inherit']
    command += ['--setenv='+name+'='+value for name,value in environment.items()]
    command += [str(ROOT/'environments/curobo/bin/python'),'-m',
        'modules.pipeline.experiments.indexed_grasp_planning.replay','run','--root',str(root)]
    subprocess.run(command,check=True)
    write_json(root/'controller.json',{'at':now(),'unit':unit,'run':str(root.relative_to(ROOT)),
        'physical_grasps':0,'cuda_environment':environment})
    print(json.dumps({'unit':unit,'run':str(root.relative_to(ROOT)),
        'planning_requests':read_json(root/'registration.json')['planning_requests']},ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--cases',nargs='+')
    args = parser.parse_args()
    launch(ensure_run_path(args.root),args.gpu,args.cases)
