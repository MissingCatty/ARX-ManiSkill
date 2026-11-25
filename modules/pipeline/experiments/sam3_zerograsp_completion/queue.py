"""Wait for GraspGenX, execute two authorized makeup cases, publish and push."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT, now
from .prepare import prepare
from .finalize import dirty_paths, publish_final, push_git


def unit_active(unit):
    value = subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
    return value.stdout.strip() in ['active','activating','deactivating','reloading']


def run(configuration):
    config = read_json(configuration)
    queue_root = ROOT/config['queue_root']
    queue_root.mkdir(parents=True,exist_ok=True)
    if (queue_root/'started.json').exists():
        raise RuntimeError('Completion queue must not be automatically restarted')
    atomic_json(queue_root/'started.json',{'at':now(),'configuration':str(configuration)})
    atomic_json(queue_root/'git_paths.json',{'at':now(),'paths':dirty_paths()})
    predecessor, root = ROOT/config['predecessor'], ROOT/config['completion_root']
    try:
        while True:
            if (predecessor/'STOP').exists() or (predecessor/'ABORT').exists():
                raise RuntimeError('GraspGenX predecessor stopped before completion')
            summary = read_json(predecessor/'progress.json')
            busy = [unit for unit in config['predecessor_units'] if unit_active(unit)]
            accepted = predecessor/'acceptance.json'
            stopped = predecessor/'services_stopped.json'
            if not busy and accepted.exists() and stopped.exists():
                if not read_json(accepted)['passed'] or not read_json(stopped)['all_owned_exited']:
                    raise RuntimeError('Predecessor acceptance or service exit gate failed')
                break
            atomic_json(queue_root/'status.json',{'at':now(),'phase':'waiting_for_graspgenx_and_acceptance',
                'completed':{m:v['completed'] for m,v in summary['methods'].items()},'waiting_units':busy,
                'new_grasps_started':0})
            time.sleep(10)
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'registering_two_makeup_cases'})
        prepare(root,predecessor,config)
        previous_pointer = read_json(ROOT/'tmp/sam3_zerograsp_comparison/active_io_repeat.json')
        atomic_json(root/'previous_active_pointer.json',previous_pointer)
        atomic_json(ROOT/'tmp/sam3_zerograsp_comparison/active_io_repeat.json',{'at':now(),
            'run':config['completion_root'],'progress':config['completion_root']+'/progress.json',
            'controller':'arx-v6-complete74-20261126-01.service',
            'queue_status':config['queue_root']+'/status.json'})
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'running_two_authorized_makeup_cases',
            'run':config['completion_root'],'cases':config['authorized_makeup_cases']})
        with (root/'campaign.log').open('x') as stream:
            subprocess.run([sys.executable,'-u','-m',
                'modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.campaign',
                'run','--root',str(root)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'independent_final_acceptance'})
        from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.audit import audit
        acceptance = audit(root)
        if not acceptance['passed'] or not acceptance['task_complete']:
            raise RuntimeError('Final 148-case independent acceptance failed')
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'publishing_final_results'})
        summary = publish_final(root,config)
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'pushing_git'})
        commit = push_git(root,config,queue_root)
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'complete','commit':commit,
            'completed':{m:v['completed'] for m,v in summary['methods'].items()},
            'successes':{m:v['successes'] for m,v in summary['methods'].items()},
            'public_updated':True,'git_pushed':True})
    except BaseException:
        atomic_json(queue_root/'STOP',{'at':now(),'error':traceback.format_exc()})
        atomic_json(queue_root/'status.json',{'at':now(),'phase':'stopped','error':traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--configuration',type=Path,required=True)
    run(parser.parse_args().configuration.resolve())


if __name__=='__main__':
    main()
