"""Finish the pilot, review it, then run the already authorized 69-input matrix."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from .launch import launch
from .review import review


def wait_for_exit(run, status):
    unit = read_json(run/'controller.json')['unit']
    while True:
        value = subprocess.check_output(['systemctl','--user','show',unit,
            '-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],text=True)
        fields = dict(line.split('=',1) for line in value.strip().splitlines())
        progress = read_json(run/'progress.json')
        write_json(status,{'at':now(),'phase':'following_planning','run':str(run.relative_to(ROOT)),
            'completed_requests':progress['completed_requests'],'total_requests':progress['total_requests'],
            'current':progress.get('current'),'unit':unit})
        if fields.get('MainPID','0') == '0':
            if (run/'STOP').exists() or fields.get('ExecMainStatus') != '0':
                raise RuntimeError('Planning stopped; preserve this run and do not retry: '+str(run))
            if progress['phase'] != 'complete':
                raise RuntimeError('Planning exited without completing its registered requests')
            return
        time.sleep(15)


def publish(root):
    from website.tools.publish_indexed_planning_plan import publish as refresh
    from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.publication import sync_public
    refresh()
    node = max((Path.home()/'.nvm/versions/node').glob('v*/bin/node'),
        key=lambda p:tuple(map(int,p.parents[1].name[1:].split('.'))))
    with (root/'website_build.log').open('a') as stream:
        subprocess.run([str(node),'build.mjs'],cwd=ROOT/'website',stdout=stream,stderr=subprocess.STDOUT,check=True)
    sync_public(root,Path.home()/'.ssh/arx_public_repeat_continue')


def try_publish(root):
    """Keep website network failures separate from the planning comparison."""
    try:
        publish(root)
    except Exception:
        history = root/'publication_attempts'
        history.mkdir(exist_ok=True)
        write_json(history/f'{len(list(history.glob("*.json")))+1:03d}.json',
            {'at':now(),'published':False,'error':traceback.format_exc()})
        return False
    return True


def follow(root,pilot,full,gpu):
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'pilot':str(pilot),'full':str(full)})
    status = root/'status.json'
    wait_for_exit(pilot,status)
    reviewed = review(pilot)
    if not reviewed['passed'] or not all(reviewed['control_successes_retained'].values()):
        raise RuntimeError('Pilot FK or original-success retention needs diagnosis before full replay')
    write_json(root/'pilot_ready.json',{'at':now(),'pilot':str(pilot.relative_to(ROOT)),
        'independent_review_passed':True,'original_successes_retained':True})
    try_publish(root)
    deadline = time.monotonic()+6*3600
    while True:
        free = int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free >= 10000:break
        write_json(status,{'at':now(),'phase':'waiting_free_gpu','gpu':gpu,'free_mib':free,
            'full_registration_not_started':True})
        if time.monotonic()>deadline:raise RuntimeError('No planning GPU memory became available before the queue deadline')
        time.sleep(15)
    launch(full,gpu)
    try_publish(root)
    wait_for_exit(full,status)
    reviewed = review(full)
    if not reviewed['passed']:
        raise RuntimeError('Full replay finished but independent physical-arm FK requires diagnosis')
    config_path = ROOT/'configs/experiments/indexed_grasp_planning.json'
    config = read_json(config_path)
    config['native_batch_implementation']['full_run'] = str(full.relative_to(ROOT))
    config['native_batch_implementation']['independent_review_passed'] = reviewed['passed']
    config['execution'].update(current_phase='native_batch_replay_complete',
        next_stage='analyze_batch_results_then_binding_reuse_and_geometric_indexed_variants')
    config['implementation_status'] = 'native_batch_replay_complete_other_variants_pending'
    for variant in config['variants']:
        if variant['id']=='native_batched':variant['state']='planning_replay_complete'
    write_json(config_path,config)
    published = try_publish(root)
    write_json(status,{'at':now(),'phase':'complete','full_run':str(full.relative_to(ROOT)),
        'planning_requests':read_json(full/'acceptance.json')['planning_requests'],
        'independent_review_passed':reviewed['passed'],'website_published':published,'physical_grasps':0})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    args = parser.parse_args();root = ensure_run_path(args.root)
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('The V8 follow-up has already started; preserve its records')
    try:follow(root,ensure_run_path(args.pilot),ensure_run_path(args.full),args.gpu)
    except BaseException:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'physical_grasps':0})
        raise
