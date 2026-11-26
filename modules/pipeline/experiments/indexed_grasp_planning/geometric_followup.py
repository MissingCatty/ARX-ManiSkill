"""Wait for C, run a bounded D pilot, and expand only an accepted improvement."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from .followup import try_publish, wait_for_exit
from .geometric_replay import launch
from .review import review


def wait_for_c(root, predecessor):
    controller=read_json(predecessor/'controller.json')['unit']
    while True:
        if (predecessor/'STOP').exists():raise RuntimeError('C follow-up stopped; preserve it before starting D')
        status=read_json(predecessor/'status.json')
        unit=subprocess.check_output(['systemctl','--user','show',controller,'-p','MainPID'],text=True).strip()
        if status['phase']=='complete' and unit=='MainPID=0':return
        if unit=='MainPID=0':raise RuntimeError('C follow-up exited without completion')
        write_json(root/'status.json',{'at':now(),'phase':'waiting_for_C_completion',
            'preceding_progress':status,'D_planning_requests':0})
        time.sleep(15)


def finish(root,run,phase):
    path=ROOT/'configs/experiments/indexed_grasp_planning.json';config=read_json(path)
    config['execution'].update(current_phase=phase,next_stage='analyze_geometric_results_then_index_reuse_and_combination')
    config['implementation_status']=phase+'_index_and_combination_pending'
    config['geometric_ik']['accepted_run']=str(run.relative_to(ROOT))
    for variant in config['variants']:
        if variant['id']=='geometric_ik':variant['state']=phase
    write_json(path,config)
    published=try_publish(root)
    write_json(root/'status.json',{'at':now(),'phase':'complete','run':str(run.relative_to(ROOT)),
        'stage_result':phase,'website_published':published,'physical_grasps':0})


def follow(root,pilot,full,gpu,predecessor):
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'started.json',{'at':now(),'predecessor':str(predecessor),'pilot':str(pilot),'full':str(full)})
    wait_for_c(root,predecessor)
    deadline=time.monotonic()+6*3600
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free>=10000:break
        write_json(root/'status.json',{'at':now(),'phase':'waiting_free_gpu','gpu':gpu,'free_mib':free})
        if time.monotonic()>deadline:raise RuntimeError('D resource queue deadline expired')
        time.sleep(15)
    policy=read_json(ROOT/'configs/experiments/indexed_grasp_geometric_ik.json')
    launch(pilot,gpu,policy['pilot_cases']);try_publish(root)
    wait_for_exit(pilot,root/'status.json')
    checked=review(pilot);accepted=read_json(pilot/'acceptance.json')
    if not checked['passed'] or not all(accepted['control_successes_retained'].values()):
        raise RuntimeError('D pilot failed physical-arm FK or original success retention')
    control,geometric=(accepted['summaries'][key] for key in ['control','geometric'])
    saving=1-geometric['total_seconds']/control['total_seconds']
    admitted=saving>=policy['full_replay_gate']['minimum_total_wall_reduction_fraction']
    write_json(root/'pilot_gate.json',{'at':now(),'independent_review_passed':True,
        'original_successes_retained':True,'total_wall_reduction_fraction':saving,
        'selection_changes':sum(not c['same_selection'] for r in accepted['paired_cases'] for c in r['comparison'].values()),
        'full_replay_admitted':admitted,'physical_grasps':0})
    if not admitted:
        finish(root,pilot,'geometric_pilot_no_material_speedup');return
    launch(full,gpu);try_publish(root)
    wait_for_exit(full,root/'status.json')
    checked=review(full)
    if not checked['passed']:raise RuntimeError('D full replay failed independent physical-arm FK')
    retained=all(read_json(full/'acceptance.json')['control_successes_retained'].values())
    finish(root,full,'geometric_IK_replay_complete' if retained else 'geometric_replay_original_success_regression')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','pilot','full','predecessor']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',required=True);args=parser.parse_args();root=ensure_run_path(args.root)
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('D queue already started')
    try:follow(root,ensure_run_path(args.pilot),ensure_run_path(args.full),args.gpu,ensure_run_path(args.predecessor))
    except BaseException:
        root.mkdir(parents=True,exist_ok=True);write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()});raise
