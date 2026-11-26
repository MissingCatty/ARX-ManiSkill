"""Review the five-case pilot, then finish the authorized 69-case comparison."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT,ensure_run_path,now,read_json,write_json
from .planning import CONFIG,prepare
from .planning_launch import watch,controller_unit
from .planning_review import review


def follow_run(run,status):
    while True:
        progress=read_json(run/'progress.json')
        write_json(status,{'at':now(),'phase':'following_planning','run':str(run.relative_to(ROOT)),
            'completed_requests':progress['completed_requests'],'total_requests':progress['total_requests'],
            'current':progress.get('current')})
        if (run/'STOP').exists():raise RuntimeError('V7 planning stopped; preserve this registration')
        if (run/'controller.json').exists():
            result=subprocess.check_output(['systemctl','--user','show',controller_unit(run),
                '-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],text=True)
            fields=dict(line.split('=',1) for line in result.strip().splitlines())
            if fields.get('MainPID')=='0' and fields.get('ActiveState') not in ['activating','active']:
                if fields.get('ExecMainStatus')!='0' or progress['phase']!='complete':
                    raise RuntimeError('V7 controller exited before finishing registered requests')
                return review(run)
        time.sleep(5)


def publish(root):
    from website.tools.publish_graspgenx_postprocessing import publish as refresh
    from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.publication import sync_public
    try:
        refresh()
        node=max((Path.home()/'.nvm/versions/node').glob('v*/bin/node'),
            key=lambda p:tuple(map(int,p.parents[1].name[1:].split('.'))))
        with (root/'website_build.log').open('a') as stream:
            subprocess.run([str(node),'build.mjs'],cwd=ROOT/'website',stdout=stream,stderr=subprocess.STDOUT,check=True)
        sync_public(root,Path.home()/'.ssh/arx_public_repeat_continue')
        write_json(root/'website_publication.json',{'at':now(),'passed':True})
    except Exception:
        write_json(root/'website_publication.json',{'at':now(),'passed':False,'error':traceback.format_exc()})


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'started.json').exists():raise ValueError('Do not duplicate the V7 planning follow-up')
    write_json(root/'started.json',{'at':now()});cfg=read_json(CONFIG)
    pilot=ROOT/cfg['pilot']['run'];full=ROOT/cfg['full']['run'];status=root/'status.json'
    checked=follow_run(pilot,status)
    write_json(root/'pilot_review.json',checked)
    if not checked['passed'] or not checked['control_successes_retained']:
        write_json(status,{'at':now(),'phase':'pilot_needs_analysis','full_run_started':False})
        publish(root);return
    prepare(full,'full');watch(full)
    write_json(ROOT/'tmp/graspgenx_postprocessing/planning_active.json',{'at':now(),'run':str(full.relative_to(ROOT)),'phase':'full'})
    publish(root)
    checked=follow_run(full,status)
    publish(root)
    write_json(status,{'at':now(),'phase':'complete','run':str(full.relative_to(ROOT)),
        'planning_requests':read_json(full/'acceptance.json')['planning_requests'],
        'independent_review_passed':checked['passed'],'control_successes_retained':checked['control_successes_retained'],
        'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=ensure_run_path(parser.parse_args().root)
    try:run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()});raise
