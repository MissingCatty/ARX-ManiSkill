"""Collect new physical ARX wrist data by replaying a verified cuRobo teacher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(ROOT/'scripts/curobo_arx/static_pick')]
from client import Client
from init_clear_bottle import initialize
from settling import assess_settling


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,required=True)
    parser.add_argument('--inspect-pregrasp-only',action='store_true')
    args=parser.parse_args();source=args.source_run.resolve();out=args.output.resolve()
    read=lambda p:json.loads(p.read_text())
    assert read(source/'verification.json')['pass']
    initialize(out);os.environ['ARX_CUROBO_VALIDATION_DIR']=str(args.mailbox.resolve())
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
    deadline=time.monotonic()+1200
    while not (args.mailbox/'ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('ARX simulator startup did not complete')
        time.sleep(2)
    subprocess.run([sys.executable,str(ROOT/'scripts/curobo_arx/static_pick/reproduce_setup.py'),
        '--output',str(out),'--mailbox',str(args.mailbox.resolve()),'--additional-render-sensor'],check=True)
    os.environ.update(ARX_RESEARCH_GUARD_FILE=str(out/'guard.json'),ARX_RESEARCH_AUDIT='0')
    client=Client(args.mailbox);selected=read(source/'selected_grasp.json')
    initial=client.request('snapshot');save('capture_snapshot.json',initial)
    if 'right_wrist' not in initial['camera_times']:
        client.request('grasp_feedback_camera',enabled=True);time.sleep(2)
    original_z=initial['objects']['test_object']['world'][2][3]
    stages=('pregrasp',) if args.inspect_pregrasp_only else ('pregrasp','approach','close','lift','hold')
    contacts=[];assessments={};timings={};recording=False;video=False;error=None;hold=None
    save('teacher_provenance.json',{'source_run':str(source),'new_zerograsp_inference':False,
        'source':'Previously physically verified ZeroGrasp + cuRobo commands; executed anew, with new wrist RGB-D and feedback',
        'command_sha256':{name:hashlib.sha256((source/f'{name}_command.json').read_bytes()).hexdigest() for name in stages}})
    try:
        client.request('grasp_lilo_record_start',directory=str(out/'teacher_raw'));recording=True
        client.request('grasp_video_start',directory=str(out/'frames'));video=True
        for name in stages:
            command=read(source/f'{name}_command.json');save(f'{name}_command.json',command)
            result=client.request('trajectory',timeout=300,**command);save(f'{name}_execution.json',result)
            contacts.extend(result['contacts'])
            goal=selected[{'pregrasp':'pregrasp_world','approach':'grasp_world','close':'grasp_world','lift':'lift_world','hold':'lift_world'}[name]]
            assessment=assess_settling(result,goal);assessments[name]=assessment
            timings[name]={key:result[key] for key in ('sim_duration_s','wall_duration_s')}
            save(f'{name}_settling.json',assessment)
            assert assessment['accepted'],f'{name} failed original convergence criteria'
            target=initial['objects']['test_object']['path']
            bad=[e for e in result['contacts'] if e['num_contacts']>0 and not (
                '/World/Robot/' in e['a'] and '/World/Robot/' in e['b']) and not (
                target in e['a']+e['b'] and any(f in e['a']+e['b'] for f in ('right_link27','right_link28')))]
            assert not bad,f'Forbidden contact during {name}'
            if name=='pregrasp':
                image=client.request('grasp_lilo_observation',directory=str(out/'pregrasp_view'))
                save('pregrasp_observation.json',image)
            if name=='close':
                fingers={f for e in result['contacts'] if e['num_contacts']>0 and target in e['a']+e['b'] for f in ('right_link27','right_link28') if f in e['a']+e['b']}
                assert len(fingers)==2,'No bilateral bottle contact; lift is not executed'
            if name=='hold':hold=result
            print(f'{name}: original convergence/contact criteria passed',flush=True)
    except Exception as exc:
        error=str(exc);raise
    finally:
        if recording:save('teacher_recording.json',client.request('grasp_lilo_record_stop'))
        if video:save('recording_stop.json',client.request('grasp_video_stop',timeout=120))
        minimum=min(r['objects']['test_object'][2][3] for r in hold['records']) if hold else original_z
        success=bool(hold and minimum>original_z+.06 and error is None)
        save('verification.json',{'pass':success,'physical_grasp_success':success,'error':error,
            'new_zerograsp_inference':False,'teacher_source':'verified cuRobo command replay',
            'held_min_rise_m':minimum-original_z,'configuration_audit_enabled':False,
            'historical_physics_matched_before_execution':True,
            'pregrasp_inspection_only':args.inspect_pregrasp_only})
        save('grasp_report.json',{'success':success,'stage_convergence':assessments,'stage_timing':timings})
    if video:
        subprocess.run([sys.executable,str(ROOT/'scripts/curobo_arx/static_pick/encode_reproduction.py'),str(out)],check=True)


if __name__=='__main__':main()
