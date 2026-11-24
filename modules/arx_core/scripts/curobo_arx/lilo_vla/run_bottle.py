"""Run fresh ZeroGrasp reaching and actual LiLo-VLA local interaction on ARX."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(ROOT/'scripts/curobo_arx/static_pick')]
from client import Client
from init_clear_bottle import initialize,BUNDLE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,required=True)
    parser.add_argument('--policy-url',default='http://127.0.0.1:8008')
    parser.add_argument('--max-steps',type=int,default=60)
    args=parser.parse_args();out=args.output.resolve();mailbox=args.mailbox.resolve()
    initialize(out)
    environment=os.environ.copy();environment.update(ARX_CUROBO_VALIDATION_DIR=str(mailbox),
        ARX_LILO_POLICY_URL=args.policy_url,ARX_LILO_MAX_STEPS=str(args.max_steps))
    neural=os.environ['ARX_NEURAL_PYTHON']
    deadline=time.monotonic()+1200
    while not (mailbox/'ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('ARX simulator startup did not complete')
        time.sleep(2)
    client=Client(mailbox);client.request('grasp_lilo_status')
    subprocess.run([neural,str(ROOT/'scripts/curobo_arx/static_pick/reproduce_setup.py'),
        '--output',str(out),'--mailbox',str(mailbox),'--additional-render-sensor'],env=environment,check=True)
    snapshot=client.request('snapshot')
    if 'right_wrist' not in snapshot['camera_times']:
        client.request('grasp_feedback_camera',enabled=True);time.sleep(2)
    (out/'execution_mode.json').write_text(json.dumps({'reaching':'Fresh ZeroGrasp + cuRobo',
        'interaction':'Actual LiLo-VLA inference; no teacher action fallback',
        'success':'Unchanged lift/contact/hold and convergence checks','policy_url':args.policy_url})+'\n')
    subprocess.run([neural,str(ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py'),
        '--output',str(out),'--reference-run',str(BUNDLE),'--mailbox',str(mailbox),
        '--curobo-python',os.environ['CUROBO_PYTHON'],'--neural-gpu',os.environ['ARX_NEURAL_GPU'],
        '--planner-gpu',os.environ['ARX_PLANNER_GPU'],'--trajopt-policy','interactive','--no-audit'],env=environment,check=True)


if __name__=='__main__':main()
