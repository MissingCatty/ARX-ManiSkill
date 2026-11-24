"""Collect synchronized ARX wrist demonstrations from the verified bottle teacher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
sys.path.insert(0,str(ROOT/'scripts/curobo_arx/static_pick'))
from client import Client
from init_clear_bottle import initialize, BUNDLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve(); initialize(out)
    mailbox = args.mailbox.resolve()
    environment = os.environ.copy()
    environment.update(ARX_CUROBO_VALIDATION_DIR=str(mailbox),ARX_LILO_TEACHER_DIR=str(out/'teacher_raw'))
    neural = os.environ['ARX_NEURAL_PYTHON']; planner = os.environ['CUROBO_PYTHON']
    deadline=time.monotonic()+1200
    while not (mailbox/'ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('ARX simulator startup did not complete')
        time.sleep(2)
    client = Client(mailbox)
    client.request('grasp_lilo_status')
    subprocess.run(['bash',str(ROOT/'scripts/curobo_arx/apply_curobo_performance_patch.sh')],check=True)
    subprocess.run([neural,str(ROOT/'scripts/curobo_arx/static_pick/reproduce_setup.py'),
        '--output',str(out),'--mailbox',str(mailbox),'--additional-render-sensor'],env=environment,check=True)
    had_wrist = 'right_wrist' in client.request('snapshot')['camera_times']
    try:
        enabled = client.request('grasp_feedback_camera',enabled=True)
        (out/'wrist_camera.json').write_text(json.dumps(enabled,indent=2)+'\n')
        deadline = time.monotonic()+120
        while True:
            probe = client.request('grasp_camera_probe')
            if probe.get('right_wrist',{}).get('frame_time') is not None:
                break
            if time.monotonic()>deadline:
                raise TimeoutError(f'Wrist camera did not warm up: {probe}')
            time.sleep(.2)
        subprocess.run([neural,str(ROOT/'scripts/curobo_arx/static_pick/reproduce_record.py'),
            '--output',str(out),'--reference-run',str(BUNDLE),'--mailbox',str(mailbox),
            '--curobo-python',planner,'--neural-gpu',os.environ['ARX_NEURAL_GPU'],
            '--planner-gpu',os.environ['ARX_PLANNER_GPU'],'--trajopt-policy','interactive','--no-audit'],
            env=environment,check=True)
        record = json.loads((out/'teacher_recording.json').read_text())
        if record.get('error') or not record.get('samples'):
            raise RuntimeError(f'Teacher motion and dataset recording are separate: {record}')
    finally:
        if not had_wrist:
            (out/'wrist_camera_lifecycle.json').write_text(json.dumps({
                'kept_active':True,'reason':'Reuse the same calibrated sensor across trials; avoid destroying/recreating RTX sensors during simulation'})+'\n')


if __name__=='__main__':
    main()
