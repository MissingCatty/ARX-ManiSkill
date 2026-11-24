"""One bounded handoff from the authorized D retrain to the native-mask30 run.

No service/experiment restart, retry, extra training, or additional trial slots.
"""
import argparse
import itertools
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, verify_files, space
from scripts.curobo_arx.data_factory.launch import devices
from scripts.curobo_arx.seed_augmentation.campaign import available_memory
from .mask_repair_physical import prepare


def register(root, physical):
    root=Path(root).resolve();out=root/'followthrough'
    out.mkdir(exist_ok=False)
    if read(root/'selection.json')['reuse_D544']:raise ValueError('This handoff expects the triggered D retrain')
    paths=[Path(__file__),ROOT/'scripts/curobo_arx/nearfield/mask_repair_physical.py',
        ROOT/'scripts/curobo_arx/nearfield/native_mask_preflight.py',
        ROOT/'scripts/curobo_arx/nearfield/advisor31_campaign.py',
        ROOT/'scripts/curobo_arx/nearfield/advisor32_spec.py',
        ROOT/'scripts/curobo_arx/nearfield/visual_grounding_spec.py',
        ROOT/'scripts/curobo_arx/nearfield/closedloop30_policy.py',
        ROOT/'training/nearfield_arx/lateral_campaign.py',
        ROOT/'training/nearfield_arx/mask_repair_eval.py',ROOT/'training/nearfield_arx/native_mask.py',
        ROOT/'scripts/curobo_arx/data_factory/common.py',ROOT/'scripts/curobo_arx/data_factory/launch.py',
        root/'selection.json',root/'verification.json',root/'training/registration.json']
    write(out/'registration.json',{'created_unix':time.time(),'deadline_unix':time.time()+8*3600,
        'diagnostic':str(root),'physical':str(Path(physical).resolve()),
        'files':{str(p):sha(p) for p in paths},'max_new_physical_slots':30,
        'training_retries':0,'physical_retries':0,'restart_services_on_failure':False,
        'resource_scope':'Any available coolas-8 GPU; distinct simulation device; original memory floors',
        'training_unit':'arx-native-mask-train-20261111-03.service'})
    write(out/'state.json',{'status':'prepared'})
    return str(out)


def choose(inventory):
    choices=[]
    for policy,sim in itertools.permutations(inventory,2):
        if policy['free_mb']<9500 or sim['free_mb']<9000:continue
        computes=[c for c in inventory if c['uuid']!=sim['uuid'] and
            c['free_mb']>=(12000 if c['uuid']==policy['uuid'] else 5000)]
        for compute in computes:
            separate=compute['uuid']!=policy['uuid']
            score=(separate,policy['index']==2,sim['index']==3,
                   min(policy['free_mb']-9500,sim['free_mb']-9000),compute['free_mb'])
            choices.append((score,(policy['index'],compute['index'],sim['index'])))
    return max(choices)[1] if choices and available_memory()>=40*2**30 else None


def run(folder):
    out=Path(folder).resolve();reg=read(out/'registration.json');root=Path(reg['diagnostic'])
    if (out/'STARTED.json').exists() or (out/'STOP').exists():raise RuntimeError('No handoff retry')
    write(out/'STARTED.json',{'wall':time.time(),'registration_sha256':sha(out/'registration.json')})
    child=None
    def interrupted(signum, frame):
        raise InterruptedError('Handoff received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    def check():
        if time.time()>reg['deadline_unix'] or any(p.exists() for p in
                (out/'STOP',root/'STOP',root/'training/STOP')):raise RuntimeError('Stopped/deadline/training fault')
    try:
        verify_files(reg['files'])
        while True:
            check();p=root/'training/D/state.json';state=read(p) if p.exists() else {'status':'starting'}
            if state['status']=='fault':raise RuntimeError('D training failed; no physical launch')
            props=subprocess.check_output(['systemctl','--user','show',reg['training_unit'],
                '-p','MainPID','-p','ExecMainStatus'],text=True)
            values=dict(x.split('=',1) for x in props.strip().splitlines())
            if values['MainPID']=='0':
                if state['status']!='complete' or values['ExecMainStatus']!='0':
                    raise RuntimeError('Training exited without a complete successful receipt')
                break
            write(out/'state.json',{'status':'waiting_for_training','training':state,'wall':time.time()})
            time.sleep(15)
        stable=0;selected=None
        while stable<3:
            check();space(root);inventory=devices();choice=choose(inventory)
            stable=stable+1 if choice is not None and choice==selected else int(choice is not None)
            selected=choice
            write(out/'state.json',{'status':'waiting_for_physical_resources','inventory':inventory,
                'selected_indices':selected,'stable_samples':stable,'wall':time.time()})
            if stable<3:time.sleep(10)
        verify_files(reg['files'])
        write(out/'state.json',{'status':'preparing_physical_registration','indices':selected,'wall':time.time()})
        physical=Path(prepare(reg['physical'],root,*selected));p=read(physical/'registration.json')
        write(out/'physical_registration.json',{'path':str(physical),'sha256':sha(physical/'registration.json'),
            'slots':30,'training_completed_before_launch':True})
        command=[sys.executable,'-m','scripts.curobo_arx.nearfield.advisor33_launch',str(physical)]
        with (out/'physical.log').open('x') as log:
            child=subprocess.Popen(command,cwd=p['source_snapshot'],stdout=log,stderr=subprocess.STDOUT)
            write(out/'physical_process.json',{'pid':child.pid,'command':command,'wall':time.time()})
            write(out/'state.json',{'status':'physical_running','campaign':str(physical),'wall':time.time()})
            while child.poll() is None:
                if (out/'STOP').exists() or (root/'STOP').exists():
                    write(physical/'STOP',{'reason':'Explicit upstream stop; no continuation'})
                time.sleep(5)
            code=child.returncode
        if code:raise RuntimeError('Physical coordinator stopped; no retry: '+str(code))
        result=read(physical/'advisor33_verification.json')
        write(out/'state.json',{'status':'complete','completed':result['completed'],
            'successes':result['successes'],'unknown':result['unknown'],'unexecuted':result['unexecuted'],
            'record_integrity_pass':result['record_integrity_pass'],'wall':time.time()})
    except BaseException:
        error=traceback.format_exc();write(out/'FAULT.json',{'error':error,'wall':time.time()})
        write(out/'STOP',{'reason':'Bounded handoff failed; no restart','error':error})
        write(out/'state.json',{'status':'fault','error':error,'wall':time.time()})
        if child is not None and child.poll() is None:
            write(Path(reg['physical'])/'STOP',{'reason':'Handoff interrupted; coordinator must stop safely'})
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                write(out/'unconfirmed_child_exit.json',{'pid':child.pid,'manual_review_required':True})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','run'));p.add_argument('root')
    p.add_argument('--physical');a=p.parse_args()
    print(register(a.root,a.physical)) if a.action=='register' else run(a.root)
