"""Launch bounded production, update coverage, and export accepted data at exit."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from scripts.curobo_arx.data_factory.common import ROOT,read,write
from scripts.curobo_arx.data_factory.launch import launch
from .report import report
from .export import export


def monitor(campaign):
    campaign=Path(campaign).resolve();stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    while not stop:
        report(campaign)
        state=read(campaign/'state.json')
        if state['status'] in ('fault','stopped','budget_exhausted','quota_complete'):break
        for _ in range(30):
            if stop:break
            time.sleep(1)


def final_export(campaign, summary):
    """Never publish an incomplete terminal snapshot while a group can still commit."""
    campaign=Path(campaign);state=read(campaign/'state.json')
    reason=None
    if state.get('unresolved_groups'):reason='Simulator shutdown is unresolved'
    else:
        for path in (campaign/'group_registrations').glob('*.process.json'):
            name=read(path)['container']
            result=subprocess.run(['podman','container','exists',name],capture_output=True,timeout=15)
            if result.returncode!=1:
                reason='Registered simulator container remains or its absence cannot be confirmed: '+name
                break
    if reason:
        write(campaign/'EXPORT_DEFERRED.json',{'reason':reason,'wall':time.time(),
            'committed_successes_at_check':summary['full_ARX_successful_demonstrations'],
            'scope':'Committed raw episodes retained; reconcile export only after confirmed shutdown'})
        return None
    if not summary['full_ARX_successful_demonstrations']:return None
    target=campaign/'exports'/('final_'+str(time.time_ns()))
    try:
        result=export(campaign,target);write(campaign/'final_export.json',{'path':str(target),**result})
        return result
    except BaseException:write(campaign/'EXPORT_FAULT.json',{'error':traceback.format_exc(),'path':str(target)})


def run(campaign,compute_gpu=None,sim_gpus=None):
    campaign=Path(campaign).resolve();reg=read(campaign/'registration.json')
    if reg.get('stage')!='sdg_stage1' or not reg.get('production_ready'):raise ValueError('Not a prepared SDG stage1 registration')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=(campaign/'coverage_monitor.log').open('x')
    child=subprocess.Popen([sys.executable,'-m','scripts.curobo_arx.sdg_stage1.run',str(campaign),'--monitor'],
        cwd=reg['source_snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();write(campaign/'coverage_monitor_process.json',{'pid':child.pid,'kind':'cpu_coverage_monitor'})
    try:launch(campaign,compute_gpu=compute_gpu,sim_gpus=sim_gpus)
    except BaseException:
        write(campaign/'RUNNER_FAULT.json',{'error':traceback.format_exc(),'wall':time.time()})
        raise
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                write(campaign/'monitor_cleanup_unknown.json',{'pid':child.pid});raise
        final=report(campaign)
        final_export(campaign,final)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');p.add_argument('--compute-gpu')
    p.add_argument('--sim-gpus',nargs='+');p.add_argument('--monitor',action='store_true');a=p.parse_args()
    if a.monitor:monitor(a.campaign)
    else:run(a.campaign,compute_gpu=a.compute_gpu,sim_gpus=a.sim_gpus)
