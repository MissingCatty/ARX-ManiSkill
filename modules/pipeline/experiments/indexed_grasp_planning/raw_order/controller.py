"""Own each phase, review it independently, then dispatch the authorized follow-up."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT,now,sha256,write_json
from arx_project.client import Client
from arx_project.artifacts import close_owned_transports
from ..physical.controller import environment,wait_ready
from ..physical.registration import guard,raw


def run(root):
    reg=guard(root);physical=reg['campaign_stage']=='physical'
    if reg['device_assignment_pending'] or (root/'started.json').exists() or list((root/'claims').glob('*.json')):
        raise ValueError('A raw-order phase must be fresh and CUDA assigned')
    if physical:
        from .registration import physical_gate
        if not physical_gate(raw(ROOT/reg['planning_gate_source']/'independent_review.json')):
            raise ValueError('Physical follow-up gate changed')
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':reg['gpu']})
    progress=raw(root/'progress.json');progress.update(at=now(),phase='warming');write_json(root/'progress.json',progress)
    socket_root=ROOT/'tmp/rpc'/sha256(root/'registration.json')[:12];socket_root.mkdir(parents=True)
    planner_socket=socket_root/'plan.sock';proxy_socket=socket_root/'order.sock'
    children={};streams=[]

    def start(name,command,env):
        log=(root/(name+'.log')).open('x');streams.append(log)
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children[name]=child
        write_json(root/(name+'_process.json'),{'at':now(),'pid':child.pid,'command':command})
        return child

    try:
        descriptor=root/'gripper_descriptor.json' if physical else ROOT/'configs/modules.json'
        args=['--kind','curobo_indexed_planning','--root',str(root/'resident'),'--socket',str(planner_socket),
            '--warmup-trial',str(root/'warmup'),'--descriptor',str(descriptor)]
        child=start('planner',[str(ROOT/'environments/curobo/bin/python'),'-u','-m',
            'modules.curobo.indexed_planning.execution_service',*args],environment(reg['gpu']))
        wait_ready(child,root/'resident/curobo_indexed_planning_ready.json')
        if physical:
            args=['--root',str(root/'proxy'),'--socket',str(proxy_socket),'--warmup-trial',str(root/'warmup'),
                '--descriptor',str(descriptor),'--campaign',str(root),'--planner-socket',str(planner_socket)]
            child=start('proxy',[str(ROOT/'environments/geometric_ik/bin/python'),'-u','-m',
                'modules.pipeline.experiments.indexed_grasp_planning.raw_order.proxy',*args],
                dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            wait_ready(child,root/'proxy/curobo_indexed_planning_ready.json',timeout=120)
            child=start('simulation',[str(ROOT/'environments/simulation/bin/python'),'-u','-m',
                'modules.pipeline.experiments.indexed_grasp_planning.physical.simulation',
                '--root',str(root),'--socket',str(proxy_socket)],environment(reg['gpu']))
            while child.poll() is None:
                if any(p.poll() is not None for name,p in children.items() if name!='simulation'):
                    raise RuntimeError('Resident service exited during the physical phase')
                time.sleep(1)
            if child.returncode!=0 or not (root/'execution_complete.json').exists():
                raise RuntimeError('Physical phase stopped; preserve simulation receipts')
        else:
            from .planning import run as planning
            planning(root,planner_socket)
        guard(root)
    finally:
        close_owned_transports()
        for name,child in reversed(list(children.items())):
            if child.poll() is None:
                socket=planner_socket if name=='planner' else proxy_socket if name=='proxy' else None
                if socket:
                    try:Client(socket,timeout=10).call({'command':'shutdown'})
                    except Exception:child.terminate()
                else:child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        for stream in streams:stream.close()
        write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':all(p.poll() is not None for p in children.values()),
            'returncodes':{name:p.returncode for name,p in children.items()}})
    from .review import planning_review,physical_review
    checked=physical_review(root) if physical else planning_review(root)
    progress=raw(root/'progress.json');progress.update(at=now(),phase='complete');write_json(root/'progress.json',progress)
    pointer=ROOT/'tmp/indexed_grasp_planning/funnel_active.json';active=raw(pointer)
    key='raw_order_physical_run' if physical else 'raw_order_run'
    active.update(at=now(),phase='raw_order_physical_complete' if physical else 'raw_order_planning_complete',
        **{key:str(root.relative_to(ROOT))});write_json(pointer,active)
    if not physical:
        from .registration import physical_gate,prepare_physical
        from .launch import watch
        if physical_gate(checked):
            following=ROOT/reg['followup_physical_directory'];prepare_physical(following,root)
            active.update(at=now(),phase='raw_order_physical_waiting',raw_order_physical_run=str(following.relative_to(ROOT)))
            write_json(pointer,active);watch(following)
        else:
            write_json(root/'physical_not_started.json',{'at':now(),'reason':'planning correctness or latency gate did not pass',
                'physical_slots_executed':0,'automatic_retry':False})
    from .report import report,publish
    report()
    try:publish(root)
    except Exception:
        write_json(root/'publication_error.json',{'at':now(),'error':traceback.format_exc(),'experiment_retry_required':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    try:run(args.root)
    except BaseException:
        if not (args.root/'STOP').exists():
            write_json(args.root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False})
        raise
