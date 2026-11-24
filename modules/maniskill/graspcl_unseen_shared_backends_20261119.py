"""One ZeroGrasp/cuRobo pair retained for all 350 slots; frozen banks per arm."""
from arx_project.layout import canonical_path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import time

import graspcl_pipeline as common
import graspcl_unseen_bench as bench
from arx_service_readiness import wait_ready
from benchmark_lifecycle import mark_ready
from resident_backend import rpc

CODE=Path(__file__).resolve().parent
NEURAL_GPU='GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
PLANNER_GPU='GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'
MANAGER=None


class Manager:
    def __init__(self,experiment,epoch):
        self.experiment=experiment;self.epoch=epoch
        self.shared={};self.all_processes={};self.streams=[]

    def launch(self,kind,group,reg):
        root=group/'resident'
        key=hashlib.sha256((str(group)+self.epoch+kind).encode()).hexdigest()[:14]
        sock=CODE/'tmp'/('shared5_'+key+'.sock')
        assert not sock.exists()
        python=canonical_path(CODE.parent/{'zerograsp':'tmp/arx-fullchain/env/bin/python',
            'curobo':'curobo/.venv/bin/python','graspcl':'tmp/arx-graspcl-env/bin/python'}[kind])
        gpu=NEURAL_GPU if kind=='zerograsp' else PLANNER_GPU
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,ARX_NEURAL_GPU=NEURAL_GPU,ARX_PLANNER_GPU=PLANNER_GPU,
            ARX_SIM_GPU=PLANNER_GPU,LILO_GPU=PLANNER_GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            GRASP_TRAJOPT_POLICY='interactive')
        env.pop('PYTHONPATH',None)
        command=['bash','-c','source "$1"; shift; exec "$@"','shared-frozen-service',
            str(CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'),str(python),
            str(CODE/'graspcl_unseen_shared_services_20261119.py'),'--kind',kind,'--root',str(root),
            '--socket',str(sock),'--warmup-trial',reg['warmup_trial']]
        if kind=='graspcl':
            env['CUDA_VISIBLE_DEVICES']=''
            command.extend(['--bank',reg['bank'],'--policy-registration',str(group/'registration.json')])
        common.save(root/(kind+'_startup_progress.json'),{'phase':'loading','component':kind,'at':bench.core.now(),
            'gpu':gpu if kind!='graspcl' else 'CPU','wait_policy':'Published ready PID and successful health RPC'})
        stream=(root/(kind+'.log')).open('x');self.streams.append(stream)
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record={'process':process,'socket':sock,'owner':root,'kind':kind,'group':group.name}
        self.all_processes[group.name+':'+kind]=record
        health=wait_ready(sock,root/(kind+'_ready.json'),process,kind,timeout=1800)
        print('SHARED RESIDENT READY',kind,process.pid,health['startup_seconds'],flush=True)
        return record

    def component(self,kind,group,reg):
        if kind in {'zerograsp','curobo'}:
            if kind not in self.shared:self.shared[kind]=self.launch(kind,group,reg)
            record=self.shared[kind]
            health=rpc(record['socket'],{'command':'health'},timeout=120)
            if record['process'].poll() is not None or health['pid']!=record['process'].pid:
                raise RuntimeError('Shared service exited; no reload or replay')
            if record['owner']!=group/'resident':
                for suffix in ['_ready.json','_stopped.json']:
                    (group/'resident'/(kind+suffix)).symlink_to(record['owner']/(kind+suffix))
                if kind=='curobo':
                    (group/'resident/planner_warmup').symlink_to(record['owner']/'planner_warmup',target_is_directory=True)
            return record
        return self.launch(kind,group,reg)

    def stop_record(self,record):
        process=record['process']
        if process.poll() is None:
            try:
                rpc(record['socket'],{'command':'shutdown'},timeout=15)
                process.wait(timeout=30)
            except Exception:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()

    def stop(self):
        for record in self.all_processes.values():self.stop_record(record)
        for stream in self.streams:stream.close()
        receipt={name:{'pid':r['process'].pid,'exited':r['process'].poll() is not None,'kind':r['kind'],
            'socket':str(r['socket'])} for name,r in self.all_processes.items()}
        common.save(self.experiment/self.epoch/'model_services_stopped.json',{'services':receipt,
            'all_owned_model_services_exited':all(r['exited'] for r in receipt.values()),
            'zero_model_loads':int('zerograsp' in self.shared),'curobo_pool_loads':int('curobo' in self.shared)})


class Backends(bench.core.Backends):
    def start(self):
        root=self.campaign/'resident';root.mkdir(exist_ok=False)
        common.save(root/'startup_progress.json',{'phase':'loading','components':self.reg['required_components'],
            'at':bench.core.now(),'handshake':'published ready plus live health'})
        missing=[kind for kind in ['zerograsp','curobo'] if kind not in MANAGER.shared]
        if missing:
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures={kind:pool.submit(MANAGER.launch,kind,self.campaign,self.reg) for kind in missing}
                for kind,future in futures.items():MANAGER.shared[kind]=future.result()
        self.records={}
        for kind in self.reg['required_components']:
            record=MANAGER.component(kind,self.campaign,self.reg)
            self.records[kind]=record;self.children[kind]=record['process']
        self.config={'owner_pid':os.getpid(),'required_components':self.reg['required_components'],
            'gpu_by_component':{'zerograsp':NEURAL_GPU,'curobo':PLANNER_GPU,'graspcl':'CPU'},
            'sockets':{k:str(r['socket']) for k,r in self.records.items()},
            'shared_across_phases':['zerograsp','curobo']}
        common.save(root/'backend_config.json',self.config)
        mark_ready(root,self.reg['required_components'],self.config,self.children)
        if 'graspcl' not in self.reg['required_components']:
            self.config['sockets']['graspcl']=str(root/'local_adapter/graspcl.sock')
            self.config['local_adapters']={'graspcl':'Native candidate adapter; no GraspCL inference'}
            common.save(root/'backend_config.json',self.config)
        common.save(root/'startup_progress.json',{'phase':'ready','at':bench.core.now(),
            'shared_model_pids':{k:r['process'].pid for k,r in self.records.items()}})

    def stop(self):
        # Frozen B/C banks are loaded once for their own 100-case arm. The
        # common ZeroGrasp/cuRobo pair belongs to the whole suite manager.
        for kind,record in getattr(self,'records',{}).items():
            if kind=='graspcl':MANAGER.stop_record(record)
