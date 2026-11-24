"""Execute retained registered resident workers sequentially on one GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_adapters import SUPPORTED_METHODS

PROJECT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).parent/'runs/five_methods_50_20261122_01'


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    registration=json.loads((RUN/'registration.json').read_text())
    removed=set(registration['methods'])-set(SUPPORTED_METHODS)
    if removed:raise ValueError('Registration contains retired methods: '+', '.join(sorted(removed)))
    slots=len(registration['methods'])*len(registration['cases'])
    with (RUN/'suite_started.json').open('x') as f:json.dump({'started_at':time.time(),'pid':os.getpid()},f)
    python_map={'zerograsp':PROJECT.parent/'tmp/arx-fullchain/env/bin/python',
                'spar3d':PROJECT/'envs/spar3d/bin/python'}
    gpu=registration['resources']['gpu_uuid'];finished=[]
    for method in registration['methods']:
        while True:
            raw=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            free=int(raw.strip())
            if free>=18000:break
            save(RUN/'progress.json',{'stage':'waiting_for_memory','method':method,'free_mib':free,'finished_methods':finished,'physical_trials':0})
            time.sleep(30)
        save(RUN/'progress.json',{'stage':'running','current_method':method,'finished_methods':finished,'registered_slots':slots,'physical_trials':0})
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',USE_SAGEATTN='0')
        python=python_map.get(method,PROJECT/'envs/reconstruction/bin/python')
        log=RUN/(method+'.log')
        with log.open('x') as f:
            process=subprocess.run([str(python),'-u',str(Path(__file__).parent/'run_method.py'),'--method',method],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=PROJECT.parent)
        if process.returncode:
            stop={'stage':'stopped','method':method,'exit_code':process.returncode,'finished_methods':finished,'log':str(log),'physical_trials':0}
            save(RUN/'STOP.json',stop);save(RUN/'progress.json',stop)
            subprocess.run([str(PROJECT/'envs/reconstruction/bin/python'),str(Path(__file__).parent/'summarize_results.py')])
            raise RuntimeError('Registered method stopped; no automatic retries: '+method)
        complete=json.loads((RUN/'methods'/method/'complete.json').read_text())
        assert len(complete['results'])==50
        finished.append(method)
        subprocess.run([str(PROJECT/'envs/reconstruction/bin/python'),str(Path(__file__).parent/'summarize_results.py')],check=True)
    save(RUN/'suite_complete.json',{'methods':finished,'registered_slots':slots,'result_records':slots,'physical_trials':0,'completed_at':time.time()})
    save(RUN/'progress.json',{'stage':'complete','finished_methods':finished,'result_records':slots,'physical_trials':0})


if __name__=='__main__':main()
