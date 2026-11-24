"""Continue the real ARX LiLo pilot through download, adaptation and execution.

Run with the sourced cuRobo environment and the neural Python interpreter.
Progress, child-process IDs, errors and unabridged logs survive terminal exit.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from gpu_policy import POLICY_GPU,resolve,compute_env,verify_health


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,required=True)
    parser.add_argument('--lilo-python',type=Path,
        default=ROOT.parent/'lilo-vla/.venv/bin/python')
    parser.add_argument('--cdn-proxy',action='append',default=[])
    parser.add_argument('--download-workers',type=int,default=8)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--max-actions',type=int,default=60)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dataset=args.dataset.resolve();checkpoint=args.checkpoint.resolve()
    # Keep the venv interpreter path: resolving its symlink bypasses pyvenv.cfg.
    env=os.environ.copy();lilo=str(args.lilo_python.absolute())
    gpu=resolve(env.get('LILO_GPU',POLICY_GPU))['uuid']
    env=compute_env(env,gpu)
    port=int(env.get('LILO_PORT','8008'));url=f'http://127.0.0.1:{port}'
    progress={'status':'running','pid':os.getpid(),'started_at':time.time(),
        'dataset':str(dataset),'checkpoint':str(checkpoint),'mailbox':str(args.mailbox.resolve()),
        'policy_url':url,'stages':{},'physical_execution_pass':None}
    def save():
        progress['updated_at']=time.time()
        temporary=out/'status.tmp';temporary.write_text(json.dumps(progress,indent=2)+'\n')
        temporary.replace(out/'status.json')
    def stage(name,command,environment=None):
        progress['stage']=name
        detail={'status':'running','started_at':time.time(),'log':name+'.log'}
        progress['stages'][name]=detail;save()
        with (out/(name+'.log')).open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=environment or env,
                stdout=stream,stderr=subprocess.STDOUT)
            detail['pid']=child.pid;save();code=child.wait()
        detail.update(status='passed' if code==0 else 'failed',returncode=code,
            finished_at=time.time());save()
        if code:raise RuntimeError(f'{name} exited with {code}; see {name}.log')
    def get_json(endpoint,payload=None):
        body=json.dumps(payload).encode() if payload is not None else None
        request=urllib.request.Request(url+endpoint,data=body,
            headers={'Content-Type':'application/json'} if body else {})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=180) as response:
            return json.load(response)
    save()
    try:
        # Record implementation identity without changing a simulation property.
        files={}
        for folder in (ROOT/'servers/lilo_vla',Path(__file__).parent):
            for path in folder.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'implementation_files.json').write_text(json.dumps(files,indent=2)+'\n')
        for required in ('ARX_NEURAL_PYTHON','CUROBO_PYTHON','ARX_NEURAL_GPU','ARX_PLANNER_GPU'):
            if not env.get(required):raise ValueError(f'Source scripts/curobo_arx/environment.sh: missing {required}')
        spec=json.loads((dataset/'dataset.json').read_text())
        if not any(e['split']=='validation' for e in spec['episodes']):
            raise ValueError('This pilot requires a held-out ARX trajectory')
        download=[lilo,'servers/lilo_vla/download_checkpoint.py',str(checkpoint),
            '--workers',str(args.download_workers),'--range-mib','8']
        for proxy in args.cdn_proxy:download+=['--cdn-proxy',proxy]
        receipt_path=checkpoint/'download_progress.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        reused=bool(receipt.get('complete') and receipt.get('revision')==
            'f54c56709004f6680d2ce8bc9e06d8cc96567ad8')
        if reused:
            metadata=json.loads((checkpoint/'.lilo-checkpoint-files.json').read_text())
            sizes={f['rfilename']:f['size'] for f in metadata['siblings']}
            reused=bool(sum(sizes[n] for n in receipt['verified'])==receipt['expected_bytes']
                and all((checkpoint/n).stat().st_size==sizes[n]
                    and (checkpoint/n).stat().st_mtime<=receipt_path.stat().st_mtime
                    for n in receipt['verified']))
        if reused:
            (out/'verified_checkpoint_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            progress['stages']['download']={'status':'reused_verified_checkpoint',
                'receipt':'verified_checkpoint_receipt.json'};save()
        else:
            for attempt in range(3):
                try:
                    stage(f'download_{attempt+1}',download);break
                except RuntimeError:
                    if attempt==2:raise
                    time.sleep(15)
        training_env=env|{'CUDA_VISIBLE_DEVICES':gpu,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        stage('adapt_action_head',[lilo,str(Path(__file__).with_name('finetune_head.py')),
            '--checkpoint',str(checkpoint),'--dataset',str(dataset),
            '--output',str(out/'adapter'),'--steps',str(args.steps)],training_env)
        # Bind only a free local port. Never replace another user's service.
        import socket
        with socket.socket() as probe:probe.bind(('127.0.0.1',port))
        service_env=env|{'LILO_PYTHON':lilo,'LILO_CHECKPOINT':str(checkpoint),
            'LILO_ADAPTER':str(out/'adapter'),'LILO_GPU':gpu,'LILO_PORT':str(port)}
        progress['stage']='start_policy';save()
        with (out/'policy_service.log').open('a') as stream:
            service=subprocess.Popen(['bash','scripts/run_lilo_vla_server.sh'],cwd=ROOT,
                env=service_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        progress['service_pid']=service.pid;save();deadline=time.monotonic()+900
        while True:
            if service.poll() is not None:raise RuntimeError('Policy service failed; see policy_service.log')
            try:
                health=get_json('/health');break
            except OSError:
                if time.monotonic()>deadline:
                    os.killpg(service.pid,signal.SIGTERM)
                    raise TimeoutError('Policy startup exceeded 15 minutes')
                time.sleep(2)
        if not health.get('checkpoint_verified'):
            raise ValueError('Actual policy must load the verified checkpoint')
        verify_health(health,gpu)
        (out/'policy_health.json').write_text(json.dumps(health,indent=2)+'\n')
        progress['stage']='actual_arx_inference_probe';save()
        fixture=ROOT/'servers/lilo_vla/tests/fixtures'
        state=json.loads((fixture/'arx_fixture.json').read_text())['observation']['state']
        prediction=get_json('/predict',{'image_png_base64':base64.b64encode(
            (fixture/'arx_wrist_pregrasp.png').read_bytes()).decode(),'state':state,
            'instruction':'pick up the bbq sauce bottle'})
        (out/'arx_inference_probe.json').write_text(json.dumps(prediction,indent=2)+'\n')
        import numpy as np
        actions=np.asarray(prediction['actions'])
        if actions.shape!=(8,7) or not np.isfinite(actions).all():
            raise ValueError('Real LiLo inference did not return finite 8x7 actions')
        stage('execute_arx',[env['ARX_NEURAL_PYTHON'],str(Path(__file__).with_name('run_bottle.py')),
            '--output',str(out/'execution'),'--mailbox',str(args.mailbox.resolve()),
            '--policy-url',url,'--max-steps',str(args.max_actions)])
        verification=json.loads((out/'execution/verification.json').read_text())
        progress['physical_execution_pass']=bool(verification['pass'])
        if not verification['pass']:raise RuntimeError('Actual ARX execution failed the original physical criteria')
        progress.update(status='completed',stage='finished',finished_at=time.time());save()
    except Exception:
        verification=out/'execution/verification.json'
        if verification.exists():
            progress['physical_execution_pass']=bool(json.loads(verification.read_text()).get('pass'))
        progress.update(status='failed',error=traceback.format_exc(),finished_at=time.time());save()
        raise


if __name__=='__main__':main()
