"""Shared local RPC lifecycle. Algorithms implement request(), not socket code."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from .layout import ROOT,ensure_run_path,now,write_json
from .registry import create_worker, module_records


def read_message(stream):
    line=stream.readline(1024*1024)
    if not line or len(line)>=1024*1024:raise ValueError('Invalid RPC message')
    return json.loads(line)


def serve(args,worker=None):
    started=time.perf_counter()
    args.root=ensure_run_path(args.root);args.root.mkdir(parents=True,exist_ok=True)
    worker=create_worker(args.kind,args) if worker is None else worker
    if module_records().get(args.kind, {}).get('role') != 'motion_planning':
        warm=worker.request(ensure_run_path(args.warmup_trial))
        if not warm.get('success') or not warm.get('target_found',True):raise RuntimeError('Module warmup failed: '+str(warm))
    memory = None
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        memory = {'allocated_bytes': torch.cuda.memory_allocated(),
                  'reserved_before_release_bytes': torch.cuda.memory_reserved()}
        # Release unused warmup buffers before the shared-device admission gate.
        # Live model tensors and captured CUDA graph allocations stay resident.
        torch.cuda.empty_cache()
        memory.update(reserved_after_release_bytes=torch.cuda.memory_reserved(),
                      device_free_bytes=torch.cuda.mem_get_info()[0])
    ready={'status':'ready','kind':args.kind,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'startup_seconds':time.perf_counter()-started,'model_load_count':1,'requests_completed':0,'warmup_passed':True,'at':now(),
        'cuda_memory': memory}
    args.socket=Path(args.socket);ensure_run_path(args.socket)
    args.socket.parent.mkdir(parents=True,exist_ok=True)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(args.socket));os.chmod(args.socket,0o600);server.listen(4)
        write_json(args.root/(args.kind+'_ready.json'),ready);print(ready,flush=True)
        try:
            while True:
                with server.accept()[0] as connection,connection.makefile('rwb') as stream:
                    try:
                        message=read_message(stream);command=message.pop('command','request')
                        if command=='health':response=ready
                        elif command=='shutdown':
                            stream.write(b'{"status":"stopped"}\n');stream.flush();break
                        elif command=='request':
                            directory=ensure_run_path(message.pop('directory'))
                            if (directory/'result.json').exists():raise ValueError('Trial already finalized')
                            result=worker.request(directory,**message)
                            response={'status':'ok','returncode':0 if result.get('success') else 2,'result':result}
                            ready['requests_completed']+=1;write_json(args.root/(args.kind+'_ready.json'),ready)
                        else:raise ValueError('Unknown RPC command')
                    except Exception:response={'status':'error','returncode':1,'error':traceback.format_exc()}
                    stream.write(json.dumps(response,default=lambda v:v.tolist()).encode()+b'\n');stream.flush()
        finally:args.socket.unlink(missing_ok=True)
    write_json(args.root/(args.kind+'_stopped.json'),ready)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',required=True)
    for name in ['root','socket','warmup-trial','descriptor']:parser.add_argument('--'+name,type=Path,required=True)
    serve(parser.parse_args())


if __name__=='__main__':main()
