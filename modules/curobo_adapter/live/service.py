"""Owned resident planner with no request retries."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

from paths import BASE, MS, ARX, read, save, guard
from resident_backend import read_message




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--kind',choices=['curobo'],required=True)
    parser.add_argument('--socket',type=Path,required=True)
    args = parser.parse_args()
    out = args.campaign.resolve()
    root = out/'resident'
    reg = read(out/'registration.json')
    guard(reg)
    began = time.monotonic()
    from planning import Pool
    worker = Pool(root/'planner',reg)
    details = {'model_load_count':worker.loads,'runtime_robot_rebuilds':0,'version':'0.9.0'}
    ready = {'status':'ready','kind':args.kind,'pid':os.getpid(),'details':details,
             'ready_at':datetime.now().astimezone().isoformat(),'startup_seconds':time.monotonic()-began,
             'requests_completed':0,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    args.socket.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if args.socket.exists():
        raise RuntimeError('Resident socket already exists; no replacement')
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(args.socket));os.chmod(args.socket,0o600);server.listen(4)
            save(root/(args.kind+'_ready.json'),ready)
            while True:
                with server.accept()[0] as connection:
                    with connection.makefile('rwb') as stream:
                        message = read_message(stream)
                        command = message.pop('command','request')
                        if command=='shutdown':
                            stream.write(b'{"status":"stopped"}\n');stream.flush();break
                        if command=='health':
                            answer = ready
                        elif command=='request':
                            identity = ready['requests_completed']+1
                            if identity>reg['maximum_resident_requests']:
                                raise RuntimeError('Registered resident request budget exhausted')
                            prefix = root/'requests'/args.kind/f'{identity:04d}'
                            save(prefix.with_name(prefix.name+'_request.json'),message)
                            started = time.monotonic()
                            try:
                                if (out/'STOP').exists():
                                    raise RuntimeError('Campaign stopped before resident request')
                                guard(reg)
                                result = worker.request(message)
                                answer = {'status':'ok','result':result}
                            except Exception as error:
                                answer = {'status':'error','error':str(error),'traceback':traceback.format_exc()}
                            answer.update(pid=os.getpid(),request_id=identity,wall_seconds=time.monotonic()-started)
                            save(prefix.with_name(prefix.name+'_response.json'),answer)
                            ready['requests_completed'] = identity
                            save(root/(args.kind+'_ready.json'),ready)
                        else:
                            answer = {'status':'error','error':'Unknown resident command'}
                        stream.write((__import__('json').dumps(answer)+'\n').encode());stream.flush()
    finally:
        if hasattr(worker,'close'):
            worker.close()
        if args.socket.exists():
            args.socket.unlink()
        save(root/(args.kind+'_stopped.json'),{'pid':os.getpid(),'requests_completed':ready['requests_completed']})


if __name__=='__main__':
    main()
