"""One preloaded ManiSkill runtime per scene worker; unchanged episode functions."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import faulthandler
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

START=time.monotonic()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--socket',type=Path,required=True)
parser.add_argument('--owner-pid',type=int,required=True)
args=parser.parse_args()
args.root.mkdir(parents=True,exist_ok=False)
faulthandler.enable()
faulthandler.register(signal.SIGUSR1,all_threads=True)
faulthandler.dump_traceback_later(60,repeat=True)


def save(path,value):
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def stage(name,**values):
    record={'stage':name,'elapsed_seconds':time.monotonic()-START,'pid':os.getpid(),**values}
    save(args.root/'stage.json',record)
    print(json.dumps(record),flush=True)


sys.path.insert(0,'/home/zhaoyechong/projects/ARX-ManiSkill/modules/maniskill')
import arx_runtime_imports
arx_runtime_imports.install()
stage('loading_python_modules')
import graspcl_unseen_bench as bench
import graspcl_unseen_inputs as data
stage('loading_simulation_modules')
run,baseline=bench.core.batch.original_modules()
import scene_tools
from unittest.mock import patch
bench.lift.install_execution_policy(run,baseline)
stage('simulation_runtime_ready')
assert arx_runtime_imports.PROOF['omitted_import_count']==1
save(args.root/'import_scope_verification.json',arx_runtime_imports.PROOF)
ready={'status':'ready','kind':'maniskill_simulation','pid':os.getpid(),'owner_pid':args.owner_pid,
    'runtime_import_count':1,'library_load_count':{'numpy_scipy':1,'torch':1,'sapien':1},
    'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'startup_seconds':time.monotonic()-START,
    'requests_completed':0,'physical_grasp_functions_unchanged':True}
save(args.root/'ready.json',ready)
assert not args.socket.exists()
args.socket.parent.mkdir(parents=True,exist_ok=True)
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
    server.bind(str(args.socket));os.chmod(args.socket,0o600);server.listen(8)
    while True:
        with server.accept()[0] as connection:
            with connection.makefile('rwb') as stream:
                line=stream.readline(1024*1024)
                message=json.loads(line)
                command=message.get('command','health')
                if command=='health':
                    reply=ready
                elif command=='shutdown':
                    reply={'status':'stopped','pid':os.getpid(),'requests_completed':ready['requests_completed']}
                    stream.write(json.dumps(reply).encode()+b'\n');stream.flush()
                    break
                else:
                    os.kill(args.owner_pid,0)
                    started=time.monotonic()
                    try:
                        output=Path(message['output']).resolve()
                        if bench.CODE not in output.parents:raise RuntimeError('Output outside registered workspace')
                        stage('request_running',command=command,output=str(output))
                        log=Path(message['log'])
                        with log.open('x') as log_stream,redirect_stdout(log_stream),redirect_stderr(log_stream):
                            if command=='admit':
                                inputs=Path(message['inputs'])
                                plan=data.common.read(inputs/'preparation.json')
                                current=data.original.install_sampler(plan)
                                def visibility(*values,**kwargs):
                                    result=scene_tools.visibility(*values,**kwargs)
                                    result['maximum_distractor_occlusion_fraction']=.40
                                    result['admitted_view']=bool(result['admitted_view'] and result['occlusion_fraction']<=.40)
                                    return result
                                with patch.object(current,'visibility',visibility):
                                    current.admit(data.common.read(message['entry']),output,max_variants=1,start_variant=message['variant'])
                            elif command=='execute':
                                bench.execute(output,Path(message['entry']))
                            else:raise RuntimeError('Unknown simulator request')
                        reply={'status':'ok','returncode':0,'pid':os.getpid(),'runtime_import_count':1}
                    except BaseException:
                        reply={'status':'error','returncode':1,'error':traceback.format_exc(),'pid':os.getpid()}
                    reply['wall_seconds']=time.monotonic()-started
                    ready['requests_completed']+=1
                    save(args.root/'ready.json',ready)
                    save(args.root/'last_request.json',{'request':message,'response':reply})
                    stage('request_complete',command=command,status=reply['status'])
                stream.write(json.dumps(reply).encode()+b'\n');stream.flush()
faulthandler.cancel_dump_traceback_later()
args.socket.unlink()
save(args.root/'stopped.json',{'pid':os.getpid(),'requests_completed':ready['requests_completed']})
