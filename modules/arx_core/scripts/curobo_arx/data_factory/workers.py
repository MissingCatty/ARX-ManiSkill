"""Bounded asynchronous file RPC to resident inference/planning processes."""
import argparse
import os
from pathlib import Path
import time
import traceback
import uuid
from .common import read, write


class WorkerClient:
    def __init__(self, directory, capacity=32):
        self.directory, self.capacity = Path(directory), capacity
        for name in ('requests','responses'):
            (self.directory/name).mkdir(parents=True,exist_ok=True)

    def submit(self, action, **payload):
        if len(list((self.directory/'requests').glob('*.json'))) >= self.capacity:
            raise RuntimeError('Resident worker bounded request queue full')
        ident = uuid.uuid4().hex
        write(self.directory/'requests'/f'{time.time_ns()}_{ident}.json',{'id':ident,'action':action,**payload})
        return ident

    def poll(self, ident):
        path = self.directory/'responses'/(ident+'.json')
        if not path.is_file():
            return None
        value = read(path)
        if value['id'] != ident:
            raise RuntimeError('Worker response identity mismatch')
        return value


def serve(kind, directory, registration):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    client = WorkerClient(directory)
    if kind == 'planner':
        from .teacher import Teacher
        if registration.get('authored_finger_proposal_filter'):
            from scripts.curobo_arx.graspness_teacher.finger_overlap import guarded_teacher
            Teacher=guarded_teacher(Teacher)
        if registration.get('local_goal_refinement'):
            if not registration.get('authored_finger_proposal_filter'):
                raise ValueError('Local-goal refinement requires registered actual-finger proposal checks')
            from scripts.curobo_arx.graspness_teacher.goal_refinement import refining_teacher
            Teacher=refining_teacher(Teacher)
        if registration.get('load_aware_lift'):
            from scripts.curobo_arx.graspness_teacher.load_lift import lift_teacher
            Teacher=lift_teacher(Teacher)
        backend = Teacher(registration,directory)
    elif kind == 'sdg':
        from scripts.curobo_arx.sdg_stage1.sampler import SDGSampler
        backend = SDGSampler(registration)
    elif kind == 'anygrasp':
        from scripts.curobo_arx.anygrasp_teacher.backend import AnyGrasp
        backend = AnyGrasp(registration,directory)
    elif kind == 'graspness':
        from scripts.curobo_arx.graspness_teacher.backend import Graspness
        backend = Graspness(registration,directory)
    else:
        from .zerograsp_backend import ZeroGrasp
        backend = ZeroGrasp(registration)
    write(directory/'ready.json',{'pid':os.getpid(),'kind':kind,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
            'resident':True,'model_initialization':('Official CPU antipodal sampler; no neural model'
                if kind=='sdg' else 'Graspness checkpoint strictly loaded; first actual frame performs CUDA warmup'
                if kind=='graspness' else 'lazy first valid observation, followed by fixed-shape warmup before motion')})
    while not (directory/'STOP').exists():
        jobs = sorted((directory/'requests').glob('*.json'))
        if not jobs:
            time.sleep(.02)
            continue
        path = jobs[0]
        request = read(path)
        running = path.with_suffix('.running')
        path.rename(running)
        started = time.monotonic()
        try:
            result = backend.handle(request)
            response = {'id':request['id'],'ok':True,'result':result,'seconds':time.monotonic()-started}
        except BaseException:
            response = {'id':request['id'],'ok':False,'error':traceback.format_exc(),
                        'seconds':time.monotonic()-started,'fault':True}
        write(directory/'responses'/(request['id']+'.json'),response)
        running.rename(running.with_suffix('.done'))
        if not response['ok']:
            write(directory/'FAULT.json',response)
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind',choices=['planner','zerograsp','sdg','anygrasp','graspness'])
    p.add_argument('--directory',required=True)
    p.add_argument('--registration',required=True)
    args = p.parse_args()
    serve(args.kind,args.directory,read(args.registration))
