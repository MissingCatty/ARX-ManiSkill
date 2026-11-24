"""Separate bounded planner mailbox; never accesses production worker queues."""
import argparse
from pathlib import Path
import os
import time
import traceback
from scripts.curobo_arx.data_factory.common import read, write
from .teacher import Teacher


def serve(directory, registration):
    directory = Path(directory)
    for name in ('requests', 'responses'):
        (directory/name).mkdir(parents=True, exist_ok=True)
    teacher = Teacher(registration, directory)
    write(directory/'ready.json', {'pid': os.getpid(), 'kind': 'success_seed_planner',
          'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'no_neural_inference': True})
    while not (directory/'STOP').exists():
        jobs = sorted((directory/'requests').glob('*.json'))
        if not jobs:
            time.sleep(.02); continue
        path = jobs[0]; request = read(path); running = path.with_suffix('.running'); path.rename(running)
        started = time.monotonic()
        try:
            response = {'id': request['id'], 'ok': True, 'result': teacher.handle(request), 'seconds': time.monotonic()-started}
        except BaseException:
            response = {'id': request['id'], 'ok': False, 'error': traceback.format_exc(), 'seconds': time.monotonic()-started}
        write(directory/'responses'/(request['id']+'.json'), response)
        running.rename(running.with_suffix('.done'))
        if not response['ok']:
            write(directory/'FAULT.json', response); break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--registration', required=True)
    args = parser.parse_args(); serve(args.directory, read(args.registration))
