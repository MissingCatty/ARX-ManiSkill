"""Persistent CPU worker; its environment never imports CUDA or a simulator."""
import json
import sys
import time
import traceback

from .kinematics import GeometricSolver


def serve():
    solver = GeometricSolver('right')
    print(json.dumps({'ready': True, 'protocol': 1, 'family': solver.family,
        'urdf_sha256': solver.chain.urdf_sha256, 'axis_signs': solver.axis_signs.tolist()}), flush=True)
    for line in sys.stdin:
        began = time.perf_counter()
        message = json.loads(line)
        targets = message['targets']
        if not isinstance(targets, list) or not 1 <= len(targets) <= 32:
            raise ValueError('Geometric worker accepts between one and 32 original targets')
        answers = [solver.solve(target, message['start']) for target in targets]
        print(json.dumps({'id': message['id'], 'answers': answers,
            'worker_seconds': time.perf_counter()-began}, allow_nan=False), flush=True)


if __name__ == '__main__':
    try:
        serve()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
