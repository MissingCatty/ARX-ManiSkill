"""Small persistent pipe interface to the isolated EAIK environment."""
import json
import os
from pathlib import Path
import select
import subprocess
import time

from arx_project.assets import resource_path
from arx_project.layout import ROOT, sha256


class GeometricClient:
    def __init__(self, directory, timeout=60.):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.requests = 0
        self.log = (self.directory/'cpu_worker.log').open('w')
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
            'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        self.process = subprocess.Popen([str(ROOT/'environments/geometric_ik/bin/python'), '-m',
            'modules.curobo.indexed_planning.geometric.worker'], cwd=ROOT, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        try:
            self.ready = self.read()
            expected = sha256(resource_path('arx_robot')/'robot.urdf')
            if (not self.ready.get('ready') or self.ready.get('protocol') != 1
                    or self.ready.get('urdf_sha256') != expected):
                raise RuntimeError('Geometric worker has the wrong protocol or robot model')
        except BaseException:
            self.close()
            raise

    def read(self):
        readable, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not readable:
            raise TimeoutError('Geometric worker response timed out; do not retry this planning request')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Geometric worker exited without a complete response')
        return json.loads(line)

    def solve(self, targets, start):
        began = time.perf_counter()
        if self.process.poll() is not None:
            raise RuntimeError('Geometric worker exited; do not silently replace it')
        self.requests += 1
        message = {'id': self.requests, 'targets': targets, 'start': start}
        self.process.stdin.write(json.dumps(message, allow_nan=False)+'\n')
        self.process.stdin.flush()
        result = self.read()
        if result.get('id') != self.requests or len(result.get('answers', [])) != len(targets):
            raise RuntimeError('Geometric worker returned a mismatched target batch')
        result['roundtrip_seconds'] = time.perf_counter()-began
        return result

    def close(self):
        try:
            if self.process.stdin is not None and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()
            self.log.close()


class GeometricSequence:
    def __init__(self, candidates, context, state, client, capacity=16):
        if not 1 <= capacity <= 32: raise ValueError('Unsupported geometric prefetch capacity')
        self.candidates, self.context, self.state, self.client = candidates, context, state, client
        self.capacity, self.cursor = capacity, 0
        self.cache, self.batches, self.events = {}, [], []

    @staticmethod
    def target(candidate):
        import numpy as np
        grasp = np.asarray(candidate['pose'], dtype=float).copy()
        grasp[:3, 3] -= .043*grasp[:3, 2]
        return grasp

    def next(self, grasp, start):
        import numpy as np
        index = self.cursor
        if index >= len(self.candidates): raise RuntimeError('Geometric sequence exceeded original candidates')
        if not np.allclose(grasp, self.target(self.candidates[index]), atol=1e-10, rtol=0):
            raise RuntimeError('Geometric query changed the original endpoint or order')
        if index not in self.cache:
            stop = min(len(self.candidates), index+self.capacity)
            targets = [(self.context.invbase@self.target(self.candidates[i])).tolist()
                       for i in range(index, stop)]
            result = self.client.solve(targets, start.detach().cpu().numpy().tolist())
            self.cache = dict(zip(range(index, stop), result['answers']))
            self.batches.append({'start_index': index, 'stop_index_exclusive': stop,
                'targets': stop-index, 'worker_seconds': result['worker_seconds'],
                'roundtrip_seconds': result['roundtrip_seconds'],
                'EAIK_seconds': sum(a['native_solver_seconds'] for a in result['answers'])})
        self.cursor += 1
        return self.cache.pop(index)

    def summary(self):
        return {'capacity': self.capacity, 'consumed_candidates': self.cursor,
            'computed_candidates': sum(b['targets'] for b in self.batches),
            'prefetched_unconsumed': len(self.cache), 'batches': self.batches, 'events': self.events,
            'candidate_order_and_world_targets_unchanged': True,
            'empty_or_colliding_geometric_output_uses_original_IK': True}
