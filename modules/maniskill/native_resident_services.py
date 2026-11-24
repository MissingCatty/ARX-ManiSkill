"""Resident ZeroGrasp, GraspCL and fast ARX cuRobo services for native-post benches."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

import resident_backend as service


class GraspCLWorker:
    def __init__(self, root, bank):
        from graspcl_pipeline import seed, load_model, read
        seed()
        self.root = root
        self.ids = read(bank / 'bank_complete.json')['source_ids'] if (bank / 'bank_complete.json').exists() else []
        if not self.ids:
            raise RuntimeError('Frozen feedback provenance is missing')
        started = time.perf_counter()
        runtime = root / 'graspcl_model'
        shutil.copytree(bank / 'model', runtime / 'model')
        self.module = load_model(runtime)
        scorer = self.module.grasp_scoring_module_cl
        self.memory = (scorer.X_on, scorer.y_on, scorer.w_on, scorer.tree_on,
                       list(self.module.grasp_recall_module_object._db),
                       list(self.module.grasp_recall_module_patch._db))
        self.load_seconds = time.perf_counter()-started

    def warmup(self, trial, policy):
        directory = self.root / 'graspcl_warmup' / trial.name
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copytree(trial / 'camera', directory / 'camera')
        shutil.copytree(trial / 'network/raw_outputs', directory / 'network/raw_outputs')
        for name in ['raw_grasps.npy', 'state.json']:
            shutil.copy2(trial / name, directory / name)
        result = self.request(directory, policy)
        service.save(self.root / 'graspcl_warmup.json', result)
        return result

    def restore(self):
        scorer = self.module.grasp_scoring_module_cl
        scorer.X_on, scorer.y_on, scorer.w_on, scorer.tree_on = self.memory[:4]
        self.module.grasp_recall_module_object._db = list(self.memory[4])
        self.module.grasp_recall_module_patch._db = list(self.memory[5])

    def request(self, directory, policy):
        from graspcl_pipeline import seed, save
        from graspcl_batch import exclude_self
        from postprocess_pipeline import infer, write_inference
        seed()
        self.restore()
        directory = Path(directory)
        started = time.perf_counter()
        exclusion = exclude_self(self.module, self.ids, directory.name)
        save(directory / 'self_case_exclusion.json', exclusion)
        try:
            result = infer(self.module, directory, policy)
            for recalled in result['recalled_before_filter']:
                if recalled['metadata'].get('source_trial_id') == directory.name:
                    raise RuntimeError('Self-case demonstration leaked')
            output = directory / 'graspcl'
            output.mkdir(exist_ok=False)
            (output / 'zg_outputs').mkdir()
            write_inference(result, directory, output)
            return {'success': True, 'counts': result['counts'],
                'wall_seconds': time.perf_counter()-started, 'encoder_load_count': 1,
                'self_case_exclusion': exclusion, 'bank_updates': 0}
        finally:
            self.restore()


def serve_graspcl(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    worker = GraspCLWorker(root, args.bank.resolve())
    policy = json.loads(args.policy_registration.read_text())['candidate_policy']
    warmup = worker.warmup(args.warmup_trial.resolve(), policy)
    ready = {'kind': 'graspcl', 'status': 'ready', 'pid': os.getpid(), 'device': 'cpu',
        'encoder_load_count': 1, 'bank_load_count': 1, 'bank_updates': 0,
        'startup_seconds': time.perf_counter()-started, 'model_load_seconds': worker.load_seconds,
        'warmup_passed': warmup['success'], 'warmup_seconds': warmup['wall_seconds'],
        'requests_completed': 0, 'ready_at': datetime.now().astimezone().isoformat()}
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise RuntimeError('Existing GraspCL socket must not be replaced')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(8)
        service.save(root / 'graspcl_ready.json', ready)
        print(json.dumps(ready), flush=True)
        while True:
            with server.accept()[0] as connection:
                with connection.makefile('rwb') as stream:
                    message = service.read_message(stream)
                    command = message.pop('command', 'request')
                    if command == 'health':
                        reply = ready
                    elif command == 'shutdown':
                        stream.write(json.dumps({'status': 'stopped', 'pid': os.getpid()}).encode()+b'\n')
                        stream.flush()
                        break
                    else:
                        started = time.perf_counter()
                        try:
                            directory = Path(message['directory']).resolve()
                            if service.ROOT not in directory.parents:
                                raise RuntimeError('GraspCL input outside workspace')
                            result = worker.request(directory, message['policy'])
                            reply = {'status': 'ok', 'returncode': 0, 'result': result}
                        except Exception:
                            reply = {'status': 'error', 'returncode': 1, 'error': traceback.format_exc()}
                        reply.update(pid=os.getpid(), wall_seconds=time.perf_counter()-started)
                        ready['requests_completed'] += 1
                        service.save(root / 'graspcl_ready.json', ready)
                    stream.write(json.dumps(reply).encode()+b'\n')
                    stream.flush()
    args.socket.unlink()
    service.save(root / 'graspcl_stopped.json', {'pid': os.getpid(), 'requests_completed': ready['requests_completed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['zerograsp', 'graspcl', 'curobo'], required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--warmup-trial', type=Path)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--policy-registration', type=Path)
    args = parser.parse_args()
    if args.kind == 'graspcl':
        serve_graspcl(args)
    else:
        if args.kind == 'zerograsp':
            from postprocess_backend import NativePostNeural
            service.NeuralWorker = NativePostNeural
        else:
            import resident_planner
            from postprocess_planner import PostprocessPlanner
            from fast_resident_planner import FastPlannerPool
            resident_planner.GraspCLPlanner = PostprocessPlanner
            resident_planner.PlannerPool = FastPlannerPool
        service.serve(args)
