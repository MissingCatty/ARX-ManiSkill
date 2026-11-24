"""Benchmark-owned persistent ZeroGrasp/cuRobo workers over local Unix sockets."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def read_message(stream):
    line = stream.readline(1024 * 1024)
    if not line or len(line) >= 1024 * 1024:
        raise RuntimeError('Missing or oversized resident request')
    return json.loads(line)


def rpc(path, message, timeout=1800):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        with connection.makefile('rwb') as stream:
            stream.write(json.dumps(message).encode() + b'\n')
            stream.flush()
            return read_message(stream)


class NeuralWorker:
    def __init__(self, root, warmup_trial):
        package = ROOT / 'arx_core/servers/zerograsp/vendor/zerograsp_mainline_minimal'
        sys.path[:0] = [str(package), str(ROOT / 'arx_core/shared'), str(ROOT / 'arx_core/servers/zerograsp')]
        import demo
        import torch
        from zerograsp.utils.config import parse_config
        self.demo = demo
        self.package = package
        self.config = parse_config(str(package / 'configs/demo_640x480.yaml'))
        self.config.update_octree = True
        self.config.checkpoint = str(package / 'checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt')
        self.config.img_path = str(warmup_trial / 'camera/color.png')
        self.config.depth_path = str(warmup_trial / 'camera/depth.png')
        self.config.mask_path = str(warmup_trial / 'camera/mask.png')
        self.config.output_dir = str(root / 'warmup_network')
        started = time.perf_counter()
        self.model = demo.BaseTrainer.load_from_checkpoint(self.config.checkpoint, config=self.config,
                                                         strict=False, map_location='cuda')
        self.model.to('cuda')
        self.model.eval()
        self.loads = 1
        self.load_seconds = time.perf_counter() - started
        self.current_config = None
        # Execute the exact frozen demo, replacing only its repeated model load.
        demo.BaseTrainer.load_from_checkpoint = lambda *args, **kwargs: self.model
        demo.parse_config = lambda: self.current_config
        warmup = root / 'neural_warmup'
        warmup.mkdir(parents=True, exist_ok=False)
        shutil.copytree(warmup_trial / 'camera', warmup / 'camera')
        started = time.perf_counter()
        self.request(warmup)
        torch.cuda.synchronize()
        self.warmup_seconds = time.perf_counter() - started

    def request(self, directory, **ignored):
        import numpy as np
        import torch
        from arx_shared.schemas import ZeroGraspInputRequest
        from coolas_zerograsp.grasp import prepare_zerograsp_input
        from run_mainline import parse_counts, summarize_top1
        directory = Path(directory)
        started = time.perf_counter()
        prepared = prepare_zerograsp_input(ZeroGraspInputRequest(camera_dir=str(directory / 'camera'),
            mask_path=str(directory / 'camera/mask.png'), output_dir=str(directory / 'zg_input'), label='maniskill_target'))
        save(directory / 'network_input.json', prepared.model_dump())
        config = deepcopy(self.config)
        for field, value in [('img_path', prepared.rgb_path), ('depth_path', prepared.depth_path),
                             ('mask_path', prepared.mask_path), ('camera_info_path', prepared.camera_info_path),
                             ('output_dir', str(directory / 'network/raw_outputs'))]:
            setattr(config, field, value)
        self.current_config = config
        output = directory / 'network'
        logs = output / 'logs'
        logs.mkdir(parents=True, exist_ok=False)
        with (logs / 'stdout.log').open('x') as stdout, (logs / 'stderr.log').open('x') as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.demo.main()
                torch.cuda.synchronize()
        files = sorted((output / 'raw_outputs').glob('*.raw_grasps.npy'))
        if not files:
            raise RuntimeError('Resident ZeroGrasp emitted no raw candidate files')
        rows = np.concatenate([np.load(f, allow_pickle=False) for f in files])
        np.save(directory / 'raw_grasps.npy', rows)
        save(directory / 'network_candidates.json', {'fresh_inference': True, 'count': len(rows),
            'source_files': list(map(str, files)), 'mask': 'ManiSkill target instance segmentation',
            'backend_mode': 'resident', 'model_load_count': self.loads})
        grasps = sorted((output / 'raw_outputs').glob('*.grasp.npy'))
        raw_counts, filtered = parse_counts((logs / 'stdout.log').read_text() + (logs / 'stderr.log').read_text())
        report = {'status': 'ok', 'exit_code': 0, 'runtime_sec': time.perf_counter() - started,
            'backend_mode': 'resident', 'pid': os.getpid(), 'model_load_count': self.loads,
            'raw_outputs_dir': str(output / 'raw_outputs'), 'grasp_files': list(map(str, grasps)),
            'raw_counts_by_object': raw_counts, 'post_collision_counts_by_object': filtered,
            'recommended_grasp_top1': summarize_top1(grasps)}
        save(output / 'run_report.json', report)
        save(output / 'reconstruction_summary.json', {
            'ply_files': [f.name for f in sorted((output / 'raw_outputs').glob('*.ply'))],
            'note': 'Each PLY is one reconstructed object-level point cloud exported by the ZeroGrasp mainline.'})
        if report['recommended_grasp_top1'] is not None:
            save(output / 'recommended_grasp_top1.json', report['recommended_grasp_top1'])
        return {'success': True, 'count': len(rows), 'wall_seconds': report['runtime_sec']}


def serve(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    started = time.perf_counter()
    if args.kind == 'zerograsp':
        worker = NeuralWorker(root, args.warmup_trial.resolve())
        details = {'model_load_count': worker.loads, 'load_seconds': worker.load_seconds,
                   'warmup_seconds': worker.warmup_seconds}
    else:
        sys.path.insert(0, str(CODE))
        from resident_planner import PlannerPool
        worker = PlannerPool(root / 'planner_warmup', args.warmup_trial.resolve())
        details = {'model_load_count': worker.loads, 'warmup': worker.warmup}
    ready = {'status': 'ready', 'kind': args.kind, 'pid': os.getpid(),
        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'socket': str(args.socket),
        'startup_seconds': time.perf_counter() - started, 'details': details,
        'ready_at': datetime.now().astimezone().isoformat(), 'requests_completed': 0}
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise RuntimeError('Existing resident socket must be investigated, not replaced')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(8)
        save(root / (args.kind + '_ready.json'), ready)
        print(json.dumps(ready), flush=True)
        while True:
            with server.accept()[0] as connection:
                with connection.makefile('rwb') as stream:
                    message = read_message(stream)
                    command = message.pop('command', 'request')
                    if command == 'health':
                        response = ready
                    elif command == 'shutdown':
                        stream.write(json.dumps({'status': 'stopped', 'pid': os.getpid()}).encode() + b'\n')
                        stream.flush()
                        break
                    elif command == 'request':
                        call_start = time.perf_counter()
                        try:
                            directory = Path(message.pop('directory')).resolve()
                            if ROOT not in directory.parents:
                                raise RuntimeError('Resident request directory outside workspace')
                            result = worker.request(directory, **message)
                            response = {'status': 'ok', 'returncode': 0 if result.get('success') else 2,
                                        'result': result}
                        except Exception:
                            response = {'status': 'error', 'returncode': 1, 'error': traceback.format_exc()}
                            print(response['error'], flush=True)
                        response.update(pid=os.getpid(), wall_seconds=time.perf_counter() - call_start)
                        ready['requests_completed'] += 1
                        save(root / (args.kind + '_ready.json'), ready)
                    else:
                        response = {'status': 'error', 'returncode': 1, 'error': 'Unknown command'}
                    stream.write(json.dumps(response).encode() + b'\n')
                    stream.flush()
    args.socket.unlink()
    save(root / (args.kind + '_stopped.json'), {'pid': os.getpid(), 'requests_completed': ready['requests_completed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['zerograsp', 'curobo'], required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--warmup-trial', type=Path, required=True)
    serve(parser.parse_args())
