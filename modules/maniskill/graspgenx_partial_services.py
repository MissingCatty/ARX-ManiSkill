"""Batch-owned GraspGenX and unchanged-geometry ARX planner services."""
import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import graspgenx_partial_geometry as geometry
from graspgenx_partial_geometry import save

CODE = Path(__file__).resolve().parent


class GraspGenXWorker:
    def __init__(self, root, warmup_trial, descriptor):
        import torch
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.samplers.planner import run_planner_on_object
        self.root = root;self.loads = 1;self.calls = 0
        self.torch = torch;self.run_planner = run_planner_on_object
        self.descriptor = json.loads(Path(descriptor).read_text())
        checkpoints = CODE.parent / 'research_dependencies/graspgenx_checkpoints/release'
        cfg = load_model_cfg(str(checkpoints/'gen'), str(checkpoints/'dis'))
        cfg.diffusion.ptv3vanilla.enable_flash = False
        cfg.discriminator.ptv3vanilla.enable_flash = False
        started = time.perf_counter()
        torch.manual_seed(20261120)
        self.sampler = GraspGenXSampler.from_sweep_volume(cfg, self.descriptor['parameters'])
        self.load_seconds = time.perf_counter() - started
        root.mkdir(parents=True, exist_ok=True)
        _, cloud = geometry.visible_points(warmup_trial)
        cloud = self.sample_points(cloud)
        started = time.perf_counter()
        self.predict(cloud, root/'graspgenx_warmup', seed=20261120)
        torch.cuda.synchronize()
        self.warmup_seconds = time.perf_counter() - started
        save(root/'graspgenx_model.json', {'model': 'GraspGenX', 'model_load_count': 1,
            'planner': 'graspmoe', 'moe_obb_density': 'dense', 'diffusion_candidates': 512,
            'weights_modified': False, 'arx_conditioning': self.descriptor,
            'flash_attention': False, 'pytorch': torch.__version__, 'startup_warmup_is_training_lemon': True})

    @staticmethod
    def sample_points(cloud):
        _, indices = np.unique(np.floor(cloud/.002).astype(np.int64), axis=0, return_index=True)
        cloud = cloud[np.sort(indices)]
        if len(cloud) > 3500:
            cloud = cloud[np.linspace(0, len(cloud)-1, 3500, dtype=int)]
        assert len(cloud) >= 20 and np.isfinite(cloud).all()
        return cloud.astype(np.float32)

    def predict(self, cloud, output, seed):
        output = Path(output);output.mkdir(parents=True, exist_ok=False)
        self.torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        started = time.perf_counter()
        grasps, scores, tags, obb = self.run_planner(cloud, self.sampler,
            planner='graspmoe', grasp_threshold=-1., num_grasps=512,
            topk_num_grasps=-1, moe_obb_density='dense')
        self.torch.cuda.synchronize()
        assert grasps.shape == (len(scores),4,4) and len(tags) == len(scores)
        assert np.isfinite(grasps).all() and np.isfinite(scores).all()
        np.save(output/'grasps_generator_world.npy', grasps, allow_pickle=False)
        np.save(output/'scores.npy', scores, allow_pickle=False)
        np.save(output/'model_input_points_world.npy', cloud, allow_pickle=False)
        save(output/'inference.json', {'model': 'GraspGenX', 'branch_tags': tags,
             'input': 'GT visible partial points, no completion or full object mesh',
             'model_load_count': self.loads, 'num_candidates': len(scores),
             'num_diffusion': tags.count('diff'), 'num_native_obb': tags.count('obb'),
             'moe_obb_density': 'dense', 'topk_cap': None, 'obb': obb,
             'point_count': len(cloud), 'seed': seed,
             'world_frame': True, 'unit': 'meter', 'wall_seconds': time.perf_counter()-started})
        return {'success': True, 'count': len(scores), 'wall_seconds': time.perf_counter()-started}

    def request(self, directory, **ignored):
        directory = Path(directory)
        cloud = np.load(directory/'gt_partial_points_world.npy', allow_pickle=False)
        self.calls += 1
        return self.predict(self.sample_points(cloud), directory/'graspgenx_raw', 20261120+self.calls)


def install_planner():
    # Import the frozen physical-table policy without starting its benchmark.
    spec = importlib.util.spec_from_file_location('ggx_physical_policy', CODE/'tmp/physical_table_pair_20261118_01.py')
    policy = importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    policy.install_collision_policy()
    import resident_planner as resident
    import table_truncation_model as truncation
    from fast_resident_planner import FastPlannerPool
    from checked_native_services import checked_pool
    from short_lift_planner import ShortLiftPlanner

    class Selector(ShortLiftPlanner):
        def select_tcp43(self, state, rows, camera):
            result = super().select_tcp43(state, rows, camera)
            result['candidate_protocol'] = geometry.PROTOCOL
            # Any successful endpoint existed if a candidate advanced beyond qg IK.
            rejected = result['rejected']
            result['grasp_endpoint_reachable'] = bool(result.get('success') or
                any(rejected[k] for k in ['pre_or_lift_ik','path','optimizer']))
            return result

    base = truncation.make_pool(checked_pool(FastPlannerPool))

    class Pool(base):
        def refine(self, directory, mode):
            assert mode == 'fixed_orientation'
            started = time.perf_counter();directory = Path(directory)
            state = json.loads((directory/'state.json').read_text())
            context = self.bind(directory/'refinement_model', state)
            shapes = truncation.templates(context, state)
            assert len(shapes) == 1 and shapes[0][2] == 0.
            _, points = geometry.visible_points(directory)
            path = directory/'graspcl/candidate_pool.json'
            pool = json.loads(path.read_text());save(path.with_name('unrefined_candidate_pool.json'), pool)
            kept, rejected = [], []
            original_count = len(pool['candidates'])
            for index, source in enumerate(pool['candidates']):
                candidate = deepcopy(source)
                is_ggx = candidate['metadata']['origin'] == 'graspgenx'
                if is_ggx:
                    # Adapter-only argument to the old truncation function. The
                    # value is a physical contact-zone budget, never a network depth.
                    candidate['metadata']['source_network_depth_m'] = candidate['metadata']['retraction_budget_m']
                item, receipt = truncation.truncate(candidate, points,
                    state['scene_cuboids']['table'], shapes, .043, .088)
                if item is None:
                    rejected.append({'index': index, **receipt});continue
                if is_ggx:
                    meta = item['metadata']
                    for key in ['source_network_depth_m','remaining_source_insertion_m','original_source_insertion_m']:
                        meta.pop(key, None)
                    meta['retraction_budget_source'] = 'Measured ARX distal pad Z extent; not a network depth prediction'
                item['metadata']['source_pool_index'] = index
                item['metadata']['topdown_angle_deg'] = geometry.angle_from_down(item['pose'])
                assert item['metadata']['topdown_angle_deg'] <= geometry.TOPDOWN_DEGREES + 1.e-5
                kept.append(item)
            pool['candidates'] = kept
            pool['refinement'] = {'input_candidates': original_count,
                'retained': len(kept), 'rejected': rejected, 'grid_m': .002,
                'nominal_closed_precheck': False, 'table_closed_margin_m': 0.,
                'original_geometry_preserved': True, 'orientation_changes': 0,
                'protocol': geometry.PROTOCOL, 'wall_seconds': time.perf_counter()-started}
            save(path, pool);save(directory/'truncation.json', pool['refinement'])
            return {'success': True, **pool['refinement']}

    resident.GraspCLPlanner = Selector
    resident.PlannerPool = Pool


def serve_graspgenx(args):
    import resident_backend as transport
    started = time.perf_counter();root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker = GraspGenXWorker(root, args.warmup_trial, args.descriptor)
    ready = {'status': 'ready', 'kind': 'graspgenx', 'pid': os.getpid(),
        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'socket': str(args.socket),
        'details': {'model_load_count': worker.loads, 'load_seconds': worker.load_seconds,
                    'warmup_seconds': worker.warmup_seconds},
        'startup_seconds': time.perf_counter()-started,
        'ready_at': datetime.now().astimezone().isoformat(), 'requests_completed': 0,
        'model': 'GraspGenX', 'warmup_passed': True}
    assert not args.socket.exists()
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket));os.chmod(args.socket,0o600);server.listen(8)
        save(root/'graspgenx_ready.json',ready)
        print(json.dumps(ready),flush=True)
        while True:
            with server.accept()[0] as connection:
                with connection.makefile('rwb') as stream:
                    message = transport.read_message(stream)
                    command = message.pop('command','request')
                    if command == 'health':response = ready
                    elif command == 'shutdown':
                        stream.write(json.dumps({'status':'stopped','pid':os.getpid()}).encode()+b'\n')
                        stream.flush();break
                    else:
                        call_start = time.perf_counter()
                        try:
                            directory = Path(message.pop('directory')).resolve()
                            assert CODE in directory.parents and not (directory/'result.json').exists()
                            result = worker.request(directory,**message)
                            response = {'status':'ok','returncode':0,'result':result}
                        except Exception:
                            response = {'status':'error','returncode':1,'error':traceback.format_exc()}
                            print(response['error'],flush=True)
                        response.update(pid=os.getpid(),wall_seconds=time.perf_counter()-call_start)
                        ready['requests_completed'] += 1;save(root/'graspgenx_ready.json',ready)
                    stream.write(json.dumps(response).encode()+b'\n');stream.flush()
    args.socket.unlink()
    save(root/'graspgenx_stopped.json',{'pid':os.getpid(),'requests_completed':ready['requests_completed']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['graspgenx','zerograsp','curobo'], required=True)
    for name in ['root','socket','warmup-trial','descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    import resident_backend as service
    if args.kind == 'graspgenx':
        serve_graspgenx(args);return
    elif args.kind == 'zerograsp':
        from postprocess_backend import NativePostNeural
        service.NeuralWorker = NativePostNeural
    else:install_planner()
    service.serve(args)


if __name__ == '__main__':main()
