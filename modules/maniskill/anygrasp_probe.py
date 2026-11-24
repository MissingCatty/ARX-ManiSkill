"""Official licensed AnyGrasp on an archived ManiSkill ARX RGB-D frame."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--license', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--backend-lock', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    source = args.input_trial.resolve()
    out = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    license_dir = args.license.resolve()
    sdk = ROOT.parent/'research_dependencies/anygrasp_sdk_b8eaafc9'
    binary = sdk/'grasp_detection/gsnet_versions/gsnet.cpython-311-x86_64-linux-gnu.so'
    out.mkdir(parents=True, exist_ok=False)
    (out/'camera').mkdir()
    inputs = [source/'camera'/n for n in ('color.png', 'depth.png', 'mask.png', 'params.json')]
    inputs += [source/'state.json']
    registration = {
        'created_at': datetime.now().astimezone().isoformat(),
        'scope': 'One new official AnyGrasp inference on the exact saved pudding-box wrist RGB-D. No robot execution and no new camera/observation policy.',
        'model': 'Official AnyGrasp', 'sdk_revision': 'b8eaafc9eca7babd5208e7a5ade3c561060be4c5',
        'driver_sha256': sha(Path(__file__)),
        'sdk_binary': str(binary), 'sdk_binary_sha256': sha(binary),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'license_directory': str(license_dir),
        'license_file_hashes': {p.name: sha(p) for p in license_dir.iterdir() if p.is_file()},
        'input_trial': str(source), 'input_file_hashes': {str(p): sha(p) for p in inputs},
        'gpu_uuid': args.gpu, 'shared_backend_lock': str(args.backend_lock.resolve()),
        'parameters': {'max_gripper_width': .088, 'gripper_height': .03,
                       'dense_grasp': False, 'collision_detection': True,
                       'region_steering': 'Original target mask, elementwise on input points',
                       'approach_steering': None, 'approach_thresh': float(np.pi)},
        'point_cloud': 'Official pinhole RGB-D projection in optical camera frame; meters; 0 < Z < 1 meter as SDK demo',
        'ranking': 'Official SDK collision/width-filtered candidates, then official NMS and decreasing network score',
        'inference_budget': 1, 'physical_attempt_budget': 0,
        'probe_script_sha256': sha(Path(__file__)),
    }
    save(out/'registration.json', registration)
    for f in inputs:
        destination = out/'camera'/f.name if f.parent.name == 'camera' else out/f.name
        shutil.copy2(f, destination)
    params = json.loads((source/'camera/params.json').read_text())
    intr = params['intrinsics']
    depth = np.asarray(Image.open(source/'camera/depth.png'), dtype=np.float32)*params['depth_scale']
    target = np.asarray(Image.open(source/'camera/mask.png')) > 0
    v, u = np.indices(depth.shape)
    xyz = np.stack([(u-intr['ppx'])*depth/intr['fx'],
                    (v-intr['ppy'])*depth/intr['fy'], depth], axis=-1)
    valid = (depth > 0) & (depth < 1.)
    points = np.ascontiguousarray(xyz[valid], dtype=np.float32)
    region = np.ascontiguousarray(target[valid], dtype=bool)
    np.save(out/'anygrasp_points.npy', points)
    np.save(out/'anygrasp_region.npy', region)
    if region.sum() < 50:
        raise RuntimeError('Insufficient valid target pixels')
    result = {'model': 'Official AnyGrasp', 'success': False, 'inference_executed': False,
              'physical_attempt': False, 'input_points': len(points), 'target_points': int(region.sum())}
    request = {'state': 'queued', 'queued_at': datetime.now().astimezone().isoformat()}
    save(out/'shared_backend_request.json', request)
    try:
        with args.backend_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            request.update(state='running', started_at=datetime.now().astimezone().isoformat())
            save(out/'shared_backend_request.json', request)
            import torch
            import MinkowskiEngine
            from pointnet2 import pointnet2_utils
            runtime = out/'runtime'
            runtime.mkdir()
            (runtime/'license').symlink_to(license_dir, target_is_directory=True)
            local_binary = runtime/binary.name
            shutil.copy2(binary, local_binary)
            if sha(local_binary) != sha(binary):
                raise RuntimeError('Isolated SDK binary copy differs from official source')
            # The SDK checks license/ next to its loaded binary, not just CWD.
            sys.path.insert(0, str(runtime))
            import gsnet
            os.chdir(runtime)
            from argparse import Namespace
            result['license_passed'] = bool(gsnet.check_license(str(license_dir)))
            if not result['license_passed']:
                raise RuntimeError('Official SDK rejected the supplied license')
            started = time.monotonic()
            detector = gsnet.create_detector(Namespace(checkpoint_path=str(checkpoint),
                max_gripper_width=.088, gripper_height=.03))
            if detector is None:
                raise RuntimeError('Official AnyGrasp detector failed to initialize')
            result['model_load_seconds'] = time.monotonic()-started
            started = time.monotonic()
            result['inference_executed'] = True
            with torch.no_grad():
                group = detector.get_grasp(points, {'dense_grasp': False,
                    'collision_detection': True, 'region_steering': region,
                    'approach_steering': None, 'approach_thresh': float(np.pi)})
            result['inference_seconds'] = time.monotonic()-started
            if group is None or len(group) == 0:
                rows = np.zeros((0, 17))
                result.update(reason='No candidate survived official AnyGrasp filtering', candidate_count=0)
            else:
                np.save(out/'anygrasp_before_nms.npy', group.grasp_group_array)
                result['before_nms_count'] = len(group)
                group = group.nms().sort_by_score()
                rows = np.asarray(group.grasp_group_array, dtype=np.float64)
                result.update(success=True, reason='Official AnyGrasp inference completed', candidate_count=len(rows),
                              top10_scores=rows[:10, 0].tolist(), top10_width_mm=(rows[:10, 1]*1000).tolist())
            np.save(out/'raw_grasps.npy', rows)
            save(out/'network_candidates.json', {'model_name': 'AnyGrasp', 'fresh_inference': True,
                'fresh_rgbd': False, 'count': len(rows), 'source': 'Official licensed AnyGrasp SDK',
                'mask': 'Same saved simulator target instance mask as ZeroGrasp comparison',
                'post_collision_and_nms': True})
            request.update(state='complete', finished_at=datetime.now().astimezone().isoformat())
            save(out/'shared_backend_request.json', request)
    except Exception:
        result.update(reason='AnyGrasp probe failed', error=traceback.format_exc())
        request.update(state='failed', finished_at=datetime.now().astimezone().isoformat())
        save(out/'shared_backend_request.json', request)
        raise
    finally:
        save(out/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
