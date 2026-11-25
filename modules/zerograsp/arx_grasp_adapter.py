"""Native ZeroGrasp grasp inference with audited collision/refinement/NMS output."""
from pathlib import Path
import os
import random
import sys
import time

import numpy as np
from PIL import Image

from arx_project.layout import ROOT, module_dir, read_json, weights_dir, write_json


def common_candidates(rows, camera):
    """Convert GraspNet contact frames to the shared TCP43 planning contract."""
    from modules.maniskill.planning_common.grasp_selector import ARX_AXES
    rows = np.asarray(rows)
    camera = np.asarray(camera)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError('Expected finite native postprocessed (N,17) grasps')
    result = []
    for index in np.argsort(-rows[:, 0], kind='stable'):
        row = rows[index]
        rotation = camera[:3, :3] @ row[4:13].reshape(3, 3)
        center = camera[:3, :3] @ row[13:16] + camera[:3, 3] + row[3]*rotation[:, 0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation @ ARX_AXES, center
        result.append({'pose': pose.tolist(), 'score': float(row[0]), 'width': float(row[1]),
            'point1': (center-row[1]/2*pose[:3, 1]).tolist(),
            'point2': (center+row[1]/2*pose[:3, 1]).tolist(),
            'metadata': {'origin': 'zerograsp', 'postprocessed_row_index': int(index),
                'source_network_row': row.tolist(), 'source_network_depth_m': float(row[3]),
                'source_model_score': float(row[0]), 'storage_tcp_m': .043,
                'coordinate_conversion_only': True, 'finger_flip': False,
                'pose_refinement_applied': False}})
    return result


class NativeGraspAdapter:
    def __init__(self):
        package = module_dir('zerograsp')/'server/vendor/zerograsp_mainline_minimal'
        sys.path[:0] = [str(package), str(module_dir('maniskill')),
                       str(module_dir('maniskill')/'planning_common')]
        from zerograsp.pipeline import ZeroGraspPipeline
        self.pipeline = ZeroGraspPipeline(
            checkpoint_path=str(weights_dir('zerograsp')/'mainline/zerograsp_synthetic.ckpt'),
            config_path=str(package/'configs/demo_640x480.yaml'), device='cuda')
        if not self.pipeline._config.use_collision_detection:
            raise RuntimeError('Native ZeroGrasp collision detection must be enabled')

    def request(self, directory, *, seed=20261125):
        import torch
        import zerograsp.pipeline as native
        from planning_common.candidates import (
            digest_rows, export_candidates, finalize_postprocessing, load_candidates)
        directory = Path(directory)
        from arx_project.artifacts import publish_candidate_pool
        if (directory/'network').exists() or (directory/'candidates').exists():
            raise RuntimeError('ZeroGrasp outputs already exist; no inference replay')
        state = read_json(directory/'state.json')
        rgb = np.asarray(Image.open(directory/'camera/color.png').convert('RGB'))
        mask = np.asarray(Image.open(directory/'camera/mask.png')) > 0
        depth = np.load(directory/'camera/depth_m.npy', allow_pickle=False).squeeze()
        if not mask.any():
            raise ValueError('Comparison requires an already successful SAM3 mask')
        detections = []
        original_detector = native.ModelFreeCollisionDetector

        class AuditedDetector(original_detector):
            def detect(self, group, *args, **kwargs):
                raw = group.grasp_group_array.copy()
                collision, width, refined_depth = super().detect(group, *args, **kwargs)
                refined = raw.copy()
                refined[:, 1] += np.asarray(width)
                refined[:, 3] = np.asarray(refined_depth)
                detections.append({'raw_rows': raw, 'raw_rows_sha256': digest_rows(raw),
                    'accepted_rows': refined[~np.asarray(collision, dtype=bool)]})
                return collision, width, refined_depth

        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.synchronize()
        started = time.perf_counter()
        native.ModelFreeCollisionDetector = AuditedDetector
        try:
            prediction = self.pipeline.predict(rgb, depth.astype(np.float32)*1000.,
                mask.astype(np.int32), np.asarray(state['intrinsics']), depth_scale=1.)
        finally:
            native.ModelFreeCollisionDetector = original_detector
        raw = directory/'network/raw_outputs'
        raw.mkdir(parents=True)
        if len(prediction.objects) != len(detections):
            raise RuntimeError('ZeroGrasp collision call ledger differs from objects')
        clouds = []
        for obj, proof in zip(prediction.objects, detections):
            name = f'object_{obj.object_index:03d}'
            np.save(raw/(name+'.raw_grasps.npy'), proof['raw_rows'], allow_pickle=False)
            np.save(raw/(name+'.grasp.npy'), obj.grasp_group_array, allow_pickle=False)
            clouds.append(obj.point_cloud.astype(np.float32)/1000.)
        audit = finalize_postprocessing(raw, detections)
        exported = export_candidates(directory/'network', directory, fresh_inference=True)
        exported['mask'] = 'frozen successful SAM3 prediction, shared with the paired baseline'
        write_json(directory/'network_candidates.json', exported)
        rows, provenance = load_candidates(directory)
        candidates = common_candidates(rows, state['camera_world_optical'])
        dest = directory/'candidates'
        dest.mkdir()
        # Legacy planner reads this path; its file is identical to the common pool.
        (directory/'graspcl').symlink_to(dest, target_is_directory=True)
        transport = publish_candidate_pool(directory, {'candidates': candidates,
            'protocol': 'sam3_zerograsp_native_tcp43_v1', 'native_score_order': True,
            'retained_count': len(candidates), 'raw_count': sum(o['raw_count'] for o in audit['objects']),
            'geometric_prefilters': False, 'additional_orientation_search': False,
            'raw_fallback': False, 'graspcl_invoked': False, 'tcp_m': .043,
            'postprocessing_receipt': provenance['postprocessing_receipt']})
        np.save(directory/'raw_grasps.npy', rows, allow_pickle=False)
        np.save(dest/'raw_grasps.npy', rows, allow_pickle=False)
        points = np.concatenate(clouds) if clouds else np.empty((0, 3), np.float32)
        np.save(directory/'completed_points_camera.npy', points, allow_pickle=False)
        camera = np.asarray(state['camera_world_optical'])
        np.save(directory/'completed_points_world.npy', points @ camera[:3, :3].T+camera[:3, 3], allow_pickle=False)
        torch.cuda.synchronize()
        report = {'success': True, 'candidate_count': len(candidates), 'point_count': len(points),
            'seed': int(seed), 'collision_filter': True, 'width_depth_refinement': True,
            'nms': True, 'raw_fallback': False, 'seconds': time.perf_counter()-started,
            'mask_source': 'frozen_successful_SAM3', 'tcp_m': .043}
        write_json(directory/'zerograsp.json', report)
        report['candidate_transport'] = transport
        return report


def create(args):
    return NativeGraspAdapter()
