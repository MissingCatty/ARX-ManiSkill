"""用 CLARO 的实例点提示和同帧 RGB-D 定位挡物。"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SAM3_SHA256 = '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_point(receipt, size):
    decision = receipt['decision']
    if decision['tool'] != 'locate_object':
        raise ValueError('选择记录没有返回可定位的挡物')
    graph, = [c['parse']['parsed_scene_graph'] for c in receipt['calls'] if c['tool'] == 'survey_scene']
    matches = [o for o in graph['objects'] if o['name'] == decision['object_name']]
    if 'object_id' in decision:
        matches = [o for o in matches if o['obj_id'] == decision['object_id']]
    if len(matches) != 1:
        raise ValueError('挡物名称必须对应一个场景图实例')
    obj = matches[0]
    point = np.asarray(obj['point_norm'], dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or not ((0 <= point) & (point <= 1)).all():
        raise ValueError('场景图点位不在归一化图像范围内')
    height, width = size
    row, col = point
    return obj, (round(col * (width - 1)), round(row * (height - 1)))


def points_in_base(depth, mask, rgb, params, state):
    intrinsics = params['intrinsics']
    height, width = mask.shape
    if (depth.shape != mask.shape or rgb.shape[:2] != mask.shape
            or (width, height) != (intrinsics['width'], intrinsics['height'])):
        raise ValueError('RGB、掩码、深度与内参尺寸不一致')
    camera = np.asarray(state['camera_world_optical'], dtype=float)
    base = np.asarray(state['links']['body4'], dtype=float)
    for transform in (camera, base):
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError('缺少有效的相机或机械臂基座变换')
    recorded_k = np.asarray(state['intrinsics'], dtype=float)
    fx, fy, cx, cy = (intrinsics[key] for key in ('fx', 'fy', 'ppx', 'ppy'))
    k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    scale = float(params['depth_scale'])
    if not (np.isfinite(scale) and scale > 0 and fx > 0 and fy > 0 and np.allclose(k, recorded_k)):
        raise ValueError('相机内参或深度单位与同帧记录不符')
    meters = depth.astype(float) * scale
    # ARX 腕部相机记录的有效深度范围，单位米。
    low, high = state['camera_alignment']['camera_definition']['depth_valid_range_m']
    v, u = np.nonzero(mask & np.isfinite(meters) & (meters > low) & (meters <= high))
    if len(u) == 0:
        raise ValueError('掩码内没有有效深度')
    uv = np.c_[u, v]
    optical = (np.c_[uv, np.ones(len(u))] @ np.linalg.inv(k).T) * meters[v, u, None]
    base_from_camera = np.linalg.solve(base, camera)
    xyz = optical @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    return xyz, optical, uv, rgb[v, u]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '权重' / 'sam3.pt')
    parser.add_argument('--sam3-source', type=Path, default=ROOT / '依赖' / 'SAM3源码')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    receipt = read(args.selection)
    if receipt['scene'] != args.trial.name.split('_')[0]:
        raise ValueError('挡物选择记录和场景编号不一致')
    color, depth_path, params_path, state_path = [args.trial / name for name in (
        'camera/color.png', 'camera/depth.png', 'camera/params.json', 'state.json')]
    if sha256(color) != receipt['image_sha256']:
        raise ValueError('定位图和 VLM 选挡图不一致')
    if not args.checkpoint.is_file() or sha256(args.checkpoint) != SAM3_SHA256:
        raise ValueError('需要与 Meta 原版哈希一致的 sam3.pt')
    commit = subprocess.check_output(['git', '-C', str(args.sam3_source), 'rev-parse', 'HEAD'], text=True).strip()
    frame = Image.open(color).convert('RGB')
    rgb = np.asarray(frame)
    depth = np.asarray(Image.open(depth_path))
    state, params = read(state_path), read(params_path)
    obj, (x, y) = selected_point(receipt, rgb.shape[:2])
    args.output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.sam3_source.resolve()))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    gpu = args.device.startswith('cuda')
    began = time.monotonic()
    with torch.inference_mode():
        model = build_sam3_image_model(device=args.device, checkpoint_path=str(args.checkpoint),
                                       load_from_HF=False, enable_inst_interactivity=True)
        processor = Sam3Processor(model, device=args.device)
        with (torch.autocast('cuda', dtype=torch.bfloat16) if gpu else nullcontext()):
            image_state = processor.set_image(frame)
            masks, scores, _ = model.predict_inst(image_state, point_coords=np.array([[x, y]], dtype=np.float32),
                                                  point_labels=np.array([1]), multimask_output=True)
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if masks.shape != (len(scores), *rgb.shape[:2]) or not np.isfinite(scores).all():
        raise ValueError('SAM3 返回的掩码尺寸或分数无效')
    valid = np.flatnonzero(masks[:, y, x])
    if not valid.size:
        raise ValueError('SAM3 无候选掩码覆盖 VLM 定位点')
    best = int(valid[np.argmax(scores[valid])])
    mask = masks[best]
    xyz, optical, uv, colors = points_in_base(depth, mask, rgb, params, state)
    v, u = np.nonzero(mask)
    box = [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]

    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / '挡物掩码.png')
    preview = rgb.copy()
    preview[mask] = (0.63 * preview[mask] + 0.37 * np.array([38, 218, 109])).astype(np.uint8)
    overlay = Image.fromarray(preview)
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='#ffdc41', outline='black', width=1)
    overlay.save(args.output / '挡物定位叠加.png')
    np.savez_compressed(args.output / '候选掩码.npz', masks=masks, scores=scores)
    np.savez_compressed(args.output / '挡物点云.npz', xyz_base_m=xyz, xyz_optical_m=optical, uv=uv, rgb=colors)
    record = {
        'scene': receipt['scene'], 'object_name': obj['name'], 'survey_object_id': obj['obj_id'],
        'point_xy': [x, y], 'bbox_xyxy_exclusive': box, 'sam3_commit': commit,
        'sam3_checkpoint_sha256': SAM3_SHA256, 'torch_version': torch.__version__,
        'device': torch.cuda.get_device_name() if gpu else args.device,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES') if gpu else None,
        'cuda_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20) if gpu else None,
        'source_sha256': {str(p.name): sha256(p) for p in (color, depth_path, params_path, state_path, args.selection)},
        'predicted_iou': scores.tolist(), 'candidate_pixel_counts': masks.sum(axis=(1, 2)).tolist(),
        'candidate_contains_prompt': masks[:, y, x].tolist(), 'selected_candidate': best,
        'mask_pixels': int(mask.sum()), 'depth_points': len(xyz), 'base_frame': 'body4',
        'pointcloud_min_base_m': xyz.min(0).tolist(), 'pointcloud_max_base_m': xyz.max(0).tolist(),
        'wall_seconds': round(time.monotonic() - began, 3), 'robot_motion_commands': 0,
    }
    (args.output / '定位记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
