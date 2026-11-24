"""Inspect SAM3 on all geometry-admitted scenes without replacing any scene."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from scene_builder import PROJECT, DEFAULT, LAYOUTS, LABELS, save, read, sha, now


def report(root, results):
    summary = {'completed': len(results), 'registered': 200,
        'with_prediction': sum(r['has_prediction'] for r in results),
        'iou_ge_50': sum(r['iou'] >= .50 for r in results),
        'iou_ge_75': sum(r['iou'] >= .75 for r in results),
        'median_iou': float(np.median([r['iou'] for r in results])) if results else None,
        'by_layout': {layout: {'completed': sum(r['layout'] == layout for r in results),
            'with_prediction': sum(r['layout'] == layout and r['has_prediction'] for r in results),
            'iou_ge_50': sum(r['layout'] == layout and r['iou'] >= .50 for r in results)} for layout in LAYOUTS},
        'at': now(), 'physical_grasps': 0, 'scene_replacement_from_SAM3_results': False}
    save(root / 'sam3_progress.json', summary)
    return summary


def document(root, summary):
    reg = read(root / 'registration.json')
    lines = ['# 头部感知场景构造结果', '',
        '新构造200个场景，五类各40个。沿用原38个目标型号及各类中的型号配额，升降固定45.2厘米，使用向下的头部D405观察。', '',
        '| 场景类别 | 已构造 | SAM3有预测掩码 | 预测掩码与可见目标IoU至少50% |',
        '| --- | ---: | ---: | ---: |']
    for layout, label in zip(LAYOUTS, LABELS):
        s = summary['by_layout'][layout]
        lines.append(f"| {label} | 40 | {s['with_prediction']}/40 | {s['iou_ge_50']}/40 |")
    lines += ['', f"已检查{summary['completed']}/200例，有预测掩码{summary['with_prediction']}例，IoU至少50%的有{summary['iou_ge_50']}例。", '',
        'SAM3检查结果不用于替换场景；尚未进行新的抓取实验。', '', '## 场景缩略图', '']
    doc = PROJECT / 'docs/head_perception_scenes/场景构造结果.md'
    resources = doc.parent / 'resources'; resources.mkdir(exist_ok=True)
    import shutil
    for layout, label in zip(LAYOUTS, LABELS):
        image = root / 'gallery' / (layout + '.jpg')
        shutil.copy2(image, resources / image.name)
        lines += [f'### {label}', '', f'![{label}的40个场景](resources/{image.name})', '']
    lines += ['## 数据目录', '', f'`{root}`', '',
        '每个场景的`camera/color.png`是模型输入原图，`overview.jpg`是第三人称截图。`sam3/<场景ID>/overlay.jpg`显示预测区域。']
    doc.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--wait-for-manifest', action='store_true'); args = parser.parse_args()
    root = args.root.resolve(); deadline = time.monotonic() + 3600
    while not (root / 'manifest.json').exists():
        if not args.wait_for_manifest or (root / 'halt.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Scene geometry is not complete; SAM3 check not started')
        time.sleep(5)
    reg = read(root / 'registration.json'); manifest = read(root / 'manifest.json')
    assert manifest['registration_sha256'] == sha(root / 'registration.json')
    sys.path.insert(0, str(PROJECT.parent / 'claro/依赖/SAM3源码'))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    checkpoint = PROJECT.parent / 'claro/权重/sam3.pt'
    save(root / 'sam3_registration.json', {'at': now(), 'checkpoint': str(checkpoint),
        'checkpoint_size': checkpoint.stat().st_size, 'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
        'threshold': .5, 'text_catalog': {e['model_id']: e['target_text'] for e in reg['schedule']},
        'source_sha256': sha(Path(__file__)), 'input_manifest_sha256': sha(root / 'manifest.json'),
        'selection': 'maximum SAM3 score for the registered target text; no GT matching',
        'object_or_scene_replacement': False, 'physical_grasps': 0})
    model = build_sam3_image_model(device='cuda', checkpoint_path=str(checkpoint),
        load_from_HF=False, enable_inst_interactivity=False)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=.5)
    results = []
    for entry in manifest['schedule']:
        source = root / 'cases' / entry['id']; out = root / 'sam3' / entry['id']; out.mkdir(parents=True, exist_ok=False)
        frame = Image.open(source / 'camera/color.png').convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(frame)
            state = processor.set_text_prompt(prompt=entry['target_text'], state=state)
            scores = state['scores'].detach().float().cpu().numpy()
            masks = state['masks'].detach().cpu().numpy().astype(bool).reshape(-1, frame.height, frame.width)
        chosen = int(np.argmax(scores)) if len(scores) else None
        mask = masks[chosen] if chosen is not None else np.zeros((frame.height, frame.width), bool)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / 'mask.png')
        np.savez_compressed(out / 'all_predictions.npz', masks=masks, scores=scores)
        rgb = np.asarray(frame).copy(); rgb[mask] = (.6 * rgb[mask] + .4 * np.array([50, 220, 80])).astype(np.uint8)
        Image.fromarray(rgb).save(out / 'overlay.jpg', quality=88)
        # Independent GT is opened only after inference and candidate selection.
        gt = np.asarray(Image.open(source / 'target_gt.png')) > 0
        inter, union = int((mask & gt).sum()), int((mask | gt).sum())
        result = {'id': entry['id'], 'layout': entry['layout'], 'prompt': entry['target_text'],
            'scores': scores.tolist(), 'chosen_index': chosen, 'has_prediction': bool(mask.any()),
            'iou': inter / max(1, union), 'target_recall': inter / max(1, int(gt.sum())),
            'ground_truth_used_for_selection': False, 'model_load_count': 1,
            'input_rgb_sha256': sha(source / 'camera/color.png')}
        save(out / 'result.json', result); results.append(result)
        summary = report(root, results)
        print('SAM3', len(results), entry['id'], result['has_prediction'], round(result['iou'], 3), flush=True)
    save(root / 'sam3_complete.json', summary)
    document(root, summary)


if __name__ == '__main__': main()
