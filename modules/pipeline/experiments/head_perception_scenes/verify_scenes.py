"""Independent file/geometry checks for the constructed head-view dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scene_builder import DEFAULT, MS, LAYOUTS, read, save, sha, project, now


def verify(root):
    reg, manifest = read(root / 'registration.json'), read(root / 'manifest.json')
    assert len(manifest['schedule']) == 200
    assert manifest['registration_sha256'] == sha(root / 'registration.json')
    assert Counter(e['layout'] for e in manifest['schedule']) == dict.fromkeys(LAYOUTS, 40)
    db = read(MS / 'assets/data/assets/mani_skill2_ycb/info_pick_v0.json')
    def corners(info):
        lo, hi = np.asarray(info['bbox']['min']), np.asarray(info['bbox']['max'])
        return np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]]) * info.get('scales', [1.])[0]
    baseline = SimpleNamespace(DB=db, corners=corners)
    geometry_keys = set(); records = []
    for entry in manifest['schedule']:
        out = root / 'cases' / entry['id']
        admission = read(out / 'admission.json'); assert admission['passed']
        assert entry == admission['entry']
        state = read(out / 'state.json'); camera = np.asarray(state['camera_world_optical']); k = np.asarray(state['intrinsics'])
        assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
        assert state['perception_camera'] == 'head'
        for actor in entry['scene']['actors']:
            assert project(actor, baseline, camera, k) is not None, entry['id'] + ':' + actor['name']
        for name, digest in admission['camera_input_sha256'].items(): assert sha(out / 'camera' / name) == digest
        native = np.asarray(Image.open(out / 'target_gt_native.png')) > 0
        mask = np.asarray(Image.open(out / 'target_gt.png')) > 0
        rgb = np.asarray(Image.open(out / 'camera/color.png')); depth = np.asarray(Image.open(out / 'camera/depth.png')) * .0001
        assert native.shape == mask.shape == depth.shape == (480, 640) and rgb.shape == (480, 640, 3)
        q = entry['admission']
        assert int(native.sum()) == q['visible_pixels'] == q['target_mask_pixels']
        assert int(native.sum()) >= 64 and mask.any()
        valid = np.isfinite(depth) & (depth > .10) & (depth < 2.)
        fraction = float((valid & mask).sum() / mask.sum())
        assert fraction >= .98, (entry['id'], fraction)
        assert .20 <= q['occlusion_fraction'] <= .40 if entry['layout'] == 'S5_occluded' else q['occlusion_fraction'] <= .15
        counts = reg['distractor_counts'][entry['layout']]
        assert counts[0] <= len(entry['scene']['actors']) - 1 <= counts[1]
        canonical = [(a['model_id'], np.round(a['pose'], 6).tolist()) for a in entry['scene']['actors']]
        key = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        assert key not in geometry_keys, 'Duplicate physical scene'; geometry_keys.add(key)
        records.append({'id': entry['id'], 'visible_pixels': int(native.sum()),
            'target_depth_median_m': q['target_depth_median_m'], 'valid_model_depth_fraction': fraction,
            'occlusion_fraction': q['occlusion_fraction'], 'all_actors_in_frame': True})
    for layout in LAYOUTS:
        assert Counter(e['model_id'] for e in manifest['schedule'] if e['layout'] == layout) == reg['target_quotas_by_layout'][layout]
    result = {'passed': True, 'at': now(), 'scenes': 200, 'unique_scenes': len(geometry_keys),
        'target_models': len({e['model_id'] for e in manifest['schedule']}),
        'layout_counts': dict(Counter(e['layout'] for e in manifest['schedule'])),
        'same_object_quotas_as_source': True, 'all_actors_inside_head_frame': True,
        'snapshot_mask_counts_match': True, 'physical_grasps': 0, 'records': records}
    save(root / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=DEFAULT)
    verify(p.parse_args().root.resolve())
