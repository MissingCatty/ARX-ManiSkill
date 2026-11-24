"""Verify and hash-freeze geometry-only scenes before any diagnostic runs."""
import argparse
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from asset_catalog import HERE, PROJECT, read, save, sha
from verify_dataset import LABELS


def verify(root):
    reg, manifest = read(root / 'registration.json'), read(root / 'manifest.json')
    catalog = read(root / 'object_catalog.json')
    db = catalog['models']
    assert reg['construction_independent_of_ik'] and reg['source_identity_fields_only']
    assert not reg['source_positions_used'] and not reg['admission']['IK_selection']
    assert manifest['registration_sha256'] == sha(root / 'registration.json')
    assert reg['object_catalog_sha256'] == sha(root / 'object_catalog.json')
    assert reg['sampling_region_sha256'] == sha(root / 'sampling_region.json')
    for path, digest in reg['source_hashes'].items():
        assert sha(path) == digest, path
    source = Path(reg['source_dataset'])
    assert sha(source / 'manifest.json') == reg['source_manifest_sha256']
    old = {e['id']: e for e in read(source / 'manifest.json')['schedule']}
    rows = manifest['schedule']
    assert len(rows) == len({e['id'] for e in rows}) == 200 and set(old) == {e['id'] for e in rows}
    region = reg['sampling_region']
    reference = read(root / 'preflight/head_camera.json')
    records, distances, angles, model_counts, layouts = [], [], [], Counter(), Counter()
    for entry in rows:
        folder = root / 'cases' / entry['id']
        admission, state, construction = [read(folder / name) for name in
                                         ['admission.json', 'state.json', 'construction_check.json']]
        assert admission['passed'] and admission['entry'] == entry
        assert construction['passed'] and construction['construction_independent_of_ik'] and construction['IK_queries'] == 0
        assert entry['model_id'] == old[entry['id']]['model_id'] and entry['layout'] == old[entry['id']]['layout']
        assert entry['semantic_class'] == db[entry['model_id']]['semantic_class']
        actors = entry['scene']['actors']
        assert [(a['name'], a['model_id']) for a in actors] == [(a['name'], a['model_id']) for a in old[entry['id']]['scene']['actors']]
        assert state['perception_camera'] == 'head'
        assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
        assert np.allclose(state['links']['base_link'], reference['links']['base_link'], atol=3e-5)
        assert np.allclose(state['camera_world_optical'], reference['camera_world_optical'], atol=3e-5)
        assert np.allclose(state['intrinsics'], reference['intrinsics'], atol=1e-6)
        camera, k = np.array(state['camera_world_optical']), np.array(state['intrinsics'])
        boxes = []
        for actor in actors:
            b, p = db[actor['model_id']]['bbox'], np.array(actor['pose'])
            rotation = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
            points = np.array(list(product(*zip(b['min'], b['max'])))) @ rotation.T + p[:3]
            lo, hi = points.min(0), points.max(0)
            center = (lo + hi) / 2
            assert (center[:2] >= np.array(region['center_roi_min_xy_m']) - .002).all()
            assert (center[:2] <= np.array(region['center_roi_max_xy_m']) + .002).all()
            assert (lo[:2] >= np.array(region['table_min_xy_m']) + .012).all()
            assert (hi[:2] <= np.array(region['table_max_xy_m']) - .012).all()
            local = (np.array(region['shoulder_world_m']) - p[:3]) @ rotation
            nearest = np.clip(local, b['min'], b['max']) @ rotation.T + p[:3]
            assert np.linalg.norm(nearest - region['shoulder_world_m']) <= region['contact_length_upper_bound_m'] + .002
            xyz = (points - camera[:3, 3]) @ camera[:3, :3]
            assert xyz[:, 2].min() > .10 and xyz[:, 2].max() < 2.
            uv = xyz @ k.T
            uv = uv[:, :2] / uv[:, 2:]
            assert (uv.min(0) >= [12, 12]).all() and (uv.max(0) <= [628, 468]).all()
            boxes.append((lo, hi))
        for i, (lo, hi) in enumerate(boxes):
            for other_lo, other_hi in boxes[:i]:
                assert (hi[:2] + .003 < other_lo[:2]).any() or (other_hi[:2] + .003 < lo[:2]).any()
        if entry['layout'] == 'S2_edge':
            assert .087 <= boxes[0][0][0] - region['table_min_xy_m'][0] <= .113
        quality = entry['admission']
        assert quality['visible_pixels'] >= 64 and quality['valid_target_depth_fraction'] >= .98
        assert (.20 <= quality['occlusion_fraction'] <= .40) if entry['layout'] == 'S5_occluded' else quality['occlusion_fraction'] <= .15
        for name, digest in admission['camera_input_sha256'].items():
            assert sha(folder / 'camera' / name) == digest
        native = np.asarray(Image.open(folder / 'target_gt_native.png')) > 0
        assert int(native.sum()) == quality['visible_pixels']
        mask = np.asarray(Image.open(folder / 'target_gt.png')) > 0
        depth = np.asarray(Image.open(folder / 'camera/depth.png')) * .0001
        valid = np.isfinite(depth) & (depth > .10) & (depth < 2.)
        assert (valid & mask).sum() / mask.sum() >= .98
        assert not (folder / 'workspace_check.json').exists() and not (folder / 'plan_result.json').exists()
        distances.append(construction['actors']['target']['center_distance_m'])
        angles.append(float(Rotation.from_quat(np.array(actors[0]['pose'])[[4, 5, 6, 3]]).as_euler('xyz', degrees=True)[2]))
        layouts[entry['layout']] += 1
        model_counts[entry['model_id']] += 1
        records.append({'id': entry['id'], 'passed': True, 'visible_pixels': quality['visible_pixels'],
                        'occlusion_fraction': quality['occlusion_fraction'], 'center_distance_m': distances[-1]})
    assert layouts == dict.fromkeys(reg['layouts'], 40)
    assert len(model_counts) == 80 and set(model_counts.values()) == {2, 3}
    for layout in reg['layouts']:
        assert Counter(e['semantic_class'] for e in rows if e['layout'] == layout) == dict.fromkeys(reg['semantic_classes'], 1)
    assert not list((root / 'preflight').glob('ik*')) and not (root / 'diagnostics').exists()
    histogram, bins = np.histogram(angles, bins=np.linspace(-180, 180, 13))
    result = {'passed': True, 'scenes': 200, 'categories': 40, 'models': 80, 'layouts': dict(layouts),
              'construction_independent_of_ik': True, 'construction_IK_queries': 0,
              'physical_grasps': 0, 'records': records,
              'target_distance_min_median_max_m': [min(distances), float(np.median(distances)), max(distances)],
              'target_centers_beyond_70cm_retained': sum(d > .70 for d in distances),
              'yaw_bins_degrees': bins.tolist(), 'yaw_counts': histogram.tolist(),
              'verifier_source_sha256': sha(Path(__file__))}
    save(root / 'verification.json', result)
    files = [root / name for name in ['registration.json', 'manifest.json', 'object_catalog.json',
                                      'sampling_region.json', 'verification.json', 'preflight/head_camera.json']]
    files += [p for p in sorted((root / 'cases').rglob('*')) if p.is_file()]
    frozen = {'at': datetime.now().astimezone().isoformat(), 'scenes': 200,
              'construction_independent_of_ik': True, 'IK_evaluations_at_freeze': 0,
              'files': {str(p.relative_to(root)): sha(p) for p in files},
              'post_freeze_changes_allowed': ['diagnostics/', 'progress.json', 'documentation outputs']}
    save(root / 'frozen_dataset.json', frozen)
    save(root / 'progress.json', {'phase': 'frozen', 'completed': 200, 'registered': 200,
                                'construction_independent_of_ik': True, 'physical_grasps': 0})
    active = {'dataset': str(root), 'object_catalog': str(root / 'object_catalog.json'),
              'environment_id': 'ARXPickCatalog-v1', 'adapter': str(HERE / 'asset_environment.py'),
              'runtime_factory': str(HERE / 'pipeline.py') + ':runtime', 'scenes': 200, 'categories': 40, 'models': 80,
              'state': 'verified_independent_sampling', 'grasp_tests_authorized': False, 'grasp_test_budget': 0,
              'construction_independent_of_ik': True, 'verification': str(root / 'verification.json'),
              'freeze': str(root / 'frozen_dataset.json')}
    save(HERE / 'active_dataset.json', active)
    save(PROJECT / 'configs/scene_dataset.json', active)
    write_report(root, reg, result)
    print('GEOMETRY_FROZEN', {k: v for k, v in result.items() if k != 'records'}, flush=True)


def write_report(root, reg, result):
    docs = PROJECT / 'docs/diverse_object_scenes'
    resource = docs / 'resource'
    lines = ['# 独立采样场景结果', '', '200个场景已按预先固定的物理区域重新随机生成，并在任何IK诊断开始前完成核验和散列冻结。', '',
             '| 检查项 | 结果 |', '| --- | --- |', '| 布局 | 五类各40例，共200例 |', '| 物体 | 40类、80个型号 |',
             '| 构造阶段IK调用 | 0次 |', '| 物理抓取 | 0次 |',
             f'| 目标距肩部范围 | {result["target_distance_min_median_max_m"][0]*100:.1f}–{result["target_distance_min_median_max_m"][2]*100:.1f}厘米 |',
             f'| 距肩部超过70厘米且按几何规则保留的目标 | {result["target_centers_beyond_70cm_retained"]}例 |', '',
             '位置在桌面近侧一半、中央80厘米宽的区域内均匀采样，目标朝向使用连续随机角度。桌边组距近侧桌沿9–11厘米。构造只按静态物理、完整视野、有效深度和布局遮挡要求准入。', '',
             '机器人升降45.2厘米，底盘、桌面、头部D405参数、物体资产和物理配置沿用原设置。原IK准入批次保留为历史数据。', '',
             '## 各组截图', '']
    labels = ['开阔桌面', '靠近桌边', '稀疏混放', '中度杂乱', '部分遮挡']
    for layout, label in zip(reg['layouts'], labels):
        name = 'geometry_' + layout + '.jpg'
        shutil.copy2(root / 'gallery' / (layout + '.jpg'), resource / name)
        lines += [f'### {label}', '', f'![{label}40例](resource/{name})', '']
    lines += ['## 文件位置', '', f'- 场景目录：`{root}`', '- 生成前规则：`registration.json`、`sampling_region.json`',
              '- 冻结文件和散列：`frozen_dataset.json`', '- 每例几何核查：`cases/场景编号/construction_check.json`',
              '- 事后IK诊断单独保存到`diagnostics/`，所有成功与失败均保留，诊断结果不会改变场景。']
    (docs / '独立采样场景结果.md').write_text('\n'.join(lines) + '\n')
    main = ['# 多类别物体与场景', '', '当前使用独立随机采样的200个场景：40类、80个型号，五类布局各40例。场景先完成几何核验和散列冻结，再单独进行IK诊断。', '',
            '- [独立采样方案](独立采样场景方案.md)', '- [本批次结果与截图](独立采样场景结果.md)', '',
            '## 物体种类', '', '![40类80个型号](resource/object_gallery.jpg)', '',
            '| 类别 | 型号数 | 场景数 |', '| --- | ---: | ---: |']
    main += [f'| {LABELS[c]} | 2 | 5 |' for c in reg['semantic_classes']]
    main += ['', '## 场景目录', '', f'`{root}`', '', '统一入口为`configs/scene_dataset.json`。本轮物理抓取为0次，抓取测试保持停止。']
    (docs / '多类别物体与场景.md').write_text('\n'.join(main) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root.resolve())
