"""Read back scene geometry, replay saved FK, and independently check table collision."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
import csv
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS = PROJECT.parent / 'maniskill'
URDF = PROJECT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collision_check(root):
    import torch
    sys.path[:0] = [str(MS), str(MS / 'arx_pipeline')]
    from planner import Planner
    state = read(root / 'preflight/head_camera.json')
    state['scene_meshes'] = {}
    directory = root / 'independent_collision_model'
    directory.mkdir(exist_ok=True)
    planner = Planner(directory, state)
    records = []
    for entry in read(root / 'manifest.json')['schedule']:
        certificate = read(root / 'cases' / entry['id'] / 'workspace_check.json')['preparation_pose_ik']
        q = torch.tensor([certificate[n] for n in ['q_goal', 'q_preparation', 'q_lift']],
                         dtype=torch.float32, device='cuda')
        valid = planner.checker.validate(q[:, None]).flatten().tolist()
        assert all(valid), (entry['id'], valid)
        records.append({'id': entry['id'], 'goal_preparation_lift_valid': valid})
    result = {'passed': True, 'checked_scenes': len(records), 'checked_joint_states': len(records) * 3,
              'collision_model': 'original robot collision spheres, original table and ground',
              'physical_grasps': 0, 'records': records,
              'model_sha256': sha(directory / 'arx.yml'), 'scene_sha256': sha(directory / 'scene.json')}
    save(root / 'independent_collision_validation.json', result)
    print('COLLISION_VERIFIED', len(records), flush=True)


def fk(chain, base, joints):
    value = np.asarray(base).copy()
    for joint in chain:
        origin = joint.find('origin')
        local = np.eye(4)
        local[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        local[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        value = value @ local
        if joint.get('type') in ('revolute', 'continuous'):
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * joints[joint.get('name')]).as_matrix()
            value = value @ motion
        elif joint.get('type') != 'fixed':
            raise AssertionError('Unexpected movable joint in fixed-lift arm')
    return value


def verify(root):
    from PIL import Image
    reg, manifest, catalog = [read(root / n) if n != 'object_catalog.json' else load_object_catalog(root / n) for n in ['registration.json', 'manifest.json', 'object_catalog.json']]
    db = catalog['models']
    assert not reg['development_only'] and len(manifest['schedule']) == 200
    assert manifest['registration_sha256'] == sha(root / 'registration.json')
    assert reg['object_catalog_sha256'] == sha(root / 'object_catalog.json')
    for file, digest in reg['source_hashes'].items():
        assert sha(file) == digest, file
    source = Path(reg['source_dataset'])
    assert sha(source / 'manifest.json') == reg['source_manifest_sha256']
    assert sha(source / 'registration.json') == reg['source_registration_sha256']
    assert sha(source / 'object_catalog.json') == sha(root / 'object_catalog.json')
    original = {e['id']: e for e in read(source / 'manifest.json')['schedule']}
    assert set(original) == {e['id'] for e in manifest['schedule']}
    bychild = {j.find('child').get('link'): j for j in ET.parse(URDF).getroot().findall('joint')}
    chain, link = [], 'right_eef_link'
    while link != 'body4':
        joint = bychild[link]
        chain.append(joint)
        link = joint.find('parent').get('link')
    chain.reverse()
    rows, records, labels = [], [], Counter()
    max_fk_error, max_rot_error = 0., 0.
    for entry in manifest['schedule']:
        folder = root / 'cases' / entry['id']
        admission, state, workspace = [read(folder / n) for n in ['admission.json', 'state.json', 'workspace_check.json']]
        assert admission['passed'] and workspace['passed'] and admission['entry'] == entry
        old = original[entry['id']]
        assert entry['model_id'] == old['model_id'] and entry['layout'] == old['layout']
        actors = entry['scene']['actors']
        assert [(a['name'], a['model_id']) for a in actors] == [(a['name'], a['model_id']) for a in old['scene']['actors']]
        assert state['perception_camera'] == 'head'
        assert abs(state['qpos'][state['joint_names'].index('joint4')] - .452) < .001
        assert np.allclose(state['camera_world_optical'], read(root / 'preflight/head_camera.json')['camera_world_optical'], atol=3e-5)
        camera, k, pivot = np.array(state['camera_world_optical']), np.array(state['intrinsics']), np.array(reg['shoulder_pivot_world_m'])
        boxes = []
        for actor in actors:
            b = db[actor['model_id']]['bbox']
            corners = np.array(list(product(*zip(b['min'], b['max']))))
            p = np.array(actor['pose'])
            world = corners @ Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix().T + p[:3]
            lo, hi = world.min(0), world.max(0)
            center = (lo + hi) / 2
            assert np.linalg.norm(center - pivot) <= .70001, entry['id']
            assert (lo[:2] >= [-.7125, -1.197]).all() and (hi[:2] <= [.4725, 1.197]).all()
            xyz = (world - camera[:3, 3]) @ camera[:3, :3]
            assert xyz[:, 2].min() > .10 and xyz[:, 2].max() < 2.
            uv = xyz @ k.T
            uv = uv[:, :2] / uv[:, 2:]
            assert (uv.min(0) >= [12, 12]).all() and (uv.max(0) <= [628, 468]).all()
            boxes.append((lo, hi))
        # Settling must not introduce overlap into the source nonoverlap layout.
        for i, (lo, hi) in enumerate(boxes):
            for a, b in boxes[:i]:
                assert (hi[:2] + .003 < a[:2]).any() or (b[:2] + .003 < lo[:2]).any()
        if entry['layout'] == 'S2_edge':
            assert .087 <= boxes[0][0][0] + .7245 <= .113
        q = entry['admission']
        assert q['visible_pixels'] >= 64 and q['valid_target_depth_fraction'] >= .98
        assert (.20 <= q['occlusion_fraction'] <= .40) if entry['layout'] == 'S5_occluded' else q['occlusion_fraction'] <= .15
        for name, digest in admission['camera_input_sha256'].items():
            assert sha(folder / 'camera' / name) == digest
        native = np.array(Image.open(folder / 'target_gt_native.png')) > 0
        assert int(native.sum()) == q['visible_pixels']
        depth = np.array(Image.open(folder / 'camera/depth.png')) * .0001
        mask = np.array(Image.open(folder / 'target_gt.png')) > 0
        valid = np.isfinite(depth) & (depth > .10) & (depth < 2.)
        assert float((mask & valid).sum() / mask.sum()) >= .98
        certificate = workspace['preparation_pose_ik']
        assert certificate['passed'] and certificate['joint_limits_passed'] and certificate['table_collision_passed']
        measured = dict(zip(state['joint_names'], state['qpos']))
        measured_fk = fk(chain, state['links']['body4'], measured)
        assert np.linalg.norm(measured_fk[:3, 3] - np.array(state['links']['right_eef_link'])[:3, 3]) < 2e-5
        for qname, goalname in [('q_goal', 'goal_world'), ('q_preparation', 'preparation_world'), ('q_lift', 'lift_world')]:
            joints = dict(measured)
            for name, angle in zip(certificate['joint_names'], certificate[qname]):
                limits = state['joint_limits'][state['joint_names'].index(name)]
                assert limits[0] <= angle <= limits[1]
                joints[name] = angle
            actual = fk(chain, state['links']['body4'], joints)
            expected = np.array(certificate[goalname])
            error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            rot_error = float(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude())
            max_fk_error, max_rot_error = max(max_fk_error, error), max(max_rot_error, rot_error)
            assert error <= .0022 and rot_error <= .031, (entry['id'], error, rot_error)
        labels[entry['layout']] += 1
        target_check = workspace['actors']['target']
        target = actors[0]
        previous = old['scene']['actors'][0]
        row = {'id': entry['id'], 'layout': entry['layout'], 'model': entry['model_id'],
               'old_x_m': previous['pose'][0], 'old_y_m': previous['pose'][1],
               'new_x_m': target['pose'][0], 'new_y_m': target['pose'][1],
               'new_center_distance_m': target_check['center_distance_m'],
               'placement_changed': workspace['placement_changed'], 'preparation_ik_passed': True,
               'visible_pixels': q['visible_pixels'], 'occlusion_fraction': q['occlusion_fraction']}
        rows.append(row)
        records.append({'id': entry['id'], 'passed': True, 'actors': len(actors), 'preparation_ik': True})
    assert labels == dict.fromkeys(['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter', 'S5_occluded'], 40)
    assert len({e['model_id'] for e in manifest['schedule']}) == 80
    assert Counter(e['semantic_class'] for e in manifest['schedule']) == dict.fromkeys(reg['semantic_classes'], 5)
    command = [str(PROJECT.parent / 'curobo/.venv/bin/python'), str(Path(__file__)), str(root), '--collision-only']
    with (root / 'independent_collision.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    collision = read(root / 'independent_collision_validation.json')
    assert collision['passed'] and collision['checked_scenes'] == 200
    with (root / 'placement_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'passed': True, 'scenes': 200, 'categories': 40, 'models': 80,
              'layouts': dict(labels), 'placement_changed': sum(r['placement_changed'] for r in rows),
              'maximum_reference_fk_position_error_m': max_fk_error,
              'maximum_reference_fk_orientation_error_rad': max_rot_error,
              'all_actor_center_limit_m': .70, 'physical_grasps': 0,
              'reference_ik_and_table_collision_passed': 200,
              'old_dataset_unchanged': True, 'records': records,
              'verifier_source_sha256': sha(Path(__file__))}
    save(root / 'verification.json', result)
    publish(root, reg, rows, result)
    print('VERIFIED', json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


def publish(root, reg, rows, result):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle, Rectangle
    font = PROJECT.parent / 'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'
    matplotlib.font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    docs = PROJECT / 'docs/diverse_object_scenes'
    resource = docs / 'resource'
    resource.mkdir(exist_ok=True)
    pivot = np.array(reg['shoulder_pivot_world_m'])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.add_patch(Rectangle((-.7245, -1.209), 1.209, 2.418, color='#f2e6d4', ec='#9c7a4d'))
    ax.add_patch(Circle(pivot[:2], reg['eef_radius_upper_bound_m'], color='#d5e9fd', ec='#437eb2', alpha=.6))
    ax.scatter([r['old_x_m'] for r in rows], [r['old_y_m'] for r in rows], s=20, color='#cf5644', alpha=.6, label='原位置')
    ax.scatter([r['new_x_m'] for r in rows], [r['new_y_m'] for r in rows], s=20, color='#176bb3', alpha=.7, label='修正后位置')
    ax.scatter(*pivot[:2], marker='*', s=200, color='#153754', label='右臂肩部')
    ax.set(xlim=(-1.15, .60), ylim=(-1.26, 1.26), xlabel='世界坐标 X（米）', ylabel='世界坐标 Y（米）', title='200 个目标的位置：修正前后')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(resource / 'workspace_before_after.png')
    plt.close(fig)
    labels = {'S1_open': '开阔桌面', 'S2_edge': '靠近桌边', 'S3_sparse': '稀疏混放', 'S4_clutter': '中度杂乱', 'S5_occluded': '部分遮挡'}
    lines = ['# 场景位置修正', '', '200 个场景已逐个检查并调整。保留原40类、80个型号，每类在五种布局中各出现一次；每例的目标与干扰物型号保持原样。', '',
             '机器人升降固定45.2厘米，底盘、桌面、头部D405角度、相机参数和物体物理配置保持原样。所有物体中心距右臂肩部不超过70厘米，并完整处于头部视野内。', '',
             '每个目标上方8厘米的末端准备点、再上方8厘米的位置和向上3厘米的位置，都有满足关节限位及桌面碰撞检查的静态IK解。场景核验期间机械臂保持初始姿态，物理抓取为0次。', '',
             '桌边组移到靠机器人一侧的真实桌沿，物体外缘距桌沿9–11厘米。原1.5–3.5厘米位置无法被当前头部相机完整看到；远端桌沿则超出机械臂工作范围。', '',
             '## 检查结果', '', '| 布局 | 场景数 | 调整位置 | 准备点IK与桌面碰撞通过 | 头部观察通过 |', '| --- | ---: | ---: | ---: | ---: |']
    for layout, label in labels.items():
        subset = [r for r in rows if r['layout'] == layout]
        lines.append(f'| {label} | {len(subset)} | {sum(r["placement_changed"] for r in subset)} | 40 | 40 |')
    lines += ['', f'参考姿态最大位置误差：{result["maximum_reference_fk_position_error_m"] * 1000:.2f}毫米；最大朝向误差：{np.degrees(result["maximum_reference_fk_orientation_error_rad"]):.2f}°。', '',
              '![位置对比](resource/workspace_before_after.png)', '', '## 逐组截图', '']
    for layout, label in labels.items():
        name = 'workspace_' + layout + '.jpg'
        shutil.copy2(root / 'gallery' / (layout + '.jpg'), resource / name)
        lines += [f'### {label}', '', f'![{label}40例](resource/{name})', '']
    book = root / 'cases/D0043_S2_edge_book'
    shutil.copy2(book / 'overview.jpg', resource / 'workspace_book_overview.jpg')
    lines += ['## 书本例子', '', '此前4832个抓点全部越过伸展上限的书本已移到机器人近侧桌边。', '',
              '![书本修正后的摆放](resource/workspace_book_overview.jpg)', '',
              '## 文件位置', '', f'- 新数据集：`{root}`',
              '- [逐例位置对照表](resource/workspace_placement_comparison.csv)',
              '- [逐例核查记录](resource/workspace_verification.json)', '',
              '原场景与已停止测试的结果保留。新的抓取测试保持停止。']
    shutil.copy2(root / 'placement_comparison.csv', resource / 'workspace_placement_comparison.csv')
    shutil.copy2(root / 'verification.json', resource / 'workspace_verification.json')
    (docs / '场景位置修正.md').write_text('\n'.join(lines) + '\n')
    main = docs / '多类别物体与场景.md'
    text = main.read_text()
    text = text.replace('# 多类别物体与场景\n', '# 多类别物体与场景\n\n当前使用逐例修正并核验后的200个场景，详见[场景位置修正](场景位置修正.md)。原批次留档，抓取测试保持停止。\n', 1)
    text = text.replace('场景构造不使用SAM3、抓取模型、IK或抓取成败筛选。',
                        '场景构造加入工作范围、关节限位与桌面碰撞的静态IK检查；保留原物体与布局。')
    text = text.replace(str(Path(reg['source_dataset'])), str(root))
    for layout in labels:
        text = text.replace('resource/' + layout + '.jpg', 'resource/workspace_' + layout + '.jpg')
    main.write_text(text)
    active = {'dataset': str(root), 'object_catalog': str(root / 'object_catalog.json'),
              'environment_id': 'ARXPickCatalog-v1', 'adapter': str(HERE / 'asset_environment.py'),
              'runtime_factory': str(HERE / 'pipeline.py') + ':runtime',
              'scenes': 200, 'categories': 40, 'models': 80, 'state': 'verified_workspace_relocation',
              'grasp_tests_authorized': False, 'grasp_test_budget': 0,
              'verification': str(root / 'verification.json'), 'source_dataset': reg['source_dataset']}
    save(HERE / 'active_dataset.json', active)
    save(PROJECT / 'configs/scene_dataset.json', active)
    save(root / 'progress.json', {'phase': 'verified', 'completed': 200, 'registered': 200,
                                'physical_grasps': 0, 'grasp_tests_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--collision-only', action='store_true')
    args = parser.parse_args()
    if args.collision_only:
        collision_check(args.root.resolve())
    else:
        verify(args.root.resolve())
