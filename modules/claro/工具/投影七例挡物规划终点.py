"""Project archived CLARO endpoint goals onto original RGB and video frames, on CPU."""
from pathlib import Path
import hashlib
import json
import math
from functools import lru_cache
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/06_做配对验证/resources'
DIAG = ROOT / 'docs/03_夹走并放下挡物/规划失败核查/resources/20261119_01'
OUT = BASE / '七例规划终点投影_20261120'
URDF = ROOT.parent / 'maniskill/assets/arx/robot.urdf'
FONT = ROOT / '缓存/对比视频字体/NotoSansCJKsc-Regular.otf'
CASES = ['I0015', 'I0018', 'I0038', 'I0044', 'I0063', 'I0119', 'I0130']
COLORS = {'G': (245, 127, 28), 'H': (40, 154, 246), 'L': (201, 98, 247),
          'C': (255, 255, 255), 'S': (213, 222, 234)}
AXES = [(244, 65, 65), (68, 211, 104), (69, 139, 255)]
SOURCES = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    if str(path) not in SOURCES:
        SOURCES[str(path)] = digest(path)
    return path


def read(path):
    return json.loads(source(path).read_text())


@lru_cache(maxsize=32)
def font(size):
    return ImageFont.truetype(str(FONT), size)


def text(image, xy, value, size=22, fill=(34, 50, 73), stroke=False):
    ImageDraw.Draw(image).text(xy, value, font=font(size), fill=fill,
                              stroke_width=2 if stroke else 0,
                              stroke_fill=(13, 21, 33) if stroke else None)


def last_frame(path):
    source(path)
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert cap.isOpened() and count > 0, path
    cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
    ok, rgb = cap.read()
    cap.release()
    assert ok, path
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), count - 1, count


def project(points, camera, intrinsics):
    points = np.asarray(points).reshape(-1, 3)
    optical = (np.linalg.inv(camera) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    pixels = optical @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:3], optical[:, 2]


def third_camera(state, definition):
    base = np.asarray(state['links']['base_link'])[:3, 3]
    eye = base + definition['eye_local']
    target = base + definition['target_local']
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack([right, down, forward])
    camera[:3, 3] = eye
    width, height = definition['width'], definition['height']
    focal = height / (2 * np.tan(definition['vertical_fov_rad'] / 2))
    return camera, np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])


def goal_poses(candidate):
    contact = np.array(candidate['pose'])
    grasp = contact.copy()
    grasp[:3, 3] -= .043 * contact[:3, 2]
    pre = grasp.copy()
    pre[:3, 3] -= .08 * contact[:3, 2]
    lift = grasp.copy()
    lift[2, 3] += .03
    return {'G': grasp, 'H': pre, 'L': lift, 'C': contact}


def meshes_in_eef(state):
    root = ET.parse(source(URDF)).getroot()
    frames = {'base_link': np.array(state['links']['base_link'])}
    qs = dict(zip(state['joint_names'], state['qpos']))
    joints = root.findall('joint')
    for _ in range(len(joints)):
        added = False
        for joint in joints:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in frames or child in frames:
                continue
            origin = joint.find('origin')
            t, motion = np.eye(4), np.eye(4)
            if origin is not None:
                t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                q = qs[joint.get('name')]
                if joint.get('type') == 'prismatic': motion[:3, 3] = q * axis
                else: motion[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
            frames[child] = frames[parent] @ t @ motion
            added = True
        if not added: break
    fk_error = float(np.abs(frames['right_eef_link'] - state['links']['right_eef_link']).max())
    assert fk_error < 2e-5, fk_error
    result = []
    for name in ['right_link26', 'right_link27', 'right_link28']:
        visual = root.find(f"link[@name='{name}']/visual")
        item = visual.find('geometry/mesh')
        path = source(URDF.parent / item.get('filename'))
        mesh = trimesh.load(path, force='mesh', process=False)
        transform = np.eye(4)
        origin = visual.find('origin')
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        scale = np.fromstring(item.get('scale', '1 1 1'), sep=' ')
        local = np.linalg.inv(state['links']['right_eef_link']) @ frames[name] @ transform
        vertices = (np.c_[np.asarray(mesh.vertices) * scale, np.ones(len(mesh.vertices))] @ local.T)[:, :3]
        result.append((name, vertices, np.asarray(mesh.faces)))
    for joint in ['right_joint27', 'right_joint28']:
        assert abs(state['qpos'][state['joint_names'].index(joint)] - .044) < 1e-5
    return result, fk_error


def ghost_mesh(image, pose, geometry, camera, intrinsics, color, opacity=.065, width=2):
    # A projected visual-mesh silhouette; includes occluded portions and is not a collision certificate.
    h, w = image.shape[:2]
    for name, vertices, faces in geometry:
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        pixels, z = project(world, camera, intrinsics)
        valid = (z[faces] > .02).all(axis=1)
        polygons = np.round(np.clip(pixels[faces[valid]], -50000, 50000)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, polygons, 255)
        selected = mask > 0
        image[selected] = np.round(image[selected] * (1 - opacity) + np.array(color) * opacity).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, width, cv2.LINE_AA)


def dashed(image, start, end, color, width=2):
    start, end = np.asarray(start), np.asarray(end)
    length = np.linalg.norm(end - start)
    for d in np.arange(0., length, 12.):
        a = start + (end - start) * d / max(length, 1.)
        b = start + (end - start) * min(d + 7., length) / max(length, 1.)
        cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, width, cv2.LINE_AA)


def overlay(raw, camera, k, state, goals, geometry, queried, blocker_mesh, wrist_mask=None):
    image = raw.copy()
    h, w = image.shape[:2]
    width = max(2, round(w / 640 * 2))
    if wrist_mask is not None:
        contours, _ = cv2.findContours(wrist_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 218, 70), width, cv2.LINE_AA)
    else:
        actor = state['scene_meshes']['distractor_00']
        t = np.asarray(actor['world'])
        pixels, z = project(blocker_mesh @ t[:3, :3].T + t[:3, 3], camera, k)
        hull = cv2.convexHull(np.round(pixels[z > .02]).astype(np.int32))
        cv2.polylines(image, [hull], True, (255, 218, 70), width, cv2.LINE_AA)
    if 'H' in queried:
        ghost_mesh(image, goals['H'], geometry, camera, k, COLORS['H'], opacity=.035, width=width)
    ghost_mesh(image, goals['G'], geometry, camera, k, COLORS['G'], width=width)

    positions, depths = {}, {}
    for name, pose in {**goals, 'S': np.asarray(state['links']['right_eef_link'])}.items():
        uv, z = project([pose[:3, 3]], camera, k)
        positions[name] = uv[0]
        depths[name] = float(z[0])
    for a, b, color in [('H', 'G', COLORS['H']), ('G', 'L', COLORS['L']), ('G', 'C', COLORS['G'])]:
        if a in queried | {'C'} and b in queried | {'C'} and min(depths[a], depths[b]) > .02:
            dashed(image, positions[a], positions[b], color, width)
    origin = goals['G'][:3, 3]
    for index, color in enumerate(AXES):
        uv, z = project([origin, origin + .027 * goals['G'][:3, index]], camera, k)
        if z.min() > .02:
            cv2.arrowedLine(image, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), color, width,
                           cv2.LINE_AA, tipLength=.18)
    info = {}
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    offsets = {'G': (18, 12), 'H': (15, -30), 'L': (23, -36), 'C': (-65, -10), 'S': (10, 8)}
    ratio = w / 640
    for name in ['S', 'H', 'L', 'G', 'C']:
        uv = positions[name]
        visible = depths[name] > .02 and 0 <= uv[0] < w and 0 <= uv[1] < h
        info[name] = {'pixel_xy': uv.tolist(), 'camera_z_m': depths[name], 'inside_image': bool(visible),
                      'queried': name in queried}
        if name not in queried | {'C', 'S'} or depths[name] <= .02:
            continue
        if name == 'S' and wrist_mask is not None:
            continue
        point = np.clip(uv, [15 * ratio, 15 * ratio], [w - 17 * ratio, h - 17 * ratio])
        radius = 5 * ratio
        color = COLORS[name]
        draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                     fill=color, outline=(20, 27, 42), width=max(1, round(ratio)))
        label = name + ('：视野外' if not visible else '')
        if wrist_mask is None:
            # Callouts avoid overlapping labels when G/C/L project onto the same small object.
            callouts = {'G': (150, 10), 'H': (150, -100), 'L': (-165, -100), 'C': (-165, 95)}
            label_point = positions['G'] + np.asarray(callouts[name]) if name in callouts else point + [55, -80]
            tx, ty = label_point
            label_size = 36
            end = (tx + 10, ty + label_size / 2)
            draw.line([tuple(point), end], fill=color, width=3)
        else:
            dx, dy = offsets[name]
            tx = min(max(4, point[0] + dx), w - 130) if not visible else min(max(4, point[0] + dx), w - 28)
            ty = min(max(3, point[1] + dy), h - 28)
            label_size = 20
            if visible:
                draw.line([tuple(point), (tx + 8, ty + 10)], fill=color, width=1)
        text(pil, (tx, ty), label, size=label_size, fill=color, stroke=True)
    return pil, info


def status_line(events, index):
    by = {row['stage']: row for row in events if row['candidate'] == index}
    labels = []
    queried = set()
    for stage, symbol in [('grasp', 'G'), ('pregrasp', 'H'), ('lift', 'L')]:
        if stage in by:
            queried.add(symbol)
            passed = any(by[stage]['solver_success'])
            labels.append(f'{symbol}：' + ('复核找到解' if passed else '复核未找到解'))
        else:
            labels.append(f'{symbol}：未进入求解')
    return '；'.join(labels), queried


def pair_figure(case, index, count, blocker_name, line, wrist, third):
    height, margin, gap, top, bottom = 600, 24, 24, 160, 150
    left_width, right_width = 800, round(third.width * height / third.height)
    canvas = Image.new('RGB', (left_width + right_width + 2 * margin + gap, top + height + bottom), (249, 251, 255))
    text(canvas, (margin, 16), f'{case}  挡物：{blocker_name}  |  候选 #{index} / 原池 {count} 个（编号从 0 开始）', 28)
    text(canvas, (margin, 58), line + '  |  未执行；不是已选中的可执行方案', 22)
    text(canvas, (margin, 94), '分步状态来自 11 月 19 日保存的原求解器复核；目标位姿与原候选及原协议逐项核对。', 18, (97, 113, 134))
    text(canvas, (margin, 127), '腕部原图 · 黄色轮廓是可见挡物实例', 20)
    text(canvas, (margin + left_width + gap, 127), '第三人称原录像末帧 · 黄色轮廓是挡物网格投影', 20)
    canvas.paste(wrist.resize((left_width, height), Image.Resampling.LANCZOS), (margin, top))
    canvas.paste(third.resize((right_width, height), Image.Resampling.LANCZOS), (margin + left_width + gap, top))
    y = top + height + 15
    for x, name, desc in [(24, 'G', '抓取 EEF 目标'), (370, 'H', '预抓取 EEF 目标'), (760, 'L', '名义抬升 EEF 目标'),
                          (1240, 'C', '动作 TCP 中心'), (1630, 'S', '实际初态 EEF')]:
        color = COLORS[name] if name not in {'C', 'S'} else (81, 91, 112)
        text(canvas, (x, y), name + '  ' + desc, 21, color)
    text(canvas, (24, y + 37), '原协议：H 沿接近轴后退 8 cm；L 沿世界 Z 抬升 3 cm；C = G 的局部 +Z 方向 43 mm。', 20)
    text(canvas, (24, y + 70), '橙／蓝外形为真实 ARX 开爪视觉网格的目标投影（含遮挡部分），不是实际到位或碰撞通过的证明。', 20)
    text(canvas, (24, y + 103), '红／绿／蓝短轴分别为 G 的局部 X／Y／Z；虚线仅连接目标点，不代表规划轨迹。未求解的 H／L 不绘制。', 18, (97, 113, 134))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate_path = BASE / 'S5_固定挡物接续01/配对合并验收.json'
    rows = read(aggregate_path)['rows']
    source(ROOT.parent / 'maniskill/short_lift_planner.py')
    source(__file__)
    summary = {'scope': 'CPU-only projection of archived original scenes and candidates; no simulator, IK or physical execution.',
               'physical_attempts': 0, 'new_ik_queries': 0, 'cases': []}
    for case in CASES:
        row = next(row for row in rows if row['case'] == case)
        directory = BASE / row['run_directory']
        grasp_dir = directory / '01_抓挡'
        state = read(grasp_dir / 'state.json')
        observed = read(directory / '00_进场/observed_state.json')
        assert state['qpos'] == observed['qpos'] and state['objects'] == observed['objects']
        result = read(grasp_dir / 'plan_result.json')
        assert not result['success'] and 'selected' not in result
        pool = read(grasp_dir / 'candidates/candidate_pool.json')['candidates']
        assert len(pool) == result['count']
        events = read(DIAG / case / '原求解器分步记录.json')['events']
        valid_grasps = [event['candidate'] for event in events if event['stage'] == 'grasp' and any(event['solver_success'])]
        primary = valid_grasps[0] if valid_grasps else 0
        selected_policy = ('分步复核中首个 G 通过的原候选' if valid_grasps else '全部 G 未通过，展示原池最高优先级候选')
        wrist_path = source(grasp_dir / 'camera/color.png')
        wrist = np.array(Image.open(wrist_path).convert('RGB'))
        mask = np.array(Image.open(source(grasp_dir / 'camera/mask.png'))) > 0
        third, frame_index, frame_count = last_frame(directory / '00_进场/execution.mp4')
        wrist_video, wrist_index, wrist_count = last_frame(directory / '00_进场/right_wrist.mp4')
        assert frame_count == wrist_count
        mae = float(np.abs(wrist.astype(float) - wrist_video).mean())
        assert mae < 5., (case, mae)
        camera_definition = read(directory / '00_进场/recording_camera.json')
        third_t, third_k = third_camera(state, camera_definition)
        assert third.shape[1] == camera_definition['width'] and third.shape[0] == camera_definition['height']
        wrist_t, wrist_k = np.array(state['camera_world_optical']), np.array(state['intrinsics'])
        geometry, fk_error = meshes_in_eef(state)
        actor = state['scene_meshes']['distractor_00']
        mesh_path = source(actor['file_path'])
        assert digest(mesh_path) == actor['file_sha256']
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        actor_t = np.asarray(actor['world'])
        hull_pixels, hull_z = project(np.asarray(mesh.vertices) @ actor_t[:3, :3].T + actor_t[:3, 3], wrist_t, wrist_k)
        silhouette = np.zeros(mask.shape, np.uint8)
        hull = cv2.convexHull(np.round(hull_pixels[hull_z > .02]).astype(np.int32))
        cv2.fillConvexPoly(silhouette, hull, 1)
        coverage = float((silhouette.astype(bool) & mask).sum() / mask.sum())
        assert coverage > .98, (case, coverage)
        info = {'case': case, 'source_trial': str(directory), 'blocker_model': row['blocker_model'],
                'target_model': row['target_model'], 'original_candidate_count': len(pool), 'original_selected': None,
                'representative_candidate_index': primary, 'representative_policy': selected_policy,
                'original_rejections': result['rejected'], 'third_frame_index': frame_index,
                'wrist_video_frame_index': wrist_index, 'video_frame_counts_equal': True,
                'observed_and_planning_qpos_objects_exactly_equal': True,
                'wrist_video_vs_raw_rgb_mean_absolute_difference_8bit': mae,
                'independent_robot_fk_max_element_error': fk_error,
                'visible_wrist_instance_coverage_by_archived_mesh_hull': coverage,
                'wrist_camera_world_optical': wrist_t.tolist(), 'wrist_intrinsics': wrist_k.tolist(),
                'third_camera_world_optical': third_t.tolist(), 'third_intrinsics': third_k.tolist(),
                'third_camera_source': camera_definition, 'candidates': []}
        figures = []
        for index, candidate in enumerate(pool):
            goals = goal_poses(candidate)
            line, queried = status_line(events, index)
            for event in [e for e in events if e['candidate'] == index]:
                symbol = {'grasp': 'G', 'pregrasp': 'H', 'lift': 'L'}[event['stage']]
                assert np.max(np.abs(goals[symbol] - event['world_goal'])) < 1e-10
            wrist_image, wrist_pixels = overlay(wrist, wrist_t, wrist_k, state, goals, geometry, queried, mesh.vertices, mask)
            third_image, third_pixels = overlay(third, third_t, third_k, state, goals, geometry, queried, mesh.vertices)
            name = '罐头' if row['blocker_model'] == '002_master_chef_can' else '木块'
            figure = pair_figure(case, index, len(pool), name, line, wrist_image, third_image)
            filename = f'{case}_候选{index:02d}.jpg'
            figure.save(OUT / filename, quality=92, optimize=True, subsampling=0)
            figures.append(figure)
            info['candidates'].append({'index': index, 'score': candidate['score'], 'status': line, 'queried': sorted(queried),
                'world_goals_eef': {key: value.tolist() for key, value in goals.items()},
                'wrist_projection': wrist_pixels, 'third_projection': third_pixels, 'image': filename})
            if index == primary:
                preview = figure.copy()
                preview.thumbnail((1120, 600), Image.Resampling.LANCZOS)
                preview.save(OUT / f'{case}_缩略图.jpg', quality=85, optimize=True)
        cell_w = 1280
        cell_h = round(figures[0].height * cell_w / figures[0].width)
        sheet = Image.new('RGB', (cell_w * 2, cell_h * math.ceil(len(figures) / 2)), (233, 238, 247))
        for index, figure in enumerate(figures):
            sheet.paste(figure.resize((cell_w, cell_h), Image.Resampling.LANCZOS), ((index % 2) * cell_w, (index // 2) * cell_h))
        sheet.save(OUT / f'{case}_全部候选.jpg', quality=92, optimize=True, subsampling=0)
        summary['cases'].append(info)
        print(case, len(pool), 'representative', primary, 'wrist_video_mae', round(mae, 4), flush=True)
    summary['source_sha256'] = SOURCES
    summary['source_hashes_unchanged'] = all(digest(path) == expected for path, expected in SOURCES.items())
    assert summary['source_hashes_unchanged']
    summary['total_original_candidates'] = sum(case['original_candidate_count'] for case in summary['cases'])
    (OUT / '投影依据.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
