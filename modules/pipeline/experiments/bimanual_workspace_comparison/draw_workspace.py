"""Render actual robot visuals, tabletop and measured high-density voxel regions."""
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from arx_project.assets import resource_path
from arx_project.layout import ROOT, read_json, write_json, sha256
from recommend_workspace import OUTPUT, STATE

RIGHT = '#3689c9'
LEFT = '#e49a36'
SHARED = '#9565be'
CORE = '#169969'
CUBE = np.asarray(list(product([0., 1.], repeat=3)))
FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def origin(element):
    frame = np.eye(4)
    if element is not None:
        frame[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        frame[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return frame


def box_quads(lower, upper):
    vertices = np.asarray(lower)+CUBE*(np.asarray(upper)-lower)
    return vertices[np.asarray(FACES)]


def triangles(quads):
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def shell(cells, step, height_offset):
    occupied = {tuple(cell) for cell in cells}
    quads = []
    for cell in cells:
        base = np.asarray(cell)*step
        base[2] += height_offset
        vertices = base+CUBE*step
        for index, (axis, direction) in enumerate([(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]):
            neighbor = list(cell); neighbor[axis] += direction
            if tuple(neighbor) not in occupied:
                quads.append(vertices[FACES[index]])
    return triangles(np.asarray(quads)) if quads else np.empty((0, 3, 3))


def robot_meshes(state, workspace):
    asset = resource_path('arx_robot')
    tree = ET.parse(asset/'robot.urdf').getroot()
    parents = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
    q = dict(zip(state['joint_names'], state['qpos']))
    cache = {'base_link': np.asarray(state['links']['base_link'])}

    def frame(link):
        if link in cache:
            return cache[link]
        joint = parents[link]
        move = np.eye(4)
        if joint.get('type') in ['revolute', 'continuous', 'prismatic']:
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            value = q.get(joint.get('name'), 0.)
            if joint.get('type') == 'prismatic':
                move[:3, 3] = value*axis
            else:
                move[:3, :3] = Rotation.from_rotvec(axis*value).as_matrix()
        cache[link] = frame(joint.find('parent').get('link'))@origin(joint.find('origin'))@move
        return cache[link]

    body = np.asarray(workspace['body4_world'])
    if not np.allclose(frame('body4'), body, atol=1e-5):
        raise ValueError('Rendered robot FK differs from the measured body4 frame')
    parts = []; sources = {}; original_faces = rendered_faces = 0
    for link in tree.findall('link'):
        name = link.get('name')
        for visual in link.findall('visual'):
            element = visual.find('geometry/mesh')
            if element is None:
                continue
            path = asset/element.get('filename')
            mesh = trimesh.load_mesh(path, process=False)
            vertices = np.asarray(mesh.vertices)*np.fromstring(element.get('scale', '1 1 1'), sep=' ')
            original_faces += len(mesh.faces)
            # Display-only vertex clustering. The source robot meshes stay intact.
            spacing = 0.0025 if name.startswith(('left_', 'right_')) else 0.006
            _, inverse = np.unique(np.round(vertices/spacing).astype(np.int64), axis=0, return_inverse=True)
            count = np.bincount(inverse)
            reduced = np.column_stack([np.bincount(inverse, weights=vertices[:, axis])/count for axis in range(3)])
            faces = inverse[mesh.faces]
            faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])]
            _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
            faces = faces[keep]
            transform = frame(name)@origin(visual.find('origin'))
            world = reduced@transform[:3, :3].T+transform[:3, 3]
            local = (world-body[:3, 3])@body[:3, :3]
            local[:, 2] += body[2, 3]-workspace['table_height_world_m']
            color = RIGHT if name.startswith('right_') else LEFT if name.startswith('left_') else '#686d75'
            parts.append((local[faces], color, 1.0))
            rendered_faces += len(faces)
            sources[str(path.relative_to(ROOT))] = sha256(path)
    return parts, {'original_visual_triangles': original_faces, 'display_triangles': rendered_faces,
                   'source_visual_mesh_sha256': sources}


def projected(ax, parts, matrix, title):
    faces = np.concatenate([part[0] for part in parts])
    color_groups = []
    light = np.array([0.4, -0.6, 0.8]); light /= np.linalg.norm(light)
    for polygons, color, alpha in parts:
        rgba = np.tile(matplotlib.colors.to_rgba(color, alpha), (len(polygons), 1))
        if alpha == 1.0:
            normal = np.cross(polygons[:, 1]-polygons[:, 0], polygons[:, 2]-polygons[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            rgba[:, :3] *= (0.60+0.40*np.abs(normal@light))[:, None]
        color_groups.append(rgba)
    colors = np.concatenate(color_groups)
    centers = faces.mean(axis=1)
    order = np.argsort(centers@matrix[:, 2], kind='stable')
    xy = faces@matrix[:, :2]
    ax.add_collection(PolyCollection(xy[order], facecolors=colors[order], edgecolors='none', rasterized=True))
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_facecolor('#f7f9fc')
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.13)
    return ax


def main():
    workspace = read_json(OUTPUT/'workspace.json')
    state = read_json(STATE)
    atlas = ROOT/'tmp/reachability/products/前方精细_20261121_01'
    with np.load(atlas/'概览体素统计.npz', allow_pickle=False) as saved:
        cells, counts = saved['cells'].astype(int), saved['sample_count']
    height_offset = workspace['body4_world'][2][3]-workspace['table_height_world_m']
    height = (cells[:, 2]+0.5)*0.05+height_offset
    visible = (height >= 0.02) & (height <= 0.55)
    threshold = float(np.quantile(counts[visible], 0.75))
    right = {tuple(cell) for cell in cells[visible & (counts >= threshold)]}
    left = {tuple(np.asarray(cell)+[0, 10, 0]) for cell in right}
    common = right & left
    parts, proof = robot_meshes(state, workspace)
    for region, color in [(right-common, RIGHT), (left-common, LEFT), (common, SHARED)]:
        parts.append((shell(np.asarray(sorted(region)), 0.05, height_offset), color, 0.21))
    body = np.asarray(workspace['body4_world'])
    box = workspace['table_geometry_world']
    table_low = np.asarray(box['center'])-np.asarray(box['dims'])/2
    table_high = np.asarray(box['center'])+np.asarray(box['dims'])/2
    table_low[:2] -= body[:2, 3]; table_high[:2] -= body[:2, 3]
    table_low[2] -= workspace['table_height_world_m']; table_high[2] -= workspace['table_height_world_m']
    parts.append((triangles(box_quads(table_low, table_high)), '#c69860', 0.14))
    lower, upper = np.asarray(workspace['lower_body4_m']), np.asarray(workspace['upper_body4_m'])
    lower[2] += height_offset; upper[2] += height_offset
    core = box_quads(lower, upper)
    parts.append((triangles(core), CORE, 0.19))
    font = ROOT/'tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    eye = np.array([1., -1.5, 1.0]); eye /= np.linalg.norm(eye)
    horizontal = np.cross([0., 0., 1.], eye); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(eye, horizontal)
    views = [('斜视图', np.column_stack([horizontal, vertical, eye]), 'oblique', '投影横向（米）', '投影纵向（米）'),
        ('俯视图', np.eye(3), 'top', '机身前方 X（米）', '机身左方 Y（米）'),
        ('正视图', np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), 'front', '机身左方 Y（米）', '高于桌面 Z（米）'),
        ('侧视图', np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), 'side', '机身前方 X（米）', '高于桌面 Z（米）')]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, (title, matrix, label, xlabel, ylabel) in zip(axes.flat, views):
        projected(ax, parts, matrix, title)
        for face in core:
            closed = np.vstack([face, face[0]])@matrix[:, :2]
            ax.plot(closed[:, 0], closed[:, 1], color=CORE, linewidth=1.6)
        if label == 'top':
            ax.text(0.34, -0.23, '推荐框 20×30厘米', color='#08724d', fontsize=11)
        if label == 'side':
            ax.text(0.62, 0.13, 'TCP高于桌面\n2～30厘米', color='#08724d', fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    handles = [Patch(facecolor=RIGHT, alpha=0.6, label='右臂高密度'),
               Patch(facecolor=LEFT, alpha=0.6, label='左臂高密度'),
               Patch(facecolor=SHARED, alpha=0.6, label='双臂共同高密度'),
               Patch(facecolor=CORE, edgecolor=CORE, alpha=0.5, label='推荐TCP操作空间'),
               Patch(facecolor='#c69860', alpha=0.5, label='当前桌子')]
    fig.suptitle('ARX双臂高可达区域与推荐操作空间', fontsize=19)
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.012, '升降45.2厘米 · TCP43 · 高密度取桌面上方2～55厘米内采样数前25%的5厘米格', ha='center', fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, hspace=0.26, wspace=0.18)
    fig.savefig(OUTPUT/'workspace_multiview.png', dpi=170)
    for text in fig.texts:
        text.set_visible(False)
    for legend in fig.legends:
        legend.set_visible(False)
    for ax, (_, _, label, _, _) in zip(axes.flat, views):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(OUTPUT/f'workspace_{label}.png', dpi=170, bbox_inches=bbox.expanded(1.06, 1.06))
    plt.close(fig)
    write_json(OUTPUT/'figure_provenance.json', {**proof, 'high_density_threshold_samples': threshold,
        'display_height_above_table_m': [0.02, 0.55], 'right_high_cells': len(right),
        'left_high_cells': len(left), 'both_high_cells': len(common), 'physical_attempts': 0,
        'sources': {str(path.relative_to(ROOT)): sha256(path) for path in [STATE, OUTPUT/'workspace.json',
            atlas/'概览体素统计.npz', resource_path('arx_robot')/'robot.urdf', Path(__file__)]}})
    print('Workspace figures saved', OUTPUT/'workspace_multiview.png', flush=True)
    print('Visual triangles', proof['original_visual_triangles'], 'display', proof['display_triangles'], flush=True)


if __name__ == '__main__':
    main()
