"""Render the intrinsic body-relative workspace, retaining a reference table."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from arx_project.assets import resource_path
from arx_project.layout import ROOT, read_json, write_json, sha256
from intrinsic_workspace import OUTPUT, ATLAS, STATE
from draw_workspace import (RIGHT, LEFT, SHARED, CORE, robot_meshes, shell,
                            triangles, box_quads, projected)


def main():
    selected = read_json(OUTPUT/'workspace.json')
    state = read_json(STATE)
    body = np.asarray(state['links']['body4'])
    table = state['scene_cuboids']['table']
    table_top = table['center'][2]+table['dims'][2]/2
    # Existing mesh reader returns Z above the table. Convert it to body4 Z;
    # all region selection has already completed without table/camera inputs.
    reference = {'body4_world': body.tolist(), 'table_height_world_m': table_top}
    parts, mesh_proof = robot_meshes(state, reference)
    for vertices, _, _ in parts:
        vertices[..., 2] -= body[2, 3]-table_top
    with np.load(ATLAS/'概览体素统计.npz', allow_pickle=False) as source:
        cells = source['cells'].astype(int)
        counts = source['sample_count']
    threshold = float(np.quantile(counts, .75))
    right = {tuple(cell) for cell in cells[counts >= threshold]}
    left = {tuple(np.asarray(cell)+[0, 10, 0]) for cell in right}
    common = right & left
    for region, color in [(right-common, RIGHT), (left-common, LEFT), (common, SHARED)]:
        parts.append((shell(np.asarray(sorted(region)), .05, 0.), color, .19))
    lower_world = np.asarray(table['center'])-np.asarray(table['dims'])/2
    upper_world = np.asarray(table['center'])+np.asarray(table['dims'])/2
    table_vertices = triangles(box_quads(lower_world, upper_world))
    table_vertices = (table_vertices-body[:3, 3])@body[:3, :3]
    parts.append((table_vertices, '#c69860', .10))
    core = selected['regions']['shared']
    core_faces = box_quads(core['lower_body4_m'], core['upper_body4_m'])
    parts.append((triangles(core_faces), CORE, .23))
    font = ROOT/'tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    eye = np.array([1., -1.5, 1.0]); eye /= np.linalg.norm(eye)
    horizontal = np.cross([0., 0., 1.], eye); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(eye, horizontal)
    views = [('斜视图', np.column_stack([horizontal, vertical, eye]), 'oblique', '投影横向（米）', '投影纵向（米）'),
        ('俯视图', np.eye(3), 'top', 'body4前方 X（米）', 'body4左方 Y（米）'),
        ('正视图', np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), 'front', 'body4左方 Y（米）', 'body4上下 Z（米）'),
        ('侧视图', np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), 'side', 'body4前方 X（米）', 'body4上下 Z（米）')]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    dims = np.rint(np.asarray(core['dimensions_m'])*100).astype(int)
    center = np.asarray(core['center_body4_m'])
    for ax, (title, matrix, label, xlabel, ylabel) in zip(axes.flat, views):
        projected(ax, parts, matrix, title)
        for face in core_faces:
            points = np.vstack([face, face[0]])@matrix[:, :2]
            ax.plot(points[:, 0], points[:, 1], color=CORE, linewidth=1.7)
        origin = np.zeros(3)@matrix[:, :2]
        ax.plot(origin[0], origin[1], marker='+', color='#292e36', markersize=8, markeredgewidth=1.5)
        if label in ['front', 'side']:
            ax.axhline(0., color='#536171', linestyle=':', linewidth=.8, alpha=.8)
        if label == 'top':
            ax.annotate(f'优先核心区 {dims[0]}×{dims[1]}厘米', xy=center[:2], xytext=(.58, .15),
                        fontsize=10, color='#08724d', arrowprops={'arrowstyle': '-', 'color': CORE})
        if label == 'side':
            ax.annotate(f'相对body4\nZ：{core["lower_body4_m"][2]*100:.0f}～{core["upper_body4_m"][2]*100:.0f}厘米',
                xy=center[[0, 2]], xytext=(.65, -.06), fontsize=10, color='#08724d',
                arrowprops={'arrowstyle': '-', 'color': CORE})
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    handles = [Patch(facecolor=RIGHT, alpha=.6, label='右臂高密度区'),
               Patch(facecolor=LEFT, alpha=.6, label='左臂高密度区'),
               Patch(facecolor=SHARED, alpha=.6, label='双臂共同高密度区'),
               Patch(facecolor=CORE, edgecolor=CORE, alpha=.6, label='双臂优先核心区'),
               Patch(facecolor='#c69860', alpha=.4, label='桌子（仅作位置参照）')]
    fig.suptitle('ARX双臂推荐操作空间 · 不按桌面或相机视野裁剪', fontsize=18)
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .035))
    fig.text(.5, .014, '坐标原点为升降躯干body4；区域随机器人位姿和升降一起变换 · 全前方地图 · TCP43',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.07, right=.98, top=.93, bottom=.14, hspace=.25, wspace=.18)
    fig.savefig(OUTPUT/'workspace_multiview.png', dpi=170)
    for text in fig.texts:
        text.set_visible(False)
    for legend in fig.legends:
        legend.set_visible(False)
    for ax, (_, _, label, _, _) in zip(axes.flat, views):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(OUTPUT/f'workspace_{label}.png', dpi=170, bbox_inches=bbox.expanded(1.06, 1.06))
    plt.close(fig)
    proof = {**mesh_proof, 'whole_front_high_density_threshold': threshold,
        'right_high_cells': len(right), 'left_high_cells': len(left), 'both_high_cells': len(common),
        'density_height_clipped': False, 'reference_table_used_for_selection': False,
        'display_pose_lift_m': state['qpos'][state['joint_names'].index('joint4')],
        'display_table_top_body4_z_m': table_top-body[2, 3],
        'sources': {str(path.relative_to(ROOT)): sha256(path) for path in
            [STATE, OUTPUT/'workspace.json', ATLAS/'概览体素统计.npz',
             resource_path('arx_robot')/'robot.urdf', Path(__file__),
             Path(__file__).with_name('draw_workspace.py')]}, 'physical_attempts': 0}
    write_json(OUTPUT/'figure_provenance.json', proof)
    print('New intrinsic figures:', OUTPUT/'workspace_multiview.png', flush=True)


if __name__ == '__main__':
    main()
