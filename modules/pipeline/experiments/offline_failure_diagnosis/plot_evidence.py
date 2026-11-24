"""Draw saved geometry and velocity evidence; no simulator imports."""
from arx_project.assets import load_object_catalog
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from inspect_failures import DEFAULT, PROJECT, read, surface_mesh


def main():
    out = PROJECT / 'docs/diverse_grasp_benchmark/debug_20261123/resources'
    out.mkdir(parents=True, exist_ok=True)
    font = '/home/zhaoyechong/projects/ARX-ManiSkill/tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10})
    catalog = load_object_catalog(DEFAULT / 'object_catalog.json')['models']
    desc = read(DEFAULT / 'gripper_descriptor.json')
    ext = np.asarray(desc['parameters']['extents_open'])
    off = np.asarray(desc['parameters']['offset_open'])
    rot = np.asarray(desc['T_eef_generator'])[:3, :3]
    cases = read(PROJECT / 'experiments/offline_failure_diagnosis/results_20261123/cases.json')
    lookup = {r['id']: r for r in cases}
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    fig.suptitle('已保存失败记录：夹持位置与闭合接触核查', fontsize=19)
    for col, (slot, title) in enumerate([
            ('D0002_S1_open_bottled_water', '瓶装水：夹持区域偏在物体之外'),
            ('D0042_S2_edge_lemon', '柠檬：指尖只到顶面，指腹中心仍偏高')]):
        d = DEFAULT / 'trials' / slot
        state = read(d / 'state.json')
        entry = read(d / 'entry.json')
        mesh = surface_mesh(catalog[entry['model_id']])
        mesh.apply_transform(state['object_world'])
        vertices = np.asarray(mesh.vertices)[::max(1, len(mesh.vertices)//3000)]
        eef = np.asarray(read(d / 'plan_result.json')['selected']['grasp_world'])
        local = np.asarray(list(product([-1, 1], repeat=3))) * ext/2 + off
        corner = local @ rot.T @ eef[:3, :3].T + eef[:3, 3]
        center = eef[:3, 3] + eef[:3, :3] @ rot @ off
        origin = np.asarray(state['object_center'])
        v, c, m = [(p-origin)*1000 for p in [vertices, corner, center]]
        ax = fig.add_subplot(grid[0, col], projection='3d')
        ax.scatter(*v.T, s=1, color='#657b83', alpha=.45, label='真实物体表面')
        for i in range(8):
            for j in range(i+1, 8):
                if sum(a != b for a, b in zip(list(product([-1, 1], repeat=3))[i],
                                            list(product([-1, 1], repeat=3))[j])) == 1:
                    ax.plot(*np.stack([c[i], c[j]]).T, color='#e08713', lw=2)
        ax.scatter(*m, s=90, color='#da6814', marker='x', label='指腹区域中心')
        ax.plot([], [], [], color='#e08713', lw=2, label='张开时的指腹夹持区域')
        combined = np.concatenate([v, c]); middle = (combined.max(0)+combined.min(0))/2
        half = max(np.ptp(combined, axis=0))/2+5
        ax.set(xlim=(middle[0]-half, middle[0]+half), ylim=(middle[1]-half, middle[1]+half),
               zlim=(middle[2]-half, middle[2]+half), xlabel='相对目标中心 X / mm', ylabel='相对目标中心 Y / mm',
               zlabel='相对目标中心 Z / mm', title=title+'\n指腹中心至表面约 '+
               f"{lookup[slot]['pad_center_to_surface_mm']:.1f} mm")
        ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=19, azim=-60)
        ax.legend(fontsize=8, loc='upper left')
    for col, (slot, label) in enumerate([
            ('D0003_S1_open_mushroom', '蘑菇'), ('D0042_S2_edge_lemon', '柠檬')]):
        rows = [json.loads(s) for s in (DEFAULT/'trials'/slot/'execution.jsonl').open()]
        first = next(i for i, r in enumerate(rows) if r['contacts'])
        xyz = np.asarray([r['object_pose'][:3] for r in rows])
        t = np.asarray([r['tick'] for r in rows])/120
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)/np.diff(t)
        i0, i1 = max(1, first-10), min(len(t), first+32)
        ax = fig.add_subplot(grid[1, col])
        ax.plot(t[i0:i1]-t[first], speed[i0-1:i1-1], color='#c33e3e', lw=2)
        ax.axvline(0, color='#59646e', ls='--', label='首个记录到的夹指接触')
        ax.set(xlabel='相对首次接触的时间 / s', ylabel='由相邻位置计算的物体速度 / m/s',
               title=f"{label}：闭爪时物体速度跳变，最高 {lookup[slot]['max_object_speed_mps']:.2f} m/s")
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.savefig(out/'contact_and_grasp_geometry.png', dpi=145)
    print(out/'contact_and_grasp_geometry.png')


if __name__ == '__main__':
    main()
