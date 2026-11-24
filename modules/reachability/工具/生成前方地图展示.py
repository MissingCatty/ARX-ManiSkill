"""Render the validated front-half atlas; no table geometry or robot connection."""
import base64
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from 公共 import ROOT, read, save, sha


def main():
    directory = ROOT/'产物/前方精细_20261121_01'
    complete = read(directory/'完成.json')
    assert complete['passed']
    for name, digest in complete['file_sha256'].items():
        assert sha(directory/name) == digest
    meta = read(directory/'地图元数据.json')
    out = ROOT/'docs/可达性地图/resources/前方精细_01'
    out.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(directory/'概览体素统计.npz', allow_pickle=False))
    p = (data['cells'].astype(float)+.5)*float(data['resolution_m'])
    assert (p[:, 0] >= 0).all()
    fig = plt.figure(figsize=(13, 5.8))
    axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122)]
    metric = data['margin_mean_rad']
    ax = axes[0]
    image = ax.scatter(*p.T*100, c=metric, cmap='viridis', s=2, alpha=.7, rasterized=True)
    ax.scatter([0], [0], [0], marker='+', c='black', s=70)
    ax.set(xlabel='body4 X (cm)', ylabel='body4 Y (cm)', zlabel='body4 Z (cm)',
           title='Front-half kinematic witnesses (no scene obstacles)')
    ax.view_init(elev=23, azim=-54)
    ax.set_box_aspect(np.ptp(p, axis=0))
    fig.colorbar(image, ax=ax, shrink=.55, label='Mean joint-limit margin (rad)', pad=.08)
    ax = axes[1]
    ax.scatter(p[:, 0]*100, p[:, 1]*100, c=metric, cmap='viridis', s=4, alpha=.65)
    ax.axvline(0, color='#d05a38', lw=1.8)
    ax.annotate('Robot forward +X', (50, 0), (5, 0), arrowprops={'arrowstyle': '->'})
    ax.scatter([0], [0], marker='+', c='black', s=70)
    ax.set(xlabel='body4 X (cm)', ylabel='body4 Y (cm)', title='Top view: only X >= 0 retained')
    ax.set_aspect('equal')
    ax.grid(alpha=.15)
    fig.suptitle(f"ARX TCP43 | {meta['witnesses']:,} front witnesses | 5 mm query cells", fontsize=13)
    fig.text(.5, .015, 'Display: 5 cm overview cells. Limits checked; table, ground, self-collision and paths not checked.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.015, right=.98, bottom=.12, top=.88, wspace=.12)
    fig.savefig(out/'前方可达性概览.png', dpi=150)
    plt.close(fig)
    payload = np.column_stack([p, data['orientation_count']/1152, data['down_orientation_count']/288,
                               data['sample_count'], data['margin_mean_rad'], data['condition_mean'],
                               p[:, 2], np.ones(len(p))]).astype('<f4')
    template = Path(__file__).with_name('地图页面模板.html').read_text()
    template = template.replace('ARX 朝向可达性地图', 'ARX 前方精细可达性地图')
    old = '固定升降 0.22 m · action TCP43 · 419 万组独立关节采样 · 327 万组通过原静态检查'
    assert old in template
    template = template.replace(old, f"前方 180° · TCP43 · {meta['sampled_joint_states']:,} 组采样 · {meta['witnesses']:,} 个关节见证 · 无桌面约束")
    template = re.sub(r'<label for="roi">.*?</label>\n', '', template)
    template = template.replace('min="3" max="120" value="30"', 'min="-100" max="120" value="120"')
    template = template.replace('<span id="height-value">30</span>', '<span id="height-value">120</span>')
    template = template.replace('只看桌面区域（上方 3–30 cm）', '')
    old = '概览格为 5 cm，原始查询格为 2 cm。橙线为原桌面边界；X/Y/Z 为 body4 基座坐标，单位厘米。关节余量取六关节中最小值，再按格内样本求平均。'
    assert old in template
    template = template.replace(old, '概览显示格为 5 cm，查询统计格为 5 mm；完整位姿匹配使用原始位置与四元数，不按朝向格吸附。前方定义为 body4 +X 半空间。坐标单位厘米，关节余量单位弧度。')
    old = '当前杂物和运动路径未计入此通用地图；执行仍须通过原场景 IK、碰撞及轨迹检查。原自碰撞策略保持关闭。本页不连接仿真服务，也不发送运动命令。'
    assert old in template
    template = template.replace(old, '本图仅验证运动学与关节限位，不检查桌面、地面、杂物和自碰撞。升降通过实际 body4 坐标变换适配。抓取几何、场景碰撞与路径分别检查。空白表示未知，本页不连接仿真。')
    template = re.sub(r'<section id="regions">.*?</section>\n', '', template, flags=re.S)
    old = "(!el('roi').checked||data[i*w+9]>.5)&&data[i*w+8]<=Number(el('height').value)/100"
    assert old in template
    template = template.replace(old, "data[i*w+2]<=Number(el('height').value)/100")
    old = " for(let i=0;i<4;i++)line(config.table[i],config.table[(i+1)%4],W,H,'#c15b27',1.7);\n"
    assert old in template
    template = template.replace(old, '')
    template = template.replace("el('roi').onchange=refresh;", '')
    template = template.replace('桌上高度：', 'body4 高度：')
    template = re.sub(r"el\('region-rows'\)\.innerHTML=.*?;\n", '', template)
    template = template.replace('__MAP_CONFIGURATION__', json.dumps({'rows': len(p), 'columns': 10}))
    template = template.replace('__MAP_DATA__', base64.b64encode(payload.tobytes()).decode())
    template = template.replace('url(data:font/woff;base64,__MAP_FONT__)', 'url(/notosc.woff)')
    style = re.search(r'<style>(.*?)</style>', template, re.S)
    script = re.search(r'<script>(.*?)</script>', template, re.S)
    assert style and script and 'region-rows' not in script.group(1) and "el('roi')" not in script.group(1)
    (out/'前方地图.css').write_text(style.group(1))
    (out/'前方地图.mjs').write_text(script.group(1))
    template = template.replace(style.group(0), '<link rel="stylesheet" href="/media/reachability-front-map.css">')
    template = template.replace(script.group(0), '<script type="module" src="/media/reachability-front-map.mjs"></script>')
    (out/'前方交互地图.html').write_text(template)
    save(out/'展示依据.json', {'map': str(directory), 'map_hashes': complete['file_sha256'],
                              'overview_cells': len(p), 'table_rendered': False,
                              'rear_cells_rendered': 0, 'code_sha256': sha(Path(__file__)),
                              'html_bytes': len(template.encode()), 'physical_attempts': 0})
    print('FRONT_VISUALS_COMPLETE', len(p), len(template.encode()), flush=True)


if __name__ == '__main__':
    main()
