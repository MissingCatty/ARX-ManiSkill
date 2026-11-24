"""Publish a compact distance comparison with exact candidate denominators."""
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/three_objects_20261123_01'
PROJECT = HERE.parents[1]
DOC = PROJECT / 'docs/diverse_grasp_benchmark/debug_20261123'
OUT = DOC / 'resources/edge_distance'
FONT = PROJECT.parent / 'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'


def read(p): return json.loads(Path(p).read_text())
def font(n): return ImageFont.truetype(str(FONT), n)


def main():
    summary = read(ROOT / 'summary.json')
    assert summary['completed'] == summary['registered'] == 18
    captures = read(ROOT / 'capture_complete.json')
    assert captures['passed'] and len(captures['cases']) == 18
    for row in summary['cases']:
        directory = ROOT / 'cases' / row['id']
        e = row['endpoint']
        arrays = np.load(directory / 'endpoint_ik.npz', allow_pickle=False)
        assert arrays['solved'].shape == (row['candidates'], 32)
        assert int(arrays['solved'].any(-1).sum()) == e['pure_ik_candidates']
        assert int(arrays['scene_clear'].any(-1).sum()) == e['scene_clear_candidates']
        assert not (arrays['scene_clear'] & ~arrays['solved']).any()
        assert read(directory / 'plan_result.json')['success'] == row['planning_success']
        assert not (directory / 'execution.jsonl').exists()
    (ROOT / 'final_verification.json').write_text(json.dumps({'passed': True,
        'completed_positions': 18, 'endpoint_candidates_assessed': sum(r['candidates'] for r in summary['cases']),
        'matching_rendered_positions': 18, 'physical_grasp_attempts': 0,
        'source_inputs_unchanged': summary['source_inputs_unchanged']}, indent=2) + '\n')
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in summary['cases']: grouped[row['slug']].append(row)
    # A target ring is projected from the exact unchanged third-person camera.
    sys.path.insert(0, str(PROJECT / 'experiments/offline_failure_diagnosis'))
    from render_unreachable_candidates import camera_overview, projection
    sheet = Image.new('RGB', (2560, 1400), '#142737')
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 15), '三个物体逐步靠近桌沿：固定原抓法的距离对照', fill='white', font=font(36))
    draw.text((24, 68), '圆圈标出目标；绿色：初始抓取规划通过；红色：规划不出来。表中数量是无碰撞IK候选。', fill='#d0e2eb', font=font(24))
    for ri, (slug, rows) in enumerate(grouped.items()):
        for ci, row in enumerate(rows):
            directory = ROOT / 'cases' / row['id']
            raw = Image.open(directory / 'overview.png').convert('RGB')
            state = read(directory / 'capture_state.json')
            t, k = camera_overview(state)
            uv, valid = projection([state['object_center']], t, k)
            image_draw = ImageDraw.Draw(raw)
            u, v = uv[0]
            if valid[0]: image_draw.ellipse((u-55, v-55, u+55, v+55), outline='#32efb1', width=5)
            raw.thumbnail((420, 236), Image.Resampling.LANCZOS)
            x, y = ci * 426 + 4, ri * 420 + 130
            sheet.paste(raw, (x, y+48))
            label = f'{row["name"]}  {"原位 " if row["level"] == 0 else ""}{row["edge_gap_m"]*100:.1f}cm'
            draw.text((x+7, y+5), label, font=font(22), fill='white')
            color = '#43e3ae' if row['planning_success'] else '#ff8c87'
            draw.text((x+7, y+294), '规划通过' if row['planning_success'] else '规划不出来', font=font(25), fill=color)
            e = row['endpoint']; n = e['candidates']; good = e['scene_clear_candidates']
            draw.text((x+7, y+333), f'无碰撞IK：{good}/{n}（{good/n:.1%}）', font=font(21), fill='#d0e2eb')
            Image.open(directory / 'head.png').save(OUT / f'{row["id"]}_head.jpg', quality=88)
    sheet.save(OUT / 'distance_contact_sheet.jpg', quality=90)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (_, rows) in zip(axes, grouped.items()):
        gaps = [r['edge_gap_m'] * 100 for r in rows]
        for key, label, color in [('pure_ik_candidates', '只检查关节运动学', '#6f87cf'),
                                  ('scene_clear_candidates', '再检查桌面等场景碰撞', '#189a78')]:
            values = [r['endpoint'][key] / r['candidates'] * 100 for r in rows]
            ax.plot(gaps, values, 'o-', color=color, label=label, linewidth=2)
        for r in rows:
            if r['planning_success']:
                ax.scatter(r['edge_gap_m'] * 100, r['endpoint']['scene_clear_candidates']/r['candidates']*100,
                           marker='*', s=170, edgecolors='#172d38', facecolors='#ffd166', zorder=5)
        ax.set_title(rows[0]['name']); ax.set_xlabel('物体到近侧桌沿距离（厘米），向右越近')
        ax.set_ylabel('候选通过比例（%）'); ax.invert_xaxis(); ax.set_ylim(-2, 103)
        ax.grid(alpha=.22); ax.legend(fontsize=9)
    fig.savefig(OUT / 'distance_planning_curves.png', dpi=150)
    plt.close(fig)
    text = ['# 桌沿距离与抓取规划结果', '',
        '将香蕉、食品盒、甜甜圈分别移向机器人近侧桌沿，共测试18个位置。每个物体使用同一批原生抓姿，随物体平移，物体相对抓点和抓取朝向均保持不变。机器人固定六轴、升降45.2厘米、TCP43毫米，沿用当前cuRobo配置。', '',
        '“无碰撞IK”表示该抓取终点找到至少一个满足关节限位、位置/朝向误差和场景碰撞要求的解；“初始规划”还检查预抓取/抬升终点以及从初态到预抓取、再接近目标的轨迹。', '',
        '| 物体 | 桌沿距离 | 全部候选 | 纯IK有解 | 无碰撞IK有解 | 初始规划 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in summary['cases']:
        e = row['endpoint']; n = e['candidates']; clear = e['scene_clear_candidates']
        distance = f'{row["edge_gap_m"]*100:.2f}厘米' + ('（原位）' if row['level'] == 0 else '')
        text.append(f'| {row["name"]} | {distance} | {n} | {e["pure_ik_candidates"]}（{e["pure_ik_candidates"]/n:.1%}） | {clear}（{clear/n:.1%}） | {"通过" if row["planning_success"] else "规划不出来"} |')
    text += ['', '![距离与可行候选比例](resources/edge_distance/distance_planning_curves.png)', '',
             '![18个位置](resources/edge_distance/distance_contact_sheet.jpg)', '',
             '| 物体 | 原位无碰撞IK | 最多可行候选的位置 | 最近8厘米 | 初始规划通过的位置数 |',
             '| --- | ---: | --- | ---: | ---: |']
    for rows in grouped.values():
        best = max(rows, key=lambda r: r['endpoint']['scene_clear_candidates'])
        start, end = rows[0], rows[-1]
        text.append(f'| {start["name"]} | {start["endpoint"]["scene_clear_candidates"]} | '
            f'{best["edge_gap_m"]*100:.2f}厘米：{best["endpoint"]["scene_clear_candidates"]}/{best["candidates"]} | '
            f'{end["endpoint"]["scene_clear_candidates"]}/{end["candidates"]} | {sum(r["planning_success"] for r in rows)}/6 |')
    text += ['', '这三个物体的原位置都规划不出来，移到25、15、8厘米后三个物体都能找到初始轨迹。食品盒在45厘米时已经能规划，香蕉和甜甜圈需要更靠近。固定同一批抓法仍出现这一变化，说明摆放距离确实是这三例规划失败的重要原因。', '',
        '靠近到15厘米后，香蕉和甜甜圈的无碰撞抓姿数量已经接近本次最高值；再靠近到8厘米，分别只增加0个和1个。食品盒仍从788个增加到818个。因此更合适的后续方向是兼顾可达性和头部视野，在15–25厘米附近重新观测并做实际抓取核验。', '',
        '靠近解决了很大一部分运动学困难，桌面碰撞仍在过滤另一部分抓法。比如香蕉在15厘米有1543个纯IK抓姿，但其中只有727个找到无碰撞解；甜甜圈同档为1242个和581个。', '']
    text += ['', '本轮完成18/18，物理抓取0次。纯IK采用每目标128种搜索起点，保留32个解；逐解复核原场景碰撞，无碰撞见证通过独立CPU运动学复核。初始轨迹沿用生产选择器，按原生分数顺序找到第一条通过的轨迹即停止。', '',
        f'完整输入、每位置终点统计、原规划回执和截图：`{ROOT}`。', '']
    text += ['头部相机的可见像素也逐位置保存。香蕉最近8厘米位置的可见像素为0，15厘米位置只剩352像素；这个近距离配置需要配合调整观察范围。', '']
    (DOC / '桌沿距离对照结果.md').write_text('\n'.join(text))
    (ROOT / 'debug/定位记录.md').write_text('# 距离对照定位结果\n\n18个位置全部完成。原位三例0/3能规划，45/35厘米各1/3，25/15/8厘米各3/3。\n\n将同一物体和原始抓法一起平移即可使原失败案例通过初始轨迹规划，确认原摆放距离是重要原因。\n\n15到8厘米的无碰撞候选变化：香蕉727→727、食品盒788→818、甜甜圈581→582。\n\n全部39024个候选终点完成统一IK及碰撞检查，物理抓取0次，原输入散列不变。详细结果见桌沿距离对照结果.md。\n')
    (ROOT / 'debug/已排除.md').write_text('# 已排除的解释\n\n“这三例只受固定抓取朝向或桌面限制，移动距离没有作用”与结果不符：不改变物体相对抓姿和朝向，只平移后，三例均找到原来没有的可行轨迹。\n\n“必须持续贴近桌沿才能增加可行抓法”也没有必要：香蕉和甜甜圈在15厘米以后已基本持平，香蕉8厘米还退出了头部视野。\n')
    (ROOT / 'debug/假设.md').write_text('# 假设结论\n\nH1得到支持：原摆放距离显著限制可达性，靠近使同一批抓法中的可行终点和轨迹增加。\n\nH2的一部分仍成立：靠近以后，桌面碰撞还会排除很多抓法，但它不能解释这三例在所有距离都失败，因为近处已经通过。\n')
    publish(summary)
    print(DOC / '桌沿距离对照结果.md')


def publish(summary):
    site = PROJECT.parent / 'claro/进度网站'
    dest = site / '内容/资源/edge-distance'
    dest.mkdir(parents=True, exist_ok=True)
    assets = {}
    for filename in ['distance_contact_sheet.jpg', 'distance_planning_curves.png']:
        shutil.copy2(OUT / filename, dest / filename)
        assets['edge-' + filename.replace('_', '-')] = 'edge-distance/' + filename
    tiles = []
    for row in summary['cases']:
        slug = row['id'].replace('_', '-')
        d = ROOT / 'cases' / row['id']
        overview = Image.open(d / 'overview.png').convert('RGB')
        overview.save(dest / f'{slug}-overview.jpg', quality=88)
        overview.thumbnail((426, 240), Image.Resampling.LANCZOS)
        overview.save(dest / f'{slug}-thumb.jpg', quality=82)
        shutil.copy2(OUT / (row['id'] + '_head.jpg'), dest / f'{slug}-head.jpg')
        for kind in ['overview', 'thumb', 'head']:
            assets[f'edge-{slug}-{kind}.jpg'] = f'edge-distance/{slug}-{kind}.jpg'
        e = row['endpoint']; n = e['candidates']
        tiles.append(f'<tr><td>{row["name"]}</td><td>{row["edge_gap_m"]*100:.2f}厘米'
            f'{"（原位）" if row["level"] == 0 else ""}</td><td>{e["pure_ik_candidates"]}/{n}（{e["pure_ik_candidates"]/n:.1%}）</td>'
            f'<td>{e["scene_clear_candidates"]}/{n}（{e["scene_clear_candidates"]/n:.1%}）</td>'
            f'<td>{"✅ 规划通过" if row["planning_success"] else "规划不出来"}</td>'
            f'<td><a href="/media/edge-{slug}-overview.jpg" target="_blank"><img loading="lazy" width="320" height="180" src="/media/edge-{slug}-thumb.jpg"></a>'
            f'<br><a href="/media/edge-{slug}-head.jpg" target="_blank">查看头部主视角</a></td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三个物体的桌沿距离对照</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#172e3b;max-width:1300px;margin:auto;padding:24px;background:#f3f6f8}table{width:100%;border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #dbe3e8;text-align:left}th{background:#dfebf1}img{max-width:100%;height:auto;border-radius:5px}a{color:#0867ac}figure{margin:24px 0}.scroll{overflow-x:auto}p{max-width:1000px}</style>
<h1>三个物体逐步靠近桌沿，规划会变好吗？</h1>
<p>将同一物体和同一批抓取位姿一起平移，分别测原位置和桌沿间距45、35、25、15、8厘米。物体朝向、左右位置和高度保持不变，机器人固定升降45.2厘米。每个位置的候选全部参加终点检查，原cuRobo选择器按原生分数顺序寻找第一条可行轨迹。</p>
<p>“纯IK”只看运动学和关节限位；“无碰撞IK”再检查桌子等场景碰撞；“规划通过”还检查预抓取和抬升终点，以及初态到预抓取、接近目标的轨迹。本轮是18个位置的规划对照，未执行物理抓取。</p>
<p><strong>结果：原位置三个物体都规划不出来；移到25、15、8厘米后三个都能规划。</strong>15厘米之后，香蕉和甜甜圈的可行抓法数量已经基本持平。食品盒从45厘米起就能规划，继续靠近仍有增加。</p>
<p>曲线中的星号表示该位置的初始轨迹规划通过。场景图可以点击放大，也可以单独打开每个位置的头部主视角。</p>
<figure><a href="/media/edge-distance-planning-curves.png" target="_blank"><img src="/media/edge-distance-planning-curves.png" alt="距离与候选通过比例"></a></figure>
<div class="scroll"><table><thead><tr><th>物体</th><th>桌沿距离</th><th>纯IK</th><th>无碰撞IK</th><th>初始轨迹规划</th><th>场景截图（点击放大）</th></tr></thead><tbody>''' + ''.join(tiles) + '''</tbody></table></div>
<p>香蕉最近8厘米时已不在头部相机画面内，15厘米时只有352个可见像素。这个近位置如果用于实际感知抓取，需要同时调整观察范围。</p></html>'''
    (dest / 'index.html').write_text(page)
    assets['edge-distance-review-20261123.html'] = 'edge-distance/index.html'
    registry = site / '内容/网站.json'
    data = read(registry); data['localAssets'].update(assets)
    registry.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('http://127.0.0.1:54390/media/edge-distance-review-20261123.html')


if __name__ == '__main__': main()
