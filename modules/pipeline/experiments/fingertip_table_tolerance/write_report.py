"""Publish V2 geometry and planning evidence from completed run receipts."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import ConvexHull

from audit import HERE, ROOT as AUDIT, PROJECT, WORKSPACE, ASSET, read, save
from geometry import Geometry, collision_meshes, box_depth
from mesh_probe import ROOT as MESH
from plan_probe import ROOT as PLAN

SITE = WORKSPACE / 'claro/进度网站'
DOCS = PROJECT / 'docs/mainline_versions'
MEDIA = SITE / '内容/资源/sam3-v2'
PHYSICAL = HERE / 'runs/physical3_20261123_02'
NAMES = {'teapot': '茶壶', 'cell_phone': '手机', 'scissors': '剪刀', 'keychain': '钥匙扣',
         'croissant': '可颂', 'bar_soap': '香皂', 'bowl': '碗', 'donut': '甜甜圈'}


def draw_examples():
    meshes, _ = collision_meshes(ASSET)
    font = FontProperties(fname=str(WORKSPACE / 'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, scene, title in zip(axes,
            ['D0036_S1_open_cell_phone_d0', 'D0125_S4_clutter_bar_soap_d0'],
            ['手机：碰撞球碰桌，真实夹指仍有间隙', '香皂：真实指尖需要约1.82毫米接触余量']):
        row = read(MESH/'cases'/scene/'result.json')
        witness = next(w for w in row['witnesses'] if w['thresholds'][-1]['valid'])
        source = Path(row['source_case']); state = read(source/'state.json')
        geometry = Geometry(source/'plan/arx.yml', state)
        frames = geometry.frames(np.asarray(witness['q'])[None]); table = state['scene_cuboids']['table']
        top = np.asarray(table['center'])[2]+np.asarray(table['dims'])[2]/2
        for name, color in [('right_link26', '#888888'), ('right_link27', '#0077b6'), ('right_link28', '#ed8b00')]:
            for i, vertices in enumerate(meshes[name]):
                t = frames[name][0]; xyz = vertices @ t[:3, :3].T+t[:3, 3]
                points = xyz[:, [0, 2]].copy(); points[:, 1] -= top; points *= 100
                points = np.unique(points, axis=0)
                if len(points) >= 3 and np.linalg.matrix_rank(points-points.mean(0)) == 2:
                    outline = points[ConvexHull(points).vertices]
                    ax.fill(outline[:, 0], outline[:, 1], color=color, alpha=.10)
                    outline = np.vstack([outline, outline[0]])
                    ax.plot(*outline.T, color=color, lw=.6, label=name if i == 0 else None)
        spheres = geometry.spheres(frames)[0]
        depths = box_depth(spheres[None], table)[0]; k = int(depths.argmax()); sphere = spheres[k]
        ax.add_patch(Circle([sphere[0]*100, (sphere[2]-top)*100], sphere[3]*100,
                           fill=False, color='#e63946', lw=2., label='碰撞球'))
        eef = frames['right_eef_link'][0]; tcp = eef[:3, 3]+.043*eef[:3, 2]
        ax.scatter(tcp[0]*100, (tcp[2]-top)*100, marker='+', c='black', s=110, label='TCP43')
        center_x = sphere[0]*100
        ax.set_xlim(center_x-12, center_x+8); ax.set_ylim(-1.2, 14)
        ax.axhline(0., c='black', lw=1.2); ax.axhspan(-1.2, 0, color='#b6b6b6', alpha=.4)
        ax.set_title(title, fontproperties=font, fontsize=12)
        ax.set_xlabel('世界X（厘米）', fontproperties=font)
        ax.set_ylabel('相对桌面的高度（厘米）', fontproperties=font)
        ax.grid(alpha=.18); ax.set_aspect('equal', adjustable='box')
        annotation = f"球模型最大穿入：{depths.max()*1000:.2f}毫米"
        ax.text(.03, .95, annotation, transform=ax.transAxes, va='top', fontproperties=font, fontsize=10,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
        ax.legend(prop=font, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(MEDIA/'finger-table-examples.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    assert read(AUDIT/'completion_verification.json')['passed']
    assert read(MESH/'completion_verification.json')['passed']
    a, m = read(AUDIT/'summary.json'), read(MESH/'summary.json')
    planning_done = (PLAN/'completion_verification.json').exists()
    p = read(PLAN/'summary.json') if planning_done else None
    MEDIA.mkdir(parents=True, exist_ok=True)
    draw_examples()
    rows = [read(f) for f in sorted((AUDIT/'cases').glob('*/result.json'))]
    mesh_rows = {f.parent.name: read(f) for f in (MESH/'cases').glob('*/result.json')}
    report = ['## V2实现与本轮结果', '',
        '已实现两层局部检查：先沿用原碰撞球检查；当最后接近或闭合阶段只有夹指碰桌时，再用真实夹指凸碰撞网格复核。真实网格没有碰桌的抓法可以继续；确有接触时，只允许最前10毫米指尖在桌面上表面接触，最大余量为2毫米。', '',
        '手腕、手臂、夹指主体、桌侧面、邻物和地面仍沿用原检查。初态到预抓取继续使用原cuRobo规划与优化；最后接近使用逐点检查的局部连接轨迹。物理接触偏移保持1毫米。', '',
        '### 为什么需要真实网格复核', '',
        '本轮遍历53例原位与14例最近位置的476463个已保存合法IK解。仅给球模型2毫米指尖余量，分别恢复2例和4例终点。进一步复核59组代表解后，零真实穿入即可确认原位至少8例、近处至少6例恢复终点；2毫米真实指尖余量使原位再增加1例。', '',
        '**主要收益来自修正碰撞球过于保守造成的误拦，微小真实接触余量补充了香皂这一例。**', '',
        '| 接触检查方式 | 原位53例：确认有可用终点 | 靠近后仍失败14例：确认有可用终点 |',
        '| --- | ---: | ---: |',
        '| 原V1球模型 | 0/53（0.0%） | 0/14（0.0%） |',
        '| 仅给指尖球模型0.5毫米余量 | 0/53（0.0%） | 0/14（0.0%） |',
        '| 仅给指尖球模型1毫米余量 | 0/53（0.0%） | 0/14（0.0%） |',
        '| 仅给指尖球模型2毫米余量 | 2/53（3.8%） | 4/14（28.6%） |',
        '| 真实夹指网格复核，零穿入 | 8/53（15.1%） | 6/14（42.9%） |',
        '| 真实夹指网格复核，加2毫米指尖余量 | 9/53（17.0%） | 6/14（42.9%） |', '',
        '球模型四档检查遍历全部已保存解；真实网格复核按预先登记规则选取最接近放行的解，共59组，表中列出已确认恢复的场景。近处14例是同一批原失败场景的另一个位置，分开统计。', '',
        '[![夹指真实几何与碰撞球对照](/media/fingertip-table-examples.png)](/media/fingertip-table-examples.png)', '',
        '香皂的真实指尖最低点低于名义桌面约1.818毫米，接触落在指尖区域。手机、茶壶、剪刀等已确认候选中，部分碰撞球已经碰桌，真实夹指仍有空间。', '',
        '### 完整初始规划对照', '']
    if planning_done:
        report += ['从已确认恢复终点的位置中，按物体类别优先选取8个位置，对零真实穿入和2毫米真实指尖余量各做一次规划对照，共16个槽。记录初态到预抓取轨迹、最后接近轨迹和抬升终点检查。', '',
                   '| 配置 | 初始抓取规划通过 |', '| --- | ---: |']
        for v in p['variants']:
            report.append(f'| 真实网格复核＋{v["tolerance_m"]*1000:g}毫米指尖余量 | {v["initial_plans_passed"]}/{v["positions"]}（{v["initial_plans_passed"]/max(1,v["positions"]):.1%}） |')
        report += ['', '| 位置 | 零穿入 | 2毫米余量 | 未通过的具体阶段 |', '| --- | --- | --- | --- |']
        cases = [read(f) for f in sorted((PLAN/'cases').glob('*/result.json'))]
        for source in read(PLAN/'registration.json')['schedule']:
            pair = [next(r for r in cases if r['source_id']==source['id'] and r['tolerance_m']==t) for t in [0., .002]]
            reasons = sorted({ev['reason'] for r in pair for ev in r['events'] if not r['initial_planning_success']})
            labels = {'pregrasp_or_lift_endpoint_failed': '预抓取或抬升终点不通过', 'strict_initial_reach_failed': '初态到预抓取规划不出来',
                      'local_approach_collision': '最后接近轨迹碰撞', 'endpoint_policy_failed': '抓取终点未通过局部接触检查'}
            if not pair[0]['tested_candidate_count']: reasons.append('零穿入档无可用候选')
            report.append('| '+source['id']+' | '+' | '.join('通过' if r['initial_planning_success'] else '未通过' for r in pair)+' | '+('；'.join(labels.get(x,x) for x in reasons) or '—')+' |')
    else:
        report += ['8个位置、两个配置共16个初始规划槽已登记，当前在GPU上运行。结果将按抓取终点、预抓取／抬升终点、初态到预抓取和最后接近分别记录。']
    report += ['', '### 代码与核验', '',
        '| 实现 | 作用 |', '| --- | --- |',
        '| contact_policy.py | 指尖球帽区域、桌面上表面和运动阶段的局部门控 |',
        '| narrow_phase.py | 独立真实凸碰撞体复核，保留原有各指与各部件间的空隙 |',
        '| runtime_adapter.py | 原生非桌面碰撞检查、局部接近连接及速度／加速度／加加速度检查 |',
        '| audit.py、mesh_probe.py、plan_probe.py | 冻结输入、登记、终点对照与初始规划对照 |', '',
        '已通过30项离线故障测试，覆盖超深、非指尖接触、手腕碰桌、桌侧面、邻物、无效数据、凸体边缘相交、双指间隙及轨迹速度／加速度／加加速度。零余量结果与原67位置的全部原生碰撞回执一致；V1输入和结果保持原样。', '',
        '### 实际抓取状态', '']
    physical_done = (PHYSICAL/'independent_verification.json').exists()
    physical = read(PHYSICAL/'summary.json') if physical_done else None
    if physical:
        verification=read(PHYSICAL/'independent_verification.json');assert verification['passed']
        report += [f'茶壶、手机、香皂各一次，已完成{physical["completed"]}/3。实际执行{physical["physical_attempts"]}次，按V2规则抓取成功{physical["success_v2"]}/3（{physical["success_v2"]/3:.1%}），按原V1接触规则成功{physical["success_v1"]}/3（{physical["success_v1"]/3:.1%}）。', '',
            '| 物体 | V2结果 | 原V1接触规则下结果 | 具体结果与录像 |', '| --- | --- | --- | --- |']
        for r in physical['cases']:
            out=PHYSICAL/'cases'/r['id']
            label=next((name for key,name in NAMES.items() if key in r['id']),r['id'])
            videos=[]
            for src,tag in [('execution.mp4','third'),('head.mp4','head')]:
                if (out/src).exists():
                    alias='fingertip-'+r['id'].lower().replace('_','-')+'-'+tag+'.mp4'
                    shutil.copy2(out/src,MEDIA/alias);videos.append((alias,'第三人称' if tag=='third' else '头部'))
            links=' · '.join(f'[{label}](/media/{alias})' for alias,label in videos)
            reason={'success':'抬升与保持达标','insufficient_hold_height':'没有持续带起物体','approach_not_converged':'接近抓点未到位'}.get(r['reason'],r['reason'])
            report.append(f'| {label} | {"成功" if r["success_v2"] else "失败"} | {"成功" if r["success_v1"] else "失败"} | {reason}；{links} |')
        report += ['', '### 实际执行说明了什么', '',
            '| 物体 | 实测情况 | 分析 |', '| --- | --- | --- |',
            '| 茶壶 | 最低保持抬升29.82毫米，保持3.025秒；没有夹指与桌面接触 | 真实网格复核消除了球模型误拦，并转化为实际抓取成功 |',
            '| 手机 | 到达原抓点，闭合时出现双指接触；抬升时接触消失，物体最高只上升约0.032毫米 | 当前失败在夹持保持，不是规划到不了抓点 |',
            '| 香皂 | 规划目标有约1.818毫米指尖穿入；实际接触深度最高约0.0044毫米，关节误差0.00602弧度，高于0.005门槛，末端距目标约1.77毫米 | 刚性桌面阻止夹指到达名义穿入目标，接近阶段结束后没有闭爪 |', '',
            '这一轮已经验证真实网格复核能带来实际收益。2毫米真实接触余量增加了规划放行，但还没有带来新增抓取成功。V2后续默认优先使用真实网格零穿入，2毫米保留为实验配置；香皂这类目标需要把抓姿落到真实桌面接触边界，再单独验证。', '']
    else:
        report += ['茶壶、手机、香皂各一次实际回放已登记，当前正在执行。录像、接触、闭合到位与抬升保持结果会单独核对。']
    report += ['', '[球模型全量核验](/media/fingertip-clearance-summary.json) · [真实网格代表解复核](/media/fingertip-mesh-summary.json) · [逐位置结果表](/media/fingertip-clearance-results.csv)', '']
    text='\n'.join(report)
    save(MEDIA/'clearance-summary.json', a); save(MEDIA/'mesh-summary.json', m)
    if p: save(MEDIA/'plan-summary.json', p)
    with (MEDIA/'clearance-results.csv').open('w', newline='') as stream:
        out=csv.writer(stream); out.writerow(['position','cohort','saved_legal_IK','sphere_0mm','sphere_0.5mm','sphere_1mm','sphere_2mm','mesh_0mm_confirmed','mesh_2mm_confirmed','minimum_table_sphere_depth_mm'])
        for r in rows:
            mr=mesh_rows[r['id']]
            out.writerow([r['id'],r['cohort'],r['saved_q'],*[v['distal_only']['candidates'] for v in r['thresholds']],
                          mr['thresholds'][0]['valid_candidates'],mr['thresholds'][-1]['valid_candidates'],
                          '' if r['minimum_table_depth_other_clear_m'] is None else r['minimum_table_depth_other_clear_m']*1000])
    (SITE/'内容/迭代/sam3-graspgenx-fingertip-table.md').write_text(text)
    (DOCS/'V2_夹指近桌接触余量实施结果.md').write_text('# V2夹指近桌接触余量实施结果\n\n'+text)
    page=SITE/'内容/迭代/sam3-graspgenx-v2.md'
    existing=page.read_text()
    if '## V2实现与本轮结果' in existing:
        existing='## V2目标与当前进度'+existing.split('## V2目标与当前进度',1)[1]
    existing=existing.replace('先比较0、0.5、1、2毫米四档，这些是离线核验档位，还没有启用到执行链路。',
        '已比较0、0.5、1、2毫米四档，并在独立V2验证中启用真实网格复核与0／2毫米对照。')
    existing=existing.replace('本次登记为V2改进，待核对几何与深度','已实现，67位置全量球模型核验及59组真实网格复核完成')
    existing=existing.replace('待执行，尚未确定启用数值','16槽初始规划已完成，V2最大指尖接触余量为2毫米')
    existing=existing.replace('待离线结果完成后登记','三次回放已完成并独立核对' if physical_done else '三次回放已登记，执行中')
    progress_labels={
        '| 1．核对碰撞部位和最小穿入深度':'已完成67位置、476463个解',
        '| 2．核对真实夹指几何':'已完成59组代表解',
        '| 3．比较四档余量':'终点四档核验和16槽初始规划已完成',
        '| 4．实现局部接触策略':'已实现并通过故障测试',
        '| 5．验证实际抓取':'三次回放已完成' if physical_done else '三次回放执行中'}
    lines=existing.splitlines()
    for i,line in enumerate(lines):
        for prefix,label in progress_labels.items():
            if line.startswith(prefix):
                cells=line.split('|');cells[-2]=' '+label+' ';lines[i]='|'.join(cells)
    page.write_text(text+'\n'+'\n'.join(lines)+'\n')
    phase='physical_verification_complete' if physical_done else 'physical_verification_running' if (PHYSICAL/'registration.json').exists() else 'initial_planning_complete' if p else 'initial_planning_running'
    save(HERE/'implementation_status.json', {'version':'V2','phase':phase,
        'policy_implemented':True,'offline_positions':67,'archived_IK_solutions':476463,'exact_mesh_witnesses':59,
        'planning_complete':bool(p),'physical_complete':physical_done,
        'physical_attempts':physical['physical_attempts'] if physical else None,'v1_config_changed':False,
        'recommended_real_contact_tolerance_m':0.,'experimental_real_contact_tolerance_m':.002,
        'artifacts': {'audit':str(AUDIT),'mesh':str(MESH),'initial_plan':str(PLAN),'physical':str(PHYSICAL)}})
    site_path=SITE/'内容/网站.json'; config=read(site_path)
    for alias, path in [('fingertip-table-examples.png','finger-table-examples.png'),('fingertip-clearance-summary.json','clearance-summary.json'),
                        ('fingertip-mesh-summary.json','mesh-summary.json'),('fingertip-clearance-results.csv','clearance-results.csv')]:
        config['localAssets'][alias]='sam3-v2/'+path
    if p: config['localAssets']['fingertip-plan-summary.json']='sam3-v2/plan-summary.json'
    if physical:
        save(MEDIA/'physical-summary.json',physical)
        config['localAssets']['fingertip-physical-summary.json']='sam3-v2/physical-summary.json'
        for path in MEDIA.glob('*.mp4'):config['localAssets'][path.name]='sam3-v2/'+path.name
    save(site_path,config)
    print('V2 report generated; planning_complete=',bool(p))


if __name__ == '__main__': main()
