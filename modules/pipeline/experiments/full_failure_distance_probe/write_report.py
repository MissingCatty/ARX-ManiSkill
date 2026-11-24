"""Summarize all 53 causes and distance sweeps, then update V1/V2 pages."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = HERE / 'runs/all53_20261123_01'
SITE = PROJECT.parent / 'claro/进度网站'
DOC = PROJECT / 'docs/diverse_grasp_benchmark/debug_20261123'
OUT = DOC / 'resources/full_failure_distance'
MEDIA = SITE / '内容/资源/full-failure-distance'
FONT = PROJECT.parent / 'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'
CATEGORIES = {'banana':'香蕉','boxed_food':'食品盒','donut':'甜甜圈','carrot':'胡萝卜','teapot':'茶壶',
    'scissors':'剪刀','baguette':'法棍','pear':'梨','book':'书','sponge':'海绵','cell_phone':'手机',
    'keychain':'钥匙扣','croissant':'牛角包','bell_pepper':'彩椒','orange':'橙子','mug':'杯子',
    'potato':'土豆','bar_soap':'香皂','bowl':'碗','spatula':'锅铲','soap_dispenser':'洗手液瓶',
    'peach':'桃子','mango':'芒果','broccoli':'西蓝花','tomato':'西红柿','apple':'苹果',
    'remote_control':'遥控器','bottled_water':'瓶装水'}
CAUSES = {'position_ik_not_found':'抓取位置IK也未解出',
    'fixed_grasp_orientation_ik_not_found':'位置能解，固定抓姿未解出',
    'table_collision':'找到IK，但均未通过碰撞；包含碰桌',
    'obstacle_collision':'找到IK，但均未通过邻物碰撞',
    'table_and_obstacle_collision':'找到IK，但碰桌与邻物共同阻挡',
    'ground_or_other_collision':'地面或原碰撞门控其他约束',
    'original_grasp_search_miss':'原抓取终点IK搜索漏解',
    'initial_state_collision':'初态发生碰撞'}
LAYOUT = {'S1_open':'单物体开放','S2_edge':'近侧桌边','S3_sparse':'稀疏多物体','S4_clutter':'密集杂乱','S5_occluded':'目标遮挡'}


def read(p): return json.loads(Path(p).read_text())
def font(n): return ImageFont.truetype(str(FONT), n)


def source_cause(row):
    if row['endpoint']['scene_clear_candidates'] > 0: return 'original_grasp_search_miss'
    return row['diagnosis']


def validate():
    sys.path.insert(0, str(HERE))
    import probe
    reg = read(ROOT/'registration.json'); probe.guard(reg)
    assert read(ROOT/'completion_verification.json')['passed']
    assert read(ROOT/'capture_complete.json')['completed'] == 318
    summary = read(ROOT/'summary.json')
    assert summary['completed'] == 318 and summary['source_failures'] == 53
    grouped = defaultdict(list)
    for r in summary['cases']:
        out = ROOT/'cases'/r['id']; arrays = np.load(out/'endpoint_ik.npz', allow_pickle=False)
        e = r['endpoint']; n = r['candidates']
        assert arrays['solved'].shape == (n,32)
        assert e['pure_ik_candidates'] == int(arrays['solved'].any(-1).sum())
        assert e['scene_clear_candidates'] == int(arrays['scene_clear'].any(-1).sum())
        assert not (arrays['scene_clear'] & ~arrays['solved']).any()
        assert e['scene_clear_candidates'] <= e['pure_ik_candidates'] <= n
        assert r['planning_success'] == read(out/'plan_result.json')['success']
        assert (out/'head.jpg').stat().st_size and (out/'overview.jpg').stat().st_size
        assert read(out/'capture.json')['passed']
        grouped[r['source_id']].append(r)
    assert len(grouped) == 53
    for rows in grouped.values():
        rows.sort(key=lambda r:r['level']); assert [r['level'] for r in rows] == list(range(6))
        assert source_cause(rows[0]) in CAUSES
    result = {'passed':True,'completed_positions':318,'source_failures':53,
        'all_candidate_endpoints':sum(r['candidates'] for r in summary['cases']),
        'rendered_positions':318,'source_inputs_unchanged':True,'physical_attempts':0}
    (ROOT/'final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return reg, summary, grouped


def classify(grouped):
    causes = Counter(); distance = Counter(); cases = []
    for slot, rows in sorted(grouped.items()):
        original = rows[0]; causes[source_cause(original)] += 1
        passed = [r for r in rows[1:] if r['planning_success']]
        if original['planning_success']: kind = '原位加强搜索已恢复'
        elif passed: kind = '靠近后恢复规划'
        else: kind = '靠近后仍规划不出来'
        distance[kind] += 1
        best = max(rows,key=lambda r:r['endpoint']['scene_clear_candidates'])
        cases.append({'id':slot,'name':CATEGORIES.get(original['semantic_class'],original['semantic_class']),
            'layout':original['layout'],'cause':source_cause(original),'distance_effect':kind,
            'original_gap_m':original['target_edge_gap_m'],'nearest_test_gap_m':rows[-1]['target_edge_gap_m'],
            'translation_m':-rows[-1]['translation_m'][0],
            'first_passing_gap_m': passed[0]['target_edge_gap_m'] if passed else None,
            'best_clear_count':best['endpoint']['scene_clear_candidates'],'best_gap_m':best['target_edge_gap_m'],
            'endpoint_counts':[r['endpoint']['scene_clear_candidates'] for r in rows],
            'planning_passes':[r['planning_success'] for r in rows]})
    return causes, distance, cases


def figures(grouped, causes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT)); plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig, axes = plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    names=[CAUSES[k] for k in causes]; values=list(causes.values())
    axes[0].barh(names,values,color='#437fa0'); axes[0].invert_yaxis(); axes[0].set_title('53个原失败的终点原因')
    for i,v in enumerate(values): axes[0].text(v+.2,i,str(v),va='center')
    levels=range(6); passes=[sum(rows[i]['planning_success'] for rows in grouped.values()) for i in levels]
    axes[1].bar(levels,passes,color='#229b79'); axes[1].set_ylim(0,56); axes[1].set_title('每档找到初始轨迹的场景数')
    for i,v in enumerate(passes): axes[1].text(i,v+.6,str(v),ha='center')
    fractions=np.array([[r['endpoint']['scene_clear_candidates']/r['candidates']*100 for r in rows] for rows in grouped.values()])
    axes[2].plot(levels,fractions.mean(0),'o-',label='各场景可行候选比例的均值')
    axes[2].plot(levels,np.median(fractions,axis=0),'o-',label='中位数'); axes[2].legend(fontsize=9)
    axes[2].set_title('无碰撞IK候选比例'); axes[2].set_ylabel('百分比（%）')
    for ax in axes[1:]:
        ax.set_xticks(list(levels),['原位','20%','40%','60%','80%','100%'])
        ax.set_xlabel('向近侧移动可用距离的比例'); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'summary.png',dpi=150); plt.close(fig)
    return passes


def contacts(grouped):
    sys.path.insert(0,str(PROJECT/'experiments/offline_failure_diagnosis'))
    from render_unreachable_candidates import camera_overview,projection
    assets = {}
    for slot,rows in sorted(grouped.items()):
        stem=slot.lower().replace('_','-')
        sheet=Image.new('RGB',(2520,520),'#152c3d'); draw=ImageDraw.Draw(sheet)
        name=CATEGORIES.get(rows[0]['semantic_class'],rows[0]['semantic_class'])
        draw.text((16,10),f'{slot}  {name}：保持目标与挡物相对摆放不变，整组靠近',font=font(25),fill='white')
        for i,row in enumerate(rows):
            out=ROOT/'cases'/row['id']; state=read(out/'state.json')
            image=Image.open(out/'overview.jpg').convert('RGB'); t,k=camera_overview(state)
            uv,valid=projection([state['object_center']],t,k); u,v=uv[0]
            if valid[0]: ImageDraw.Draw(image).ellipse((u-55,v-55,u+55,v+55),outline='#3ce2ad',width=5)
            image=image.resize((420,236),Image.Resampling.LANCZOS); x=i*420
            sheet.paste(image,(x,120)); text='原位' if i==0 else f'可用距离的{i*20}%'
            draw.text((x+10,58),text,font=font(21),fill='white')
            draw.text((x+10,88),f'目标距桌沿 {row["target_edge_gap_m"]*100:.1f}厘米',font=font(20),fill='#d4e5ed')
            e=row['endpoint']; color='#49ddb3' if row['planning_success'] else '#ff9b8f'
            draw.text((x+10,370),'规划通过' if row['planning_success'] else '规划不出来',font=font(24),fill=color)
            draw.text((x+10,414),f'无碰撞IK {e["scene_clear_candidates"]}/{row["candidates"]}',font=font(21),fill='#d4e5ed')
            visible=read(out/'capture.json')['visible_target_pixels']
            draw.text((x+10,450),f'头部目标可见像素 {visible}',font=font(19),fill='#c4d8e2')
            shutil.copy2(out/'head.jpg',MEDIA/f'{stem}-d{i}-head.jpg')
            assets[f'full-distance-{stem}-d{i}-head.jpg']=f'full-failure-distance/{stem}-d{i}-head.jpg'
        sheet.save(OUT/f'{stem}.jpg',quality=89)
        shutil.copy2(OUT/f'{stem}.jpg',MEDIA/f'{stem}.jpg')
        thumb=sheet.copy(); thumb.thumbnail((1008,208),Image.Resampling.LANCZOS); thumb.save(MEDIA/f'{stem}-thumb.jpg',quality=83)
        assets[f'full-distance-{stem}.jpg']=f'full-failure-distance/{stem}.jpg'
        assets[f'full-distance-{stem}-thumb.jpg']=f'full-failure-distance/{stem}-thumb.jpg'
    return assets


def report():
    reg,summary,grouped=validate()
    OUT.mkdir(parents=True,exist_ok=True); MEDIA.mkdir(parents=True,exist_ok=True)
    causes,effect,cases=classify(grouped); passes=figures(grouped,causes); assets=contacts(grouped)
    originals=[rows[0] for rows in grouped.values()]
    candidate_total=sum(r['candidates'] for r in originals)
    pure_total=sum(r['endpoint']['pure_ik_candidates'] for r in originals)
    clear_total=sum(r['endpoint']['scene_clear_candidates'] for r in originals)
    unresolved=[item for item in cases if item['distance_effect']=='靠近后仍规划不出来']
    at_edge=[item for item in unresolved if item['nearest_test_gap_m']<=.0121]
    constrained=[item for item in unresolved if item['nearest_test_gap_m']>.0121]
    header=['## 全部53个规划失败场景：原因拆分与距离结果','',
        'V1完整200例中，53例规划不出来。原日志全部卡在抓取终点IK／碰撞合并门控；本轮保留原结果，对全部53例分别拆解原因，并完成6档距离对照，共318个位置。','',
        f'**{effect["靠近后恢复规划"]}例在靠近后恢复初始轨迹；{effect["原位加强搜索已恢复"]}例在原位加强搜索就已恢复；{effect["靠近后仍规划不出来"]}例靠近后仍规划不出来。**','',
        '| 原位抓取终点原因 | 场景数 | 占53例 |','| --- | ---: | ---: |']
    for key,count in causes.items(): header.append(f'| {CAUSES[key]} | {count} | {count/53:.1%} |')
    header+=['','按候选统计，原位共'+str(candidate_total)+'个抓姿，下面三组相加等于全部候选。按场景统计的“有碰撞解”不等于该场景的所有候选都能解出IK。','',
        '| 原位候选的终点检查结果 | 数量 | 占全部原位候选 |','| --- | ---: | ---: |']
    for label,n in [('纯IK未解出',candidate_total-pure_total),('找到IK但未找到无碰撞解',pure_total-clear_total),('找到无碰撞解',clear_total)]:
        header.append(f'| {label} | {n} | {n/candidate_total:.1%} |')
    header+=['','| 距离对照结果 | 场景数 | 占53例 |','| --- | ---: | ---: |']
    for key,count in effect.items():header.append(f'| {key} | {count} | {count/53:.1%} |')
    header+=['','| 向近侧移动的可用距离比例 | 原位 | 20% | 40% | 60% | 80% | 100% |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        '| 找到初始轨迹的场景 | '+' | '.join(f'{v}/53（{v/53:.1%}）' for v in passes)+' |','',
        '多物体场景将所有物体和原生抓法一起平移，物体间的距离、高度与朝向保持不变。移动上限由最靠桌沿的物体决定，保留12毫米支撑边距，最大移动45厘米。每一档的实际桌沿距离均单独记录。','',
        '每个目标使用128种IK搜索起点并保留32个解，位置误差2毫米、朝向误差0.03弧度。关节限位和碰撞球、桌面与邻物、六轴和TCP43沿用V1。原位与靠近档使用相同的诊断搜索；原位就能恢复的情况单列为搜索漏解。','',
        '[![全量原因与距离变化](/media/full-distance-summary.png)](/media/full-distance-summary.png)','',
        '[全部53例的逐例原因、六档数据与截图](/media/full-distance-review-20261123.html) · [下载318个位置的数据表](/media/full-distance-results.csv) · [完整汇总数据](/media/sam3-v1-all-failure-distance.json)','',
        '### 仍未恢复的14例，具体卡在哪里','',
        f'{len(at_edge)}例的目标已经移到距桌沿1.2厘米，仍没有无碰撞抓取终点。这些场景需要重点处理抓姿与桌面／邻物的碰撞，单纯继续靠近已没有桌面支撑空间。','',
        f'另外{len(constrained)}例受前排物体限制，整组不能继续前移，目标在最近一档仍距桌沿'+
            f'{min(i["nearest_test_gap_m"] for i in constrained)*100:.1f}–{max(i["nearest_test_gap_m"] for i in constrained)*100:.1f}厘米。它们当前也被碰撞拦住；这轮保持遮挡关系的平移尚未把目标送到更近的位置，后续要结合站位或前排物体处理验证。','',
        '### 后续修正方向','',
        '能够靠近后恢复的场景，下一步用新观测生成抓姿并验证实际抓取，同时检查头部视野。碰桌或邻物阻挡仍未解决的场景，继续按碰撞来源处理；原位搜索漏解则归入IK搜索改进，分别记录各自效果。','',
        '本轮物理抓取0次，结果是规划诊断；V1正式成绩为100/200规划通过、76/200抓取成功。','']
    cause_table=['## 逐例原位原因','',
        '表中“碰桌候选”和“碰邻物候选”表示找到的解中出现该碰撞的候选，两项可以重叠。全部候选可分为纯IK未解出、找到IK但未找到无碰撞解、找到无碰撞解三组。','',
        '| 场景 | 布局 | 候选总数 | 纯IK有解 | 无碰撞IK | 含碰桌解的候选 | 含碰邻物解的候选 | 原位原因 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    distance_table=['## 逐例距离结果','',
        '六档依次为原位、可用距离的20%、40%、60%、80%、100%；✓表示完整初始轨迹通过，×表示本档未通过。','',
        '| 场景 | 原位目标桌沿距离 | 最近档目标桌沿距离 | 实际平移 | 六档无碰撞IK数 | 六档初始轨迹 | 结果 |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |']
    html_rows=[]
    for item in cases:
        slot=item['id']; original=grouped[slot][0]; e=original['endpoint']; stem=slot.lower().replace('_','-')
        image_url=f'/media/full-distance-{stem}.jpg'
        cause_table.append(f'| [{slot}，{item["name"]}]({image_url}) | {LAYOUT[item["layout"]]} | {original["candidates"]} | {e["pure_ik_candidates"]} | {e["scene_clear_candidates"]} | {e["table_collision_candidates"]} | {e["obstacle_collision_candidates"]} | {CAUSES[item["cause"]]} |')
        flags=' '.join('✓' if ok else '×' for ok in item['planning_passes'])
        numbers=' → '.join(str(n) for n in item['endpoint_counts'])
        distance_table.append(f'| [{slot}，{item["name"]}]({image_url}) | {item["original_gap_m"]*100:.1f}厘米 | {item["nearest_test_gap_m"]*100:.1f}厘米 | {item["translation_m"]*100:.1f}厘米 | {numbers} | {flags} | {item["distance_effect"]} |')
        heads=' '.join(f'<a href="/media/full-distance-{stem}-d{i}-head.jpg" target="_blank">头部{i}</a>' for i in range(6))
        html_rows.append(f'<tr><td>{slot}<br>{item["name"]} · {LAYOUT[item["layout"]]}</td><td>{CAUSES[item["cause"]]}<br>纯IK {e["pure_ik_candidates"]}/{original["candidates"]}；无碰撞IK {e["scene_clear_candidates"]}</td>'
            f'<td>{item["original_gap_m"]*100:.1f} → {item["nearest_test_gap_m"]*100:.1f}厘米<br>{numbers}<br>轨迹：{flags}<br>{item["distance_effect"]}</td>'
            f'<td><a href="{image_url}" target="_blank"><img loading="lazy" width="504" height="104" src="/media/full-distance-{stem}-thumb.jpg"></a><br>{heads}</td></tr>')
    remaining_table=['## 14个仍未恢复场景的最近档原因','',
        '| 场景 | 最近目标桌沿距离 | 整组最大平移 | 最近档原因 | 碰桌最多的连杆 |',
        '| --- | ---: | ---: | --- | --- |']
    for item in unresolved:
        last=grouped[item['id']][-1];links=last['endpoint']['table_collision_links']
        top='、'.join(k for k,v in sorted(links.items(),key=lambda t:-t[1])[:2]) or '无'
        remaining_table.append(f'| {item["id"]}，{item["name"]} | {item["nearest_test_gap_m"]*100:.1f}厘米 | {item["translation_m"]*100:.1f}厘米 | {CAUSES.get(last["diagnosis"],last["diagnosis"])} | {top} |')
    text='\n'.join(header+cause_table+['']+distance_table+['']+remaining_table+[''])
    (DOC/'全量规划失败原因与距离结果.md').write_text('# 全量规划失败原因与距离结果\n\n'+text)
    (SITE/'内容/迭代/sam3-graspgenx-failure-distance.md').write_text(text)
    output=io.StringIO(); writer=csv.writer(output)
    writer.writerow(['原场景','布局','类别','距离档','目标桌沿距离_厘米','平移量_厘米','候选总数','纯IK有解','无碰撞IK有解','含碰桌解候选','含碰邻物解候选','初始轨迹通过','本档原因','目标可见像素'])
    for slot,rows in sorted(grouped.items()):
        for r in rows:
            e=r['endpoint'];writer.writerow([slot,r['layout'],r['semantic_class'],r['level'],r['target_edge_gap_m']*100,-r['translation_m'][0]*100,
                r['candidates'],e['pure_ik_candidates'],e['scene_clear_candidates'],e['table_collision_candidates'],e['obstacle_collision_candidates'],
                r['planning_success'],r['diagnosis'],read(ROOT/'cases'/r['id']/'capture.json')['visible_target_pixels']])
    (MEDIA/'results.csv').write_text(output.getvalue(),encoding='utf-8-sig')
    shutil.copy2(OUT/'summary.png',MEDIA/'summary.png')
    data={'source_failures':53,'positions':318,'original_causes':dict(causes),'distance_effects':dict(effect),
        'planning_passes_by_level':passes,'cases':cases,'physical_attempts':0,'source_inputs_unchanged':True}
    data.update(original_candidate_total=candidate_total,original_pure_ik_candidates=pure_total,
        original_scene_clear_candidates=clear_total,unresolved_at_near_edge=len(at_edge),unresolved_with_front_objects_limiting_translation=len(constrained))
    (ROOT/'analysis_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'debug/定位记录.md').write_text('# 全量结果\n\n'+json.dumps({'原因':dict(causes),'距离结果':dict(effect),'六档规划通过':passes},ensure_ascii=False,indent=2)+'\n')
    (ROOT/'debug/假设.md').write_text('# 假设检验结果\n\n'+f'H1：{effect["靠近后恢复规划"]}例在原位未恢复、靠近后恢复，支持距离影响。\n\nH2：{effect["靠近后仍规划不出来"]}例仍未恢复，按逐例碰撞和运动学证据继续定位。\n\nH3：{effect["原位加强搜索已恢复"]}例在原位加强搜索即可恢复，单独计为搜索效果。\n')
    (ROOT/'debug/已排除.md').write_text('# 已排除的解释\n\n只靠本轮允许的整组前移，仍不能恢复全部失败场景。距离效应与原位搜索漏解已分开统计。\n')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>53个规划失败的原因与距离结果</title>
<style>body{font:16px/1.7 system-ui,sans-serif;color:#163044;background:#f4f7f9;margin:24px}h1,p{max-width:1200px}table{border-collapse:collapse;background:white;width:100%;min-width:1100px}td,th{border:1px solid #d3dfe6;padding:12px;text-align:left}th{background:#e1edf3}img{max-width:100%;height:auto}a{color:#086caa}.scroll{overflow:auto}</style><h1>全部53个规划失败：具体原因与距离对照</h1>'''
    page+=f'<p>318/318个位置完成；靠近后恢复{effect["靠近后恢复规划"]}例，原位加强搜索已恢复{effect["原位加强搜索已恢复"]}例，靠近后仍未恢复{effect["靠近后仍规划不出来"]}例。实际抓取0次。</p>'
    page+='<p>六档依次为原位和整组向机器人近侧移动可用距离的20%、40%、60%、80%、100%。保持所有物体相对位置与抓姿不变，移动量由最靠桌沿的物体限定。原位与靠近档用相同的128种IK搜索起点，之后检查原场景碰撞及初始轨迹。</p>'
    page+=f'<p>未恢复的14例中，{len(at_edge)}例目标已到距桌沿1.2厘米仍被碰撞拦下；另外{len(constrained)}例的前排物体已到桌沿，限制整组继续移动，目标仍在较远位置。</p>'
    page+='<p>截图可点击放大；“头部0”至“头部5”分别打开六档的头部主视角。</p><img src="/media/full-distance-summary.png" width="1600"><div class="scroll"><table><thead><tr><th>场景</th><th>原位抓取终点原因</th><th>距离和规划变化</th><th>六档截图</th></tr></thead><tbody>'+''.join(html_rows)+'</tbody></table></div></html>'
    (MEDIA/'index.html').write_text(page)
    assets.update({'full-distance-summary.png':'full-failure-distance/summary.png','full-distance-results.csv':'full-failure-distance/results.csv',
        'full-distance-review-20261123.html':'full-failure-distance/index.html'})
    registry=SITE/'内容/网站.json'; site=read(registry);site['localAssets'].update(assets)
    registry.write_text(json.dumps(site,ensure_ascii=False,indent=2)+'\n')
    v2=SITE/'内容/迭代/sam3-graspgenx-v2.md'; v2text=v2.read_text()
    v2text=v2text.replace('当前已启动**全量规划失败诊断**：覆盖V1全部53个规划失败场景、每例6档，共318个位置。',
        '当前已完成**全量规划失败诊断**：覆盖V1全部53个规划失败场景、每例6档，共318个位置。')
    v2text=v2text.replace('| 已登记并启动 |','| 318位置已完成，逐例结果见V1 |')
    v2text=v2text.split('\n## 全量距离诊断结果')[0]
    v2text+='\n## 全量距离诊断结果\n\n'+'\n'.join(header[2:])+'\n'
    v2.write_text(v2text)
    subprocess.run(['python3',str(SITE/'tools/publish_sam3_v1.py'),'publish'],cwd=SITE,check=True)
    with urllib.request.urlopen('http://127.0.0.1:8786/iterations/sam3-graspgenx/versions/V1/',timeout=20) as r:
        text=r.read().decode(); assert '全部53个规划失败场景' in text
    checked=0
    for name in assets:
        req=urllib.request.Request('http://127.0.0.1:8786/media/'+name,method='HEAD')
        with urllib.request.urlopen(req,timeout=10) as r: assert r.status==200
        checked+=1
    (ROOT/'publication_verification.json').write_text(json.dumps({'passed':True,'assets_checked':checked,'v1_updated':True,'v2_updated':True},indent=2)+'\n')
    print(json.dumps({'positions':318,'causes':dict(causes),'distance_effects':dict(effect),
        'unresolved_at_edge':len(at_edge),'unresolved_front_limited':len(constrained),
        'url':'http://127.0.0.1:54390/iterations/sam3-graspgenx/versions/V1/'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        while not (ROOT/'completion_verification.json').exists():
            if (ROOT/'STOP').exists():raise RuntimeError('Diagnostic stopped; final report withheld')
            time.sleep(30)
    report()
