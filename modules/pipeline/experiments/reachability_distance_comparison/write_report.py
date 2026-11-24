"""Render the atlas/distance comparison as figures, tables and V1 evidence."""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ROOT=HERE/'runs/map_distance_20261123_01'
ATLAS=PROJECT.parent/'reachability/产物/前方精细_20261121_01'
SITE=PROJECT.parent/'claro/进度网站'
DOC=PROJECT/'docs/diverse_grasp_benchmark/debug_20261123'
OUT=DOC/'resources/reachability_distance'
MEDIA=SITE/'内容/资源/reachability-distance'
FONT=PROJECT.parent/'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'


def read(p):return json.loads(Path(p).read_text())


def plots(rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    values=[r for r in rows if r['cohort']=='all53'];coords=np.array([r['tcp_mean_body4'] for r in values])
    coarse=dict(np.load(ATLAS/'概览体素统计.npz'));centers=(coarse['cells']+.5)*.05
    zlo=np.floor(coords[:,2].min()/.05)*.05;zhi=np.ceil(coords[:,2].max()/.05)*.05
    mask=(centers[:,2]>=zlo)&(centers[:,2]<=zhi)
    cells=coarse['cells'][mask,:2].astype(int);lo=cells.min(0);hi=cells.max(0)
    grid=np.zeros(tuple(hi-lo+1));np.add.at(grid,tuple((cells-lo).T),coarse['sample_count'][mask])
    xx=np.arange(lo[0],hi[0]+2)*5.;yy=np.arange(lo[1],hi[1]+2)*5.
    fig,ax=plt.subplots(figsize=(7.6,8.2),constrained_layout=True)
    image=ax.pcolormesh(xx,yy,np.log1p(grid).T,cmap='Blues',shading='flat')
    cbar=fig.colorbar(image,ax=ax,shrink=.8);cbar.set_label('该高度带的地图采样数，log(1+样本数)')
    groups={}
    for r in values:groups.setdefault(r['source_id'],[]).append(r)
    for source,series in groups.items():
        series.sort(key=lambda r:r['level']);a=np.array(series[0]['tcp_mean_body4'][:2])*100;b=np.array(series[-1]['tcp_mean_body4'][:2])*100
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#687b89','alpha':.32,'lw':.8})
    for ok,color,label,marker in [(False,'#d96c58','规划不出来','x'),(True,'#168956','规划通过','o')]:
        selected=np.array([r['tcp_mean_body4'] for r in values if r['planning_success']==ok])
        ax.scatter(selected[:,0]*100,selected[:,1]*100,s=24,c=color,marker=marker,label=label,alpha=.8)
    source=Path(read(PROJECT/'experiments/full_failure_distance_probe/runs/all53_20261123_01/registration.json')['sources'][0]['source_trial'])
    state=read(source/'state.json');base=np.array(state['links']['body4']);invbase=np.linalg.inv(base)
    near_edge=(invbase@[-.7245,0.,0.,1.])[0]*100
    ax.axvline(near_edge,color='#687882',ls='--',lw=1);ax.text(near_edge+1,yy[0]+5,'近侧桌沿',fontsize=10,color='#435663',rotation=90)
    ax.scatter([0],[0],c='black',marker='+',s=90,label='body4原点')
    for source_id,series in groups.items():
        if source_id not in ['D0014_S1_open_scissors','D0036_S1_open_cell_phone','D0080_S2_edge_keychain']:continue
        r=series[-1];x,y=np.array(r['tcp_mean_body4'][:2])*100
        ax.annotate(source_id.split('_')[0],(x,y),xytext=(x+5,y+4),fontsize=9,
            arrowprops={'arrowstyle':'-','color':'#6a5450','lw':.7})
    ax.set_xlabel('body4前方 X（厘米）');ax.set_ylabel('body4左右 Y（厘米）')
    ax.set_title(f'距离试验与前方可达性地图叠加\n图中高度带：TCP Z = {zlo*100:.0f} 至 {zhi*100:.0f}厘米，点为候选TCP均值')
    ax.set_xlim(-5,max(95,coords[:,0].max()*100+5));ax.set_ylim(yy[0],yy[-1]);ax.set_aspect('equal');ax.legend(loc='upper right',fontsize=9)
    fig.savefig(OUT/'spatial_overlay.png',dpi=160,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    density=np.array([r['coarse_density_mean'] for r in values]);levels=np.array([r['level'] for r in values])
    for key,title,label,name,rho_key in [
        ('ik_fraction','地图越密，纯IK通过比例越高','纯IK通过比例（%）','density_vs_ik.png','spearman_pure_ik'),
        ('clear_fraction','地图密度与无碰撞终点的关系','无碰撞IK通过比例（%）','density_vs_scene.png','spearman_scene_clear')]:
        fig,ax=plt.subplots(figsize=(5.7,4.4),constrained_layout=True)
        p=ax.scatter(np.maximum(density,.1),np.array([r[key] for r in values])*100,c=levels,cmap='viridis',s=25,alpha=.75)
        ax.set_xscale('log');ax.set_xlabel('候选所在5厘米格的平均地图样本数（对数轴）');ax.set_ylabel(label)
        rho=summary['all53']['correlations']['coarse_density_mean'][rho_key]
        ax.set_title(f'{title}\n排序相关系数 ρ = {rho:.2f}');ax.grid(alpha=.2)
        cb=fig.colorbar(p,ax=ax,ticks=range(6));cb.set_ticklabels(['原位','20%','40%','60%','80%','100%']);cb.set_label('向近侧移动的可用距离比例')
        fig.savefig(OUT/name,dpi=150);plt.close(fig)
    groups=summary['all53']['quartiles']['coarse_density_mean']['groups']
    fig,ax=plt.subplots(figsize=(5.7,4.4),constrained_layout=True)
    rates=[g['planning_passes']*100/g['positions'] for g in groups]
    ax.bar(range(4),rates,color=['#b8d7e7','#85b5cf','#4a8faf','#21667f'])
    for i,g in enumerate(groups):ax.text(i,rates[i]+1.2,f'{g["planning_passes"]}/{g["positions"]}\n{rates[i]:.1f}%',ha='center',fontsize=10)
    ax.set_xticks(range(4),['最低25%','25–50%','50–75%','最高25%']);ax.set_ylim(0,85)
    ax.set_ylabel('找到完整初始轨迹的位置比例（%）');ax.set_xlabel('按地图密度划分318个测试位置')
    ax.set_title('地图密度与实际规划通过率');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'density_groups.png',dpi=150);plt.close(fig)
    return {'z_band_body4_m':[float(zlo),float(zhi)],'plotted_test_positions':318}


def main():
    assert read(ROOT/'completion_verification.json')['passed']
    summary=read(ROOT/'summary.json');rows=read(ROOT/'rows.json');alignment=read(ROOT/'alignment_verification.json')
    OUT.mkdir(parents=True,exist_ok=True);MEDIA.mkdir(parents=True,exist_ok=True)
    spatial=plots(rows,summary)
    all53=summary['all53'];pilot=summary['pilot3'];cor=all53['correlations'];q=all53['quartiles']['coarse_density_mean']['groups']
    grouped={}
    for row in rows:
        if row['cohort']=='all53':grouped.setdefault(row['source_id'],[]).append(row)
    for series in grouped.values():series.sort(key=lambda r:r['level'])
    hard=sorted([s[-1] for s in grouped.values() if not any(r['planning_success'] for r in s)],key=lambda r:-r['coarse_density_mean'])
    record=['## 距离测试与可达性地图对照','',
        '**高通过率位置整体上位于地图更密集、位姿更容易匹配的区域。** 对53个原失败场景的318个位置，地图密度与纯IK通过比例的排序相关系数为'+f'{cor["coarse_density_mean"]["spearman_pure_ik"]:.2f}。'
        '原5毫米格统计的相关系数为'+f'{cor["fine_density_mean"]["spearman_pure_ik"]:.2f}，趋势一致。','',
        '地图颜色表示原始关节采样在空间中的密集程度。本次使用已保存的真实IK、碰撞与路径结果，检查它与实际通过率的对应关系。','',
        '| 地图密度分组 | 测试位置数 | 平均纯IK通过比例 | 平均无碰撞IK比例 | 找到完整初始轨迹 |',
        '| --- | ---: | ---: | ---: | ---: |']
    labels=['最低25%','25–50%','50–75%','最高25%']
    for label,g in zip(labels,q):
        record.append(f'| {label} | {g["positions"]} | {g["mean_pure_ik_fraction"]:.1%} | {g["mean_scene_clear_fraction"]:.2%} | {g["planning_passes"]}/{g["positions"]}（{g["planning_passes"]*100/g["positions"]:.1f}%） |')
    record+=['','最高密度组的平均IK通过比例达到'+f'{q[-1]["mean_pure_ik_fraction"]:.1%}，最低组为{q[0]["mean_pure_ik_fraction"]:.1%}。'
        f'完整规划通过率从{q[0]["planning_passes"]*100/q[0]["positions"]:.1f}%升至{q[-1]["planning_passes"]*100/q[-1]["positions"]:.1f}%；最高组略低于50–75%组的{q[2]["planning_passes"]*100/q[2]["positions"]:.1f}%，碰撞仍会拦下一部分位于高密度区的抓法。','',
        '[![可达性地图与318个测试位置叠加](/media/reach-distance-spatial-overlay.png)](/media/reach-distance-spatial-overlay.png)','',
        '图中箭头表示同一场景向近侧移动的方向，底图是实际抓取高度带的地图密度。绿色为初始轨迹通过，红色为规划不出来；每个点代表该位置全部候选TCP的平均位置。统计计算使用每个候选真实的三维坐标。','',
        '| 地图密度与纯IK | 地图密度与无碰撞终点 | 各密度组的轨迹通过率 |',
        '| --- | --- | --- |',
        '| [![地图密度与纯IK](/media/reach-distance-density-vs-ik.png)](/media/reach-distance-density-vs-ik.png) | [![地图密度与无碰撞终点](/media/reach-distance-density-vs-scene.png)](/media/reach-distance-density-vs-scene.png) | [![密度分组](/media/reach-distance-density-groups.png)](/media/reach-distance-density-groups.png) |','',
        '### 同时核对朝向后，趋势是否还成立','',
        '每个原场景按原生评分排名等距取64个候选，所有距离使用相同编号；在原地图的邻近关节见证中，同时比较位置与朝向。抓取、预抓取和抬升终点共核对64512个位姿。','',
        '| 地图指标 | 与纯IK通过比例的相关系数 | 与无碰撞IK比例的相关系数 |',
        '| --- | ---: | ---: |',
        f'| 位置密度，原5毫米格 | {cor["fine_density_mean"]["spearman_pure_ik"]:.3f} | {cor["fine_density_mean"]["spearman_scene_clear"]:.3f} |',
        f'| 位置密度，5厘米范围汇总 | {cor["coarse_density_mean"]["spearman_pure_ik"]:.3f} | {cor["coarse_density_mean"]["spearman_scene_clear"]:.3f} |',
        f'| 位置与朝向共同匹配的相近度 | {cor["pose_g_affinity_mean"]["spearman_pure_ik"]:.3f} | {cor["pose_g_affinity_mean"]["spearman_scene_clear"]:.3f} |',
        f'| 抓取、预抓取、抬升三点的平均相近度 | {cor["pose_hgl_mean_affinity"]["spearman_pure_ik"]:.3f} | {cor["pose_hgl_mean_affinity"]["spearman_scene_clear"]:.3f} |',
        f'| 三点都相近的最佳采样候选 | {cor["pose_hgl_best_affinity"]["spearman_pure_ik"]:.3f} | {cor["pose_hgl_best_affinity"]["spearman_scene_clear"]:.3f} |','',
        '同时考虑朝向后，相关性仍然很高。对原位与最近档作配对比较，53个场景的地图密度和纯IK通过比例全部提高；52个场景的位置与朝向平均相近度也提高。最终39例恢复轨迹，14例仍被当前约束拦住。','',
        '### 高密度区里，哪些仍然失败','',
        '下面五例已经进入很密集的地图区域，也解出了不少IK，但无碰撞终点仍为0。它们对应上一轮移到近侧桌沿后仍未恢复的剪刀、手机和钥匙扣场景。','',
        '| 最近档场景 | 地图平均样本数 | 纯IK通过比例 | 无碰撞IK比例 | 完整轨迹 |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in hard[:5]:
        record.append(f'| {row["source_id"]} | {row["coarse_density_mean"]:.0f} | {row["ik_fraction"]:.1%} | {row["clear_fraction"]:.1%} | 规划不出来 |')
    record+=['','这说明地图对判断运动学可达性有帮助。桌面与挡物决定这些运动学解能否继续变成无碰撞轨迹，需要当前场景的独立检查。','',
        '### 原三物体18位置对照','',
        f'原香蕉、食品盒、甜甜圈的18位置对照单独统计。地图密度与纯IK比例的相关系数为{pilot["correlations"]["coarse_density_mean"]["spearman_pure_ik"]:.3f}，与无碰撞终点比例为{pilot["correlations"]["coarse_density_mean"]["spearman_scene_clear"]:.3f}；三个物体靠近后都进入更密集的区域并恢复轨迹。','',
        '### 对V2的具体启发','',
        '距离修正可以把地图作为站位与目标区域的运动学先验，优先考察原抓法更容易到达的位置与朝向。保留“原抓法先做IK”的顺序，IK有解就保留原抓法，再检查碰撞和路径；仅在原抓法未解出IK时，才使用地图见证辅助或小幅修正。','',
        '当前结果支持用地图帮助选择合适的相对站位。单纯把目标送进高密度区，仍会留下碰桌和邻物阻挡的案例；这部分按原位原因分别处理。','',
        '### 数据与坐标核对','',
        '| 项目 | 本次使用 |', '| --- | --- |',
        '| 地图 | 11月21日冻结的前方纯运动学地图，16777216个关节采样，其中10559134个前方见证；原位置格5毫米 |',
        '| 主对照 | 53个原失败场景、318个距离位置，所有778416个候选均做位置查表 |',
        '| 三物体对照 | 18个位置、39024个候选，独立统计 |',
        '| 连续朝向检查 | 每原场景64个固定候选编号，抓取、预抓取、抬升三种终点，共64512次地图查询 |',
        '| 坐标 | 每次试验的实际body4局部系，已补齐原地图TCP工具轴的90°旋转，位置偏移均为43毫米 |',
        '| 构型核对 | 当前URDF与原地图一致，六轴关节限制一致；512个地图见证和312个试验见证通过独立FK核对 |',
        '| 场景碰撞 | 原地图不包含桌面和杂物，碰撞标签来自已完成的距离试验 |',
        '| 不确定性 | 按53个原场景成组重采样500次，保留同一场景六档的关联；密度与纯IK相关系数95%区间为'+
            f'{cor["coarse_density_mean"]["scene_bootstrap_95ci_spearman_pure_ik"][0]:.3f}–{cor["coarse_density_mean"]["scene_bootstrap_95ci_spearman_pure_ik"][1]:.3f} |',
        '| 执行 | 本次只读CPU分析，新增IK与实际抓取均为0，原地图及试验数据未修改 |','',
        '[下载318个位置的地图对照数据](/media/reach-distance-all53.csv) · [下载三物体18位置数据](/media/reach-distance-pilot3.csv) · [完整统计](/media/reach-distance-summary.json)','']
    raw='\n'.join(record)
    (DOC/'距离测试与可达性地图对照.md').write_text('# 距离测试与可达性地图对照\n\n'+raw)
    (SITE/'内容/迭代/sam3-graspgenx-reachability-distance.md').write_text(raw)
    keys=['id','source_id','level','candidates','edge_gap_m','fine_density_mean','coarse_density_mean',
        'coarse_orientation_fraction_mean','pose_g_affinity_mean','pose_hgl_best_affinity','ik_fraction','clear_fraction','planning_success']
    assets={}
    for cohort in ['all53','pilot3']:
        output=io.StringIO();writer=csv.writer(output);writer.writerow(keys)
        writer.writerows([[r[k] for k in keys] for r in rows if r['cohort']==cohort])
        (MEDIA/(cohort+'.csv')).write_text(output.getvalue(),encoding='utf-8-sig')
        assets[f'reach-distance-{cohort}.csv']='reachability-distance/'+cohort+'.csv'
    for filename in ['spatial_overlay.png','density_vs_ik.png','density_vs_scene.png','density_groups.png']:
        shutil.copy2(OUT/filename,MEDIA/filename);assets['reach-distance-'+filename.replace('_','-')]='reachability-distance/'+filename
    shutil.copy2(ROOT/'summary.json',MEDIA/'summary.json');assets['reach-distance-summary.json']='reachability-distance/summary.json'
    registry=SITE/'内容/网站.json';site=read(registry);site['localAssets'].update(assets)
    registry.write_text(json.dumps(site,ensure_ascii=False,indent=2)+'\n')
    v2=SITE/'内容/迭代/sam3-graspgenx-v2.md';text=v2.read_text()
    text=text.split('\n## 可达性地图对照的启发')[0]
    text+='\n## 可达性地图对照的启发\n\n'
    text+='对318个距离位置的对照，地图密度与纯IK通过比例的排序相关系数为0.930；最低密度组平均IK通过0.9%，最高组54.8%。53个场景靠近后，地图密度与纯IK比例全部提高。\n\n'
    text+='V2可以用地图给站位与目标区域提供运动学先验，同时核对抓取朝向。原抓法先做IK，有解就保留并继续检查碰撞、路径；原IK未解出时，再查询地图辅助或小幅修正。5例进入很密集区域仍碰撞的场景，继续按抓姿和场景几何处理。\n\n'
    text+='[查看V1的地图叠加图、分组表与完整分析](/iterations/sam3-graspgenx/versions/V1/#section-3d53841876)\n'
    v2.write_text(text)
    save={'passed':True,'primary_positions':318,'pilot_positions':18,'alignment_verified':alignment['passed'],
        'correlation_density_ik':cor['coarse_density_mean']['spearman_pure_ik'],
        'correlation_density_clear':cor['coarse_density_mean']['spearman_scene_clear'],**spatial}
    (ROOT/'report_verification.json').write_text(json.dumps(save,ensure_ascii=False,indent=2)+'\n')
    subprocess.run(['python3',str(SITE/'tools/publish_sam3_v1.py'),'publish'],cwd=SITE,check=True)
    for name in assets:
        req=urllib.request.Request('http://127.0.0.1:8786/media/'+name,method='HEAD')
        with urllib.request.urlopen(req,timeout=10) as response:assert response.status==200
    with urllib.request.urlopen('http://127.0.0.1:8786/iterations/sam3-graspgenx/versions/V1/',timeout=10) as response:
        text=response.read().decode();assert '距离测试与可达性地图对照' in text and '0.93' in text
    (ROOT/'publication_verification.json').write_text(json.dumps({'passed':True,'media_assets_checked':len(assets),'v1_updated':True,'v2_updated':True},indent=2)+'\n')
    print('REPORT COMPLETE; rho density/IK',cor['coarse_density_mean']['spearman_pure_ik'],flush=True)


if __name__=='__main__':main()
