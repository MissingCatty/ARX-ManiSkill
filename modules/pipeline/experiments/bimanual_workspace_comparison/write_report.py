"""Draw and publish positional overlap of actual ARX left and right arms."""
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ROOT=HERE/'runs/dual_arm_20261123_01'
ATLAS=PROJECT.parent/'reachability/产物/前方精细_20261121_01'
DISTANCE=PROJECT/'experiments/reachability_distance_comparison/runs/map_distance_20261123_01'
SITE=PROJECT.parent/'claro/进度网站'
DOC=PROJECT/'docs/diverse_grasp_benchmark/debug_20261123'
OUT=DOC/'resources/bimanual_workspace'
MEDIA=SITE/'内容/资源/bimanual-workspace'
FONT=PROJECT.parent/'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'


def read(p):return json.loads(Path(p).read_text())


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    coarse=dict(np.load(ATLAS/'概览体素统计.npz'));right=coarse['cells'].astype(int);left=right+[0,10,0]
    count=coarse['sample_count'];ids=coarse['voxel_id'];shared=np.isin(ids,ids+40960)
    rows=[r for r in read(DISTANCE/'rows.json') if r['cohort']=='all53']
    median_z=float(np.median([r['tcp_mean_body4'][2] for r in rows]));layer=int(np.floor(median_z/.05))
    rmask=right[:,2]==layer;lmask=left[:,2]==layer
    lo=np.minimum(right[rmask,:2].min(0),left[lmask,:2].min(0));hi=np.maximum(right[rmask,:2].max(0),left[lmask,:2].max(0))
    rgrid=np.zeros(tuple(hi-lo+1));lgrid=rgrid.copy()
    rgrid[tuple((right[rmask,:2]-lo).T)]=count[rmask]
    lgrid[tuple((left[lmask,:2]-lo).T)]=count[lmask]
    state=np.zeros_like(rgrid,dtype=int);state[rgrid>0]=1;state[lgrid>0]=2;state[(rgrid>0)&(lgrid>0)]=3
    x=np.arange(lo[0],hi[0]+2)*5.;y=np.arange(lo[1],hi[1]+2)*5.
    xc=(np.arange(lo[0],hi[0]+1)+.5)*5.;yc=(np.arange(lo[1],hi[1]+1)+.5)*5.
    threshold=summary['resolutions'][0]['scopes']['table_height_band']['right_high_density_threshold']
    maxlog=np.log1p(max(rgrid.max(),lgrid.max()))

    def shoulders(ax):
        ax.scatter([5.9949],[25],s=40,c='#e29439',marker='s');ax.scatter([5.9949],[-25],s=40,c='#207eb0',marker='s')
        ax.text(8,26,'左肩',fontsize=10);ax.text(8,-29,'右肩',fontsize=10)
        ax.axhline(0,color='#6b7f88',ls=':',lw=.8)
        ax.set_xlabel('body4前方 X（厘米）');ax.set_ylabel('body4左右 Y（厘米）');ax.set_aspect('equal')
        ax.set_xlim(x[0],x[-1]);ax.set_ylim(y[0],y[-1])

    for name,grid,title,cmap in [('right_density.png',rgrid,'右臂采样密度','Blues'),('left_density.png',lgrid,'左臂采样密度','Oranges')]:
        fig,ax=plt.subplots(figsize=(5,7.2),constrained_layout=True)
        image=ax.pcolormesh(x,y,np.ma.masked_where(grid.T==0,np.log1p(grid.T)),cmap=cmap,vmin=0,vmax=maxlog,shading='flat')
        ax.contour(xc,yc,(grid>=threshold).T,levels=[.5],colors=['#333444'],linewidths=1.1)
        shoulders(ax);ax.set_title(f'{title}\nTCP Z层：{layer*5}至{(layer+1)*5}厘米')
        cb=fig.colorbar(image,ax=ax,shrink=.7);cb.set_label('log(1+采样数)；深色线圈出高密度格')
        fig.savefig(OUT/name,dpi=150,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,7.2),constrained_layout=True)
    colors=['white','#78b4d1','#f1ba74','#ac91ce']
    ax.pcolormesh(x,y,state.T,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4.5),4),shading='flat')
    ax.contour(xc,yc,(rgrid>=threshold).T,levels=[.5],colors=['#145c86'],linewidths=1.4)
    ax.contour(xc,yc,(lgrid>=threshold).T,levels=[.5],colors=['#b06b16'],linewidths=1.4)
    shoulders(ax);ax.set_title(f'左右臂位置覆盖交集\nTCP Z层：{layer*5}至{(layer+1)*5}厘米；轮廓线为各臂高密度区')
    ax.legend(handles=[Patch(color=colors[1],label='仅右臂覆盖'),Patch(color=colors[2],label='仅左臂覆盖'),Patch(color=colors[3],label='两臂均覆盖')],loc='upper right',fontsize=9)
    fig.savefig(OUT/'coverage_overlap.png',dpi=150,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    fig=plt.figure(figsize=(8,6.8),constrained_layout=True);ax=fig.add_subplot(111,projection='3d')
    rpoints=(right+.5)*.05;lpoints=(left+.5)*.05
    for points,color,label in [(rpoints[~shared],'#3d8db8','仅右臂位置覆盖'),(lpoints[~np.isin(ids+40960,ids)],'#d88a32','仅左臂位置覆盖'),(rpoints[shared],'#9467bd','双臂位置覆盖交集')]:
        ax.scatter(points[:,0],points[:,1],points[:,2],s=3,c=color,alpha=.25,label=label,depthshade=False)
    ax.scatter([.059949,.059949],[-.25,.25],[.0245,.0245],s=50,c=['#1b5474','#aa6517'])
    ax.set_xlabel('body4 X（米）');ax.set_ylabel('body4 Y（米）');ax.set_zlabel('body4 Z（米）')
    ax.set_title('双臂前方位置可达体积');ax.view_init(elev=20,azim=-42);ax.legend(loc='upper left',fontsize=9)
    fig.savefig(OUT/'volume_overlap.png',dpi=150,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    return {'display_layer_body4_m':[layer*.05,(layer+1)*.05],'test_tcp_median_z_m':median_z}


def main():
    assert read(ROOT/'completion_verification.json')['passed']
    summary=read(ROOT/'summary.json');alignment=read(ROOT/'model_alignment.json')
    OUT.mkdir(parents=True,exist_ok=True);MEDIA.mkdir(parents=True,exist_ok=True)
    display=figures(summary)
    r5=summary['resolutions'][0]['scopes'];r2=summary['resolutions'][1]['scopes']
    text=['## 高密度区域与双臂空间交集','',
        '**当前桌面高度下，右臂高密度区域大部分位于两臂位置覆盖的交集中，但两臂各自的高密度核心重合较少。**', '',
        '现有可达性地图只统计右臂。通过实际完整机器人URDF核对，左右臂的六轴关节轴、限位和连杆变换相同，左肩比右肩沿body4的Y方向平移50厘米。因此，用同一组合法关节配置得到的左臂TCP位置，恰好是右臂TCP位置沿Y平移50厘米，朝向相同。', '',
        '这里比较左右臂分别能覆盖的位置；每条手臂可以采用自己的末端朝向。高密度按各范围内空间格采样数的第75百分位定义，包含同分格。下表两个比例都以右臂高密度格数量为分母。','',
        '| 比较范围 | 右臂高密度区也被左臂覆盖，5厘米格 | 两臂在这里都高密度，5厘米格 | 2厘米格复核：左臂也覆盖 | 2厘米格复核：两臂都高密度 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,label in [('all_heights','整个前方空间'),('table_height_band','当前桌面高度带')]:
        a,b=r5[name],r2[name]
        text.append(f'| {label} | {a["right_high_in_left_fraction"]:.1%} | {a["right_high_also_left_high_fraction"]:.1%} | {b["right_high_in_left_fraction"]:.1%} | {b["right_high_also_left_high_fraction"]:.1%} |')
    text+=['','当前桌面高度带取body4局部Z为−25至−10厘米，与前一轮地图叠加分析使用的抓取高度一致。','',
        '### 直观看左右臂与交集','',
        '| 右臂密度 | 左臂密度 | 左右臂位置交集 |','| --- | --- | --- |',
        '| [![右臂密度](/media/bimanual-right-density.png)](/media/bimanual-right-density.png) | [![左臂密度](/media/bimanual-left-density.png)](/media/bimanual-left-density.png) | [![双臂交集](/media/bimanual-coverage-overlap.png)](/media/bimanual-coverage-overlap.png) |','',
        '蓝色为仅右臂覆盖，橙色为仅左臂覆盖，紫色为两臂均覆盖。深色轮廓线分别标出各臂的高密度区域。图中展示距离试验候选TCP中位高度所在的空间层，全部统计使用三维格。','',
        '### 为什么大部分重叠，却不能把它们当成同一片区域','',
        f'在当前高度带，右臂共有{r5["table_height_band"]["right_occupied_cells"]}个位置覆盖格，其中{r5["table_height_band"]["shared_cells"]}格也被左臂覆盖，约占{r5["table_height_band"]["shared_fraction_of_right_volume"]:.1%}。'
        f'右臂高密度格有{r5["table_height_band"]["right_high_density_cells"]}个，其中{r5["table_height_band"]["right_high_in_left_cells"]}个位于双臂交集；但只有{r5["table_height_band"]["right_high_also_left_high_cells"]}个在左臂图中也属于高密度。','',
        '高密度来自该手臂自身合法关节配置的分布。左右地图整体相隔50厘米，密度峰值也随各自肩部位置分开；两臂能够覆盖的中央区域比较大，而各自最密集的区域只重合一部分。原图的深蓝色由右臂采样得到。','',
        '[![双臂三维位置覆盖](/media/bimanual-volume-overlap.png)](/media/bimanual-volume-overlap.png)','',
        '### 与距离试验的联系','',
        f'距离试验中131个已经通过规划的选中抓取目标，按所在5厘米空间格查询，有{summary["selected_grasp_points_in_both"]}个（{summary["selected_grasp_points_in_both"]/131:.1%}）位于双臂共同覆盖区域。'
        '因此，当前规划较容易通过的位置与双臂共同覆盖区域有较大重叠，同时也保留了单臂能够完成规划的位置。','',
        '如果后续做左右臂选择，可以比较两条臂对同一个抓取目标的地图先验，再分别核对具体抓姿的IK、碰撞和路径。当前距离试验仍是右臂执行链路。','',
        '### 模型与统计核验','',
        '| 项目 | 核对结果 |','| --- | --- |',
        '| 机器人 | 当前完整ARX机器人URDF；右臂与原地图使用的参考链条一致 |',
        '| 左右臂关系 | 六轴关节限制、关节轴与其余连杆变换相同，肩部Y方向相差0.5米 |',
        '| TCP | 两臂统一使用EEF后接43毫米工具点，采用相同工具坐标定义 |',
        '| 独立FK | 512组关节配置分别通过完整左右臂链条复算，右臂与原地图见证一致，左右差值满足固定平移关系 |',
        '| 数据 | 复用原地图10559134条合法前方见证，利用已证明的模型关系推导左臂覆盖 |',
        '| 高度与空间格 | 前方180°，三维5厘米格为主，2厘米格复核；高度带按空间格中心选择 |',
        '| 碰撞 | 与原纯运动学地图一致，本图统计各臂位置覆盖，桌面、邻物及双臂执行碰撞另行检查 |',
        '| 本次执行 | 新增IK、机器人动作和实际抓取均为0，原地图与距离试验数据保持冻结 |','',
        '[下载完整双臂比较结果](/media/bimanual-workspace-summary.json) · [下载模型核对结果](/media/bimanual-model-alignment.json)','']
    raw='\n'.join(text)
    (DOC/'双臂可达交集与高密度区域对照.md').write_text('# 双臂可达交集与高密度区域对照\n\n'+raw)
    (SITE/'内容/迭代/sam3-graspgenx-bimanual-workspace.md').write_text(raw)
    registry=SITE/'内容/网站.json';site=read(registry);assets={}
    for filename in ['right_density.png','left_density.png','coverage_overlap.png','volume_overlap.png']:
        shutil.copy2(OUT/filename,MEDIA/filename);assets['bimanual-'+filename.replace('_','-')]='bimanual-workspace/'+filename
    for name,alias in [('summary.json','bimanual-workspace-summary.json'),('model_alignment.json','bimanual-model-alignment.json')]:
        shutil.copy2(ROOT/name,MEDIA/name);assets[alias]='bimanual-workspace/'+name
    site['localAssets'].update(assets);registry.write_text(json.dumps(site,ensure_ascii=False,indent=2)+'\n')
    subprocess.run(['python3',str(SITE/'tools/publish_sam3_v1.py'),'publish'],cwd=SITE,check=True)
    for name in assets:
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8786/media/'+name,method='HEAD'),timeout=10) as r:assert r.status==200
    (ROOT/'publication_verification.json').write_text(json.dumps({'passed':True,'assets_checked':len(assets),'v1_updated':True,**display},indent=2)+'\n')
    print('BIMANUAL REPORT COMPLETE',r5['table_height_band']['right_high_in_left_fraction'],r5['table_height_band']['right_high_also_left_high_fraction'])


if __name__=='__main__':main()
