"""Generate inspectable figures and a dependency-free interactive map."""
import base64
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from 公共 import ROOT,read,save,sha,now


def main():
    directory=ROOT/'产物/首版_01'
    meta=read(directory/'地图元数据.json');complete=read(directory/'完成.json')
    assert complete['passed']
    for name,digest in complete['file_sha256'].items():assert sha(directory/name)==digest
    out=ROOT/'docs/可达性地图/resources/首版_01';out.mkdir(parents=True,exist_ok=True)
    coarse=dict(np.load(directory/'概览体素统计.npz',allow_pickle=False))
    fine=dict(np.load(directory/'精细体素统计.npz',allow_pickle=False))
    base=np.asarray(meta['reference_base_world']);inverse=np.linalg.inv(base)
    table=meta['static_world_cuboids']['table'];center=np.asarray(table['center']);dims=np.asarray(table['dims'])
    top=center[2]+dims[2]/2
    corners=np.array([[center[0]+sx*dims[0]/2,center[1]+sy*dims[1]/2,top] for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    corners=corners@inverse[:3,:3].T+inverse[:3,3]

    def coordinates(data):
        p=(data['cells'].astype(float)+.5)*float(data['resolution_m'])
        world=p@base[:3,:3].T+base[:3,3]
        height=world[:,2]-top
        roi=(np.abs(world[:,0]-center[0])<=dims[0]/2)&(np.abs(world[:,1]-center[1])<=dims[1]/2)&(height>=.03)&(height<=.30)
        return p,world,height,roi

    p,world,height,roi=coordinates(coarse)
    pf,wf,hf,rf=coordinates(fine)
    down=coarse['down_orientation_count']/288
    order=np.flatnonzero(roi)[np.argsort(-down[roi],kind='stable')]
    selected=[]
    for index in order:
        if down[index]<=0:continue
        if selected and np.min(np.linalg.norm(p[selected]-p[index],axis=1))<.10:continue
        selected.append(int(index))
        if len(selected)==20:break
    fields=['rank','body4_x_m','body4_y_m','body4_z_m','world_x_m','world_y_m','world_z_m',
            'height_above_table_m','observed_down_fraction','observed_all_fraction','joint_samples',
            'mean_joint_margin_rad','mean_condition_index']
    records=[]
    for rank,i in enumerate(selected,1):
        values=[rank,*p[i],*world[i],height[i],down[i],coarse['orientation_count'][i]/1152,
                int(coarse['sample_count'][i]),float(coarse['margin_mean_rad'][i]),float(coarse['condition_mean'][i])]
        records.append(dict(zip(fields,values)))
    with (out/'桌面高覆盖区域.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(records)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    fig=plt.figure(figsize=(13,6));axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    for ax,metric,title in zip(axes,[coarse['orientation_count']/1152,down],['All approach directions','Approach within 60 degrees of down']):
        image=ax.scatter(*p.T*100,c=metric*100,cmap='viridis',s=2,alpha=.65,rasterized=True)
        polygon=np.vstack([corners,corners[:1]])*100;ax.plot(*polygon.T,color='#bf4b19',lw=1.5)
        ax.scatter([0],[0],[0],marker='+',color='black',s=60)
        ax.set(xlabel='body4 X (cm)',ylabel='body4 Y (cm)',zlabel='body4 Z (cm)',title=title)
        ax.view_init(elev=23,azim=-54);ax.set_box_aspect(np.ptp(p,axis=0));fig.colorbar(image,ax=ax,shrink=.55,label='Observed orientation bins (%)',pad=.08)
    fig.suptitle('ARX action TCP43 | 5 cm overview cells | fixed lift 0.22 m',fontsize=14)
    fig.text(.5,.015,'Observed coverage is sample-dependent, not IK/grasp success probability. Blank = unknown.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.01,right=.96,bottom=.1,top=.89,wspace=.12)
    fig.savefig(out/'三维工作区概览.png',dpi=155);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    norm=Normalize(0,float(fine['down_orientation_count'].max()/288*100))
    for ax,h in zip(axes.flat,[.05,.10,.15,.20]):
        mask=rf&(np.abs(hf-h)<=.01001)
        image=ax.scatter(pf[mask,0]*100,pf[mask,1]*100,c=fine['down_orientation_count'][mask]/288*100,
                         cmap='viridis',norm=norm,s=12,marker='s',edgecolors='none')
        polygon=np.vstack([corners,corners[:1]])*100;ax.plot(polygon[:,0],polygon[:,1],color='#bf4b19',lw=1)
        ax.set_title(f'TCP height above table: {h*100:.0f} cm (slice +/- 1 cm)')
        ax.set_aspect('equal');ax.set_xlim(corners[:,0].min()*100-2,min(125,corners[:,0].max()*100+2))
        ax.set_ylim(-90,65);ax.grid(alpha=.15);ax.set_xlabel('body4 X (cm)');ax.set_ylabel('body4 Y (cm)')
    fig.subplots_adjust(left=.08,right=.84,bottom=.09,top=.9,wspace=.15,hspace=.25)
    cb=fig.add_axes([.88,.22,.025,.57]);fig.colorbar(image,cax=cb,label='Observed down-approach orientation bins (%)')
    fig.suptitle('Tabletop detail | 2 cm cells | same arm, limits, TCP and static checker',fontsize=13)
    fig.text(.48,.025,'Cell colors include all samples in each height slab. Orange = table boundary. Blank = unknown.',ha='center',fontsize=9)
    fig.savefig(out/'桌面高度切片.png',dpi=155);plt.close(fig)

    data=np.column_stack([p,coarse['orientation_count']/1152,down,coarse['sample_count'],
                          coarse['margin_mean_rad'],coarse['condition_mean'],height,roi.astype(float)]).astype('<f4')
    payload=base64.b64encode(data.tobytes()).decode()
    configuration={'rows':len(data),'columns':data.shape[1],'table':corners.tolist(),'regions':records}
    template=Path(__file__).with_name('地图页面模板.html').read_text()
    font=Path(__file__).with_name('资源')/'地图中文.woff'
    page=template.replace('__MAP_CONFIGURATION__',json.dumps(configuration,ensure_ascii=False)).replace('__MAP_DATA__',payload)
    page=page.replace('__MAP_FONT__',base64.b64encode(font.read_bytes()).decode())
    (out/'交互可达性地图.html').write_text(page)
    save(out/'展示依据.json',{'created_at':now(),'map_directory':str(directory),'map_hashes':complete['file_sha256'],
         'coarse_cells':len(p),'fine_cells':len(pf),'tabletop_coarse_cells':int(roi.sum()),
         'tabletop_fine_cells':int(rf.sum()),'top_regions':records,
         'region_selection':'Top down-approach observed coverage within table XY and 3-30 cm TCP height; centers >=10 cm apart. Display only, never a candidate filter.',
         'html_bytes':len(page.encode()),'physical_attempts':0,'code_sha256':sha(Path(__file__)),
         'template_sha256':sha(Path(__file__).with_name('地图页面模板.html')),'subset_font_sha256':sha(font)})
    print('VISUALS_COMPLETE',out,'html_bytes',len(page.encode()),'tabletop_cells',int(roi.sum()),flush=True)


if __name__=='__main__':main()
