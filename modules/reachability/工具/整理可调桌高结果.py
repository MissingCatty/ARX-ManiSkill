"""Validate saved scan witnesses and export a Chinese report and figure."""
from pathlib import Path
import csv
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from 公共 import ROOT, URDF, JOINTS, read, save, sha
from 计算可调桌高 import table_state


def main():
    source=ROOT/'产物/桌高与相对高度_20261120_01'
    result=read(source/'相对高度结论.json');reg=read(source/'登记.json')
    assert result['passed'] and result['physical_attempts']==0
    assert all(sha(p)==h for p,h in reg['source_sha256'].items())
    rows=sorted(result['rows'],key=lambda r:r['table_height_m'])
    assert len(rows)==result['heights']==len({r['table_height_m'] for r in rows})
    assert result['pose_queries']==len(rows)*reg['pose_probes_per_height']
    robot=ET.parse(URDF).getroot()
    joints={j.get('name'):j for j in robot.findall('joint')}
    lower=np.array([float(joints[j].find('limit').get('lower')) for j in JOINTS])
    upper=np.array([float(joints[j].find('limit').get('upper')) for j in JOINTS])
    initial=read(reg['reference_state'])
    counts=[]
    for row in rows:
        height=row['table_height_m']
        witness=np.load(source/f'见证_桌高{round(height*1000):03d}.npz')
        flags=witness['passed'];q=witness['q'][flags]
        assert flags.shape==(9264,)
        shaped=flags.reshape(193,3,2,8)
        assert shaped[:,:,0].sum()==row['vertical_pose_passes']
        assert abs(shaped.mean()-row['mixed_pose_fraction'])<1e-12
        assert abs(shaped[:,:,1].mean()-row['tilted_pose_fraction'])<1e-12
        assert np.all((q>=lower-1e-6)&(q<=upper+1e-6))
        state,delta=table_state(initial,reg['fixed_legal_joint4_m'],height)
        table=state['scene_cuboids']['table'];ground=state['scene_cuboids']['ground']
        assert abs(table['center'][2]-table['dims'][2]/2-reg['floor_world_z_m'])<1e-9
        assert state['links']['base_link']==initial['links']['base_link']
        assert ground==initial['scene_cuboids']['ground']
        assert abs(delta-(height-.5))<1e-9
        counts.append({'table_height_m':height,'vertical_passes':int(shaped[:,:,0].sum()),
            'tilted_passes':int(shaped[:,:,1].sum()),'all_saved_solutions_within_limits':True})
    best=result['best_vertical'];baseline=next(r for r in rows if r['table_height_m']==.5)
    resource=ROOT/'docs/resource/可调桌高_20261120_01';resource.mkdir(parents=True,exist_ok=True)
    verification={'passed':True,'heights_checked':len(rows),'pose_queries':result['pose_queries'],
        'source_hashes_unchanged':True,'table_bottom_and_ground_verified':True,
        'saved_witness_counts_verified':True,'saved_solutions_joint_limits_verified':True,
        'rows':counts,'physical_attempts':0}
    save(resource/'结果核验.json',verification)
    for name in ['登记.json','相对高度结论.json','桌高扫描.json','细化登记.json']:
        shutil.copy2(source/name,resource/name)
    with (resource/'桌高覆盖率.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    font=FontProperties(fname='/home/zhaoyechong/projects/ARX-ManiSkill/tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font.get_file())
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,zoom in zip(axes,[False,True]):
        for key,label,color in [('vertical_pose_fraction','竖直抓取姿态','#1976b9'),
                                 ('tilted_pose_fraction','倾斜 30° 姿态','#b56716'),
                                 ('mixed_pose_fraction','两类姿态等权合计','#318153')]:
            ax.plot([r['table_height_m']*100 for r in rows],[r[key]*100 for r in rows],
                label=label,color=color,marker='.',markersize=4,linewidth=1.5)
        ax.axvline(50,color='#767676',linestyle='--',linewidth=1,label='原桌高 50 cm')
        ax.axvline(best['table_height_m']*100,color='#b62934',linestyle=':',linewidth=1.5,label='竖直覆盖最高档')
        ax.set_xlabel('桌面离地高度（cm）');ax.set_ylabel('姿态通过率（%）');ax.grid(alpha=.2)
        if zoom:ax.set_xlim(38,54);ax.set_ylim(43,55);ax.set_title('较好区间细节')
        else:ax.set_title('全部高度：桌面并非越低越好');ax.legend(fontsize=8,loc='lower center')
    fig.suptitle('固定升降关节 45.2 cm；仅调整桌面及其上物体的高度；静态桌地碰撞 IK')
    fig.savefig(resource/'桌高与可达性.png',dpi=160);plt.close(fig)
    display_heights={.1,.2,.3,.4,.44,.46,.48,.5,.6,.7,.8,.9,best['table_height_m']}
    selected=[r for r in rows if r['table_height_m'] in display_heights]
    table='\n'.join(f"| {r['table_height_m']*100:.1f} | {r['shoulder_above_table_m']*100:.2f} | {r['vertical_pose_fraction']*100:.2f}% | {r['tilted_pose_fraction']*100:.2f}% | {r['mixed_pose_fraction']*100:.2f}% |" for r in selected)
    gain=(best['vertical_pose_fraction']-baseline['vertical_pose_fraction'])*100
    text=f'''# 可调桌高与抓取相对高度

2026 年 11 月 20 日。选择调整桌高，保持机器人底盘安装不变。前提是桌子本身可调；这样保留机器人安装和基座标定，调整桌面、支撑体及其上全部物体的共同高度。这里按当前仿真链路的改动范围选择，不估算真实硬件采购成本。

## 结论

固定已选定的合法升降关节读数 **45.2 cm**，肩轴离地约 **{result['shoulder_above_floor_m']*100:.2f} cm**。本次固定测试姿态中，竖直覆盖率最高的桌高为 **{best['table_height_m']*100:.1f} cm**，比原 50 cm 桌面降低 **{(0.5-best['table_height_m'])*100:.1f} cm**，对应肩轴高出桌面 **{best['shoulder_above_table_m']*100:.2f} cm**。

竖直姿态通过率从原桌高的 **{baseline['vertical_pose_fraction']*100:.2f}%** 变为 **{best['vertical_pose_fraction']*100:.2f}%**，增加 **{gain:.2f} 个百分点**。相对于已经提高升降关节的配置，这一收益较小；不能把它解释成实际抓取成功率增加。

相对高度定义为：**肩轴高出桌面的距离＝肩轴离地高度－桌面离地高度**。本安装关系下，肩轴离地高度约等于 **33.45 cm＋升降关节读数＋底盘额外垫高**。降低桌面与抬高底盘会同向改变该距离；地板和支撑体的碰撞关系仍需各自检查。

## 测量结果

| 桌面离地高度（cm） | 肩轴高出桌面（cm） | 竖直姿态通过率 | 倾斜 30° 通过率 | 两类等权合计 |
| --- | --- | --- | --- | --- |
{table}

![桌高与可达性](../resource/可调桌高_20261120_01/桌高与可达性.png)

## 计算口径

先扫描桌高 10–90 cm，每 2 cm 一档，再在竖直覆盖最佳粗档附近 ±12 mm 内每 2 mm 细化，共 **{result['heights']} 档、{result['pose_queries']:,} 个姿态查询**。每档使用相同的 193 个目标 XY、桌上 3／8／13 cm 三种 TCP 高度、竖直及倾斜 30° 两种朝向、8 个夹爪 yaw，共 9,264 个姿态。竖直姿态的分母为 4,632。

机器人底盘、合法升降值、关节限位、TCP43、求解参数及原碰撞策略保持一致；桌面和物体一起移动，桌体底部仍落在原地板上。每档重新检查桌体和地板碰撞。使用 32 个 IK 初值及 4 个返回分支；通过解另用 URDF 正运动学核对，最大 EEF 位置误差 {result['maximum_independent_eef_error_m']*1000:.3f} mm，最大朝向误差 {result['maximum_independent_rotation_error_rad']*180/np.pi:.3f}°。原文件散列、全部见证计数和通过解限位已复核。

这是固定测试姿态的静态几何比较，不包含杂物、完整运动路径和抓持质量，也没有进行新桌高的实际抓取。2 mm 的扫描步长不代表物理最优高度具有毫米精度。当前正在运行的原桌高 400 槽对照保持自己的登记，未改桌高或场景。

要决定新桌高是否值得用于正式抓取，应先对较好区间的冻结真实候选检查完整场景及路径；本轮计算没有据此额外启动物理试次。

## 数据与复现

- [计算登记](../resource/可调桌高_20261120_01/登记.json)
- [完整结论](../resource/可调桌高_20261120_01/相对高度结论.json)
- [全部高度表](../resource/可调桌高_20261120_01/桌高覆盖率.csv)
- [结果核验](../resource/可调桌高_20261120_01/结果核验.json)

计算入口：`工具/计算可调桌高.py`；图表及计数复核：`工具/整理可调桌高结果.py`。原始见证及代码快照在 `产物/桌高与相对高度_20261120_01/`。本轮实际抓取次数为 0。
'''
    path=ROOT/'docs/可达性地图/可调桌高与抓取相对高度.md';path.write_text(text)
    print(json.dumps({'document':str(path),'best_table_cm':best['table_height_m']*100,
        'shoulder_above_table_cm':best['shoulder_above_table_m']*100,
        'vertical_gain_percentage_points':gain,'verification_passed':True},ensure_ascii=False))


if __name__=='__main__':main()
