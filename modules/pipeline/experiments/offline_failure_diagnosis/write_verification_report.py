"""Render the verified diagnosis into a Chinese report and a geometry figure."""
from arx_project.assets import load_object_catalog
from itertools import product
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from inspect_failures import DEFAULT, PROJECT, read, surface_mesh

HERE=Path(__file__).resolve().parent
DOC=PROJECT/'docs/diverse_grasp_benchmark/debug_20261123'


def figure(data):
    font='/home/zhaoyechong/projects/ARX-ManiSkill/tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus':False,'font.size':10})
    catalog=load_object_catalog(DEFAULT/'object_catalog.json')['models']
    desc=read(DEFAULT/'gripper_descriptor.json')
    off=np.array(desc['parameters']['offset_open']);ext=np.array(desc['parameters']['extents_open'])
    r_eg=np.array(desc['T_eef_generator'])[:3,:3]
    poses={v['id']:v for v in data['reference_frame_probe']['top1_pose_checks']}
    signs=list(product([-1,1],repeat=3))
    fig=plt.figure(figsize=(13,11),layout='constrained')
    fig.suptitle('夹爪参考系对照：真实指腹夹持区域与同一物体',fontsize=18)
    for row,(slot,name) in enumerate([('D0002_S1_open_bottled_water','瓶装水'),
                                     ('D0015_S1_open_remote_control','遥控器')]):
        d=DEFAULT/'trials'/slot;state=read(d/'state.json');entry=read(d/'entry.json')
        mesh=surface_mesh(catalog[entry['model_id']]);t=np.array(state['object_world'])
        vertices=np.asarray(mesh.vertices)@t[:3,:3].T+t[:3,3]
        vertex=vertices[::max(1,len(vertices)//3500)]
        origin=np.array(state['object_center'])
        vv=(vertex-origin)*1000
        for col,suffix,title,color in [(0,'_eef_origin','原夹爪参考系','#d88a19'),
                                       (1,'_origin_back_100mm','模型参考原点后移10厘米','#16846b')]:
            case=poses[slot+suffix];eef=np.array(case['top1_actual_robot_eef_world'])
            corners=(np.array(signs)*ext/2+off)@r_eg.T@eef[:3,:3].T+eef[:3,3]
            c=(corners-origin)*1000
            ax=fig.add_subplot(2,2,row*2+col+1,projection='3d')
            ax.scatter(*vv.T,s=1,color='#697e86',alpha=.55,label='真实物体表面')
            for i in range(8):
                for j in range(i+1,8):
                    if sum(a!=b for a,b in zip(signs[i],signs[j]))==1:
                        ax.plot(*np.stack([c[i],c[j]]).T,color=color,lw=2)
            ax.plot([],[],[],color=color,lw=2,label='换算到真实ARX后的夹持区域')
            combined=np.concatenate([vv,c]);mid=(combined.max(0)+combined.min(0))/2
            half=max(np.ptp(combined,axis=0))/2+5
            ax.set(xlim=(mid[0]-half,mid[0]+half),ylim=(mid[1]-half,mid[1]+half),
                zlim=(mid[2]-half,mid[2]+half),xlabel='相对目标中心 X / mm',ylabel='相对目标中心 Y / mm',
                zlabel='相对目标中心 Z / mm',title=name+'：'+title+'\n最高分候选；夹持区目标点数 '+
                str(case['top1']['input_points_in_cavity']))
            ax.view_init(elev=20,azim=-65);ax.set_box_aspect([1,1,1]);ax.legend(fontsize=8,loc='upper left')
    fig.savefig(DOC/'resources/gripper_reference_comparison.png',dpi=145)


def main():
    data=read(HERE/'results_20261123/verification_summary.json')
    attribution=read(HERE/'results_20261123/empty_grasp_attribution.json')
    results={(r['run'],r['id']):r for r in data['physical_results']}
    def speed(run,condition,slot):return results[(run,condition+'__'+slot)]['max_speed_mps']
    run1='contact_replay_20261123_01';run2='contact_confirmation_20261123_01'
    mushroom='D0003_S1_open_mushroom';lemon='D0042_S2_edge_lemon'
    rows=[('原参数，固定末端闭爪','original'),('仅接触偏移改为1毫米','contact_offset_1mm'),
          ('仅夹指摩擦改为0.8/0.6','friction_08_06'),('仅驱动上限改为5','force_limit_5'),
          ('仅改为1.5秒渐进闭合','ramp_1500ms'),('仅提高位置/速度求解迭代至64/16','solver_64_16'),
          ('仅提高物理频率至480Hz','frequency_480hz'),('仅将去穿透速度上限设为0.2米/秒','depenetration_02')]
    lines=['# 弹飞原因与空抓来源验证','',
        '本轮完成24次独立闭爪诊断、50条历史执行记录的独立运动学与几何检查，以及六个物体的12次离线候选生成对照。原200场景批测继续停止，原66条完整结果和停止标记保留。','',
        '## 一、弹飞的主要触发因素已经定位','',
        '主要触发因素是20毫米接触偏移与当前小尺寸、多凸块碰撞几何不匹配。每个夹指有19块凸碰撞体，蘑菇和柠檬各有32块。原参数下，接触求解报告了厘米级的异常负分离距离，并施加很大的接触冲量；物体去穿透速度上限约为1e32米/秒，基本没有实际限幅。物体和夹指的恢复系数为0。','',
        '对照中使用相同物体、相同抓法、相同闭合前位置和开度，每组只改变表中对应参数。下面是直接读取刚体速度得到的峰值：','',
        '| 对照设置 | 蘑菇最高速度（米/秒） | 柠檬最高速度（米/秒） |',
        '| --- | ---: | ---: |']
    for title,condition in rows:
        lines.append(f'| {title} | {speed(run1,condition,mushroom):.3f} | {speed(run1,condition,lemon):.3f} |')
    lines += ['',
        '降低摩擦、减小驱动上限和放慢闭合都没有消除弹飞；增加求解迭代或提高物理频率也没有解决。缩小接触偏移的效果最明确；限制去穿透速度可以降低冲击，但仍保留异常的负分离距离。','',
        '随后完整重放原预抓取和接近轨迹，结果如下：','',
        '| 完整轨迹复核 | 蘑菇最高速度（米/秒） | 柠檬最高速度（米/秒） |',
        '| --- | ---: | ---: |',
        f'| 原20毫米接触偏移 | {speed(run1,"fullpath_original",mushroom):.3f} | {speed(run1,"fullpath_original",lemon):.3f} |',
        f'| 仅改为1毫米 | {speed(run2,"fullpath_contact_offset_1mm",mushroom):.3f} | {speed(run2,"fullpath_contact_offset_1mm",lemon):.3f} |','',
        '在1毫米设置下，蘑菇原抓法闭合后没有接触物体，说明此前的接触包含异常提前接触；柠檬仍有正常夹指接触，最大位移约2.33毫米。为了验证缩小偏移后还能正常接触目标，另把同一抓法向下移动15毫米作为接触对照，并对20毫米和1毫米设置分别闭合。1毫米下两物体从第1秒至结束的243条采样均保留双指接触，最大位移分别约7.65和8.69毫米。闭合瞬间仍有冲击和抖动，其中柠檬瞬时速度达到1.69米/秒；本轮定位解决的是原来的猛烈弹飞，后续还需校准闭合驱动。','',
        'PhysX在两形状距离小于接触偏移之和时就开始建立接触；两个20毫米偏移相加是40毫米。NVIDIA说明，偏移过大会破坏接触点靠近几何表面的假设，从而出现不正常行为。来源：[PhysX官方接触设置说明](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html)。','',
        '## 二、空抓主要发生在GraspGenX夹爪适配与候选生成这一层','',
        '| 待核查环节 | 验证方法 | 结果 |',
        '| --- | --- | --- |',
        '| 感知点云是否整体偏移 | 用实测深度和补全点云分别对照真实视觉网格表面 | 50例补全表面距离中位数约2.58毫米。瓶装水约1.26毫米，包围盒中心偏差约2.58毫米；剪刀等个别物体存在较大局部缺失。 |',
        f'| 夹爪是否没到规划位置 | 用URDF与实际关节记录独立计算闭合前末端位置 | 50例最大位置误差{attribution["maximum_independent_fk_error_mm"]:.3f}毫米、最大朝向误差{attribution["maximum_independent_fk_rotation_deg"]:.3f}度。24条扩散抓法在实际和规划位置都没有与真实目标表面相交的指腹夹持区域。 |',
        '| TCP或坐标换算是否重复补偿 | 逐条比对生成器原始位姿、EEF和TCP43 | 50例换算一致，TCP43没有重复补偿。 |',
        '| 原始候选是否已经偏离目标 | 在GraspGenX自身坐标系，用其原始候选和其声明的夹持区域检查输入点云 | 选中的24条扩散抓法，其夹持区域都没有包含自己的输入目标点。错误在机械臂执行前已经存在。 |',
        '| 其余8条未形成双指接触的案例 | 检查OBB顶面、指尖和真实指腹 | 全部选中零指尖切入的OBB抓法，叠加局部重建误差或夹持位置不合适；其中7例实际指腹区域与目标表面不相交，另1例只有单指瞬时接触。 |','',
        '## 三、夹爪参考系对照进一步确认了适配问题','',
        '当前用ARX的EEF作为模型参考原点，指尖在模型局部Z方向约44.97毫米处。官方ARX X5夹爪描述的指尖约为143毫米。二者都可以作为几何定义，但神经网络对参考原点的位置并不天然保持等价。','',
        '对六个物体保持输入点云、权重和随机种子相同，只把模型虚拟参考原点沿夹指反方向后移100毫米：同步将夹持区中心和指尖深度增加100毫米，并按同一变换把输出位姿还原到真实ARX的EEF。物理夹爪、实际TCP43和点云坐标都保持原定义。所有最高分输出均核对了虚拟夹持区与真实指腹区域的世界坐标一致。','',
        '| 物体 | 原参考系：512个扩散候选中夹持区含目标点的数量 | 虚拟原点后移100毫米：对应数量 |',
        '| --- | ---: | ---: |']
    probe=data['reference_frame_probe']['top1_pose_checks']
    names={'D0002_S1_open_bottled_water':'瓶装水','D0004_S1_open_carrot':'胡萝卜',
        'D0015_S1_open_remote_control':'遥控器','D0021_S1_open_apple':'苹果',
        'D0031_S1_open_bread':'面包','D0033_S1_open_bowl':'碗'}
    for slot,name in names.items():
        old=next(r for r in probe if r['slot']==slot and r['origin_behind_eef_m']==0)
        new=next(r for r in probe if r['slot']==slot and r['origin_behind_eef_m']>0)
        lines.append(f'| {name} | {old["diffusion_candidates_with_target_points"]}/512 | {new["diffusion_candidates_with_target_points"]}/512 |')
    lines += ['| 合计 | 0/3072 | 3067/3072 |','',
        '这里统计夹持区域是否覆盖目标点云。新参考系下，六例最高分候选换算到真实指腹后都与真实目标网格表面相交。这一步确认了参考系适配对候选几何有效性的影响；完整规划、闭合抬升成功率需要在适配修改后单独测试。','',
        '![同一物体的夹爪参考系对照](resources/gripper_reference_comparison.png)','',
        '## 四、修复顺序','',
        '先在独立新版执行配置中把接触偏移调整到适合这些小凸块的量级，再校准闭合瞬间的驱动响应。抓取模块优先修正GraspGenX的虚拟夹爪参考系及成对的输入/输出变换，随后检查原生评分与实际指腹接触区域的一致性。点云中局部缺损的物体继续单独定位分割与补全。完成这些修改后，再登记少量端到端验证，最后恢复批量测试。','',
        '## 五、数据与检查','',
        '原50条已执行结果、66条完整结果、全部旧领取记录和停止标记保持原样；1264个冻结执行源文件散列一致。本次24次物理操作分别记为独立闭爪诊断，未加入原200场景成功率。六物体参考系对照为离线推理，模型权重加载一次。','',
        '- [完整验证数字](resources/verification_summary.json)','- [逐例空抓来源](resources/empty_grasp_attribution.json)','',
        '原始诊断目录：`experiments/offline_failure_diagnosis/runs/contact_replay_20261123_01/`、`contact_confirmation_20261123_01/`和`gripper_frame_probe_20261123_01/`。对应源代码保存在`experiments/offline_failure_diagnosis/`。','']
    DOC.joinpath('resources').mkdir(exist_ok=True)
    for name in ['verification_summary.json','empty_grasp_attribution.json']:
        shutil.copy2(HERE/'results_20261123'/name,DOC/'resources'/name)
    (DOC/'弹飞原因与空抓来源验证.md').write_text('\n'.join(lines))
    figure(data)
    print(DOC/'弹飞原因与空抓来源验证.md')


if __name__=='__main__':main()
