"""Readable tables from the completed advisor35 CPU diagnostics."""
import argparse
import json
import re
from pathlib import Path

NAMES={'household_hope_milk_carton':'牛奶盒','household_hot3d_bbq_sauce_bottle':'BBQ瓶',
       'household_ycb_spam_can':'Spam罐','cheez_it':'Cheez-It盒','raisin_box':'葡萄干盒',
       'mustard_bottle':'芥末瓶','household_hot3d_potato_masher':'土豆压泥器','ycb_056_tennis_ball':'网球'}


def read(p):return json.loads(p.read_text())
def fmt(x):return '未记录' if x is None else f'{x:.2f}'
def label(trial):
    t=trial['trial_id'];name=NAMES[trial['object_id']]
    offset='+15' if 'plus15' in t else '−15' if 'minus15' in t else '扰动' if 'entry_offset' in t else '正常'
    arm='旧图D' if 'full_stale' in t else '固定G0' if t.endswith('_g0') else 'C' if t.endswith('_C') else '新鲜D'
    match=re.search(r'_r(\d+)_',t)
    repeat=trial.get('repeat') or (int(match.group(1)) if match else 1)
    return f'{name} {offset} {arm} R{repeat}'


def main(out):
    out=Path(out).resolve()
    inputs=read(out/'first_inputs.json');dist=read(out/'training_distribution.json')
    processor=read(out/'processor_checks.json')
    lines=['# 牛奶盒与BBQ正负15毫米：首步实际输入输出','',
        '仅复核第34轮已有8次实际推理，没有加载模型权重或新增推理。像素坐标为640×480图像，左上角为原点；质心按原生目标Mask与有效深度计算。哈希表显示前12位，JSON和CSV保留完整SHA-256。','',
        '| 条件 | RGB哈希 | 深度输入图哈希 | 目标像素数 | 质心(u,v) | 原始输出 |',
        '|---|---|---|---:|---|---|']
    outputs=[]
    for r in inputs['requests']:
        if r['prediction'] not in outputs:outputs.append(r['prediction'])
        token=chr(65+outputs.index(r['prediction']))
        lines.append(f'| {label(r)} | {r["rgb_sha256"][:12]} | {r["depth_image_sha256"][:12]} | {r["target_pixels"]} | ({r["centroid_u"]:.2f}, {r["centroid_v"]:.2f}) | {token} |')
    lines+=['','原始字符串完整保留；前三个数为末端系毫米，后三个数为旋转向量度数。','']
    for i,o in enumerate(outputs):lines.append(f'- {chr(65+i)}：`{o}`')
    lines+=['','| 物体与重复 | 正负质心水平差(px) | 垂直差(px) | Mask交并比 | RGB平均绝对差(0—255) | 输出相同 |',
            '|---|---:|---:|---:|---:|---|']
    for r in inputs['paired_comparisons']:
        lines.append(f'| {NAMES[r["object_id"]]} R{r["repeat"]} | {r["centroid_plus_minus_du_px"]:.2f} | {r["centroid_plus_minus_dv_px"]:.2f} | {r["mask_iou"]:.3f} | {r["RGB_MAE_0_255"]:.2f} | {"是" if r["same_raw_output"] else "否"} |')
    lines+=['','8次实际请求、回复和图像哈希绑定通过；按冻结处理器在CPU重新编码，得到8个不同的视觉张量，文本token哈希均与原记录相同。文本token哈希不含图像像素，所以相同的token哈希不能用来判定图像重复。原执行源码逐次构建两张图的输入，并将pixel_values送入generate；没有复用首帧视觉张量的分支。历史GPU内部张量未单独保存，此处是冻结代码与原图的CPU复算。','',
        '输出A在8条中出现7次，完整匹配芥末瓶的一条训练标签：`22617a971b32418591146f208858b514:18:actual_g0`。它不是牛奶盒的无权或加权平均动作。剩余一次BBQ正向输入得到B，表明模型并非对所有图像完全无响应，但响应没有可靠对应目标移位方向。','',
        '离线参考中，牛奶盒正负移位分别需要约+14.93和−15.04毫米的末端Y修正；四次模型均给出+0.872毫米。参考抓点只用于本项诊断，没有传给模型。','',
        '| 训练提示中的控制状态 | 样本数 | REFINE | GRASP | 牛奶盒样本数 |','|---|---:|---:|---:|---:|']
    for r in processor['training_prompt_states']:
        st=r['state'];name=('已停稳' if st['stopped'] else '运动中')+'；剩余次数'+str(st['remaining_corrections'])
        lines.append(f'| {name} | {r["rows"]} | {r["labels"].get("REFINE",0)} | {r["labels"].get("GRASP",0)} | {r["objects"].get("household_hope_milk_carton",0)} |')
    lines+=['','与在线首步完全相同的控制状态只覆盖15条训练观测，其中牛奶盒仅1条，标签为负向X约24.96毫米。图像几何发生明显变化而输出复用跨物体训练字符串，支持“条件覆盖稀疏、动作模板复用”的诊断；尚不能仅凭这些记录断定捷径只来自RGB纹理，或证明唯一根因。','',
        '完整证据：[输入记录](first_inputs.json)、[CPU处理器检查](processor_checks.json)、[离线参考残差与分离平面计算](two_step_sweep_design.json)。','']
    (out/'FIRST_INPUTS.md').write_text('\n'.join(lines))
    lines=['# 435条训练标签：分物体、六轴与控制阶段分布','',
        f'训练集435条＝{dist["motion"]}条REFINE＋{dist["GRASP"]}条GRASP，来自6个物体，每物体只有1个源抓取家族。连续轨迹中的多帧不计为独立抓取家族。GRASP单独统计，不补成零运动。','',
        '| 物体 | 全部 | 运动 | 闭合请求 | 源家族 | 源episode |','|---|---:|---:|---:|---:|---:|']
    for r in dist['objects']:lines.append(f'| {NAMES[r["object_id"]]} | {r["total"]} | {r["motion"]} | {r["GRASP"]} | {r["families"]} | {r["episodes"]} |')
    lines+=['','以下均在当前末端坐标系；平移单位毫米，旋转向量单位度。标准差为总体标准差（ddof=0）。负/零/正比例使用保存标签的精确符号；加权均值使用实际训练sample_weight。该均值描述标签分布，序列交叉熵训练没有要求模型回归到这个算术均值。','',
        '| 物体 | 轴 | 均值 | 标准差 | 负/零/正(%) | 加权均值 |','|---|---|---:|---:|---|---:|']
    for r in dist['objects']:
        for j,axis in enumerate(r['axes']):
            counts=[r[k][j]/r['motion']*100 for k in ('negative','zero','positive')]
            lines.append(f'| {NAMES[r["object_id"]]} | {axis} | {r["mean"][j]:.3f} | {r["std_population"][j]:.3f} | '+ '/'.join(f'{v:.1f}' for v in counts)+f' | {r["weighted_mean"][j]:.3f} |')
    lines+=['','为避免将零附近噪声解释成方向学习，另列有实际量级的平移符号。中间栏表示绝对值不超过0.5毫米，旋转的同类统计在CSV中按0.5度保存。','',
        '| 物体 | 轴 | 小于−0.5毫米 | ±0.5毫米内 | 大于+0.5毫米 |','|---|---|---:|---:|---:|']
    for r in dist['objects']:
        for j,axis in enumerate(('X','Y','Z')):
            lines.append(f'| {NAMES[r["object_id"]]} | {axis} | {r["negative_beyond_0p5"][j]} | {r["within_0p5"][j]} | {r["positive_beyond_0p5"][j]} |')
    lines+=['','牛奶盒45条运动中，X为负40条；按0.5毫米死区，X明显负向34条、明显正向3条。Y明显正向16条、明显负向0条。方向覆盖不对称，同时多个小偏差标签来自同一次运动的相邻帧。','',
        '实际训练已使用亮度0.85—1.15、对比度0.85—1.15、饱和度0.8—1.2，10轮中5轮灰度图；1740个RGB增强视图中870个为灰度。下一阶段继续此增强，并用配对消融评估是否有收益，不把原有增强描述为本轮新增。','',
        f'训练日志：544次更新、4350次样本呈现，训练循环{dist["timing"]["training_seconds"]/60:.2f}分钟，登记至训练结束{dist["timing"]["registration_to_finished_seconds"]/60:.2f}分钟。后续耗时估计以此为依据。','',
        '完整数据：[六轴统计CSV](training_axes.csv)、[控制阶段统计CSV](training_phases.csv)、[结构化分布](training_distribution.json)。','']
    (out/'TRAINING_DISTRIBUTION.md').write_text('\n'.join(lines))
    axial=read(out/'axial_review.json');geometry=read(out/'first_motion_geometry.json');sweep=read(out/'two_step_sweep_design.json')
    lines=['# 移除独立轴向补偿：已有轨迹与接触复盘','',
        '以下比较同一旧试验中补偿前、后的实际观测。反事实只删除“额外纯轴向动作”和“补偿后表面Z>36.5毫米拒闭”；保留可见性和指垫走廊，再判断是否可以申请原独立闭合模型检查。未执行的闭合、夹持和抬升结果保持未知。','',
        '| 9个拒闭条件 | 表面Z前→后(mm) | 走廊点前→后 | 包络内点前→后 | 物体移动(mm)/转角(°) | 额外动作中的接触侧 |','|---|---|---|---|---|---|']
    for r in axial:
        if not r['among_nine_axial_rejections']:continue
        b,a=r['before'],r['after'];sides='/'.join(k.replace('right_link','') for k,v in r['axial_motion_contacts'].items() if v['callbacks'])
        lines.append(f'| {label(r)} | {fmt(b["surface_z_mm"])}→{fmt(a["surface_z_mm"])} | {b["corridor_points"]}→{a["corridor_points"]} | {b["inside_points"]}→{a["inside_points"]} | {r["object_translation_mm"]:.2f}/{r["object_rotation_degrees"]:.2f} | 指{sides} |')
    lines+=['','9槽在补偿前、后均通过原指垫可见性门控，删除上述两项规则后都可进入独立闭合模型检查；这些历史前置姿态没有相应的闭合模型结果，因此可直接执行闭合的数量、抓取成功数量均未知。原9条失败不改分。','',
        '| 条件 | 前一动作接触事件(27/28) | 补偿动作接触事件(27/28) | 力/冲量测量 |','|---|---|---|---|']
    for r in axial:
        if not r['among_nine_axial_rejections']:continue
        before='/'.join(str(r['before_motion_contacts'][k]['callbacks']) for k in ('right_link27','right_link28'))
        after='/'.join(str(r['axial_motion_contacts'][k]['callbacks']) for k in ('right_link27','right_link28'))
        lines.append(f'| {label(r)} | {before} | {after} | 未保存 |')
    lines+=['','事件数由回调频率和动作持续时间共同影响；不能据此计算力增大或连续夹持。原始回执只有碰撞体、事件类型、接触点数与仿真时间，没有法向力、切向力或冲量。因此本轮可以量化物体被推动的位移与转角、识别接触侧，不能给出牛顿数值。','',
        'Spam扰动D的补偿将物体移动10.54毫米、转动9.62度；可见表面Z由49.11增至53.20毫米，且仍只有指27接触。这说明本例增加轴向命令没有补齐夹持几何。牛奶盒各补偿后拒闭槽物体移动4.94—10.96毫米，也都保持单侧接触。','',
        '全部12次补偿中，另外3条均完成闭合：牛奶盒正向新鲜D第二次成功、BBQ正向旧图D第一次成功、BBQ正向旧图D第二次抬升后失落。因此“12次有10次失败”描述这些槽位的最终结果，不能解释为10次失败全部由补偿单独造成。','',
        '| 新鲜D条件 | 首个指尖进入物体顶面高度层的时刻 | 首次目标接触时刻 | 该接触附近完整物体横向容纳余量(mm) |','|---|---|---|---:|']
    for r in geometry:
        first=r['first_top_layer_frame'];c=r['nearest_saved_frame_to_first_contact']
        t='H时已低于顶面' if first and first['seconds_after_H_observation']==0 else fmt(first['seconds_after_H_observation'])+'秒' if first else '未进入'
        lines.append(f'| {label(r)} | {t} | {fmt(r["first_contact_seconds_after_H"])}'+('秒' if r['first_contact_seconds_after_H'] is not None else '')+f' | {fmt(c["whole_object_lateral_clearance_mm"] if c else None)} |')
    lines+=['','时间以该槽H观测为零点；运动观测5Hz，接触事件有独立仿真时钟。横向余量为完整视觉网格在双指开口方向的保守支撑区间，负数表示尚未整体落入双指之间，并不等于已经发生碰撞。牛奶盒在H时指尖已比顶面低约39.84毫米，BBQ约22.46毫米，因此不能把每例冲突都归为首步“刚越过顶面”；还必须检查接近方向的间隔。','',
        '纯离线将首步轴向分量设为零、保留模型横向和旋转后，对8条牛奶盒/BBQ实际入口的双指视觉网格做61点旋转扫掠，并扣除点间旋转误差上界。沿H的反接近方向，最小分离距离为11.10—15.64毫米，均超过拟定10毫米预接触间隔。这个结果提供了“先在物体外对齐”的几何依据；它没有运行完整机器人IK、碰撞或规划，也不能将当前模型的错误横向输出变成正确输出。','',
        '还存在两步预算衔接问题：这8个历史H到参考G的轴向残差约78.45—81.63毫米；完全不插入的第一步结束后，剩余轴向通常仍超过当前单步65毫米上限。下一阶段应先核对正交抓点与新入口是否满足两个阶段的共同可行域。当前分析不放宽65毫米、不暗中加入第三步，也不宣称两步闭环已经可执行。','',
        '完整数据：[补偿前后JSON](axial_review.json)、[逐帧几何CSV](first_motion_frames.csv)、[首步事件JSON](first_motion_geometry.json)、[分离平面设计复算](two_step_sweep_design.json)。','']
    (out/'CONTROLLER_SUBTRACTION.md').write_text('\n'.join(lines))
    print('Three offline table documents written.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
