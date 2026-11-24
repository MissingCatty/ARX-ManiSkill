"""Readable reports from the revised advisor35 CPU diagnosis; no runtime imports."""
import argparse
import json
import os
from pathlib import Path


NAMES = {
    'cheez_it': 'Cheez-It盒', 'household_hope_milk_carton': '牛奶盒',
    'household_hot3d_potato_masher': '土豆压泥器', 'mustard_bottle': '芥末瓶',
    'raisin_box': '葡萄干盒', 'ycb_056_tennis_ball': '网球',
    'household_hot3d_bbq_sauce_bottle': 'BBQ瓶', 'household_ycb_spam_can': 'Spam罐',
}


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


def vec(values):
    return ', '.join(f'{v:.3f}' for v in values)


def main(out):
    out = Path(out).resolve()
    dest = out / 'DIAGNOSIS.md'
    if dest.exists():
        raise FileExistsError('Preserve finished diagnosis report')
    old = out.parent / 'nearfield_advisor35_offline_20261112_01'
    first = read(old / 'first_inputs.json')
    dist = read(old / 'training_distribution.json')
    processor = read(old / 'processor_checks.json')
    axial = read(out / 'axial_label_distribution.json')
    review = read(out / 'raw_output_review.json')
    rows = read(out / 'raw_output_rows.json')
    refs = read(out / 'canonical_references.json')
    closure = read(out / 'grasp_event_applicability.json')
    def link(label, path):
        return f'[{label}]({os.path.relpath(path, out)})'
    parts = [
        '# 修订回复35：输入、六轴标签及纯模型目标误差诊断\n',
        '2026年11月12日。按修订后的导师回复35完成离线核查。新增物理试验、服务启动、模型推理、训练更新均为0；旧30条物理结果及原执行器保留。本文使用已有模型输出计算目标，未启动新的纯模型闭环。\n',
        '## 1. 牛奶盒正负入口：图像不同，输出相同\n',
        '对牛奶盒和BBQ各两轮正负15毫米入口，共8次首步请求，核对保存的两幅实际输入、目标掩码质心，并用冻结版本的处理器在CPU重建视觉张量。以下哈希显示前12位，完整值保存在原始核查文件中。\n',
        table(['物体/轮次/偏移', 'RGB哈希', '深度输入图哈希', '目标质心(u,v)', '输出'], [
            [f"{NAMES[r['object_id']]} R{r['repeat']} {r['shift_mm']:+g}mm",
             r['rgb_sha256'][:12], r['depth_image_sha256'][:12],
             f"({r['centroid_u']:.2f}, {r['centroid_v']:.2f})",
             'A' if r['exact_training_label_matches'] else 'B']
            for r in first['requests']]),
        'A为完整动作 `REFINE 19.958 0.872 9.959 -5.0005 -0.0077 0.0116`，7/8次首步重复该字符串；B为 `REFINE 24.958 0.872 4.974 -4.9994 0.0126 0.0086`。平移为毫米，旋转向量为度。\n',
        '牛奶盒正负入口的目标横向质心差约74.20像素，BBQ约71.11像素。8份RGB、深度输入图及重建的视觉张量均不同。相同的文字token哈希不代表图片相同，图片在独立视觉张量中。历史GPU端张量本身没有落盘；本轮证据覆盖已存图片和CPU处理过程，不能补称已直接观测到历史GPU内部状态。\n',
        'A恰好完整匹配一条芥末瓶训练标签，而牛奶盒自身X均值为−3.156毫米、加权均值为−3.843毫米。因此“特定物体动作均值被背下来”不能精确描述这个现象；直接证据是不同视觉状态复用了跨物体动作模板。方向覆盖稀疏、提示状态与动作耦合是有依据的原因候选，尚未由训练消融确定各自贡献。\n',
        link('8次输入及4组配对', old/'first_inputs.json') + '；' +
        link('CPU处理器核查', old/'processor_checks.json') + '。\n',
        '## 2. 435条训练数据的六维分布\n',
        '435条中374条为REFINE、61条为GRASP。以下动作均值只统计REFINE，GRASP不作为零向量混入；前3轴单位毫米，后3轴为局部旋转向量、单位度。每个物体只有1个源抓取家族，合计53条展开轨迹。\n',
        table(['物体', '总数/运动/GRASP', 'dx均值', 'dy均值', 'dz均值', 'rx均值', 'ry均值', 'rz均值'], [
            [NAMES[r['object_id']], f"{r['total']}/{r['motion']}/{r['GRASP']}"] +
            [f'{x:.3f}' for x in r['mean']] for r in dist['objects']]),
        table(['物体', '六轴标准差（同上顺序）', '按实际样本权重计算的六轴均值'], [
            [NAMES[r['object_id']], vec(r['std_population']), vec(r['weighted_mean'])]
            for r in dist['objects']]),
        '牛奶盒45条运动中，X为负40条、为正5条；采用0.5毫米的实际方向阈值时，X负向34条、正向3条，Y负向0条、正向16条。实际首步提示“已停稳、剩余2次”只有15条运动样本，牛奶盒仅1条。这比全局样本数量更直接暴露了当前入口缺少反方向监督。\n',
        '原训练使用文本序列交叉熵，并未要求输出等于算术均值。均值描述数据偏向，不能单独证明模型采取了均值回归。完整标准差、正负比例、阶段和权重见' +
        link('六轴逐项表', old/'training_axes.csv') + '和' + link('分布记录', old/'training_distribution.json') + '。\n',
        '## 3. 轴向覆盖：全局小动作偏多，入口条件缺少大动作\n',
        table(['样本条件', '运动数', '轴向绝对值≤5mm', '该区间权重占比', 'dz中位数', '40–75mm', '>75mm'], [
            [label, axial[key]['motion_rows'],
             f"{axial[key]['abs_z_le5']['count']} ({axial[key]['abs_z_le5']['fraction']:.2%})",
             f"{axial[key]['abs_z_le5']['sample_weight_fraction']:.2%}",
             f"{axial[key]['z_mm_percentiles'][3]:.3f}mm",
             axial[key]['z_40_to75']['count'], axial[key]['z_gt75']['count']]
            for label,key in [('所有REFINE','all'), ('横向距离>10mm','large_lateral_gt10mm'),
                              ('已停稳、剩余2次','state:{"stopped":true,"remaining_corrections":2}'),
                              ('已停稳、预算未知','state:{"stopped":true,"remaining_corrections":"unknown"}')]]),
        '全体374条运动按轴向分层为：≤5毫米144条，5–30毫米180条，30–40毫米11条，40–75毫米22条，>75毫米17条；前3档按绝对值，后2档均为正值。各档互斥且总数为374。全局轴向中位数7.125毫米、均值14.272毫米，加权均值22.152毫米。\n',
        '“绝大多数大横向偏移样本的|dz|≤5毫米”不符合这份训练导出：该占比只有9/52=17.31%。更准确的问题是控制状态条件分布错位：在线入口对应的15条标签轴向最大仅19.994毫米，40毫米以上为0；而40毫米以上的39条标签全部带“预算未知”的提示。当前记录的入口到历史抓点轴向约79–80毫米，不能假设只需45–65毫米。\n',
        link('轴向分组统计', out/'axial_label_distribution.json') + '。\n',
        '## 4. 原样解释模型六维输出后的目标误差\n',
        '本轮逐条复算已有36次D输出：20次新鲜图像、16次固定旧图。将模型平移和旋转向量原样右乘当前实测末端变换，不替换Z、不补偿、不截断，也不把零REFINE转换成GRASP。历史目标以物体坐标系保存，再随该帧实测物体位姿变换；目标跟随物体只用于离线算误差，未送入模型或执行器。\n',
        '下面是20次新鲜图像调用的分组中位数。历史目标是旧示教参考，不预先假定它处处正交。\n',
        table(['物体/阶段', '次数', '原始预测dz', '当时到历史目标dz', '原样输出后位置误差', '朝向误差'], [
            [NAMES[r['object_id']] + ('/入口' if r['phase']=='entry' else '/中间帧'), r['N'],
             f"{r['predicted_z_median_mm']:.2f}mm", f"{r['historical_target_z_median_mm']:.2f}mm",
             f"{r['historical_position_error_median_mm']:.2f}mm", f"{r['historical_angle_error_median_deg']:.2f}°"]
            for r in review['summaries'] if r['vision_arm']=='fresh']),
        '全部36条的原始六轴输出、目标矩阵、逐轴残差及角度误差见' +
        link('逐条结果CSV', out/'raw_output_rows.csv') + '和' + link('结构化结果', out/'raw_output_rows.json') +
        '。位置残差分量在当前末端坐标系；旋转残差采用预测末端到参考末端的相对旋转向量，报告其模长作朝向误差。16次固定旧图结果单列在' + link('汇总',out/'raw_output_review.json') + '，未混入上表。\n',
        '**这些中间帧来自旧混合控制器实际执行后的观测。逐帧改算目标只能得到该观测下的单步反事实，不能拼成纯模型闭环，更不能推出它一定会成功或失败。**\n',
        '## 5. 正交参考的质量边界\n',
        '对新牛奶盒/BBQ候选，先以OBB给出方向，再在实际资产三角面上求两端交点、表面法线与中点。Spam复用此前已登记的正交抓点。5°是此次离线候选质量参考，不修改任何机器人碰撞或物理成功阈值。\n',
        table(['物体', '来源', '两端法线偏差', '离线几何结论', '新IK/物理验证'], [
            [NAMES[r['object_id']], '历史已登记' if r['registered_canonical_reused'] else '本轮离线OBB候选',
             ' / '.join(f'{v:.2f}°' for v in r['contact_normal_to_closing_degrees']),
             '通过本项局部法线核查' if r['normal_alignment_within_5deg'] else '未通过，不认作合格canonical', '均未新增']
            for r in refs]),
        '牛奶盒合格几何候选相对历史目标只改位置1.95毫米、朝向2.76°；其入口原始输出后的误差仍为74.36毫米/4.61°，中间帧26.09毫米/5.13°。Spam相应为49.63毫米/12.42°和29.56毫米/22.59°。BBQ候选局部法线偏差约15–17°，它的“canonical误差”仅保留作被拒候选诊断，不能进入正交目标验收。\n',
        '因此，本轮完成了全部36条对历史参考的六维复算，以及牛奶盒/Spam对有质量依据参考的复算；**BBQ真实局部正交目标仍需拟合与确认，6个训练物体也尚未全部完成canonical注册。**OBB方向正确不保证夹持点两侧的真实表面正交，更不等于已证明稳定夹持。详见' + link('参考来源与两端交点',out/'canonical_references.json') + '。\n',
        '## 6. GRASP事件的适用性\n',
        f"本轮复算的36次D调用全为REFINE。旧物理轮次全部44次神经调用中0次显式GRASP，是触发频次；44次中还有8次C，不能全当成D的召回率分母。新鲜D的{closure['proximity']['historical']['fresh_rows']}帧中，按历史抓点检查，符合4毫米/3°到位条件的为0帧，最近一帧位置距离仍为{closure['proximity']['historical']['minimum_position_residual_mm']:.2f}毫米；按有质量依据的牛奶盒/Spam参考检查12帧，同样0帧到位。\n",
        '所以不能由在线0次GRASP断言“到位后也不会闭合”。独立的旧离线验证中，到位标签24条仅7条预测GRASP（29.17%）才是现有闭合召回证据。需要增加密集正例与边界负例，同时测提前误闭合，避免一个永远输出GRASP的模型满足单独的召回要求。' + link('到位适用性核查',out/'grasp_event_applicability.json') + '。\n',
        '## 7. 本轮验收口径\n',
        table(['维度','结果'], [
            ['任务结果','输入与435条分布诊断、36条原样动作复算完成；纯VLM数据与训练方案另文给出。'],
            ['记录完整性','引用原始请求、回执、训练导出和资产；旧结果不改写；修订指令与新登记绑定，最终文件核查见completion_receipt.json。'],
            ['事件适用性','单步离线反事实适用；纯模型连续执行、训练收益、真实夹持成功未验证；在线GRASP召回缺少到位正例分母。'],
            ['关键事件覆盖','8个入口输入、4组正负配对、435训练标签、36个D输出；新增闭合/抬升/保持均0。']]),
        '已完成4项CPU动作解释检查：六轴保留、零REFINE不自动闭合、非法数字拒绝、精确到目标时误差归零。它们只验证离线计算，不替代ARX仿真执行。\n',
    ]
    dest.write_text('\n'.join(parts))
    print({'report':str(dest),'new_physical_trials':0})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    main(p.parse_args().out)
