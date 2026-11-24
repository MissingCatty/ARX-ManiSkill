"""One advisor40 response, refreshed from actual receipts; no experiment dispatch."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, digest

NAMES = {'household_hope_milk_carton': '牛奶盒', 'raisin_box': '葡萄干盒', 'cheez_it': 'Cheez-It盒'}
RESPONSE = ROOT / 'research_plan/chat/gpt回复/gpt回应58.md'


def optional(path, default=None):
    return read(path) if Path(path).is_file() else default


def number(value, digits=2):
    return '未得到有限结果' if value is None else f'{value:.{digits}f}'


def row_metrics(label, summary):
    p = summary['overall']
    motion = p['motion_denominator']
    numbers = [number(p[k][q]) if motion else '不适用'
               for k in ('position_mm', 'rotation_degrees') for q in ('median', 'p90')]
    recall = f'{p["correct_GRASP"]}/{p["GRASP_denominator"]}' if p['GRASP_denominator'] else '不适用'
    premature = f'{p["premature_GRASP"]}/{motion}' if motion else '不适用'
    return '| ' + ' | '.join([label, *numbers, recall, premature]) + ' |'


def report(root):
    root = Path(root).resolve(); reg = read(root / 'registration.json')
    state = read(root / 'state.json'); manifest = read(root / 'dataset/manifest.json')
    training = optional(root / 'D/training_receipt.json')
    steps = 0; samples = 0
    if training:
        steps = training['optimizer_steps']; samples = training['sample_presentations']
    elif (root / 'D/training.jsonl').is_file():
        with (root / 'D/training.jsonl').open('rb') as f:
            f.seek(0, 2); end = f.tell(); f.seek(max(0, end - 32768)); lines = f.read().splitlines()
        if lines:
            last = json.loads(lines[-1]); steps = last['step']; samples = last['samples_seen']
    curve = optional(root / 'validation_curve.json', {})
    gate = optional(root / 'offline_gate.json')
    physical = optional(root / 'physical/summary.json')
    new = optional(root / 'offline/new_D_test/metrics.json')
    initialization = optional(root / 'D/model_check.json', {})
    processor = optional(root / 'CPU_processor_preflight.json', {})
    if state['status'] == 'fault':
        status = '运行已停止，现有产物和停止记录保留。下表按实际已完成范围报告。'
    elif physical:
        status = f'训练与离线检验已完成；物理验证已有{physical["completed"]}/9个完整结果，成功{physical["successes"]}次。'
    elif gate:
        status = ('训练与离线检验已完成，四项主门槛全部通过，进入已授权的9槽闭环验证预检。' if gate['pass']
                  else '训练与离线检验已完成，四项主门槛未全部通过，9个物理槽位均未执行。')
    elif steps:
        status = f'三物体先导训练正在推进，已完成{steps}/2865个优化步。离线验收及物理验证尚未完成。'
    elif (root / 'D').exists():
        status = '训练进程已启动，正在核对输入与原始模型初始化；尚未完成优化器更新。'
    else:
        status = '三物体数据与新实验登记已完成，后台等待可用显存，优化器更新为0。资源通过后自动开训。'
    lines = ['# 针对导师回复40：三物体纯VLM先导实验', '',
        f'更新于{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}。{status}', '',
        '用户已明确授权“按回复40执行训练及达标后的验证”。本轮采用四项离线主门槛全部通过后进入物理验证的执行口径；'
        '单独观察到模板坍缩改善，不替代这四项门槛。', '',
        '## 对本轮修改意见的逐项落实', '',
        '| 导师要求 | 本次具体措施和方法 | 当前实验效果 | 分析与状态 |',
        '|---|---|---|---|',
        '| 锁定三物体先导数据 | 使用牛奶盒、葡萄干盒、Cheez-It各自最终累计独立导出；保持原配对与空间邻域拆分 | 共3058条，训练2292、验证380、测试386；1529个完整正负配对 | 数据锁定完成；回复40中3046为加总笔误，Cheez-It距离1024还差4条 |',
        '| 暂停其余采集和搜索 | 保存6个已释放规范抓点、网球与芥末瓶的搜索记录、旧登记和STOP；确认采集进程及仿真容器已退出 | 未为凑满数据继续补采 | 暂停完成，原始证据保留 |',
        f'| 纯VLM训练10轮 | 同一Qwen3.5-2B基础权重与初始化种子；语言LoRA和视觉合并层训练；完整六维REFINE及GRASP；去掉预算字段 | 已完成{steps}/2865步，样本呈现{samples}/22920次 | '+
        ('基础及LoRA初始化哈希已与原D核对一致' if initialization.get('same_original_D_initialization') else '原始初始化一致性将在首步更新前核对')+' |',
        f'| 每2轮即时验证 | 在第573、1146、1719、2292、2865步保存权重，随即运行380条验证观测，再继续训练 | 已完成{len(curve)}/5次阶段验证 | 固定用第10轮做封存测试，不用测试集选检查点 |',
        '| 封存测试整体及轴向分层 | 对386条测试观测统计位置、测地角、闭合召回、误闭合；新D、旧D原提示、旧D新提示和训练集均值分别评估 | '+('已生成新D封存测试结果' if new else '尚未执行')+' | 非法输出和误闭合保留在运动分母中，主误差按无穷大惩罚 |',
        '| 正负配对及图像交换 | 检查X/Y和三轴旋转的反号、正确方向及响应幅度；两张图一起换成配对另一端 | '+('已完成' if (root/'offline/image_swap.json').exists() else '尚未执行')+' | 图像交换后仅与供图样本一致并不足够，还需输出发生变化并指向正确动作 |',
        '| 重测牛奶盒正负15毫米历史入口 | 固定4条历史首步观测，保留两次重复；新旧模型均重新推理；历史目标与新规范目标分别计误差 | '+('已得到新D历史帧预测' if (root/'offline/new_D_historical/metrics.json').exists() else '尚未执行')+' | 直接检查约30毫米方向分离及完整轴向推进，避免把更换目标造成的误差差异算作训练收益 |',
        '| 达标后纯模型闭环 | 3物体×正常H、目标世界X方向±15毫米，共9槽；同一物体的H固定；进入H后原样解释六维输出，仅显式GRASP合爪 | '+(f'已有{physical["completed"]}个完整结果' if physical else '物理试验0次')+' | 原关节、碰撞、停稳、接触和抬升保持判据保留；无自动重试 |', '',
        '## 数据与训练口径', '',
        '| 物体 | 训练观测 | 验证观测 | 封存测试观测 | 合计 |', '|---|---:|---:|---:|---:|']
    for obj, splits in manifest['distributions'].items():
        values = [splits[s]['observations'] for s in ('train', 'validation', 'test')]
        lines.append(f'| {NAMES[obj]} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |')
    lines += ['', '这些产物是带解析标签的静态近场观测。每个源规范抓点具有7姿态物理预检证据，'
        '各观测另经图像、实际姿态、碰撞和配对检查；3058条不表示3058次逐条执行成功的抓取轨迹。', '',
        '模型只看到同步腕部RGB、完整原生实例Mask深度图和固定任务文本。目标数值位姿、规范抓点、G0及剩余动作预算均不进入模型输入。'
        '动作按当前实测TCP局部坐标系解释，完整保留毫米平移和旋转向量角度。', '',
        '训练沿用LoRA rank64、alpha128、dropout0.05，冻结视觉骨干和语言基座；AdamW，学习率1e−4，微批量1、累积8。'
        '本轮按实际2292条训练样本运行10轮，共2865步；余弦学习率覆盖这2865步。'
        '每条GRASP的样本系数为REFINE的3倍，各物体总样本权重相等，保留原动作类型/数值/结束符损失分项。'
        'RGB使用原亮度、对比度、饱和度扰动与一半灰度轮次，共9168张增强图；深度、Mask和动作标签不变，增强不增加独立观测数。', '',
        '## 验证与封存测试结果', '',
        '| 检查点/模型 | 位置中位mm | 位置P90 mm | 朝向中位° | 朝向P90° | 到位GRASP命中/正例 | 过早GRASP/运动样本 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for step, summary in sorted(curve.items(), key=lambda v: int(v[0])):
        lines.append(row_metrics(f'第{round(int(step)*8/2292)}轮验证', summary))
    for folder, label in [('new_D_test', '新D封存测试'), ('old_D544_native_prompt', '旧D544原生提示'),
                           ('old_D544_new_prompt', '旧D544新提示干预')]:
        result = optional(root / 'offline' / folder / 'metrics.json')
        if result:
            lines.append(row_metrics(label, result))
    mean = optional(root / 'offline/train_only_mean.json')
    if mean:
        lines.append(row_metrics('训练部分全局均值基线', mean['summary']))
    if not curve and not new:
        lines.append('| 尚未得到模型评估结果 | — | — | — | — | — | — |')
    if new:
        lines += ['', '新D的封存测试轴向分层如下。GRASP核心区单列闭合指标，运动误差以真实REFINE为分母。', '',
            '| 轴向层/类别 | 位置中位mm | 位置P90 mm | 朝向中位° | 朝向P90° | 到位GRASP命中/正例 | 过早GRASP/运动样本 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for layer, value in new['by_axial_layer_mm'].items():
            lines.append(row_metrics(layer, {'overall': value}))
    mirror = optional(root / 'offline/new_D_mirror.json'); swap = optional(root / 'offline/image_swap.json')
    if mirror:
        lines += ['', '| 配对方向 | 两端方向均正确/可评估配对 | 预测正负反号数 | 响应幅度比例中位数 |', '|---|---:|---:|---:|']
        for axis, v in mirror['by_axis'].items():
            lines.append(f'| {axis} | {v["both_signs_correct"]}/{v["denominator"]} | {v["opposite_predicted_signs"]} | {number(v["response_gain"]["median"])} |')
    if swap:
        lines += ['', f'交换配对图像后，{swap["changed_output_vs_original"]}/{swap["paired_observations"]}条输出发生变化，'
                  f'{swap["exact_output_matches_donor_normal"]}/{swap["paired_observations"]}条与供图样本原输出完全一致。']
    history_path = root / 'offline/new_D_historical/predictions.jsonl'
    if history_path.exists():
        history = [json.loads(s) for s in history_path.read_text().splitlines()]
        lines += ['', '| 牛奶盒历史入口 | 新D原始六维输出 | 历史目标位置误差mm | 新规范目标位置误差mm |', '|---|---|---:|---:|']
        for row in history:
            lines.append(f'| 重复{row["repeat"]}，{row["shift_world_x_mm"]:+g}mm | {row["prediction"].replace(chr(10), " ")} | '
                         f'{number(row["position_mm"])} | {number(row["canonical_reference_score"]["position_mm"])} |')
    if physical:
        lines += ['', '## 物理闭环结果', '', '| 物体 | 正常H | 目标−15mm | 目标+15mm |', '|---|---|---|---|']
        for obj in NAMES:
            values = physical['by_object'].get(obj, {})
            cells = [('成功' if values[c]['success'] else '失败：' + values[c]['outcome']) if c in values else '未得到完整结果'
                     for c in ('normal', 'minus15', 'plus15')]
            lines.append('| ' + NAMES[obj] + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', f'完整结束{physical["completed"]}次，其中成功{physical["successes"]}次；'
            f'未执行{len(physical.get("unexecuted", []))}次，中断或结果未知{len(physical.get("interrupted_unknown", []))}次。']
    lines += ['', '## 当前分析与下一步', '']
    if gate:
        names = {'position_median_below_5mm': '位置中位误差小于5毫米',
                 'rotation_median_below_3deg': '朝向中位误差小于3度',
                 'GRASP_recall_above_85_percent': '到位闭合召回超过85%',
                 'premature_GRASP_at_most_5_percent': '未到位误闭合不超过5%'}
        lines.append('四项离线门槛：' + '；'.join(f'{names[k]}：{"通过" if v else "未通过"}' for k, v in gate['checks'].items()) + '。')
        lines.append('离线误差、闭合分类和配对响应说明的是这三个已知物体在封存相对姿态上的动作预测能力。'
                     '真实抓取质量由独立物理闭环结果判断；本轮尚不覆盖未见物体泛化。')
        if not gate['pass']:
            lines.append('下一步先按物体、轴向层与REFINE/GRASP混淆定位未达标项，再确定修正方案；本轮不追加训练，也不提前运行9槽物理验证。')
    else:
        lines.append('当前完成的是数据锁定、接口检查和自动执行准备，模型是否改善视觉响应尚无新的评估结论。'
                     '下一次有实质判断价值的节点是第2轮验证：同时看六维误差、到位闭合召回与未到位误闭合，不能只看训练损失下降。')
    lines += ['', '本次同时改变了监督目标、姿态覆盖、提示词与类别权重，若结果改善，首先支持这套整体重构。'
        '将某一部分单独认定为增益来源，还需要后续保持训练预算一致的消融。', '',
        f'产物目录：[{root.name}]({root})。'
        f'固定配置见[训练登记]({root / "registration.json"})，数据来源与分布见[数据清单]({root / "dataset/manifest.json"})。', '']
    text = '\n'.join(lines)
    if RESPONSE.exists() and not RESPONSE.read_text().startswith('# 针对导师回复40：'):
        raise RuntimeError('Response58 belongs to a different advisor round')
    temporary = RESPONSE.with_suffix('.md.tmp'); temporary.write_text(text); temporary.replace(RESPONSE)
    (root / 'EXPERIMENT_REPORT.md').write_text(text)
    write(root / 'report_status.json', {'response': str(RESPONSE), 'sha256': sha(RESPONSE),
        'optimizer_steps': steps, 'validation_checkpoints': len(curve), 'offline_gate': gate,
        'wall': time.time(), 'physical_completed': physical['completed'] if physical else 0})


def watch(root):
    root = Path(root).resolve(); until = time.monotonic() + 48 * 3600; previous = None
    while time.monotonic() < until:
        state = read(root / 'state.json')
        checks = len(optional(root / 'validation_curve.json', {}))
        progress = optional(root / 'D/state.json', {})
        token = digest([state['status'], checks, bool(progress.get('optimizer_steps', 0)),
                        (root / 'offline_gate.json').exists(), (root / 'physical/summary.json').stat().st_mtime
                        if (root / 'physical/summary.json').exists() else None])
        if token != previous:
            report(root); previous = token
        if state['status'] in ('complete', 'complete_offline_thresholds_not_met', 'fault'):
            return
        time.sleep(30)
    write(root / 'report_watch_expired.json', {'wall': time.time(), 'training_or_physical_processes_modified': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('report', 'watch')); p.add_argument('root')
    a = p.parse_args(); report(a.root) if a.action == 'report' else watch(a.root)
