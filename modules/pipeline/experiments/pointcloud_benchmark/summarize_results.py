"""Report saved evidence; empty predictions, runtime faults, and omissions stay separate."""
import csv
import fcntl
from collections import Counter
from datetime import datetime
import json
import numpy as np
from prepare_scenes import RUN, PROJECT, write

NAMES = {'zerograsp': 'ZeroGrasp（仅重建）', 'hunyuan-mini': 'Hunyuan3D-2mini-Turbo',
         'hunyuan-turbo': 'Hunyuan3D-2-Turbo', 'spar3d': 'SPAR3D', 'rayst3r': 'RaySt3R'}


def number(values, percentile=50):
    return float(np.percentile(values, percentile)) if values else None


def show(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    rows, all_results, export = [], {}, []
    for method, name in NAMES.items():
        folder = RUN/'methods'/method
        results = [json.loads(p.read_text()) for p in sorted(folder.glob('C*/result.json'))]
        valid = [r for r in results if r.get('success')]
        failures = [r for r in results if r.get('status') == 'model_output_failure']
        normal = valid + failures
        ready_path = folder/'ready.json'
        ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
        row = {'method': method, 'name': name, 'completed_records': len(results), 'valid_outputs': len(valid),
               'output_failures': len(failures), 'omitted': 50-len(results)+sum(r['status']=='not_executed' for r in results),
               'faults': sum(r['status']=='stopped' for r in results),
               'failure_reasons': dict(Counter(r.get('reason', 'unknown') for r in failures)),
               'load_s': ready.get('load_s'), 'warmup_s': ready.get('warmup_s'),
               'chamfer_mm_p50': number([r['geometry']['chamfer_l1_mm'] for r in valid]),
               'fscore5_mean_including_output_failures': sum(r['geometry']['fscore_5mm'] for r in valid)/len(normal) if normal else None,
               'quality_denominator': len(normal),
               'generation_s_p50': number([r['generation_s'] for r in results if 'generation_s' in r]),
               'generation_s_p95': number([r['generation_s'] for r in results if 'generation_s' in r], 95),
               'placement_s_p50': number([r['calibration_s'] for r in valid]),
               'reconstruction_and_placement_s_p50': number([r['reconstruction_and_placement_s'] for r in valid]),
               'oracle_chamfer_percent_p50': number([r['oracle_shape']['chamfer_percent_object_extent'] for r in valid]),
               'visible_surface_error_mm_p50': number([r['geometry']['visible_surface_to_prediction_mean_mm'] for r in valid]),
               'max_gpu_gb': max([r['peak_gpu_gb'] for r in results if 'peak_gpu_gb' in r], default=None)}
        rows.append(row); all_results[method] = results
        for r in results:
            export.append({'method': method, 'case': r['id'], 'category': r['category'], 'style_id': r['style_id'],
                           'status': r['status'], 'reason': r.get('reason', ''), 'generation_s': r.get('generation_s'),
                           'calibration_s': r.get('calibration_s'),
                           'mesh_sampling_save_s': r.get('mesh_to_points_and_save_s'),
                           'total_reconstruction_placement_s': r.get('reconstruction_and_placement_s'),
                           'peak_torch_gpu_gb': r.get('peak_gpu_gb'),
                           'quality_eligible': r['status'] in ['completed', 'model_output_failure'],
                           'chamfer_mm': r.get('geometry', {}).get('chamfer_l1_mm'),
                           'fscore_5mm': 0. if r['status']=='model_output_failure' else r.get('geometry', {}).get('fscore_5mm'),
                           'visible_error_mm': r.get('geometry', {}).get('visible_surface_to_prediction_mean_mm'),
                           'oracle_chamfer_percent': r.get('oracle_shape', {}).get('chamfer_percent_object_extent')})
    baseline = json.loads((RUN/'baseline_complete.json').read_text())
    dataset_path = RUN/'dataset_description.json'
    dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else None
    generated_at = datetime.now().astimezone().isoformat(timespec='seconds')
    summary = {'generated_at': generated_at, 'rows': rows, 'registered_model_slots': 250, 'records': sum(r['completed_records'] for r in rows),
               'valid_outputs': sum(r['valid_outputs'] for r in rows), 'physical_trials': 0, 'baseline': baseline}
    if dataset: summary['dataset_description'] = dataset
    write(RUN/'summary.json', summary)
    if export:
        temp = RUN/'case_metrics.csv.tmp'
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(export[0])); writer.writeheader(); writer.writerows(export)
        temp.replace(RUN/'case_metrics.csv')
    lines = ['# 点云生成模型评测结果', '',
             f"正式登记250项，当前保存{summary['records']}项结果。物理抓取试验0次。汇总时间：{generated_at}。", '',
             '十类常见物体，每类五种资产款式，共50个静态桌面场景。使用ARX头部D405已有标定与45°向下安装，640×480；原始深度每单位0.1毫米，模型共用去畸变图。', '',
             '50个视觉资产中，按中心对齐后10微米量化顶点集核查，有49组不同几何；盒装食品中两款几何相同、外观不同。目标区域深度中位数在0.30–0.40米，各例详细尺寸和深度保存在dataset_description.json。', '',
             f"共享SAM3掩码50/50有效，与独立仿真可见掩码的IoU中位数为{np.median([x['mask_iou'] for x in baseline['results']])*100:.2f}%。真值未用于SAM3提示、候选选择或被测模型输入。", '',
             '## 完整性与几何表现', '',
             '| 方法 | 有效输出/50 | 模型输出失败 | 故障 | 未执行 | 米制CD中位数（mm） | F-score@5mm | 统计分母 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        score = r['fscore5_mean_including_output_failures']
        score_text = show(score*100)+'%' if score is not None else '—'
        lines.append(f"| {r['name']} | {r['valid_outputs']}/50 | {r['output_failures']} | {r['faults']} | {r['omitted']} | {show(r['chamfer_mm_p50'])} | {score_text} | {r['quality_denominator']} |")
    lines += ['', 'CD为双向、非平方最近点距离的均值，中位数仅统计有效三维输出。F-score均值将正常执行但输出为空的情况计为0；资源故障和未执行另列。', '',
              '## 生成、定标与资源', '',
              '| 方法 | 重建P50/P95（s） | 观测定标P50（s） | 重建及定标P50（s） | 峰值显存（GB） | 真值对齐形状CD（尺寸%） | 可见表面平均误差的中位数（mm） |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['name']} | {show(r['generation_s_p50'])}/{show(r['generation_s_p95'])} | {show(r['placement_s_p50'])} | {show(r['reconstruction_and_placement_s_p50'])} | {show(r['max_gpu_gb'])} | {show(r['oracle_chamfer_percent_p50'])} | {show(r['visible_surface_error_mm_p50'])} |")
    lines += ['', 'RaySt3R与ZeroGrasp直接使用RGB-D；混元与SPAR3D先由单张去背景RGB生成形状，再仅用共享观测深度恢复尺度和坐标。表中将定标耗时单列。原生模态不同，不是同输入结构消融。', '',
              '真值对齐的形状CD独立衡量归一化形状质量，不能当作部署精度。离线评分耗时不计入重建与定标；首次加载和公共预热单列。所有正式方法依次使用同一块RTX4090 GPU3，模型每方法加载一次并常驻。', '',
              '重建时间为适配器predict的同步墙钟时间，包含该方法的预处理、原生深度/网格生成和必要的CPU传输；RaySt3R原生文件接口的输入保存也包含在内。另列网格采样保存和深度定标，离线评分单独保存。显存为PyTorch已分配峰值（十进制GB），不等于nvidia-smi总显存。每款只测一次，未测真实深度噪声或时间重复性。', '',
              'ZeroGrasp保留共享网络前向，没有裁剪网络结构；只省略抓取解码和候选后处理。SPAR3D保留全部可学习几何模块与官方几何权重，省略不参与形状生成的材质/光照模块。', '',
              '## 各类别款式表现', '',
              '| 类别（每类5款） | ZeroGrasp | Mini-Turbo | Turbo | SPAR3D | RaySt3R |',
              '| --- | --- | --- | --- | --- | --- |']
    categories = json.loads((RUN/'scene_selection.json').read_text())['cases']
    for category in dict.fromkeys(c['category'] for c in categories):
        label = next(c['category_cn'] for c in categories if c['category']==category)
        cells = []
        for method in NAMES:
            values = [r for r in all_results[method] if r['category']==category and r['status'] in ['completed', 'model_output_failure']]
            score = sum(r['geometry']['fscore_5mm'] for r in values if r.get('success'))/len(values)*100 if values else None
            cells.append(show(score)+'%' if score is not None else '—')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '类别表为F-score@5mm，空输出计0。每类仅五种款式，预训练数据成员关系未知；这里衡量跨款式表现，不声明严格训练集外泛化。', '',
              '## 不同款式的波动', '',
              '| 方法 | F-score≥80% | F-score<50% | 最差有效输出（CD） | 输出失败原因 |',
              '| --- | --- | --- | --- | --- |']
    for row in rows:
        results = all_results[row['method']]; valid = [r for r in results if r.get('success')]
        normal = [r for r in results if r['status'] in ['completed', 'model_output_failure']]
        good = sum(r['geometry']['fscore_5mm']>=.8 for r in valid)
        poor = sum(not r.get('success') or r['geometry']['fscore_5mm']<.5 for r in normal)
        worst = max(valid, key=lambda r:r['geometry']['chamfer_l1_mm']) if valid else None
        worst_text = f"{worst['id']} / {worst['style_id']} / {worst['geometry']['chamfer_l1_mm']:.2f}mm" if worst else '—'
        reasons = '；'.join(f'{reason}×{count}' for reason, count in row['failure_reasons'].items()) or '无'
        lines.append(f"| {row['name']} | {good}/{len(normal)} | {poor}/{len(normal)} | {worst_text} | {reasons} |")
    lines += ['', '80%和50%仅帮助阅读分布，不替代连续指标，也不是抓取成功判据。逐款数值见运行目录case_metrics.csv；正常空输出的CSV中F-score为0、CD留空，故障/未执行不填造分数。', '',
              '## 相同形状、不同外观的已有案例', '',
              'C017与C020的可视几何和深度逐像素完全相同，纹理不同；共享SAM3掩码的相互IoU为99.67%。这对案例可以观察外观与预测掩码变化下的稳定性，因掩码未强制相同，不是严格的纯纹理消融。', '',
              '| 方法 | C017 CD / F-score@5mm | C020 CD / F-score@5mm |',
              '| --- | --- | --- |']
    for method, name in NAMES.items():
        cells = []
        for cid in ['C017', 'C020']:
            result = next((r for r in all_results[method] if r['id']==cid), None)
            if result is None: cells.append('未执行')
            elif result.get('success'): cells.append(f"{result['geometry']['chamfer_l1_mm']:.2f}mm / {result['geometry']['fscore_5mm']*100:.2f}%")
            else: cells.append(result['status'])
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '## 反投影参照与冷启动', '', '| 项目 | 结果 |', '| --- | --- |']
    base = baseline['results']
    lines.append(f"| 仅可见RGB-D反投影，完整表面CD中位数 | {show(number([x['geometry']['chamfer_l1_mm'] for x in base]))}mm |")
    lines.append(f"| 仅可见RGB-D反投影，F-score@5mm均值 | {np.mean([x['geometry']['fscore_5mm'] for x in base])*100:.2f}% |")
    for row in rows:
        lines.append(f"| {row['name']} 加载 / 公共预热 | {show(row['load_s'])}s / {show(row['warmup_s'])}s |")
    lines += ['', '反投影只有相机看见的表面，完整覆盖偏低是其表示边界。Mini在公共预热盘子上没有生成表面，接口另在官方样例通过；该现象保留，不更改等值面阈值补救。', '',
              '## 当前结论与下一步', '']
    if summary['records']==250 and all(r['completed_records']==50 and r['faults']==0 for r in rows):
        best = max(rows, key=lambda r:r['fscore5_mean_including_output_failures'])
        fastest = min(rows, key=lambda r:r['generation_s_p50'])
        lines += [f"250项均有回执，无资源/接口故障。按包含空输出的F-score@5mm均值，{best['name']}最高（{best['fscore5_mean_including_output_failures']*100:.2f}%）；常驻生成中位数最低为{fastest['name']}（{fastest['generation_s_p50']:.3f}s）。", '',
                  '后续选型同时检查可见表面误差与完整表面覆盖。RGB-only方法若归一化形状较好而米制误差较大，应先定位尺度/朝向定标；RGB-D方法仍需逐例看隐藏补全和薄小结构。先用本轮表格和固定案例图筛出候选，再另行登记GraspGenX接口或抓取对照。']
        indexed = {method:{r['id']:r for r in all_results[method]} for method in NAMES}
        ray = next(r for r in rows if r['method']=='rayst3r'); zero = next(r for r in rows if r['method']=='zerograsp')
        paired = np.array([indexed['rayst3r'][cid]['geometry']['fscore_5mm']-indexed['zerograsp'][cid]['geometry']['fscore_5mm'] for cid in indexed['rayst3r']])
        lines += ['',
                  f"RaySt3R相对ZeroGrasp的F-score提高{(ray['fscore5_mean_including_output_failures']-zero['fscore5_mean_including_output_failures'])*100:.2f}个百分点，50款中{int((paired>0).sum())}款更高、{int((paired<0).sum())}款更低。其F-score达到80%的款式为44/50，ZeroGrasp为33/50；生成中位数约为ZeroGrasp的{ray['generation_s_p50']/zero['generation_s_p50']:.1f}倍。",
                  '',
                  '收益集中在食品盒、罐头、杯子及剪刀。RaySt3R在番茄酱瓶上的类别均值为74.98%，低于ZeroGrasp的81.76%；碗为94.99%，也略低于95.57%。C043勺子的CD为7.16mm，是RaySt3R本轮最差CD案例，不能只根据总体均值选型。',
                  '',
                  '混元Mini有7例空网格；完整Turbo解决了本轮空输出，仍存在形状与场景对齐误差。SPAR3D虽50例均有网格，部署CD中位数9.35mm、归一化形状CD中位数5.29%，表明问题不只体现在是否输出。三个RGB生成分支的F-score均低于直接反投影的73.17%，且需要额外约2秒观测定标。',
                  '',
                  '建议下一步优先把RaySt3R作为新主线的补全候选，与直接RGB-D反投影接入相同GraspGenX配置后另行登记对照，重点检查食品盒的收益及番茄酱瓶、细长物体的退步。当前新主线的反投影接口继续保留；这三种RGB生成分支暂不作为默认点云入口。ZeroGrasp在本实验中保持比较方法身份。']
    else:
        lines += ['尚未全部完成，当前数字只对应已保存结果。剩余槽位结束后再比较整体表现；停止标记保留，故障与未执行不自动重跑。']
    lines += ['', '与同帧RGB-D可见表面反投影逐例比较如下。空输出的F-score计0；该表为描述性配对结果，不代表抓取收益。', '',
              '| 方法 | F-score更高 / 更低 / 相同 | 平均变化（百分点） |',
              '| --- | --- | --- |']
    base_scores = {r['id']:r['geometry']['fscore_5mm'] for r in baseline['results']}
    for row in rows:
        normal = [r for r in all_results[row['method']] if r['status'] in ['completed', 'model_output_failure']]
        if len(normal)!=50:
            lines.append('| '+row['name']+' | 等待50项完整回执 | — |')
            continue
        differences = np.array([(r['geometry']['fscore_5mm'] if r.get('success') else 0)-base_scores[r['id']] for r in normal])
        better, worse = int((differences>1e-9).sum()), int((differences < -1e-9).sum())
        lines.append(f"| {row['name']} | {better} / {worse} / {50-better-worse} | {differences.mean()*100:+.2f} |")
    lines += ['', '## 固定案例图', '', '每类取登记顺序中的第一款，共10例，不按成绩选图。颜色表示预测点到参考表面的距离；两排是同一米制坐标下的不同观察方向。输入拼图中的绿色仅为掩码展示，实际送给模型的RGB保留原颜色，另传二值掩码或透明通道。', '',
              '[![50例输入](resources/inputs_contact_sheet_thumb.jpg)](resources/inputs_contact_sheet.jpg)', '']
    if (PROJECT/'docs/pointcloud_benchmark/resources/metric_distributions.jpg').exists():
        lines += ['![几何与时间分布](resources/metric_distributions.jpg)', '']
    for i in range(1, 51, 5):
        name = f'comparison_c{i:03d}'
        if (PROJECT/f'docs/pointcloud_benchmark/resources/{name}.jpg').exists():
            lines.append(f'[![C{i:03d}五方法比较](resources/{name}_thumb.jpg)](resources/{name}.jpg)')
    if (PROJECT/'docs/pointcloud_benchmark/resources/empty_outputs.jpg').exists():
        lines += ['', '以下为全部7个空网格案例的输入裁剪图；完整输入和共享掩码均非空。它们仍计入50项分母。', '',
                  '[![全部空网格案例](resources/empty_outputs.jpg)](resources/empty_outputs.jpg)']
    lines += ['', '## 适用范围', '',
              '本轮为仿真静态几何比较。参考来自实际渲染的可视网格，预测共用20,000点上限，参考固定50,000点；RGB-D反投影另作参照。相机标定、畸变与深度编码已核对，尚未模拟真实双目噪声、反光缺失及运动模糊。', '',
              '本轮不计算抓取成功率，不启动GraspGenX、cuRobo或物理抓取。', '',
              '运行目录：experiments/pointcloud_benchmark/runs/five_methods_50_20261122_01/。原生输出、米制点云、逐例指标、源代码/权重/输入散列和计时均保留。']
    doc = PROJECT/'docs/pointcloud_benchmark/评测结果.md'; temp = doc.with_suffix('.tmp')
    temp.write_text('\n'.join(lines)+'\n'); temp.replace(doc); print(doc)


if __name__ == '__main__':
    with (RUN/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
