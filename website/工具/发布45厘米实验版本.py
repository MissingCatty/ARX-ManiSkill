"""将已核验的45.2厘米实验发布为独立版本；不修改试次与冻结代码。"""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil

spec=importlib.util.spec_from_file_location('height_report',Path(__file__).with_name('汇总45厘米高度实验.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def ratio(n,d):return f'{n}/{d}（{100*n/d:.1f}%）'

def document(s,gallery):
    z,c=[s['new'][g] for g in h.ARMS];paired=s['paired'];n=160
    old=s['history_matched_160'];high=old['height_452_old_bank'];low=old['height_220_old_bank']
    delta=(c['successes']-z['successes'])/n*100
    cross_z=s['historical_same_height_pair'][h.ARMS[0]]
    regression='、'.join(f'{h.REASONS.get(k,k)} {v} 例' for k,v in cross_z['regression_reasons'].items())
    a=s.get('audit');when=s.get('runtime',{}).get('time','进行中')
    text=[
        '## 版本定位',
        '',
        '从 [GraspCL 实验版本](/iterations/graspcl/) 分出的高度与示教适配实验，仍比较 ZeroGrasp＋cuRobo 与 ZeroGrasp＋GraspCL＋cuRobo。',
        '',
        '**45.2 cm 指升降关节 joint4 的读数，不是夹爪离桌面的高度。** 桌面仍为 50 cm；机器人、资产与物理保持原设置。先完成“只升高、沿用旧库”的 400 次对照，再补采匹配此高度的独立示教，完成 160 个非 S5 场景的两法对照。两个阶段分别统计。',
        '',
        '## 修改前后',
        '',
        '| 项目 | 原主线／GraspCL 分支 | 阶段一：只调整高度 | 阶段二：本轮匹配示教 |',
        '| --- | --- | --- | --- |',
        '| 初态升降读数 | 约 22 cm | 45.2 cm | 45.2 cm |',
        '| GraspCL 本地经验 | 旧 12 正例＋13 负例 | 原库保持冻结 | 新采 40 个独立物体；24 正例＋4 负例入库，其余 12 例不作干净标签 |',
        '| 工具参考点 | 执行 TCP43，旧库按既有规则转换 | 保留原转换 | 示教直接存 TCP43，召回不重复加 3 mm |',
        '| 相机 | 腕部 RGB-D＋实例掩码 | 挂载与内参不变，升高后重新采图 | 同高度腕部采图、建库、测试 |',
        '| 测试场景 | 原 200 场景，含 40 个 S5 遮挡场景 | 两法各 200 次，共 400 槽 | 原 S1–S4 各 40 场景，两法各 160 次，共 320 槽 |',
        '| 共同执行 | 原六轴，固定朝向、8 cm 预抓取、TCP43、名义抬升 3 cm | 保留 | 保留 |',
        '| 共同成功条件 | 物体实际抬高至少 1 cm、连续保持 3 秒；双指接触、动作收敛、禁止接触检查通过 | 保留 | 保留 |',
        '',
        '旧执行端本来已有 TCP40→TCP43 的转换，因此本轮不把旧库描述为已证实存在 3 mm 错位。修正的是存储定义显式化、避免新库重复补偿，以及标签、观察高度和示教来源匹配。测试期间新库冻结，不写入测试反馈；源物体及其视觉／碰撞网格与测试目标、干扰物无重合。',
        '',
        '## 阶段一：45.2 cm 高度的 400 次结果',
        '',
        '| 管线 | 初始规划通过／全部登记 | 规划通过／进入规划 | 实际成功／全部登记 |',
        '| --- | ---: | ---: | ---: |',
        '| ZeroGrasp＋cuRobo | 153/200（76.5%） | 153/161（95.0%） | 120/200（60.0%） |',
        '| ZeroGrasp＋GraspCL＋cuRobo，旧库 | 160/200（80.0%） | 160/161（99.4%） | 120/200（60.0%） |',
        '',
        '两组各 39 个场景未进入抓取规划：31 个升高后遮挡率低于旧 S5 分组的 20% 下限、7 个原登记初始化未通过、1 个腕部视野未通过。这些保留在 200 的分母，不能算成 cuRobo 的 IK／碰撞失败。高度升高同时改变了腕部的世界观察位置；这是同场景工程对照，22 cm 数据为历史参考。',
        '',
        '### 相同 160 个非 S5 场景的历史参考',
        '',
        '| 历史条件 | ZeroGrasp 初始规划 | GraspCL 初始规划 | ZeroGrasp 成功 | GraspCL 成功 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for title,group in [('22 cm／旧库',low),('45.2 cm／旧库',high)]:
        l,r=[group[g] for g in h.ARMS]
        text.append(f'| {title} | {ratio(l["planning_passes"],160)} | {ratio(r["planning_passes"],160)} | {ratio(l["successes"],160)} | {ratio(r["successes"],160)} |')
    text += ['',
        '上述 160 场景均按本轮原登记 ID 取交集，避免用 200 场景结果直接对比排除 S5 后的 160 场景结果。旧库阶段原生组实际成功增加 19 例，GraspCL 组没有净增；不能概括为升高对两条管线都提高了抓取成功率。',
        '',
        '## 阶段二：新示教库的 160 场景对照',
        '',
        '| 指标 | ZeroGrasp＋cuRobo | ZeroGrasp＋新 GraspCL 库＋cuRobo |',
        '| --- | ---: | ---: |',
        f'| 已完成测试 | {z["completed"]}/160 | {c["completed"]}/160 |',
        f'| 初始规划通过 | {ratio(z["planning_passes"],n)} | {ratio(c["planning_passes"],n)} |',
        f'| 进入抓取执行 | {ratio(z["physical_attempts"],n)} | {ratio(c["physical_attempts"],n)} |',
        f'| 实际抓取成功 | **{ratio(z["successes"],n)}** | **{ratio(c["successes"],n)}** |',
        f'| 未进入抓取执行 | {ratio(n-z["physical_attempts"],n)} | {ratio(n-c["physical_attempts"],n)} |',
        f'| 已执行但抓取失败 | {ratio(z["physical_attempts"]-z["successes"],n)} | {ratio(c["physical_attempts"]-c["successes"],n)} |',
        '',
        f'本轮 GraspCL 与原生管线的实际成功率差为 **{delta:+.1f} 个百分点**。40 次源示教采集另计，其中规划通过 30 次、成功 24 次，不加入上述测试分子或分母。',
        '',
        '### 按场景布局',
        '',
        '| 布局 | 场景数 | ZeroGrasp 规划通过 | GraspCL 规划通过 | ZeroGrasp 成功 | GraspCL 成功 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for layout,title in h.LAYOUTS.items():
        l,r=[s['new'][g]['by_layout'][layout] for g in h.ARMS]
        text.append(f'| {title} | 40 | {ratio(l["planning"],40)} | {ratio(r["planning"],40)} | {ratio(l["success"],40)} | {ratio(r["success"],40)} |')
    text += ['',
        '### 同场景配对与失败分布',
        '',
        '| 同场景结果 | 数量／160 |',
        '| --- | ---: |']
    for key,label in [('both_success','两法都成功'),('graspcl_only','仅 GraspCL 成功'),('zerograsp_only','仅 ZeroGrasp 成功'),('both_failure','两法都失败')]:
        text.append(f'| {label} | {ratio(len(paired[key]),160)} |')
    text += ['',
        '| 最终失败原因（每次只计一个；百分比以 160 为分母） | ZeroGrasp | GraspCL |',
        '| --- | ---: | ---: |']
    reasons=set(z['failure_reasons'])|set(c['failure_reasons'])
    for reason in sorted(reasons,key=lambda k:-(z['failure_reasons'].get(k,0)+c['failure_reasons'].get(k,0))):
        text.append(f'| {h.REASONS.get(reason,reason)} | {ratio(z["failure_reasons"].get(reason,0),160)} | {ratio(c["failure_reasons"].get(reason,0),160)} |')
    text += ['',
        '## 结果分析',
        '',
        f'当前的初始规划通过已达到 {ratio(z["planning_passes"],160)} 与 {ratio(c["planning_passes"],160)}。规划通过后仍分别有 {z["physical_attempts"]-z["successes"]} 与 {c["physical_attempts"]-c["successes"]} 例实际抓取失败，应重点看闭合接触、接近收敛和抬升保持，而不只继续扩大 IK 搜索。有限候选未找到可执行初始方案，也不等于证明该物体数学上不可达。',
        '',
        f'上一轮同为 45.2 cm、同一批 160 场景时，两组都成功 120 例；本轮原生组成功 {z["successes"]} 例，新库 GraspCL 组成功 {c["successes"]} 例。原生组也有跨批波动，因此“本轮两法差值”与“换库相对旧库的效果”要分开。这里每次重新观察、推理和执行，没有同期旧库第三组，现有结果不能单独隔离新库、候选变化与执行波动的贡献。',
        '',
        f'逐例对齐上一轮同高度原生组，本轮恢复 {len(cross_z["current_only"])} 例、退步 {len(cross_z["historical_only"])} 例；退步原因为{regression}。两轮 {cross_z["same_planning_outcomes"]}/160 个场景的初始规划成败一致，原生组成功数回落主要发生在实际接触和保持阶段，而不是初始规划通过率下降。逐例清单已放入汇总数据。',
        '',
        '下一步优先对两法结果相反的场景，沿候选来源、实际选择的抓法、闭合接触和保持期物体高度逐段定位。若要单独证明新库收益，再登记同高度、同观测／候选处理的原生、旧库、新库同期对照。头部初始感知与腕部近场复核仍列为[未来计划](/roadmap/#head-wrist)，未混入本次实验。',
        '',
        '## 十个代表场景',
        '',
        '覆盖四类布局，包含 3 个仅 GraspCL 成功、2 个仅原生成功、3 个都成功和 2 个都失败的案例；按布局及物体分散选择。这是便于查看差异的展示集，统计仍使用完整 160 场景。',
        '',
        '[[实验图集]]',
        '',
        '缩略图来自各次原始初始全景；点击后，上排为该场景原生组初始全景和腕部输入，下排为两条链路各自录像的结束前画面。没有重新运行仿真补画截图；成功以整段轨迹判定。两组各自重新观测，不把一组腕部图冒充两组完全相同的输入。',
        '',
        '## 记录与复核',
        '']
    if a:
        text += [f'正式测试 320/320、源采集 40/40 均完成，结束时间 {when}。逐条重新读取 {a["trace_recomputed"]} 段原轨迹，按冻结的 1 cm／3 秒、双指接触、动作收敛与禁止接触判据核对结果；检查全部领取与结果散列、新库冻结时间、零反馈回写，以及 {a["source_and_asset_hashes_checked"]} 项源文件／模型／资产散列。所属服务全部正常退出，无结果未知槽。',
            '',
            f'源采集、建库、组件预热与正式测试总墙钟约 {s["runtime"]["wall_seconds"]/60:.1f} 分钟。两组单槽平均耗时分别为 {s["slot_timing"][h.ARMS[0]]["mean_seconds"]:.1f} 秒与 {s["slot_timing"][h.ARMS[1]]["mean_seconds"]:.1f} 秒，包含观察、推理、规划、执行、录像和核验；两组并行，单槽耗时不能直接相加当成批次墙钟，也不是单独 cuRobo 的求解时间。',
            '',
            '全部 360 条记录的两路视频文件均存在；本页十个场景的 20 条第三人称录像已实际解码取帧，不宣称已逐帧审阅全部录像。']
    else:text += ['当前仅生成未发布草稿；完整结束、最终记录核验和服务退出后才登记为已完成版本。']
    text += ['',
        '[实验汇总与配对清单](/media/height452-summary.json) · [最终核验](/media/height452-audit.json) · [截图来源](/media/height452-screenshots.json)',
        '',
        '原始批次：`ManiSkill/outputs/arx_selected_height_pair_20261120_01/`（阶段一）及 `ManiSkill/outputs/arx_graspcl_compat160_20261120_01/`（阶段二）。原登记、结果和停止标记保持原样。',
        '',
        '## 参考文献与方法',
        '',
        '- [ZeroGrasp 论文与方法](/papers/zerograsp/)：由腕部 RGB-D 与实例掩码生成原生后处理候选。',
        '- [GraspCL 论文与方法](/papers/graspcl/)：使用本轮冻结的示教召回与重评分，不进行测试在线学习。',
        '- [cuRobo 论文与方法](/papers/curobo/)：检查固定六轴模型下的 IK、碰撞与完整路径。',
        '- [高度选择与阶段一完整设置](/roadmap/fixed-height/)；[示教兼容与采集细节](/roadmap/graspcl-compat/)。',
        '',
        '本版本没有七关节实际执行、MPC、CLARO 移障或 VLM 控制。']
    return '\n'.join(text)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args()
    s=h.read(h.DEST/('处理中汇总.json' if args.draft else '实验汇总.json'))
    gallery=h.read(h.SITE/'内容/45厘米高度实验图集.json')
    if not args.draft:assert s['final'] and s['audit']['passed'] and s['audit']['final']
    body=document(s,gallery)
    if args.draft:
        (h.DEST/'实验版本草稿.md').write_text(body);print('仅生成草稿');return
    registry=h.SITE/'内容/网站.json';site=h.read(registry)
    z,c=[s['new'][g] for g in h.ARMS]
    version={'id':'lift452','parent':'graspcl','type':'experiment','title':'45.2 cm 升降与匹配示教',
        'shortTitle':'45.2 cm＋新示教库','date':'2026-11-21','status':'320 次对照已核验',
        'summary':'先验证升高后的两条抓取链路，再重采 TCP43、同高度与当前判据的独立示教，在160个非遮挡分组场景对照。',
        'metric':f'原生 {ratio(z["successes"],160)} · 新库 {ratio(c["successes"],160)}',
        'file':'45厘米升降与匹配示教实验.md','gallery':'45厘米高度实验图集.json'}
    site['versions']=[v for v in site['versions'] if v['id']!=version['id']]+[version]
    for name,file in [('height452-summary.json','实验汇总.json'),('height452-audit.json','最终核验.json'),('height452-screenshots.json','截图来源.json')]:
        site['localAssets'][name]='45厘米高度实验/'+file
    for record in site['configurationRecords']:
        if record['id']=='lift-height':
            record['status']='离线扫描与45.2 cm实测完成'
            record['summary']='高度细扫后选45.2 cm完成400次对照；随后补采匹配示教，完成160场景两法对照，结果见独立版本。'
    h.save(registry,site)
    (h.SITE/'内容/迭代'/version['file']).write_text(body)
    paperfile=h.SITE/'内容/论文调研/论文目录.json';papers=h.read(paperfile)
    for paper in papers['papers']:
        if paper['id'] in ['zerograsp','graspcl','curobo'] and version['id'] not in paper['versions']:paper['versions'].append(version['id'])
    h.save(paperfile,papers)
    plan=h.SITE.parents[1]/'ARX-可达性地图/docs/未来改进计划.md'
    lines=plan.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 5 | [GraspCL'):
            lines[i]=line.rsplit(' | ',1)[0]+f' | 已完成并核验 320/320；原生成功 {z["successes"]}/160，新库成功 {c["successes"]}/160，所属服务正常退出 |'
    plan.write_text('\n'.join(lines)+'\n')
    doc=h.MS/'docs/GraspCL示教兼容修复与160场景对照.md';old=doc.read_text();cut=old.index('当前已完成清单冻结')
    old=old[:cut]+f'''## 完成状态

2026 年 11 月 21 日全部测试完成并核验。源采集 40/40，获得 24 条成功示教与 4 条干净负反馈；新库在首个测试槽前冻结。两组各 160 次，初始规划分别为 {z['planning_passes']}/160 和 {c['planning_passes']}/160，实际成功为 {z['successes']}/160（{z['successes']/160*100:.1f}%）和 {c['successes']}/160（{c['successes']/160*100:.1f}%）。结果未知 0，全部所属服务正常退出。源示教不计入测试分母。

最终只读核验重新读取全部 360 段源／测试轨迹及原判据，核对领取记录、结果散列、原始图像和双路录像文件、新库未回写及源文件／资产。完整网页报告在项目网站 `/iterations/lift452/`；本地 [实验报告](45厘米高度与匹配示教实验报告.md) 与 [汇总数据](resource/45厘米高度与新示教库对照/实验汇总.json)。原执行代码、登记、原始结果保持冻结。
'''
    doc.write_text(old)
    report=h.MS/'docs/45厘米高度与匹配示教实验报告.md'
    report.write_text('# 45.2 cm 升降与匹配示教实验报告\n\n'+body.replace('[[实验图集]]','十个代表场景的可点击图集见项目网站 `/iterations/lift452/`。'))
    for file in ['实验汇总.json','最终核验.json','截图来源.json']:
        dest=h.MS/'docs/resource/45厘米高度与新示教库对照'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(h.DEST/file,dest)
    conf=h.SITE/'内容/实验配置/升降高度与可达性比较.md'
    if '/iterations/lift452/' not in conf.read_text():
        with conf.open('a') as f:f.write('\n\n## 后续实测\n\n已在45.2 cm完成两阶段实际抓取：只升高的400次对照，以及匹配示教后的160场景两法对照。[查看独立实验版本与十场景截图](/iterations/lift452/)。两阶段分母与示教库分别列出。\n')
    print('已登记实验版本 /iterations/lift452/；请重新构建网站')

if __name__=='__main__':main()
