"""Summarize a finished, frozen planning study without changing any trials."""
from 公共 import EXPERIMENTS
from collections import Counter
from pathlib import Path
from 公共 import ROOT,read,save,sha,now
from 评估规划研究 import evaluate

P=EXPERIMENTS / '自动研究_20261120_01'


def main():
    names=['00_基线','01_地图初值','02_地图选择H','A0_不用地图短H',
           '确认_地图_1','确认_短H_1','确认_短H_2','确认_地图_2','留出_基线','留出_地图','留出_短H']
    summaries={n:read(P/('轮次/'+n+'/汇总.json')) for n in names}
    results={n:evaluate(P/'轮次'/n,P/'轮次'/('留出_基线' if n.startswith('留出') else '00_基线')) for n in names}
    protected=read(P/'登记.json')['source_sha256'];assert all(sha(p)==h for p,h in protected.items())
    frozen=read(P/'最终方案冻结.json')
    assert sha(ROOT/'工具/规划研究执行器.py')==frozen['executor_sha256']
    assert sha(ROOT/'工具/评估规划研究.py')==frozen['evaluator_sha256']
    metrics={}
    for name,d in summaries.items():
        metrics[name]={**results[name],'ik_calls_total':sum(r['counts'].get('ik_calls',0) for r in d['rows']),
             'path_pairs_total':sum(r['counts'].get('path_pairs',0) for r in d['rows']),
             'valid_plan_over_time':[r['case'] for r in d['rows'] if r['reason']=='declared_compute_budget_exhausted_after_atomic_solver_call'],
             'case_wall_total_seconds':sum(r['wall_seconds'] for r in d['rows'])}
    dev=summaries['00_基线']['rows'];heldout=summaries['留出_基线']['rows']
    def case_table(rows,methods):
        lines=[]
        for row in rows:
            values=[]
            for n in methods:
                result=next(r for r in summaries[n]['rows'] if r['case']==row['case'])
                values.append('通过' if result['success'] else '未通过')
            lines.append('| '+row['case']+' | '+' | '.join(values)+' |')
        return '\n'.join(lines)
    dev_table=case_table(dev,['00_基线','01_地图初值','02_地图选择H','A0_不用地图短H'])
    heldout_table=case_table(heldout,['留出_基线','留出_地图','留出_短H'])
    metric_lines=[]
    for n in names:
        m=metrics[n]
        metric_lines.append(f"| {n} | {m['passes']}/{m['total']} | {len(m['gains'])}/{len(m['regressions'])} | {m['ik_calls_total']} | {m['median_seconds']:.2f} |")
    controls=[]
    for name in ['02_地图选择H','A0_不用地图短H','留出_地图','留出_短H']:
        controls.append({'name':name,'failed_controls':[r['case'] for r in summaries[name]['rows'] if 'control' in r['stratum'] and not r['success']]})
    save(P/'研究汇总.json',{'finished_at':now(),'metrics':metrics,'protected_sources_unchanged':True,
        'control_results':controls,'new_scene_planning_runs':sum(d['total'] for d in summaries.values()),
        'unique_scenes':24,'development_hypotheses_used':2,'maximum_hypotheses':6,
        'physical_attempts':0,'learned_models_trained':0,'G_modified':False,
        'map_success_rate_increment_over_blind_H':results['留出_地图']['passes']-results['留出_短H']['passes'],
        'interpretation':'Development planning gains can be obtained by adaptive H without training or changing G. Map pass-rate advantage must be judged against blind H, separately from IK call savings. No physical grasp conclusion or default promotion.'})
    m=metrics['02_地图选择H'];b=metrics['A0_不用地图短H'];hm=metrics['留出_地图'];hb=metrics['留出_短H'];h0=metrics['留出_基线']
    report=f'''# 可达性地图辅助规划：本轮 AutoResearch 结果

日期：2026 年 11 月 20 日。研究对象为当前 ARX／TCP43 链路的完整初始规划，不是实际抓取成功率。所有结果保留在本目录，旧场景、旧结果、冻结脚本与默认执行器没有改写。

## 结论

开发集：固定 G、固定 8 cm H 为 5/12；仅增加地图 IK 初值仍为 5/12；允许选择 H 后，地图方案和不用地图的距离对照均为 10/12，新增同样五例、无退步。**这批成功率提升主要来自 H 的选择空间，不能说必须使用地图，更没有证据要求训练连续 G 修正网络。**五例与前次短 H 定位一致，是重新确认。

开发集地图方案使用 {m['ik_calls_total']} 次 IK，不用地图的短 H 对照使用 {b['ik_calls_total']} 次，少 {1-m['ik_calls_total']/b['ik_calls_total']:.1%}。地图可作为搜索顺序与初值辅助；是否值得保留，应同时看调用次数与实际耗时，不仅看是否“用了地图”。

冻结后留出验证：基线 {h0['passes']}/12，地图 {hm['passes']}/12，不用地图的短 H 为 {hb['passes']}/12。地图相对短 H 的新增／退步和逐例结果列在下面，不据留出结果继续调参。未启动任何网络训练，G 位置、朝向与接触描述始终保持不变。

## 评测方式

12 个开发场景含 7 个原抓取／预抓取失败、2 个进场碰撞案例及 3 个规划通过对照。12 个留出场景从原 200 例按固定散列顺序选取：3 个抓取失败、3 个预抓取／抬升失败、6 个实际成功对照；仅指未用于本轮调参，不称全新物体／全新盲测或总体随机样本。

原生后处理及已有截断后的候选集合完全冻结，不使用原始数千候选，不增加硬性地图剔除规则。H 只沿原接近轴后退 2/4/6/8 cm；名义抬升仍为 3 cm，原关节限位、碰撞几何、静态桌地、自碰撞策略、IK 容差和轨迹优化检查保持不变。

初始规划通过要求 G/H/名义抬升 IK 通过、包含目标的完整场景进场路径与优化通过、原接近世界下的接近路径与优化通过，全部执行原密集碰撞和限位检查。评估器另外从原候选重构目标，用独立 URDF FK 核对三个位姿误差和关节解，核对原始输入散列、轨迹时序及关节限位。6 项离线故障注入均拒绝错误记录。

共同基线比历史“选出方案后进场复核失败即停止”更强：本轮所有方法都允许继续尝试其余原生候选。共同基线还从进场规划开始就包含被抓物体；后续确认两个旧进场失败场景在原同一候选、完全相同 G/H 下恢复，不能将恢复归因于换候选，而是进场世界构建与路线生成的差异，不计地图收益。共同执行器先测名义抬升、再测 H，以避免无效 H 计算；所有方法顺序一致。

每场景上限 96 次 IK、12 对路径、12 秒；模型和地图批内常驻，启动计时单列。未执行的局部数值细化分支有额外 16 次起点上限，但本轮没有触发或计入结果。两轮开发假设达到预登记开发门槛后停止搜索，剩余候选方向未运行；随后完成距离消融、地图与短 H 各两次同输入复核和一次留出对照。

## 分组结果

| 运行 | 通过 | 相对同组基线新增／退步 | IK 总调用 | 单例中位耗时（秒） |
|---|---|---|---|---|
{chr(10).join(metric_lines)}

Torch／NumPy 的确认种子为 1、2，但 cuRobo 内部 reset_seed 仍使用原配置的采样序列；这些是同输入重复核验，不当作独立随机试验。并发 GPU 负载、场景绑定和缓存会影响墙钟时间，本轮计时是完整初始规划耗时，不等于纯 IK 内核耗时，也不作严格跨方法测速结论。

不用地图的第二次复核为 9/12，差异发生在 I0063：已找到并通过原检查的方案在原子求解调用结束时累计 12.295 秒，超过登记的 12 秒，因此仍按预算内规划失败保留，未删除或补跑。不能将这一例差异直接解释为地图找到了其他方法找不到的解。

留出控制案例 I0065（065-a_cups）在固定 H 基线及地图方案中都找到了通过原检查的方案，但分别累计 17.21／14.30 秒，超过 12 秒，因此按预算内失败计数。这是计算预算未通过，不是证明抓点不可达。最终各组超时清单保存在 `研究汇总.json`，不把旧实际成功标签改写成抓取失败，也不宣称所有历史成功案例均已在当前时限内保留。

## 开发场景逐例

| 场景 | 固定 H 基线 | 地图初值 | 地图选 H | 不用地图短 H |
|---|---|---|---|---|
{dev_table}

仍未恢复的 I0018／I0119 记录为本轮搜索下未找到完整方案，不认定数学不可达；不能据此把地图空白当训练负例。

## 留出场景逐例

| 场景 | 固定 H 基线 | 地图选 H | 不用地图短 H |
|---|---|---|---|
{heldout_table}

## 对位姿修正网络的判断

当前先采用离散 H 搜索与原场景核验更合适。地图能提供可达性正向见证，不能保证指垫接触与实际夹持保持；单独学习地图并输出连续位姿修正，容易把可达性和抓取质量混淆。若今后在候选／H 方法后仍存在明确可复现、保持接触约束的 G→G' 修复，而局部优化又成为耗时瓶颈，再用这些教师解训练小型建议／初值网络会更有依据。详见 [必要性评估](../../docs/可达性地图/下游位姿修正网络必要性评估.md)。

本轮只证明离线初始规划层面的效果；没有执行闭合、实际抬升或保持，不能承诺短 H 的物理抓取效果不退步。默认执行链路没有自动替换。

## 可核查产物

- [场景选择](场景选择.md)、[完整场景清单](场景清单.json)、[原始登记](登记.json)、[补充登记](补充登记.json)。
- [方案冻结](最终方案冻结.json)、[机械汇总](研究汇总.json)、[离线故障核验](离线故障核验.json)、[源码与输入快照](代码与输入快照/映射.json)。
- [研究日志](研究日志.md)、[迭代结果](迭代结果.tsv)、[进度图](进度.png)。每轮原始结果在 `轮次/`，无失败删除或成功替换。

方法参考：[DLR 可达性与能力地图](https://elib.dlr.de/97212/1/Astra-PUBLISHED.pdf)、[Dynamic Grasping 的可达性排序](https://arxiv.org/html/2103.10562)、[REACH 官方代码](https://github.com/ros-industrial/reach)。本轮为受这些思路启发的工程改动，不是其论文实验的完整复现。
'''
    (P/'最终报告.md').write_text(report)
    state=read(P/'状态.json');state.update(status='bounded_study_complete_no_default_promotion',finished_at=now(),
        completed_iterations=2,completed_planning_runs=sum(d['total'] for d in summaries.values()),heldout_opened=True,
        physical_attempts=0,network_training_started=False,default_promoted=False)
    save(P/'状态.json',state)
    print('STUDY_COMPLETE',P/'最终报告.md')


if __name__=='__main__':main()
