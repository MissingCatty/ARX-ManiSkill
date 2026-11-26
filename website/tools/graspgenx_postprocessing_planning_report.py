"""Write the planning comparison readout from recorded paired measurements."""
from collections import Counter

from arx_project.layout import ROOT


def write_report(run,reg,progress,review):
    if progress['phase']!='complete':return None
    summaries=progress['summaries'];rows=progress['results']
    title='五场景先导' if reg['phase']=='pilot' else '完整69场景'
    a=summaries['control'];b=summaries['nms15']
    before=a['postprocessing_plus_planning_seconds']['total'];after=b['postprocessing_plus_planning_seconds']['total']
    delta=1-after/before if before else 0
    lines=['# V7：NMS候选优先顺序规划对照结果','',f'{title}：两组各{len(rows)}个相同场景，共{progress["completed_requests"]}次纯规划。',
        '', '## 主要结果','','| 方法 | 规划通过 | 候选检查中位数 | 后处理中位耗时 | 完整规划中位数 / P90 | 后处理＋规划中位数 / P90 | 后处理＋规划合计 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,v in summaries.items():
        label='原候选＋批量16' if mode=='control' else 'NMS15＋后备＋批量16'
        plan=v['planning_request_seconds'];total=v['postprocessing_plus_planning_seconds']
        lines.append(f"| {label} | {v['successes']}/{v['requests']} | {v['candidate_checks']['median']:.0f} | "
            f"{v['postprocessing_seconds']['median']*1000:.1f}毫秒 | {plan['median']:.2f} / {plan['p90']:.2f}秒 | "
            f"{total['median']:.2f} / {total['p90']:.2f}秒 | {total['total']:.2f}秒 |")
    lines += ['',f"后处理加完整规划累计{'减少' if delta>=0 else '增加'}{abs(delta):.2%}。",
        f"NMS组有{b['selected_deferred']}例最终使用后备候选；{sum(not r['comparison']['same_original_selection'] for r in rows)}例选中的原候选或手臂发生变化。",'',
        '## 具体改变','','NMS用15毫米/15°选出代表，优先按原评分检查代表；剩余原候选继续按原评分保留。两组均为相同批量16 IK、16/64双臂交替调度、同一GPU和常驻模型池。',
        '抓点、评分、开度、871碰撞球、TCP43、V2桌面余量及路径检查保持。几何硬筛选关闭。','',
        '后处理计时为本次真实坐标核对、NMS、来源关系与队列构造。完整规划包含开爪、双臂搜索、IK、碰撞、路径与请求内记录；共享内存交接全过程另记。','',
        '## 逐例结果','','| 场景 | 原候选规划 | NMS规划 | 原候选检查数 | NMS检查数 | 原候选后处理＋规划 | NMS后处理＋规划 | NMS选中原编号／手臂 |','| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        control=row['variants']['control'];nms=row['variants']['nms15']
        lines.append(f"| {row['id']} | {'通过' if control['success'] else '规划不出来'} | {'通过' if nms['success'] else '规划不出来'} | "
            f"{control['candidate_checks']} | {nms['candidate_checks']} | {control['postprocessing_plus_planning_seconds']:.2f}秒 | "
            f"{nms['postprocessing_plus_planning_seconds']:.2f}秒 | {nms['selected_original_source_index']}／{nms['selected_arm']} |")
    lines+=['','## 最终失败分布','','| 方法 | 规划不出来 |','| --- | ---: |',
        f"| 原候选 | {a['requests']-a['successes']} |",f"| NMS15＋后备 | {b['requests']-b['successes']} |",'',
        '## 候选被拒绝的环节','','下表统计搜索过程中被拒绝的候选，不把它们当作最终场景失败。','',
        '| 环节 | 原候选 | NMS15＋后备 |','| --- | ---: | ---: |']
    rejected={mode:Counter() for mode in summaries}
    for row in rows:
        for mode in rejected:rejected[mode].update(row['variants'][mode]['rejected'])
    labels={'grasp_ik':'抓取终点IK或碰撞','pre_or_lift_ik':'预抓取或抬升终点IK','path':'接近预抓取位置的路径','local_approach':'最后接近路径'}
    for key,label in labels.items():lines.append(f"| {label} | {rejected['control'][key]} | {rejected['nms15'][key]} |")
    lines+=['','## 记录核验','']
    if review:
        lines.append(f"来源、完成回执及物理臂FK核验通过：{review['checked_requests']}次规划、{review['checked_endpoints']}个终点、{review['checked_original_candidate_rows']:,}行候选原值。")
    else:lines.append('独立记录与物理臂FK核验中。')
    lines+=['','本轮没有新增感知、仿真实例或实际抓取；规划通过率与物理抓取成功率分别记录。','',
        f"完整记录：`{run.relative_to(ROOT)}`。",'']
    report=run/'report.md';report.write_text('\n'.join(lines))
    if reg['phase']=='full':
        (ROOT/'modules/pipeline/docs/mainline_versions/V7_候选后处理规划对照结果.md').write_text(report.read_text())
    return report
