"""Publish the two V7 gripper/search compute comparisons as standalone files."""
import shutil

from arx_project.layout import ROOT, read_json, sha256


def section(website, site):
    active_path = ROOT/'tmp/gripper_sweep_optimization/active.json'
    if not active_path.exists():
        return '', None, []
    active = read_json(active_path)
    if active['phase'] != 'complete':
        return '', None, []
    root = (ROOT/active['candidate_run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/gripper_sweep_optimization'):
        raise ValueError('Unexpected sweep comparison root')
    snapshot = root/'publication_snapshot.json'
    evidence = read_json(snapshot)
    if not evidence['task_complete'] or not evidence['record_integrity']:
        raise ValueError('Sweep comparison is not accepted')

    def attach(source, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise RuntimeError('Sweep publication must use independent copies')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError('Sweep publication copy differs')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    data_url = attach(snapshot, 'sam3-v7-gripper-sweep-results.json')
    document = ROOT/'modules/pipeline/docs/mainline_versions/V7_开合爪批量校验与候选搜索优化.md'
    report_url = attach(document, 'sam3-v7-gripper-sweep-report.md')
    from modules.pipeline.experiments.gripper_sweep_optimization.report import timing_table, average
    a, b = evidence['phases']
    lines = ['## 开合爪与候选搜索的计算优化', '',
        '原来每改变0.5毫米开度就完整绑定一次机器人和场景。现在保留全部开度采样，'
        '先生成每个开度的原生碰撞球，再批量检查。左臂89个开爪采样完整保留。', '',
        '第一轮比较逐采样绑定和批量校验：', '']
    lines += timing_table(a)
    lines += ['', f"完整规划累计省时{a['review']['total_saving_fraction']:.1%}。双臂开爪校验的平均耗时"
              f"由{average(a,'legacy','gripper_opening'):.2f}秒降到{average(a,'batched','gripper_opening'):.2f}秒。", '',
        '第二轮两组都使用批量校验，只加快候选宽度变化时的YAML配置编解码：', '']
    lines += timing_table(b)
    lines += ['', f"第二轮累计再省{b['review']['total_saving_fraction']:.1%}。两轮分别统计，每组均为五个场景各两次规划。", '',
        '40次规划全部通过。抓姿、选臂、开度、关节终点和完整预抓取/接近轨迹完全一致；'
        '258个开合爪采样的碰撞球坐标、半径及碰撞判断一致，中途碰撞能正常拦下。', '',
        '候选搜索仍有三个优化方向：固定场景只绑定一次、候选间复用接触几何、'
        '让批量IK支持每个候选自己的夹爪宽度。详细分项计时和逐场景结果见报告。', '',
        f'[详细方法、分项计时与后续优化]({report_url}) · [完整配对结果与核验]({data_url})', '']
    status = '开合爪与候选配置优化已完成'
    batch = {'id': root.name, 'title': '开合爪批量校验与候选搜索计算优化', 'date': '2026-11-27',
             'status': status, 'summary': f"40/40规划通过；批量开合爪累计省{a['review']['total_saving_fraction']:.1%}，"
                 f"配置编解码再省{b['review']['total_saving_fraction']:.1%}。",
             'outputs': [p['run'] for p in evidence['phases']],
             'artifacts': [{'label': '方法与结果', 'url': report_url}, {'label': '配对数据', 'url': data_url}]}
    return '\n'.join(lines), status, [batch]
