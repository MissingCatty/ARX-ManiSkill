"""Publish the sphere-count diagnosis without changing the adopted runtime."""
import shutil

from arx_project.layout import ROOT, read_json, sha256
from modules.pipeline.experiments.sphere_count_diagnosis.report import full_table, micro_table, stage_table


def section(website, site):
    active = ROOT/'tmp/sphere_count_diagnosis/active.json'
    if not active.exists() or read_json(active)['phase'] != 'complete':
        return '', []
    root = (ROOT/read_json(active)['run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/sphere_count_diagnosis'):
        raise ValueError('Unexpected sphere-count publication root')
    data = read_json(root/'publication_snapshot.json')
    if not data['task_complete'] or not data['record_integrity'] or data['production_sphere_count'] != 871:
        raise ValueError('Sphere-count diagnosis review is incomplete')

    def attach(source, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise RuntimeError('Sphere diagnosis attachments must be independent copies')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError('Sphere diagnosis attachment copy differs')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name

    url = attach(root/'publication_snapshot.json', 'sam3-v7-sphere-count-results.json')
    report = attach(ROOT/'modules/pipeline/docs/mainline_versions/V7_碰撞球数量对完整规划耗时的影响.md',
                    'sam3-v7-sphere-count-report.md')
    lines = ['## 碰撞球数量对完整规划的影响', '',
        '同样五个场景、同样候选和开度，分别用871、512、256球各做10次完整规划。'
        '少球模型由原球保守合并得到，执行配置继续使用871球。', '']
    lines += full_table(data)
    lines += ['', '本轮三档选中的抓法、手臂、开度和候选拒绝原因相同。512球相对871球平均减少约23%，'
        '继续减到256球没有再提速。完整请求按512→256→871分组运行；下面的固定输入单测让三档交替运行。', '',
        '### 秒级开销来自哪里', '', '每项为完整请求的平均值，已扣除重叠的子调用。', '']
    lines += stage_table(data)
    lines += ['', '每次平均绑定13次机器人与场景。871球减到512球时，绑定从2.745秒降到1.960秒。'
        '球数增加也会放大完整配置的解析、写出、复制和校验成本。', '',
        '### 同一工作量的GPU验证', '',
        '固定同一批关节状态、同一场景和同一实际开度，三档在同一张卡上交替测。'
        '下表包括FK、关节限位和世界碰撞；每格是五个场景测量中位数的中位数。', '']
    lines += micro_table(data)
    lines += ['', '一次检查256个状态时，完整GPU验证从约0.607毫秒降到0.386毫秒；'
        '仅世界碰撞查询从约0.566毫秒降到0.349毫秒。少量状态的碰撞查询本身已经约0.01毫秒。', '',
        '### 结论与后续优化', '',
        '球数对GPU碰撞计算有影响，但目前完整请求的主要秒级开销仍在重复配置和场景处理。'
        '本轮每档1280个固定状态中，512和256球分别新增8个、11个拦截，新增放行为0。', '',
        '保留871球，下一项优先减少重复YAML往返、碰撞世界装载和接触几何排序，保持当前夹爪几何。'
        '30次普通规划、2次剖析、5组固定输入测试均完成，96个末端终点独立复核通过。', '',
        f'[完整分项与逐场景报告]({report}) · [原始统计与核验]({url})', '']
    batch = {'id': root.name, 'title': '碰撞球数量对完整规划的影响', 'date': '2026-11-27',
        'status': '诊断已完成，保留871球',
        'summary': '三档各10/10规划通过；平均5.669/4.355/4.381秒，球数继续减半未再提速。',
        'outputs': [str(root.relative_to(ROOT))],
        'artifacts': [{'label': '分项数据与核验', 'url': url}, {'label': '完整诊断报告', 'url': report}]}
    from website.tools.sphere_fit_section import section as fit_section
    fit, fit_batches = fit_section(website, site)
    return fit + '\n' + '\n'.join(lines), [batch, *fit_batches]
