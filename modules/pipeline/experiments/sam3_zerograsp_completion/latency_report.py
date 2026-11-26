"""Pair final case timings and render identical tables for the page and report."""
import math
import json
from pathlib import Path

from arx_project.json_io import read_json
from arx_project.layout import now, sha256
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.contracts import METHODS, completed_rows
from modules.pipeline.experiments.sam3_zerograsp_comparison.metrics import distribution


STAGES = [
    ('generation_delivery_seconds','生成、转换并交出共享候选'),
    ('candidate_decode_seconds','候选一次解码'),
    ('initial_planning_rpc_seconds','首次cuRobo规划请求'),
    ('generation_and_initial_planning_seconds','生成与首次规划合计'),
    ('execution_with_replanning_seconds','执行环节电脑耗时，含重规划'),
    ('generation_to_trial_return_seconds','生成开始至试验返回'),
    ('trial_wall_seconds','整次试验电脑耗时，含场景创建和恢复'),
]


def paired_groups(ids, rows):
    groups = {}
    for group in ['all_completed','both_successful']:
        eligible = [sid for sid in ids if group == 'all_completed'
                    or all(rows[m][sid]['result']['success'] for m in METHODS)]
        records = {}
        for key,_ in STAGES:
            common = []
            for sid in eligible:
                valid = True
                for method in METHODS:
                    row = rows[method][sid]
                    value = row['latency'].get('timings',{}).get(key)
                    valid &= (not row['latency'].get('common_precondition_failure')
                              and isinstance(value,(int,float)) and math.isfinite(value))
                    if key == 'execution_with_replanning_seconds':
                        valid &= bool(row['result'].get('physical_attempt'))
                if valid:
                    common.append(sid)
            records[key] = {'ids':common, **{m:distribution([
                rows[m][sid]['latency']['timings'][key] for sid in common]) for m in METHODS}}
        groups[group] = records
    return groups


def build_latency_report(root):
    root = Path(root)
    reg = read_json(root/'registration.json')
    ids = [pair['id'] for pair in reg['cohort']['pairs']]
    rows = {m:{row['id']:row for row in completed_rows(root/'methods'/m)} for m in METHODS}
    if any(set(rows[m]) != set(ids) for m in METHODS):
        raise RuntimeError('Final timing review requires the same complete cohort')
    groups = paired_groups(ids,rows)
    success_ids = groups['both_successful']['execution_with_replanning_seconds']['ids']
    simulation = {}
    for method in METHODS:
        values = []
        for sid in success_ids:
            path = root/'methods'/method/'trials'/sid/'execution.jsonl'
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-1024*1024))
                values.append(json.loads(stream.read().splitlines()[-1])['sim_seconds'])
        simulation[method] = distribution(values)
    parts = {}
    for method,keys in {
        'graspgenx':['rayst3r_rpc_seconds','graspgenx_rpc_seconds','candidate_conversion_publish_seconds'],
        'zerograsp':['zerograsp_generation_postprocessing_rpc_seconds'],
    }.items():
        parts[method] = {key:distribution([row['latency'].get('timings',{}).get(key)
            for row in rows[method].values() if not row['latency'].get('common_precondition_failure')]) for key in keys}
    report = {'at':now(),'cases':len(ids),'groups':groups,
        'generator_parts':parts,'successful_simulation_seconds':{'ids':success_ids,**simulation},
        'source_latency_sha256':{str(root/'methods'/m/'trials'/sid/'latency.json'):
            sha256(root/'methods'/m/'trials'/sid/'latency.json') for m in METHODS for sid in ids}}
    analysis_path = root/'latency_bottleneck_analysis.json'
    if analysis_path.exists():
        analysis = read_json(analysis_path)
        planning_ids = set(groups['all_completed']['initial_planning_rpc_seconds']['ids'])
        for method in METHODS:
            if {row['id'] for row in analysis['methods'][method]['records']} != planning_ids:
                raise RuntimeError('Bottleneck analysis does not match the paired planning cohort')
            execution_ids = {row['id'] for row in analysis['successful_execution_decomposition'][method]['rows']}
            if execution_ids != set(success_ids):
                raise RuntimeError('Bottleneck analysis does not match the successful execution cohort')
        report['bottleneck_analysis'] = {
            'methods':{method:{key:analysis['methods'][method][key] for key in
                ['statistics','optimizer_calls','slowest']} for method in METHODS},
            'execution':{method:analysis['successful_execution_decomposition'][method]['statistics']
                         for method in METHODS},
            'source_sha256':sha256(analysis_path),
        }
    interleaving_path = root/'interleaving_validation.json'
    if interleaving_path.exists():
        report['interleaving_validation'] = read_json(interleaving_path)
    return report


def seconds(value):
    return f"{value['median_seconds']:.4f} / {value['p90_seconds']:.4f}" if value['samples'] else '—'


def render_bottleneck_analysis(report):
    analysis = report.get('bottleneck_analysis')
    if not analysis:
        return ''
    methods = analysis['methods']
    gg, zg = (methods[m]['statistics'] for m in METHODS)
    count = gg['rpc']['n']
    search_share = gg['search_with_bind']['mean']/gg['rpc']['mean']
    lines = ['## 具体耗时分析','',
        '主要瓶颈在候选搜索和双臂搜索顺序。单次cuRobo轨迹优化已经是几十毫秒，整次规划请求还要逐个检查候选的IK、碰撞和路径。','',
        '### 首次规划花在哪里','',
        f'下面使用两条链路都有完整首次规划计时的{count}个相同场景。各项为平均耗时，单位为秒，可以逐项相加。','',
        '| 首次规划内部环节 | RaySt3R＋GraspGenX | ZeroGrasp |','| --- | ---: | ---: |']
    for key,title,digits in [('decode','候选解码',3),('opening','双臂开爪过程碰撞检查',2),
                             ('search_with_bind','候选搜索：IK、碰撞、路径生成等',2),
                             ('other_request','其他请求开销',2),('rpc','合计',2)]:
        lines.append(f"| {title} | {gg[key]['mean']:.{digits}f} | {zg[key]['mean']:.{digits}f} |")
    lines += ['',f"GraspGenX的候选搜索占首次规划平均耗时约{search_share:.0%}。平均数被少数特别慢的场景拉高；两组首次规划中位数分别为{gg['rpc']['median']:.2f}秒、{zg['rpc']['median']:.2f}秒。",'',
        '### 香蕉场景为什么等了七分多钟','']
    slow = methods['graspgenx']['slowest'][0]
    lines += [f"场景`{slow['id']}`是本轮最慢的一次首次规划。V6本轮使用的实现先搜索右臂，再搜索左臂。",'',
        '| 搜索过程 | 检查候选数量 | 耗时 |','| --- | ---: | ---: |']
    for side,title in [('right','先搜索右臂'),('left','再搜索左臂')]:
        arm = slow['arm_evidence'][side]
        result = '全部未通过' if arm['selected_index'] is None else f"第{arm['selected_index']+1}个候选通过"
        lines.append(f"| {title}，{result} | {arm['endpoint_checks']}个 | {arm['search_seconds']:.2f}秒 |")
    lines += [f"| 加上其他检查的整次规划 | — | {slow['rpc']:.2f}秒 |",'',
        f"这一个场景占了GraspGenX全部{count}例首次规划累计耗时的约{slow['rpc']/(gg['rpc']['mean']*count):.0%}。右臂扫描到底之后，左臂第一个候选就通过，暴露出固定双臂搜索顺序造成的等待。",'',
        '### 候选数量怎样影响耗时','',
        '| 指标，中位数 | GraspGenX | ZeroGrasp |','| --- | ---: | ---: |',
        f"| 输入候选池数量 | {gg['pool_count']['median']:.0f}个 | {zg['pool_count']['median']:.0f}个 |",
        f"| 两臂实际检查的候选总次数 | {gg['endpoint_checks']['median']:.0f}次 | {zg['endpoint_checks']['median']:.0f}次 |"]
    solve = {m:methods[m]['optimizer_calls']['solve']['median']*1000 for m in METHODS}
    lines += [f"| 已保存记录中的单次轨迹优化求解 | {solve['graspgenx']:.1f}毫秒 | {solve['zerograsp']:.1f}毫秒 |",'',
        '候选先经过IK和碰撞筛选，通过后才继续生成路径。因此候选池里有几千个抓姿时，通常只检查其中一部分；但某只手臂始终找不到解时，也会出现扫描完整候选池的情况。','',
        '### 生成与执行的时间怎样理解','']
    generated = report['groups']['all_completed']['generation_delivery_seconds']
    parts = report['generator_parts']['graspgenx']
    lines += [f"生成阶段整体中位数为{generated['graspgenx']['median_seconds']:.2f}秒，对照ZeroGrasp为{generated['zerograsp']['median_seconds']:.2f}秒。RaySt3R重建、GraspGenX生成和候选转换交接的分项中位数分别为{parts['rayst3r_rpc_seconds']['median_seconds']:.2f}秒、{parts['graspgenx_rpc_seconds']['median_seconds']:.2f}秒、{parts['candidate_conversion_publish_seconds']['median_seconds']:.2f}秒。各分项分别取中位数，整段时间使用直接测得的结果。",'',
        f"两组共同成功的{len(report['successful_simulation_seconds']['ids'])}个场景，实际推进的仿真时间均为{report['successful_simulation_seconds']['graspgenx']['median_seconds']:.2f}秒。接近和抬升前的两次重规划合计中位数分别为{analysis['execution']['graspgenx']['replanning']['median']:.2f}秒、{analysis['execution']['zerograsp']['replanning']['median']:.2f}秒。执行环节的电脑耗时还包含物理仿真、渲染、录像和收尾；现有日志将这些开销合并记录。",'',
        '### 后续优化顺序','',
        ('1. 已完成双臂交替搜索优化，使用首轮16个、后续64个候选的窗口；验证结果见下方。'
         if report.get('interleaving_validation') else
         '1. 优先按小批候选交替检查左右臂，让容易找到解的一侧尽早参与比较，避免另一侧先扫描到底。'),
        '2. 再验证V8登记的地图关节解复用和批量IK，减少对原抓姿的重复求解。',
        '3. 继续压缩开爪检查和候选打包交接的开销。','',
        '[下载逐场景耗时分析数据](/media/sam3-v6-complete74-latency-analysis.json)']
    return '\n'.join(lines)


def render_interleaving_validation(proof):
    lines = ['## 交替搜索优化验证','',
        '已将当前移动双臂规划器切换为首轮每臂16个、后续每批64个候选交替检查。每只手保留搜索进度，同名次候选继续比较关节位移。','',
        '5组输入的规划成败、选中臂和原始抓姿全部一致。下表单位为秒，旧搜索列来自首轮同卡测量；后续调整批次时沿用该参照，只重新执行新版。','',
        '| 输入 | 旧搜索 | 固定16个交替 | 首轮16、后续64 |','| --- | ---: | ---: | ---: |']
    names = {'M0044_S3_sparse_banana':'稀疏香蕉','M0001_S1_open_apple':'开阔苹果',
             'M0010_S1_open_boxed_food':'盒装食品','M0069_S4_clutter_orange':'杂乱橙子'}
    for first,second in zip(proof['first']['paired_cases'],proof['second']['paired_cases']):
        label = names[first['id']]+('（ZeroGrasp候选）' if first['method']=='zerograsp' else '（GraspGenX候选）')
        lines.append(f"| {label} | {first['original_seconds']:.2f} | {first['interleaved_seconds']:.2f} | {second['interleaved_seconds']:.2f} |")
    lines += ['', '香蕉原搜索检查右臂3320次、左臂1次；交替搜索改为右臂16次、左臂1次，耗时约从428秒降到8.55秒。', '',
        '盒装食品第二轮计时升高，因此追加了同会话、同一模型池的四次相邻对照，按两种顺序执行：','',
        '| 相邻对照 | 固定16个交替 | 首轮16、后续64 |','| --- | ---: | ---: |']
    control = proof['window_policy']
    for repeat in [0,1]:
        rows = {r['mode']:r for r in control['results'] if r['repeat']==repeat}
        lines.append(f"| 第{repeat+1}组 | {rows['fixed']['seconds']:.2f} | {rows['expanded']['seconds']:.2f} |")
    lines += [f"| 中位数 | {control['median_seconds']['fixed']:.2f} | {control['median_seconds']['expanded']:.2f} |",'',
        '四次均检查114个候选，选择右臂原始第57个抓姿。扩大后续批次把搜索窗口从8个减到4个，中位耗时下降约10%。','',
        '本次共19次纯规划请求，28项离线检查通过；物理抓取与仿真实例均为0。原V6的74例成绩与录像继续保留，配置已供下一次新登记运行使用。','',
        '[方法与完整验证记录](/media/sam3-v6-interleaving-method.md) · [验证数据](/media/sam3-v6-interleaving-validation.json)']
    return '\n'.join(lines)


def render_latency(report):
    group = report['groups']['all_completed']
    generated = len(group['generation_delivery_seconds']['ids'])
    planned = len(group['initial_planning_rpc_seconds']['ids'])
    executed = len(group['execution_with_replanning_seconds']['ids'])
    succeeded = len(report['groups']['both_successful']['execution_with_replanning_seconds']['ids'])
    lines = ['## 延时对照','',
        '每一行都使用两组具有该阶段计时的相同场景。数值为中位数 / P90，单位为秒。',
        f'生成与整次试验统计{generated}个场景；两组都调用首次规划的有{planned}个，两组都开始实际执行的有{executed}个。执行行包含中途抓取失败，完整成功抓取另列在下一张表。','',
        '| 环节 | 配对场景数 | RaySt3R＋GraspGenX | ZeroGrasp |','| --- | ---: | ---: | ---: |']
    for key,title in STAGES:
        row = group[key]
        lines.append(f"| {title} | {len(row['ids'])} | {seconds(row['graspgenx'])} | {seconds(row['zerograsp'])} |")
    lines += ['', '首次规划请求包含候选解码、双臂候选搜索、IK、碰撞检查、路径生成和记录交接。执行环节电脑耗时包含仿真运算、渲染、录像保存、接近与抬升重规划，以及环境收尾。',
        '整次试验从场景创建前开始，到执行函数返回结束；模型加载预热、排队等待和返回后的独立核验另外记录。','',
        '## 两组都成功的完整抓取','',f'以下使用两组都抓取成功的相同{succeeded}个场景，分别列出仿真动作时间和电脑耗时。','',
        '| 项目 | RaySt3R＋GraspGenX | ZeroGrasp |','| --- | ---: | ---: |']
    sim = report['successful_simulation_seconds']
    lines.append('| 仿真动作时间 | '+seconds(sim['graspgenx'])+' | '+seconds(sim['zerograsp'])+' |')
    for key,title in [('generation_and_initial_planning_seconds','生成与首次规划'),
                      ('execution_with_replanning_seconds','执行环节电脑耗时，含重规划'),
                      ('trial_wall_seconds','整次试验电脑耗时')]:
        row = report['groups']['both_successful'][key]
        lines.append('| '+title+' | '+seconds(row['graspgenx'])+' | '+seconds(row['zerograsp'])+' |')
    lines += ['', '两组的仿真动作时间相同，执行环节电脑耗时还受仿真运算、渲染录像和运行资源影响。','',
        '## 生成阶段拆分','', '本轮RaySt3R使用登记的原重建后端。各分项分别统计中位数 / P90，单位为秒；整段生成时间直接取完整计时。','',
        '| 链路与环节 | 例数 | 中位数 / P90 |','| --- | ---: | ---: |']
    names = [('graspgenx','rayst3r_rpc_seconds','RaySt3R重建请求'),
             ('graspgenx','graspgenx_rpc_seconds','GraspGenX生成请求'),
             ('graspgenx','candidate_conversion_publish_seconds','候选转换与共享内存发布'),
             ('zerograsp','zerograsp_generation_postprocessing_rpc_seconds','ZeroGrasp生成与原生后处理请求')]
    for method,key,title in names:
        value = report['generator_parts'][method][key]
        lines.append(f"| {title} | {value['samples']} | {seconds(value)} |")
    bottlenecks = render_bottleneck_analysis(report)
    if bottlenecks:
        lines += ['',bottlenecks]
    if report.get('interleaving_validation'):
        lines += ['',render_interleaving_validation(report['interleaving_validation'])]
    return '\n'.join(lines)


def refresh_latency_section(page, report):
    start = page.index('## 延时对照')
    stop = page.index('## 失败分布',start)
    return page[:start]+render_latency(report)+'\n\n'+page[stop:]
