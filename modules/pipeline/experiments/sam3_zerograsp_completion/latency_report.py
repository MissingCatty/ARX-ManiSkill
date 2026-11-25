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
    return {'at':now(),'cases':len(ids),'groups':groups,
        'generator_parts':parts,'successful_simulation_seconds':{'ids':success_ids,**simulation},
        'source_latency_sha256':{str(root/'methods'/m/'trials'/sid/'latency.json'):
            sha256(root/'methods'/m/'trials'/sid/'latency.json') for m in METHODS for sid in ids}}


def seconds(value):
    return f"{value['median_seconds']:.4f} / {value['p90_seconds']:.4f}" if value['samples'] else '—'


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
    return '\n'.join(lines)


def refresh_latency_section(page, report):
    start = page.index('## 延时对照')
    stop = page.index('## 失败分布',start)
    return page[:start]+render_latency(report)+'\n\n'+page[stop:]
