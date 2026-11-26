"""Read-only follow-up: independent review, report, then publish V8 results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import urllib.request

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from .validation_launch import controller_unit
from .validation_review import validate
from .followup_state import completed_controller


def report(run, checked):
    accepted = read_json(run/'acceptance.json');summaries = accepted['summaries']
    full = read_json(run/'registration.json').get('phase') == 'full'
    doc = ROOT/'modules/pipeline/docs/mainline_versions'/('V8_抓取分支批量复核完整对照结果.md' if full else 'V8_抓取分支批量复核先导.md')
    body = doc.read_text();heading = '## 本轮实测结果'
    if heading in body: body = body.split(heading)[0].rstrip()+'\n'
    lines = [heading, '', '| 方法 | 完整规划 | 规划通过 | 中位耗时 | P90耗时 | 累计耗时 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, label in [('control', '原版逐分支复核'), ('vectorized', '批量复核')]:
        value = summaries[mode]
        lines.append(f"| {label} | {value['requests']} | {value['successes']} | {value['median_seconds']:.3f}秒 | "
                     f"{value['p90_seconds']:.3f}秒 | {value['total_seconds']:.3f}秒 |")
    saving = accepted['planning_total_time_reduction']
    lines += ['', f"原抓点、选臂、三个关节终点及候选拒绝分类全部一致，{checked['checked_endpoints']}个终点独立复算通过。"
              f"本轮累计耗时{'减少' if saving >= 0 else '增加'}{abs(saving):.2%}。", '',
              '| 环节 | 原版每请求平均 | 批量版每请求平均 |', '| --- | ---: | ---: |']
    stages = [('resident_world_bind', '场景与机器人绑定'), ('contact_world_bind', '局部碰撞世界绑定'),
        ('batch_world_bind', '批量IK世界绑定'), ('grasp_batch_full_world_IK', '成批完整场景抓取IK'),
        ('grasp_batch_kinematic_IK', '成批纯运动学回退IK'), ('independent_FK', '独立FK'),
        ('local_collision_validate', '局部碰撞复核'), ('pregrasp_scalar_IK', '预抓取IK'),
        ('lift_scalar_IK', '抬升IK'), ('PRM_path', 'PRM路径搜索'),
        ('trajectory_approach', '接近轨迹优化与验证'), ('trajectory_pregrasp', '到预抓取点轨迹优化与验证')]
    for key, label in stages:
        times = [summaries[mode]['stage_totals'].get(key, {}).get('wall_seconds', 0.)/summaries[mode]['requests']
                 for mode in ['control', 'vectorized']]
        lines.append(f'| {label} | {times[0]:.4f}秒 | {times[1]:.4f}秒 |')
    lines += ['', '该表各环节采用包含子步骤的实际调用时间；独立FK包含在局部几何复核中，不能把整列直接相加。'
        '机器可读数据同时保存扣除子步骤后的时间和全部调用事件。', '',
        (('完整69场景结果已核验，下一步比较主要耗时及稳定性，再决定后续组合。' if full else
          '先导达到累计省时2%的门槛，可以登记完整69场景比较。') if accepted['timing_gate_passed'] else
         ('完整对照没有保持累计省时2%，保留原组合，按分项计时调整下一项优化。' if full else
          '先导未达到累计省时2%的门槛，不扩展完整69场景。下一步根据本轮实测的主要开销选择优化位置。')), '',
        f"独立核验文件：`{run.relative_to(ROOT)}/independent_review.json`。", '']
    if full:
        body = body.replace('正在登记并准备运行。', '138次请求及独立核验已完成。')
    else:
        start, end = body.index('## 当前进度'), body.index('## 后续顺序')
        body = body[:start]+'## 当前进度\n\n20次请求及独立核验已完成，原抓点、选臂、关节终点及拒绝分类一致。实测结果见后文。\n\n'+body[end:]
    body = body.replace('先导运行中 |', '先导与独立核验完成 |')
    doc.write_text(body.rstrip()+'\n\n'+'\n'.join(lines))
    shutil.copy2(doc, run/'report.md')


def publish(root):
    from website.tools.publish_indexed_planning_plan import publish as refresh
    from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.publication import sync_public
    refresh()
    node = ROOT/'environments/node/bin/node'
    if not node.exists():
        node = max((Path.home()/'.nvm/versions/node').glob('v*/bin/node'),
                   key=lambda p: tuple(map(int, p.parents[1].name[1:].split('.'))))
    with (root/'website_build.log').open('a') as stream:
        subprocess.run([str(node), 'build.mjs'], cwd=ROOT/'website', stdout=stream, stderr=subprocess.STDOUT, check=True)
    sync_public(root, Path.home()/'.ssh/arx_public_repeat_continue')
    urls = {'/iterations/sam3-graspgenx/versions/V8/': ROOT/'website/resources/site/iterations/sam3-graspgenx/versions/V8/index.html',
            '/media/sam3-v8-branch-validation-pilot.json': ROOT/'website/resources/media/sam3-v8-branch-validation-pilot.json',
            '/media/sam3-v8-branch-validation-pilot.md': ROOT/'website/resources/media/sam3-v8-branch-validation-pilot.md'}
    for path, local in urls.items():
        with urllib.request.urlopen('http://117.72.104.29'+path, timeout=30) as response:
            if response.status != 200 or response.read() != local.read_bytes():
                raise RuntimeError('Public V8 result differs: '+path)
    write_json(root/'website_publication.json', {'at': now(), 'passed': True,
        'checked_paths': list(urls), 'existing_public_todo_database_preserved': True})


def run(root, source):
    if root.exists(): raise ValueError('Keep the original follow-up registration')
    root.mkdir(parents=True)
    shutil.copy2(__file__, root/'source_registered.py')
    write_json(root/'registration.json', {'at': now(), 'run': str(source.relative_to(ROOT)),
        'source_sha256': sha256(__file__), 'planning_requests': 0, 'physical_grasps': 0})
    status = root/'status.json'
    expected_requests = json.loads((source/'registration.json').read_text())['planning_requests']
    def latest_progress():
        # Only progress fields are consumed here; archived path relocation is
        # unnecessary for these increasingly large, already-canonical receipts.
        return json.loads((source/'progress.json').read_text())
    while True:
        progress = latest_progress()
        write_json(status, {'at': now(), 'phase': 'following', 'run': str(source.relative_to(ROOT)),
            **{k: progress.get(k) for k in ['completed_requests', 'total_requests', 'current']}})
        if (source/'STOP').exists(): raise RuntimeError('Validation pilot stopped; do not retry this registration')
        if (source/'controller.json').exists():
            state = subprocess.check_output(['systemctl', '--user', 'show', controller_unit(source),
                '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ExecMainStatus'], text=True)
            fields = dict(line.split('=', 1) for line in state.strip().splitlines())
            if completed_controller(fields, latest_progress, expected_requests) is not None:
                break
        time.sleep(5)
    checked = validate(source);report(source, checked)
    active = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';pointer = read_json(active)
    full = read_json(source/'registration.json').get('phase') == 'full'
    pointer.update(at=now(), phase='validation_full_complete' if full else 'validation_pilot_complete', validation_run=str(source.relative_to(ROOT)),
        next_stage='review_measured_stages_before_full69_or_q_pre_batching')
    write_json(active, pointer)
    publish(root)
    write_json(status, {'at': now(), 'phase': 'complete', 'independent_review_passed': checked['passed'],
        'planning_requests': checked['checked_requests'], 'timing_gate_passed': checked['timing_gate_passed'],
        'automatic_full_comparison': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args();root = ensure_run_path(args.root)
    try: run(root, ensure_run_path(args.source))
    except BaseException:
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
