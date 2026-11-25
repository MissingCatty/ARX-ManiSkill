"""Publish the complete paired result, preserve todos, then push project changes."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import urllib.request

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT, now, sha256
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.contracts import METHODS, completed_rows
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.metrics import FAILURE_LABELS, failure_category
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.publication import publish, sync_public
from .latency_report import build_latency_report, render_latency, refresh_latency_section


def dirty_paths():
    value = subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard','-z'], cwd=ROOT)
    return sorted(set(p.decode() for p in value.split(b'\0') if p))


def publish_final(root, configuration):
    root = Path(root)
    accepted = read_json(root/'acceptance.json')
    if not accepted['passed'] or not accepted['record_integrity'] or not accepted['task_complete']:
        raise RuntimeError('Final 148-result acceptance did not pass')
    summary = publish(root)
    if any(summary['methods'][m]['completed'] != 74 for m in METHODS):
        raise RuntimeError('Both groups must have 74 completed results')
    rows = {m:{row['id']:row for row in completed_rows(root/'methods'/m)} for m in METHODS}
    paired = build_latency_report(root)
    atomic_json(root/'paired_latency_summary.json',paired)
    site_file = ROOT/'website/内容/网站.json'
    site = read_json(site_file)

    def attach(source, name):
        target = ROOT/'website/resources/media'/name
        if target.is_symlink():
            raise RuntimeError('Website attachment must be a real independent file')
        pending = target.with_name(name+f'.{os.getpid()}.pending')
        shutil.copy2(source,pending)
        pending.replace(target)
        if target.stat().st_nlink != 1 or sha256(source) != sha256(target):
            raise RuntimeError('Website attachment copy failed')
        site['localAssets'][name] = '../../resources/media/'+name
        return '/media/'+name

    report = ['# V6共享内存对照最终结果','',f'完成时间：{now()}。两组均完整结束74例，合计148条结果，独立验收通过。','',
        '| 链路 | 完整结束 | 规划通过 | 抓取成功 |','| --- | ---: | ---: | ---: |']
    labels = {'graspgenx':'RaySt3R＋GraspGenX','zerograsp':'ZeroGrasp'}
    for method in METHODS:
        v = summary['methods'][method]
        report.append(f"| {labels[method]} | 74/74 | {v['planning_passes']}/74（{v['planning_passes']/74:.1%}） | {v['successes']}/74（{v['successes']/74:.1%}） |")
    report += ['', '25例SAM3最终失败单列剔除，5例共同站位失败保留在74例分母内。原V4中断1例继续单列。','',
        '## 两例中断补跑','', '牛油果与甜甜圈按本轮明确授权各补跑一次，原中断尝试与停止记录保留；最终74例结果采用本次完整记录。','',
        '| 场景 | 原中断原因 | 本次结果 |','| --- | --- | --- |']
    for case in configuration['authorized_makeup_cases']:
        result = rows['zerograsp'][case]['result']
        reason = '抓取成功' if result['success'] else FAILURE_LABELS.get(failure_category(result),result['reason'])
        before = '规划服务显存不足' if 'avocado' in case else '另一分组资源门控触发停止'
        report.append(f'| {case} | {before} | {reason} |')
    report += ['', *render_latency(paired).splitlines()]
    report_path = ROOT/'modules/pipeline/docs/mainline_versions/V6_共享内存对照最终结果.md'
    report_path.write_text('\n'.join(report).rstrip()+'\n')
    report_url = attach(report_path,'sam3-v6-complete74-final-report.md')
    latency_url = attach(root/'paired_latency_summary.json','sam3-v6-complete74-paired-latency.json')
    acceptance_url = attach(root/'acceptance.json','sam3-v6-complete74-acceptance.json')
    makeup_url = attach(root/'inheritance_verified.json','sam3-v6-complete74-makeup-receipt.json')
    appendix = ['','## 最终完整结果与授权补跑','',*report[2:report.index('## 延时对照')],
        '',f'- [最终结果报告与配对延时]({report_url})',f'- [逐项配对延时数据]({latency_url})',
        f'- [148条结果及录像独立验收]({acceptance_url})',f'- [两例补跑承接记录]({makeup_url})','']
    for case in configuration['authorized_makeup_cases']:
        appendix += ['### '+case,'','| RaySt3R＋GraspGenX | ZeroGrasp补跑 |','| --- | --- |']
        cells = []
        for m in METHODS:
            source = root/'methods'/m/'trials'/case/'execution.mp4'
            if source.is_file():
                url = attach(source,'sam3-v6-complete74-'+case.split('_')[0].lower()+'-'+m+'.mp4')
                cells.append(f'<video controls preload="none" src="{url}"></video>')
            else:
                result = rows[m][case]['result']
                cells.append(FAILURE_LABELS.get(failure_category(result),result['reason']))
        appendix += ['| '+' | '.join(cells)+' |','']
    page = ROOT/'website/内容/迭代/sam3-graspgenx-v6.md'
    body = refresh_latency_section(page.read_text(),paired)
    body = body.replace('## 首次启动与预热','## 两例补跑的启动与预热')
    body = body.replace('### 实际资源分配','### 两例补跑的实际资源分配')
    old = '保留前批已经完成的146条结果与0条中断。本批只执行ZeroGrasp剩余2例与GraspGenX剩余0例。原场景序号、随机种子和观测保持。'
    body = body.replace(old,'继承前批146条完整结果，另行保留两条历史中断。按本轮授权补跑ZeroGrasp牛油果与甜甜圈各一次，场景序号、随机种子和观测保持。')
    page.write_text((body+'\n'.join(appendix)).rstrip()+'\n')
    atomic_json(site_file,site)
    node = sorted((Path.home()/'.nvm/versions/node').glob('v*/bin/node'))[-1]
    subprocess.run([str(node),'build.mjs'],cwd=ROOT/'website',check=True)
    socket = Path.home()/configuration['public_control_socket_relative_to_home']
    sync_public(root,socket)
    with urllib.request.urlopen('http://117.72.104.29/iterations/sam3-graspgenx/versions/V8/',timeout=30) as response:
        if response.status != 200:
            raise RuntimeError('V8 entry was lost during final publication')
    atomic_json(root/'final_publication.json',{'at':now(),'passed':True,'completed_per_method':74,
        'authorized_makeup_cases':configuration['authorized_makeup_cases'],'latest_version_preserved':'V8'})
    return summary


def push_git(root, configuration, queue_root):
    root, queue_root = Path(root), Path(queue_root)
    node = sorted((Path.home()/'.nvm/versions/node').glob('v*/bin/node'))[-1]
    subprocess.run([str(node),'tools/todo_snapshot.mjs','export'],cwd=ROOT/'website',check=True)
    base = set(read_json(queue_root/'git_paths.json')['paths'])
    extra = {'modules/pipeline/docs/mainline_versions/V6_共享内存对照最终结果.md',
             'website/resources/state/todos.snapshot.json'}
    for path in dirty_paths():
        if (path.startswith('website/resources/media/sam3-v6-')
                or path.startswith('website/resources/optimized/video/sam3-v6-')):
            extra.add(path)
    paths = sorted((base|extra)&set(dirty_paths()))
    if not paths:
        raise RuntimeError('No registered change available for the requested Git push')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != configuration['git_branch']:
        raise RuntimeError('Current Git branch changed during the experiment')
    for path in paths:
        file = ROOT/path
        if file.is_file() and file.stat().st_size >= 100*1024**2:
            raise RuntimeError('Git upload contains an oversized file: '+path)
    staged = subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT)
    if any(p.decode() not in paths for p in staged.split(b'\0') if p):
        raise RuntimeError('Unrelated staged work must remain separate')
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',configuration['git_commit_message']],cwd=ROOT,check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    atomic_json(root/'git_commit.json',{'at':now(),'commit':head,'paths':paths,'push_complete':False})
    subprocess.run(['git','push',configuration['git_remote'],configuration['git_branch']],cwd=ROOT,check=True)
    remote = subprocess.check_output(['git','ls-remote',configuration['git_remote'],
        'refs/heads/'+configuration['git_branch']],cwd=ROOT,text=True).split()[0]
    if remote != head:
        raise RuntimeError('Pushed remote revision differs')
    atomic_json(root/'git_push.json',{'at':now(),'commit':head,'remote':configuration['git_remote'],
        'branch':configuration['git_branch'],'verified':True,'paths':paths})
    return head
