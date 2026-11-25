"""Replace current V6 comparison, preserve prior results and copy all media."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from arx_project.layout import ROOT,now,sha256
from arx_project.json_io import read_json
from arx_project.artifacts import atomic_json
from .contracts import METHODS
from .metrics import summarize,FAILURE_LABELS

PAGE=ROOT/'website/内容/迭代/sam3-graspgenx-v6.md'
TITLES={'graspgenx':'RaySt3R＋GraspGenX','zerograsp':'ZeroGrasp'}


def percent(count,total):return f'{count}/{total}（{count/total:.1%}）' if total else '待执行'


def seconds(value):
    if not value or not value['samples']:return '—'
    return f"{value['median_seconds']:.4f} / {value['p90_seconds']:.4f} 秒（{value['samples']}例）"


def publish(root):
    summary=summarize(root);reg=read_json(root/'registration.json')
    prefix='sam3-v6-'+root.name.replace('_','-')
    site=read_json(ROOT/'website/内容/网站.json')
    def attach(source,name):
        target=ROOT/'website/resources/media'/name
        if target.is_symlink():raise RuntimeError('Published artifact must be an independent file')
        if not target.exists() or sha256(source)!=sha256(target):
            pending=target.with_name(name+f'.{os.getpid()}.pending')
            shutil.copy2(source,pending);pending.replace(target)
        if target.stat().st_nlink!=1:raise RuntimeError('Published media cannot share an inode with source data')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    old=attach(root/'previous_website_page.md',prefix+'-previous-page.md')
    data=attach(root/'summary.json',prefix+'-summary.json')
    phase=summary['phase']
    status={'complete':'已完成','stopped':'故障停止，保留记录','running':'按可用资源测试中',
        'warming':'已有分组正在预热','registered':'已登记','waiting_for_resources':'等待可用显卡',
        'waiting_for_parallel_resources':'等待可用显卡'}[phase]
    body=['## 本版进度','',f"{status}。已完成 {summary['completed_pairs']}/74 个配对场景。",'',
        '本次两组都重新运行模型与抓取，使用相同的74个场景、SAM3掩码、RGB-D、最后站位及物体状态。每组资源就绪后独立预热并开始，先就绪先测。',
        '', '分组进度：'+ '；'.join(f"{TITLES[m]}完成{summary['methods'][m]['completed']}/74" for m in METHODS)+'。','',
        '## 本次比较的链路','','| 方案 | 生成与候选处理 | 候选交接与规划 |','| --- | --- | --- |',
        '| RaySt3R＋GraspGenX | RaySt3R重建、GraspGenX原候选与原评分；不加后处理 | 共享内存、后台保存、一次解码 → 双臂cuRobo |',
        '| ZeroGrasp | 原生重建与抓姿、碰撞筛选、宽度深度修正、NMS | 共享内存、后台保存、一次解码 → 同一双臂cuRobo |','',
        'ZeroGrasp允许在一张满足门槛的GPU上完成生成、规划与仿真；RaySt3R独立占用一张GPU，GraspGenX与规划执行使用另一张GPU。模型在每次承接运行中加载和预热一次。实际设备分配单独记录，871球、TCP43、V2指尖2毫米余量与物理成功判据沿用上一轮。','',
        '[[流程卡片]]','','## 两组进度与结果','','| 方案 | 完整结束 | 进行中 | 中断 | 未开始 | 规划成功率 | 抓取成功率 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        value=summary['methods'][method];n=value['completed']
        body.append(f"| {TITLES[method]} | {n}/74 | {value['in_progress']} | {value['interrupted']} | {value['not_started']} | {percent(value['planning_passes'],n)} | {percent(value['successes'],n)} |")
    body+=['','25例SAM3最终失败剔除，原1例中断单列。74例中包含5例共同站位失败，两组都保留在分母中。','',
        '## 延时对照','','每项显示中位数 / P90及实际计时例数。未调用的模块不填零；执行耗时只统计实际动作。共同站位失败的5例不进入模型与规划耗时统计。','',
        '| 环节 | RaySt3R＋GraspGenX | ZeroGrasp |','| --- | ---: | ---: |']
    if reg.get('continuation'):
        location=body.index('## 延时对照')
        slots=reg['continuation']['method_slots']
        inherited_complete=sum(len(v['complete']) for v in slots.values())
        inherited_interrupted=sum(len(v['interrupted']) for v in slots.values())
        body[location:location]=['## 本次承接','',
            f"保留前批已经完成的{inherited_complete}条结果与{inherited_interrupted}条中断。本批只执行ZeroGrasp剩余{len(slots['zerograsp']['not_started'])}例与GraspGenX剩余{len(slots['graspgenx']['not_started'])}例。原场景序号、随机种子和观测保持。",
            '规划服务现在会释放已经不属于当前碰撞场景的历史网格缓存，当前障碍、抓取位姿和规划参数保持。每个新场景领取前检查显存余量。','']
    stages=[('generation_delivery_seconds','生成、转换并交出共享候选'),('candidate_decode_seconds','候选一次解码'),
        ('initial_planning_rpc_seconds','首次cuRobo规划请求'),('generation_and_initial_planning_seconds','生成与首次规划合计'),
        ('execution_with_replanning_seconds','执行抓取，含接近与抬升重规划'),
        ('generation_to_trial_return_seconds','生成开始至本次试验返回'),('trial_wall_seconds','整次试验，含场景建立与状态恢复')]
    for key,title in stages:
        body.append('| '+title+' | '+' | '.join(seconds(summary['methods'][m]['latency'].get(key)) for m in METHODS)+' |')
    body+=['','生成阶段包含模型推理、其原生候选处理以及交出候选的时间。首次规划请求包含共享候选读取、双臂搜索与后台记录完成确认。执行阶段包含开爪、接近、闭合、抬升、保持及过程中重规划。','',
        '## 失败分布','','按场景最终失败阶段计数，括号内为占该组完整结束场景的比例。','',
        '| 失败阶段 | RaySt3R＋GraspGenX | ZeroGrasp |','| --- | ---: | ---: |']
    any_failure=False
    for category,label in FAILURE_LABELS.items():
        if not any(summary['methods'][m]['failures'].get(category,0) for m in METHODS):continue
        any_failure=True
        body.append('| '+label+' | '+' | '.join(percent(summary['methods'][m]['failures'].get(category,0),summary['methods'][m]['completed']) for m in METHODS)+' |')
    if not any_failure:body.append('| 待产生完整结果 | — | — |')
    body+=['','## 首次启动与预热','','| 分组 | 模块 | 加载与预热耗时 | GPU |','| --- | --- | ---: | --- |']
    for method in METHODS:
        startup=root/'methods'/method/'preflight/startup.json'
        if startup.exists():
            for row in read_json(startup)['components']:
                body.append(f"| {TITLES[method]} | {row['kind']} | {row['seconds']:.2f} 秒 | `{row['gpu']}` |")
    ready=root/'preflight/resources.json'
    if ready.exists():
        body+=['','### 实际资源分配','']
        for method,evidence in read_json(ready).get('per_method',{}).items():
            body += [TITLES[method]+'：'+ '；'.join(f'{k}：`{v}`' for k,v in evidence['gpu_roles'].items())+'。','']
    paired=summary['paired_results']
    samples=[r for r in paired if all((root/'methods'/m/'trials'/r['id']/'execution.mp4').exists() for m in METHODS)][:3]
    if samples:body+=['','## 本轮配对视频','']
    for row in samples:
        body += ["### "+row['id'],'','| RaySt3R＋GraspGenX | ZeroGrasp |','| --- | --- |']
        urls=[attach(root/'methods'/m/'trials'/row['id']/'execution.mp4',prefix+'-'+row['id'].split('_')[0].lower()+'-'+m+'.mp4') for m in METHODS]
        body.append('| '+' | '.join(f'<video controls preload="none" src="{url}"></video>' for url in urls)+' |')
    body+=['','## 数据与历史记录','',f'- [本次逐例统计、延时与失败分布]({data})',f'- [上一轮文件交接版本的完整页面]({old})',
        '- [ZeroGrasp方法](/papers/zerograsp/)','- [GraspGenX方法](/papers/graspgenx/)']
    PAGE.write_text('\n'.join(body)+'\n')
    registry=read_json(ROOT/'website/内容/experiments.json')
    group=next(v for v in registry['experiments'] if v['id']=='sam3-graspgenx')
    version=next(v for v in group['iterations'] if v['code']=='V6')
    version.update(status=status,resultSummary='共享内存延时复测：'+'；'.join(f"{TITLES[m]} {summary['methods'][m]['completed']}/74" for m in METHODS)+'，含失败分布。')
    batch={'id':root.name,'title':'共享内存与候选读取优化后的74例延时复测','date':'2026-11-25','status':status,
        'summary':version['resultSummary'],'outputs':[str(root.relative_to(ROOT))],
        'artifacts':[{'label':'本轮对照结果','url':data},{'label':'上一轮完整页面','url':old}]}
    version['batches']=[r for r in version['batches'] if r['id']!=root.name]+[batch]
    atomic_json(ROOT/'website/内容/experiments.json',registry);atomic_json(ROOT/'website/内容/网站.json',site)
    node=max((Path.home()/'.nvm/versions/node').glob('v*/bin/node'),key=lambda p:tuple(map(int,p.parents[1].name[1:].split('.'))))
    with (root/'website_build.log').open('a') as stream:
        subprocess.run([str(node),'build.mjs'],cwd=ROOT/'website',stdout=stream,stderr=subprocess.STDOUT,check=True)
    atomic_json(root/'website_publication.json',{'at':now(),'published':True,'phase':phase,'completed_pairs':summary['completed_pairs']})
    return summary


def sync_public(root,control_socket):
    import urllib.request
    config=read_json(ROOT/'configs/deployments/public_website.json')
    ssh=['ssh','-S',str(control_socket),'-o','BatchMode=yes','-o','ConnectTimeout=15']
    host=config['ssh_user']+'@'+config['host']
    subprocess.run([*ssh,host,'true'],check=True,timeout=30)
    import shlex
    command=['rsync','-az','--no-owner','--no-group','--chmod=D755,F644','--delay-updates',
        '--exclude=node_modules/','--exclude=resources/state/','--exclude=resources/.build-*/',
        '--exclude=resources/.previous-*/','-e',shlex.join(ssh),str(ROOT/'website')+'/',host+':'+config['website_root']+'/']
    with (root/'public_sync.log').open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
    url=config['public_http_url'].rstrip('/')+'/iterations/sam3-graspgenx/versions/V6/'
    with urllib.request.urlopen(url,timeout=30) as response:
        if response.status!=200 or response.read()!=(ROOT/'website/resources/site/iterations/sam3-graspgenx/versions/V6/index.html').read_bytes():
            raise RuntimeError('Public V6 page differs from the published snapshot')
    atomic_json(root/'public_publication.json',{'at':now(),'published':True,'url':url,'public_todo_database_preserved':True})
