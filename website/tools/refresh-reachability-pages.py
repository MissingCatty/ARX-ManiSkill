"""Publish the recorded ZeroGrasp/cuRobo results for the existing V2 and V3 pages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / '内容/资源/reachability-native'
RESOURCE.mkdir(parents=True, exist_ok=True)
PROFILES = [
    ('V2','lift452','45厘米高度实验/实验汇总.json','new','固定45.2 cm高度抓取','fixed-height'),
    ('V3','joint7','七关节对照/实验汇总.json','groups','升降与右臂联合规划','lift-arm'),
]
LAYOUTS = {'S1_open':'开阔桌面','S2_edge':'靠近桌边','S3_sparse':'稀疏混放','S4_clutter':'中度杂乱'}
REASONS = {
    'no_feasible_native_post_or_recalled_candidate':'未得到可执行抓取方案',
    'insufficient_hold_height':'抬起后未保持足够高度',
    'approach_not_converged':'接近抓取点时未到位',
    'close_not_converged':'夹爪闭合动作未达到要求',
    'no_bilateral_target_contact':'两侧手指未同时接触物体',
    'forbidden_robot_contact':'机器人接触到了禁止碰撞的部位',
    'measured_start_state_in_collision':'开始运动前检测到碰撞',
    'not_executed_preexecution_record_binding_failure':'执行前记录核对未通过，未执行',
}

def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def rate(n, total=160):
    return f'{n}/{total}（{n/total*100:.1f}%）'

def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

site_path=ROOT/'内容/网站.json';site=json.loads(site_path.read_text())
catalog_path=ROOT/'内容/experiments.json';catalog=json.loads(catalog_path.read_text())
experiment=next(e for e in catalog['experiments'] if e['id']=='reachability-improvement')
old_codes=[n['code'] for n in experiment['iterations']]
source_entries={e['id']:e for e in site['versions']}
records={}
for code,key,relative,group,title,slug in PROFILES:
    path=ROOT/'内容/资源'/relative
    data=json.loads(path.read_text());assert data['audit']['passed']
    arm=data[group]['fixed_zerograsp']
    n=arm['completed'];missing=arm.get('not_executed',0)
    assert n==160 and arm['successes']+sum(arm['failure_reasons'].values())==n
    assert sum(v['n'] for v in arm['by_layout'].values())==n
    assert sum(v['success'] for v in arm['by_layout'].values())==arm['successes']
    assert sum(v['planning'] for v in arm['by_layout'].values())==arm['planning_passes']
    counts={k:arm[k] for k in ['completed','planning_passes','physical_attempts','successes','failure_reasons','by_layout']}
    counts['not_executed']=missing;counts['complete_result_records']=n-missing
    output=str(Path(data['source'])/'fixed_zerograsp');assert Path(output).is_dir()
    record={'version':code,'pipeline':'ZeroGrasp＋cuRobo','group':'fixed_zerograsp','registered_scenes':n,
            'results':counts,'output_directory':output,'source_audit_passed':True,
            'source_summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_summary':str(path.relative_to(ROOT)),
            'subset_checks':{'failure_totals_match':True,'layout_totals_match':True}}
    save(RESOURCE/(slug+'-summary.json'),record)
    records[code]={'record':record,'arm':arm,'code':code,'key':key,'title':title,'slug':slug,'output':output}

common_settings=[['测试场景','开阔桌面、靠近桌边、稀疏混放、中度杂乱各40个，共160个场景。'],
                 ['观测与候选','每个场景重新拍摄腕部彩色和深度图，使用ZeroGrasp原生后处理候选。'],
                 ['接近与抬升','预抓取点距抓取点8 cm，计划抬升3 cm；夹爪动作参考点沿末端前向偏移43 mm，记为TCP43。'],
                 ['成功条件','物体实际抬高至少1 cm并保持3秒，两侧手指接触物体，动作到位且接触检查通过。']]
for code,item in records.items():
    arm=item['arm'];rec=item['record'];count=rec['results'];key=item['key'];node=next(n for n in experiment['iterations'] if n['code']==code)
    if code=='V2':
        setup=[['升降高度','升降关节读数固定为45.2 cm，桌面高50 cm；这个读数表示升降位置。'],['参与规划的关节','右臂六个关节，升降和底盘在抓取过程中固定。']]
        description='将升降关节的初始读数设为45.2 cm，抓取过程中保持这个高度，由右臂六个关节完成接近、闭爪和抬升。'
        sequence='腕部彩色和深度图 → ZeroGrasp候选与原生后处理 → cuRobo右臂规划 → 接近、闭爪、抬升并保持物体。'
        analysis='153个场景得到完整规划，其中111个抓取成功，42个在执行或持物保持环节失败。另有7个场景没有得到可执行方案。\n\n抬升保持高度不足有21例，是本轮最多的失败原因，主要损失发生在规划通过之后。'
    elif code=='V3':
        setup=[['起始高度','升降关节从45.2 cm开始，底盘固定。'],['参与规划的关节','升降关节与右臂六个关节共同规划到预抓取点，共七个自由度。'],['靠近物体后','固定规划选定的高度，由右臂完成最后接近、闭爪和短距离抬升。']]
        description='让升降关节和右臂一起规划到预抓取点。到达后固定升降高度，再由右臂靠近物体、闭爪并抬起。'
        sequence='腕部彩色和深度图 → ZeroGrasp候选与原生后处理 → 升降与右臂联合规划 → 固定高度完成抓取。'
        analysis='152个场景进入抓取执行，116个成功，36个在执行或持物保持环节失败，其中18个未达到抬升保持要求。另有7个场景没有得到可执行方案。\n\n有1个场景在执行前的记录核对中被拦下，没有发出机器人动作，结果表中单列为未执行。'
    settings=setup+common_settings
    result_summary=f"规划通过{rate(arm['planning_passes'])}；抓取成功{rate(arm['successes'])}"
    node.update(title=item['title'],status='160例已核查' if not count['not_executed'] else '159例已核查，1例未执行',
                resultSummary=result_summary,change=description,pipeline=sequence,
                scenes=common_settings[0][1],settings=settings,contentFile=f"reachability-{code.lower()}-native.md",reportLayout=True,
                resultScope='zerograsp-curobo-only')
    source_entries[key].update(title=item['title'],shortTitle=item['title'],status=node['status'],summary=description,metric=f"抓取成功{rate(arm['successes'])}")
    batch=node['batches'][0];batch.update(title=f"{item['title']}：160场景记录",status=node['status'],summary=result_summary,
            outputs=[str(Path(item['output']).relative_to(ROOT.parent.parent))],
            artifacts=[{'label':'本管线统计与核查来源','url':f"/media/reachability-{item['slug']}-native.json"}])
    site['localAssets'][f"reachability-{item['slug']}-native.json"]='reachability-native/'+item['slug']+'-summary.json'
    lines=['## 方案改动','',description,'','## 抓取过程','',sequence,'','## 实验设置','',table(['设置','说明'],settings),'','## 抓取结果','',
        table(['统计项','场景数'],[['登记场景','160'],['有完整结果的场景',count['complete_result_records']],['未执行',count['not_executed']],['规划通过',rate(arm['planning_passes'])],['进入抓取执行',rate(arm['physical_attempts'])],['抓取成功',rate(arm['successes'])]]),
        '', '规划通过表示找到可执行的抓取、接近与抬升方案；抓取成功还要求实际夹住物体并保持高度。成功率按160个登记场景计算。','', '## 各类场景表现','',
        table(['场景布局','场景数','规划通过','抓取成功'],[[LAYOUTS[k],v['n'],rate(v['planning'],v['n']),rate(v['success'],v['n'])] for k,v in arm['by_layout'].items()]),
        '', '## 失败与未执行原因','',table(['原因','场景数'],[[REASONS[k],v] for k,v in sorted(arm['failure_reasons'].items(),key=lambda x:-x[1])]),
        '', '每个场景按最终停止原因计一次。','', '## 结果分析','',analysis]
    if code=='V3':
        compare=['V2','V3']
        lines+=['','### 同场景的已有结果','',table(['方案','规划通过','抓取成功'],[[records[c]['title'],rate(records[c]['arm']['planning_passes']),rate(records[c]['arm']['successes'])] for c in compare]),
                '', '这些方案分别在同一批160个场景中重新观察、规划并执行。升降联合规划中有1个场景未执行，各项成功率仍按160个登记场景计算。']
    lines+=['','## 记录与产物','',f"[下载本管线统计与核查来源](/media/reachability-{item['slug']}-native.json)",'',f"录像、候选和逐场景结果位于：`{batch['outputs'][0]}`。",'','## 参考方法','',
            '[ZeroGrasp：生成抓取候选](/papers/zerograsp/) · [cuRobo：检查可达性、碰撞并规划轨迹](/papers/curobo/)']
    (ROOT/'内容/迭代'/node['contentFile']).write_text('\n'.join(lines)+'\n')
assert [n['code'] for n in experiment['iterations']]==old_codes
save(site_path,site);save(catalog_path,catalog)
print('Updated existing V2 and V3 pages and native-only results; version codes unchanged.')
