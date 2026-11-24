"""只读跟进七轴批次；完整退出后核对结果并发布独立版本，不重启抓取。"""
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

SITE=Path(__file__).resolve().parents[1]
MS=SITE.parents[1]/'ManiSkill'
RUN=MS/'outputs/arx_joint7_pair_20261121_03'
PREVIOUS=MS/'outputs/arx_graspcl_compat160_20261120_01'
ARMS=['fixed_zerograsp','fixed_graspcl']
DEST=SITE/'内容/资源/七关节对照'
spec=importlib.util.spec_from_file_location('height_readback',SITE/'工具/汇总45厘米高度实验.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def state(phase,**extra):
    h.save(SITE/'数据/七关节实验发布状态.json',{'phase':phase,'at':datetime.now().astimezone().isoformat(),**extra});print(phase,extra,flush=True)

def finish():
    end=h.read(RUN/'运行结束.json');assert not end['failed'] and end['all_owned_services_exited']
    assert all(s['returncode']==0 for s in end['services'].values())
    data={};sources={};counts=Counter()
    for g in ARMS:
        folder=RUN/g;reg=h.read(folder/'registration.json');rows=h.results(RUN,g)
        assert len(rows)==160 and set(rows)=={x['id'] for x in reg['schedule']}
        assert len(list((folder/'claims').glob('*.json')))==160
        for p,d in {**reg['source_hashes'],**reg['asset_sha256']}.items():
            if p in sources:assert sources[p]==d
            sources[p]=d
        for id,r in rows.items():
            trial=folder/'trials'/id;claim=h.read(folder/'claims'/(id+'.json'));evaluation=h.read(trial/'independent_evaluation.json')
            assert claim['state']=='complete' and claim['result_sha256']==h.sha(trial/'result.json')
            assert evaluation['passed'] and evaluation['independently_successful']==r['success']
            if r.get('not_executed'):
                assert claim['inherited_preexecution_failure'] and claim['original_claim_sha256']==h.sha(claim['original_claim'])
                assert not r['physical_attempt'] and not r['planning_success'];counts['not_executed']+=1;continue
            assert h.read(trial/'source_guard.json')['sha256']==h.sha(folder/'registration.json')
            assert h.read(trial/'示教兼容核验.json')['passed']
            for name in ['execution.jsonl','execution_origin.json','execution.mp4','right_wrist.mp4','camera/color.png','camera/depth.png','camera/mask.png']:
                assert (trial/name).is_file() and (trial/name).stat().st_size>0
            z0=h.read(trial/'execution_origin.json')['object_initial_z_m'];hold=0;minimum=float('inf');bad=0;fingers=set()
            for line in (trial/'execution.jsonl').open():
                frame=json.loads(line)
                if frame['stage']=='hold':hold+=1;minimum=min(minimum,frame['object_pose'][2]-z0)
                for c in frame['contacts']:
                    bad+=not c['allowed']
                    if frame['stage']=='close' and c['allowed']:fingers.update(f for f in ['right_link27','right_link28'] if f in c['a']+c['b'])
            stages=r.get('stages',{})
            success=bool(minimum>=.01 and hold/120>=3 and bad==0 and len(fingers)==2
                and {'pregrasp','approach','close','lift','hold'}.issubset(stages) and all(s['settled'] for s in stages.values()))
            assert success==r['success'];counts['trace_recomputed']+=1
            if r['planning_success']:
                proof=h.read(trial/'七轴规划核验.json');assert proof['passed'] and proof['body_sweep']['passed']
                if r['physical_attempt']:assert h.read(trial/'七轴执行核验.json')['passed']
                assert .0<=proof['selected_lift_m']<=.4600001;counts['joint7_plans']+=1
            if g=='fixed_graspcl':
                for name in ['frozen_memory_inference.json','frozen_memory_rescore.json']:
                    memory=h.read(trial/name);assert memory['before_sha256']==memory['after_sha256'] and memory['test_feedback_added']==0
        data[g]=h.metrics(rows);data[g]['not_executed']=sum(x.get('not_executed',False) for x in rows.values())
        old=h.results(PREVIOUS,g);data[g]['history_fixed6']=h.metrics(old);data[g]['history_pair']=h.pair(old,rows)
    for p,digest in sources.items():assert h.sha(p)==digest,p
    assert counts['not_executed']==2 and counts['trace_recomputed']==318
    audit={'passed':True,'time':datetime.now().astimezone().isoformat(),**dict(counts),
        'source_hashes_checked':len(sources),'physical_retries':0,'all_services_exited':True}
    h.save(DEST/'最终核验.json',audit);h.save(DEST/'实验汇总.json',{'groups':data,'audit':audit,'source':str(RUN)})
    ratio=lambda n,d:f'{n}/{d}（{100*n/d:.1f}%）'
    body=['## 实验设置','','右臂六关节＋原升降joint4联合规划至固定8cm预抓取点，随后固定所选升降高度，执行接近、闭爪和名义3cm短抬升。初态仍为45.2cm，底盘不动，腕部重新采集RGB-D。两法各160个原S1–S4场景，复用上一轮冻结的24正例＋4负例GraspCL库。',
        '', '原右臂碰撞模型、场景物理、质量、驱动和成功判据保留；补齐原躯干／左臂凸碰撞部件的完整竖直扫掠检查。新增升降1mm收敛门槛。实际成功仍须物体抬高至少1cm并保持3秒，通过原双指接触、手臂收敛及禁止接触判定。',
        '', '## 本轮结果','', '| 链路 | 完成／登记 | 未执行 | 初始规划通过 | 进入抓取执行 | 实际抓取成功 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g,title in zip(ARMS,['ZeroGrasp＋cuRobo','ZeroGrasp＋新GraspCL＋cuRobo']):
        d=data[g];body.append(f'| {title} | 160/160 | {ratio(d["not_executed"],160)} | {ratio(d["planning_passes"],160)} | {ratio(d["physical_attempts"],160)} | {ratio(d["successes"],160)} |')
    body += ['', '两条I0001请求在机器人场景执行前被记录绑定检查拦截，原停止现场保留；每法各一个槽单列未执行，没有重抓。正式统计保留160总分母；不能把这两条称为抓取失败或无解。其余318条均完成，独立重读轨迹判据通过。',
        '', '## 相同场景的固定六轴历史参考','', '| 链路 | 固定六轴规划 | 本轮七轴规划 | 固定六轴成功 | 本轮七轴成功 |', '| --- | ---: | ---: | ---: | ---: |']
    for g,title in zip(ARMS,['ZeroGrasp＋cuRobo','ZeroGrasp＋新GraspCL＋cuRobo']):
        d=data[g];old=d['history_fixed6'];body.append(f'| {title} | {ratio(old["planning_passes"],160)} | {ratio(d["planning_passes"],160)} | {ratio(old["successes"],160)} | {ratio(d["successes"],160)} |')
    body += ['', '历史六轴为刚完成的45.2cm同场景记录，本轮重新观察、推理与执行，没有同期重跑六轴组。初始规划、实际接触和保持结果分开比较；不把离线IK或路径通过直接当作实际成功。',
        '', '## 最终原因与未执行分开列出','', '| 最终原因 | ZeroGrasp | GraspCL |', '| --- | ---: | ---: |']
    reasons=set().union(*(d['failure_reasons'] for d in data.values()))
    for reason in sorted(reasons):
        label='记录检查拦截，未执行' if reason.startswith('not_executed') else h.REASONS.get(reason,reason)
        body.append(f'| {label} | {ratio(data[ARMS[0]]["failure_reasons"].get(reason,0),160)} | {ratio(data[ARMS[1]]["failure_reasons"].get(reason,0),160)} |')
    body += ['', '[完整汇总与配对清单](/media/joint7-summary.json) · [最终核验](/media/joint7-audit.json)',
        '', '## 参考方法','', '[ZeroGrasp](/papers/zerograsp/) · [GraspCL](/papers/graspcl/) · [cuRobo](/papers/curobo/)',
        '', f'原始结果：`{RUN}`。本轮底盘保持固定。']
    markdown='\n'.join(body)+'\n';(SITE/'内容/迭代/七关节升降抓取实验.md').write_text(markdown)
    (MS/'docs/七关节升降抓取结果.md').write_text('# 七关节升降抓取结果\n\n'+markdown)
    site=h.read(SITE/'内容/网站.json');site['versions']=[v for v in site['versions'] if v['id']!='joint7']+[{
        'id':'joint7','parent':'lift452','type':'experiment','title':'升降与右臂七关节联合规划','shortTitle':'加入升降联合规划',
        'date':'2026-11-21','status':'320槽结束 · 含2未执行','summary':'同一批160非S5场景，升降与右臂联合到预抓取，最后接近和短抬升固定高度；补齐随动躯干／左臂碰撞。',
        'metric':f'原生 {data[ARMS[0]]["successes"]}/160 · 新库 {data[ARMS[1]]["successes"]}/160','file':'七关节升降抓取实验.md'}]
    site['localAssets'].update({'joint7-summary.json':'七关节对照/实验汇总.json','joint7-audit.json':'七关节对照/最终核验.json'});h.save(SITE/'内容/网站.json',site)
    paperfile=SITE/'内容/论文调研/论文目录.json';papers=h.read(paperfile)
    for paper in papers['papers']:
        if paper['id'] in ['zerograsp','graspcl','curobo'] and 'joint7' not in paper['versions']:paper['versions'].append('joint7')
    h.save(paperfile,papers)
    fontspec=importlib.util.spec_from_file_location('site_font_refresh',SITE/'工具/跟进45厘米实验发布.py');fontmod=importlib.util.module_from_spec(fontspec);fontspec.loader.exec_module(fontmod);fontmod.fonts()
    subprocess.run(['/home/zhaoyechong/.nvm/versions/node/v22.23.2/bin/node',str(SITE/'build.mjs')],cwd=SITE,check=True)

def main():
    last=None;start=time.monotonic()
    try:
        while True:
            if (RUN/'STOP.json').exists() or (RUN/'控制器故障.json').exists():raise RuntimeError('Original campaign halted; no retry or result publication')
            p=h.read(RUN/'progress.json')
            if p['test_completed']!=last:state('waiting',completed=p['test_completed'],total=320);last=p['test_completed']
            if p['phase']=='complete' and (RUN/'运行结束.json').exists():break
            if time.monotonic()-start>8*3600:raise TimeoutError('Eight-hour bounded monitor elapsed; no experiment restart')
            time.sleep(15)
        state('verifying');finish();state('published',url='/iterations/joint7/')
    except BaseException as error:state('halted',error=repr(error),automatic_retry=False);raise

if __name__=='__main__':main()
