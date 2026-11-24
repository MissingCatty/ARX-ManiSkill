"""Independent physical results for the new registered counterfactual scenes."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from scripts.curobo_arx.data_factory.common import read,write,sha,digest,verify_files
from . import closedloop30_audit as audit
from .visual_grounding_spec import ARMS,validate_trials,verify_reply,expected_reset
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw
from .prepare_v2 import predecessors
from .capture import audit_episode
from .report import video


def paired(rows,reference):
    groups={}
    for r in rows:groups.setdefault((r['case_id'],r['repeat']),{})[r['vision_arm']]=r
    results=[]
    for (case,repeat),g in sorted(groups.items()):
        if 'fresh' not in g or reference not in g:continue
        a,b=g['fresh'],g[reference]
        results.append({'case_id':case,'repeat':repeat,'fresh_success':a['task_success'],
            'reference_success':b['task_success'],'rescued':a['task_success'] and not b['task_success'],
            'regressed':b['task_success'] and not a['task_success'],
            'both_visual_controls_comparable':a.get('first_control_applicable',False) and b.get('first_control_applicable',False)})
    return {'reference':reference,'pairs':results,'complete_pairs':len(results),
        'rescued':sum(r['rescued'] for r in results),'regressed':sum(r['regressed'] for r in results),
        'net_success_gain':sum(int(r['rescued'])-int(r['regressed']) for r in results),
        'visually_comparable_pairs':sum(r['both_visual_controls_comparable'] for r in results)}


def report(campaign):
    root=Path(campaign).resolve();reg=read(root/'registration.json');summary=read(root/'summary.json')
    if (root/'visual_grounding_verification.json').exists():raise FileExistsError('Preserve finished report')
    validate_trials(reg['seed_trials']);idle=predecessors([root])
    verify_files({**reg['execution_source_files'],**reg['dependencies']})
    specs={t['trial_id']:t for t in reg['seed_trials']};details=[];evidence={};coverage=Counter()
    for trial in summary['trials']:
        spec=specs[trial['trial_id']]
        if any(trial[k]!=spec[k] for k in spec):raise RuntimeError('Executed slot changed')
        db=sqlite3.connect('file:'+str(root/'trial_ledgers'/(trial['trial_id']+'.sqlite3'))+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:attempts=[dict(r) for r in db.execute('select * from attempts order by started,id')]
        finally:db.close()
        if attempts!=trial['attempts'] or len(attempts)>1:raise RuntimeError('Ledger mismatch or retry')
        item={k:spec[k] for k in ('trial_id','vision_arm','repeat','case_id','condition','object_id','source_split','target_delta_world_m')}
        item.update(task_success=trial['success'],outcome=trial['outcome'],strict_success=False,
            physical_attempt=bool(trial['physical_dispatches']),model_inferences=0,controller_calls=0,
            first_control_applicable=False,visual_replacements=0,event_applicability={},video=None)
        if not attempts:
            if item['task_success']:raise RuntimeError('Success without execution')
            item['classification']='pre_motion_rejection';details.append(item);continue
        ep=Path(attempts[0]['episode']);physical=read(ep/'physical_receipt.json')
        if digest(audit.evaluate(ep))!=digest(physical) or item['task_success']!=physical['pass']:
            raise RuntimeError('Independent physical score differs')
        expected=expected_reset(reg['seeds'][spec['seed_id']],spec)
        if reset_signature(read(ep/'initial_conditions.json'))!=expected:
            raise RuntimeError('Reset differs beyond the specifically registered target X displacement')
        policy=audit.audit_policy(ep);raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
        capture=audit_episode(ep) if raw else {'applicable':False,'frames':0}
        controls=[]
        for request in policy['requests']:
            path=ep/'policy'/(request['request_id']+'_request.json');req=read(path);rep=read(path.with_name(request['request_id']+'_reply.json'))
            verify_reply(reg,req,rep);entry_matches_raw(raw[req['raw_index']],read(req['observation']))
            if req['inputs']['eef_world']!=raw[req['raw_index']]['snapshot']['link_world']['right_eef_link']:
                raise RuntimeError('Policy image/actual pose binding changed')
            item['model_inferences']+=int(rep['model_inference']);item['visual_replacements']+=int(rep['visual_control']['replaced'])
            controls.append(rep['visual_control'])
        if controls:item['first_control_applicable']=controls[0]['applicable']
        if item['visual_replacements']>1 or (spec['vision_arm']!='stale_first' and item['visual_replacements']):
            raise RuntimeError('Unregistered image replacement')
        if spec['vision_arm']=='g0' and item['model_inferences']:raise RuntimeError('G0 controller called VLM')
        frozen=list((ep.parent.parent/'blocks').glob('*/frozen_*.json'))
        if not frozen:raise RuntimeError('Missing runtime physical-configuration evidence')
        for path in frozen:
            proof=read(path)
            if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:
                raise RuntimeError('Robot/scene physical configuration changed')
            evidence[str(path)]=sha(path)
        occurred=set(physical['executed_stages'])
        flags={'entry':'state_generation' in occurred,'axis':'d_entry_axis' in occurred,
            'local_motion':any(s.startswith('d_refine_') for s in occurred),'close':'close' in occurred,
            'lift':'lift' in occurred,'hold':'hold' in occurred,'controller_query':bool(policy['policy_calls']),
            'model_query':bool(item['model_inferences']),'wrong_view_applied':bool(item['visual_replacements']),
            'comparable_visual_entry':item['first_control_applicable']}
        if item['task_success'] and not all(flags[k] for k in ('entry','controller_query','close','lift','hold')):
            raise RuntimeError('Credited success misses occurred events')
        coverage.update({k:int(v) for k,v in flags.items()})
        item.update(strict_success=physical['strict_original_success'],physical=physical,capture=capture,
            controller_calls=policy['policy_calls'],closure_origins=policy['closure_origins'],visual_controls=controls,
            event_applicability=flags,registered_reset_unchanged=True,
            classification='success' if item['task_success'] else 'control_not_comparable' if trial['outcome']=='visual_control_pose_pair_not_comparable' else 'executed_task_failure',
            video=video(ep,root/'videos'/(spec['trial_id']+'.mp4')) if raw else None)
        if raw and item['video'] is None:raise RuntimeError('Recorded frames lack video')
        for path in ep.rglob('*.json*'):evidence[str(path)]=sha(path)
        details.append(item)
    groups={a:{'completed':sum(t['vision_arm']==a for t in details),
        'successes':sum(t['vision_arm']==a and t['task_success'] for t in details),
        'strict_successes':sum(t['vision_arm']==a and t['strict_success'] for t in details),
        'visually_comparable':sum(t['vision_arm']==a and t['first_control_applicable'] for t in details)} for a in ARMS}
    result={'scheduled':24,'completed':len(details),'complete':len(details)==24 and not summary['incomplete_accounting'] and not summary['unexecuted'],
        'record_integrity_pass':not summary['incomplete_accounting'],'groups':groups,'trials':details,
        'unexecuted':summary['unexecuted'],'unknown_or_incomplete':summary['incomplete_accounting'],
        'classifications':dict(Counter(t['classification'] for t in details)),
        'paired':{a:paired(details,a) for a in ('g0','stale_first')},'event_coverage_trials':dict(coverage),
        'evidence':evidence,'no_automatic_promotion':True,'independent_new_object_generalization_tested':False}
    write(root/'final_idle_verification.json',{'pass':True,'proof':idle});write(root/'visual_grounding_verification.json',result)
    lines=['# 真实目标位置变化与视觉对照验收','',f'登记24次，完成{len(details)}；未执行{len(result["unexecuted"])}，未知或记录未完成{len(result["unknown_or_incomplete"])}。','',
        '| 方法 | 完成 | 物理成功 | 严格成功 | 视觉入口可配对 |','|---|---:|---:|---:|---:|']
    for a,g in groups.items():lines.append(f'| {a} | {g["completed"]} | {g["successes"]} | {g["strict_successes"]} | {g["visually_comparable"]} |')
    lines+=['','| 新鲜视觉D相对 | 完整配对 | 救回 | 退步 | 净增加成功 |','|---|---:|---:|---:|---:|']
    for a,p in result['paired'].items():lines.append(f'| {a} | {p["complete_pairs"]} | {p["rescued"]} | {p["regressed"]} | {p["net_success_gain"]} |')
    lines+=['','三组只在新登记场景的目标X位置作±15毫米变化；G0、入口、机器人与物理属性固定。stale_first只在第一次决策使用原位置的历史真实腕部观测；后续全部恢复新鲜观测。',
        '动作是否受视觉影响、视觉是否提高实际抓取成功、是否泛化到独立新对象分别判断。本轮两个物体已参与训练或开发，不能宣称新对象泛化。',
        f'任务分类：{result["classifications"]}。记录完整性：{result["record_integrity_pass"]}。',
        f'事件覆盖：{dict(coverage)}。控制入口无法配准与普通抓取失败单列，全部已登记槽保留。','']
    for t in details:
        lines.append(f'- {t["trial_id"]}：{t["outcome"]}；'+(f'[视频](videos/{t["trial_id"]}.mp4)' if t['video'] else '无已录制运动画面'))
    (root/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
    (root/'autoresearch-results.tsv').write_text('arm\tcompleted\tsuccess\tstrict\n'+''.join(f'{a}\t{g["completed"]}\t{g["successes"]}\t{g["strict_successes"]}\n' for a,g in groups.items()))
    return result
