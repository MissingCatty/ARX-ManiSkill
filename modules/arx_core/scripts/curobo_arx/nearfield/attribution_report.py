"""Independent real-ARX results and paired incremental benefit, preserving every slot."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,digest,verify_files
from . import closedloop30_audit as audit
from .attribution_spec import ARMS,validate_trials,verify_generation
from .prepare_v2 import predecessors
from .verify_validation import reset_signature
from .verify_physical import entry_matches_raw
from .capture import audit_episode
from .report import video


def paired(rows, reference):
    groups={}
    for t in rows:groups.setdefault((t['case_id'],t['repeat']),{})[t['attribution_arm']]=t
    pairs=[]
    for (case,repeat),g in sorted(groups.items()):
        if 'trained' not in g or reference not in g:continue
        a,b=g[reference],g['trained'];x,y=bool(a['task_success']),bool(b['task_success'])
        pairs.append({'case_id':case,'repeat':repeat,'reference':reference,'reference_success':x,
            'trained_success':y,'rescued':not x and y,'regressed':x and not y,'delta':int(y)-int(x),
            'reference_trial':a['trial_id'],'trained_trial':b['trial_id']})
    n=len(pairs);rescued=sum(p['rescued'] for p in pairs);regressed=sum(p['regressed'] for p in pairs)
    cases=sorted({p['case_id'] for p in pairs});interval=None
    if len(cases)>=2:
        values={c:[p['delta'] for p in pairs if p['case_id']==c] for c in cases}
        rng=np.random.default_rng(20261110);means=[]
        for _ in range(2000):
            block=[v for c in rng.choice(cases,len(cases),replace=True) for v in values[c]]
            means.append(float(np.mean(block)))
        interval=np.quantile(means,[.025,.975]).tolist()
    return {'reference':reference,'complete_pairs':n,'rescued':rescued,'regressed':regressed,
            'net_difference':rescued-regressed,'difference_fraction':(rescued-regressed)/n if n else None,
            'exploratory_case_cluster_interval_95':interval,
            'interval_scope':'Resample condition clusters retaining repetitions; only four selected objects, not population generalization',
            'pairs':pairs}


def report(campaign):
    root=Path(campaign).resolve();reg=read(root/'registration.json');summary=read(root/'summary.json')
    if (root/'attribution_verification.json').exists():raise FileExistsError('Preserve finished report')
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
        item={k:spec[k] for k in ('trial_id','attribution_arm','repeat','case_id','condition','object_id','source_split','fixed_entry_backoff_m')}
        item.update(task_success=trial['success'],outcome=trial['outcome'],strict_success=False,
            physical_attempt=bool(trial['physical_dispatches']),controller_calls=0,model_inferences=0,
            wall_seconds=trial['finished_unix']-trial['started_unix'],video=None,record_integrity_pass=True)
        if not attempts:
            if item['task_success']:raise RuntimeError('Success without execution')
            details.append(item);continue
        ep=Path(attempts[0]['episode']);physical=read(ep/'physical_receipt.json');independent=audit.evaluate(ep)
        if digest(independent)!=digest(physical) or item['task_success']!=physical['pass']:
            raise RuntimeError('Independent physical scoring differs')
        seed=reg['seeds'][spec['seed_id']]
        if reset_signature(read(ep/'initial_conditions.json'))!=reset_signature(read(Path(seed['episode'])/'initial_conditions.json')):
            raise RuntimeError('Historical physical reset changed')
        policy=audit.audit_policy(ep);raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
        capture=audit_episode(ep) if raw else {'applicable':False,'frames':0}
        events=read(ep/'interpreter_events.json') if (ep/'interpreter_events.json').exists() else []
        for request in policy['requests']:
            path=ep/'policy'/(request['request_id']+'_request.json');req=read(path);rep=read(path.with_name(request['request_id']+'_reply.json'))
            if req['trial_id']!=spec['trial_id'] or verify_generation(reg,req,rep)!=spec['attribution_arm']:
                raise RuntimeError('Wrong comparison generator used')
            item['model_inferences']+=int(rep['model_inference'])
            entry_matches_raw(raw[req['raw_index']],read(req['observation']))
            if req['inputs']['eef_world']!=raw[req['raw_index']]['snapshot']['link_world']['right_eef_link']:
                raise RuntimeError('Image/action frame differs')
        for event in events:
            if event['kind']=='physical_entry_stopped' and (event['registered_H_world']!=spec['entry_world'] or event['registered_G0_world']!=spec['G0_world']):
                raise RuntimeError('Matched entry or prior changed at runtime')
        frozen=list((ep.parent.parent/'blocks').glob('*/frozen_*.json'))
        if not frozen:raise RuntimeError('Missing frozen runtime configuration proof')
        for path in frozen:
            proof=read(path)
            if not proof['pass'] or proof['mismatches'] or not proof['additional_assets']['pass']:
                raise RuntimeError('Physical configuration changed')
            evidence[str(path)]=sha(path)
        stages=set(physical['executed_stages'])
        flags={'entry': 'state_generation' in stages,'axis': 'd_entry_axis' in stages,
            'local_motion':any(s.startswith('d_refine_') for s in stages),
            'close':'close' in stages,'lift':'lift' in stages,'hold':'hold' in stages,
            'controller_query':policy['policy_calls']>0,'model_query':item['model_inferences']>0,
            'stationary_continuation':bool(physical['certified_nonconverged_waypoints'])}
        if physical['pass'] and not all(flags[k] for k in ('entry','controller_query','close','lift','hold')):
            raise RuntimeError('Credited success misses required events')
        if spec['attribution_arm']=='g0' and item['model_inferences']:
            raise RuntimeError('Geometry branch credited as model inference')
        coverage.update({k:int(v) for k,v in flags.items()})
        item.update(strict_success=physical['strict_original_success'],physical=physical,capture=capture,
            controller_calls=policy['policy_calls'],closure_origins=policy['closure_origins'],event_applicability=flags,
            historical_reset_unchanged=True,video=video(ep,root/'videos'/(spec['trial_id']+'.mp4')) if raw else None)
        if raw and item['video'] is None:raise RuntimeError('Recorded frames lack video')
        for path in ep.rglob('*.json*'):evidence[str(path)]=sha(path)
        details.append(item)
    groups={a:{'completed':sum(t['attribution_arm']==a for t in details),
        'successes':sum(t['attribution_arm']==a and t['task_success'] for t in details),
        'strict_successes':sum(t['attribution_arm']==a and t['strict_success'] for t in details),
        'invalid_output_failures':sum(t['attribution_arm']==a and t['outcome']=='policy_invalid_action' for t in details)} for a in ARMS}
    result={'record_integrity_pass':not summary['incomplete_accounting'],'scheduled':72,'completed':len(details),
        'task_successes':sum(t['task_success'] for t in details),'ordinary_failures':sum(not t['task_success'] for t in details),
        'unexecuted':summary['unexecuted'],'unknown_or_incomplete':summary['incomplete_accounting'],
        'groups':groups,'paired':{a:paired(details,a) for a in ('g0','base')},'trials':details,
        'event_coverage_trials':dict(coverage),'evidence':evidence,'no_automatic_promotion':True}
    result['complete']=len(details)==72 and result['record_integrity_pass'] and not result['unexecuted']
    write(root/'final_idle_verification.json',{'pass':True,'proof':idle});write(root/'attribution_verification.json',result)
    lines=['# G0与微调贡献对照验收','',f'计划72次，完成{len(details)}；成功{result["task_successes"]}，普通失败{result["ordinary_failures"]}；未执行{len(result["unexecuted"])}，未知/记录未完成{len(result["unknown_or_incomplete"])}。','',
        '| 组别 | 完成 | 成功 | 原严格成功 | 动作格式失败 |','|---|---:|---:|---:|---:|']
    for a,g in groups.items():lines.append(f'| {a} | {g["completed"]}/24 | {g["successes"]} | {g["strict_successes"]} | {g["invalid_output_failures"]} |')
    lines+=['','| 训练D相对 | 完整配对 | 救回 | 破坏 | 净增成功 |','|---|---:|---:|---:|---:|']
    for a,p in result['paired'].items():lines.append(f'| {a} | {p["complete_pairs"]} | {p["rescued"]} | {p["regressed"]} | {p["net_difference"]} |')
    lines+=['','| 物体/条件/重复 | G0 | 训练D | 基础模型 |','|---|---|---|---|']
    by={(t['case_id'],t['repeat'],t['attribution_arm']):t for t in details}
    for case,repeat in sorted({(t['case_id'],t['repeat']) for t in reg['seed_trials']}):
        cells=[]
        for arm in ARMS:
            t=by.get((case,repeat,arm))
            cells.append('未完成，见未执行/未知清单' if t is None else ('成功' if t['task_success'] else '失败：'+t['outcome']))
        lines.append('| '+f'{case}/r{repeat}'+' | '+' | '.join(cells)+' |')
    lines+=['','格式失败属于接口表现，不能单独解释成视觉几何能力差异。牛奶/芥末与训练共享来源家族，BBQ/Spam属于反复使用过的验证物体；G0来自历史成功候选，本轮不宣称独立新物体泛化。',
        '每组最多两次局部运动、三次决策，共用G0、近场入口、轴向接近、闭合与物理评价。几何控制调用不计为模型推理。部分配对缺失时仅报告已完成配对，未执行与未知不伪装为抓取失败。',
        '本轮不移除G0门控、不训练、不自动推广。普通失败及成功均保留；若基础模型主要因格式失败，后续需另行评估视觉增益。','']
    for t in details:
        if t['video']:lines.append(f'- [{t["trial_id"]} 视频](videos/{t["trial_id"]}.mp4)')
    (root/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
    (root/'autoresearch-results.tsv').write_text('arm\tcompleted\tsuccesses\tstrict_successes\n'+''.join(f'{a}\t{g["completed"]}\t{g["successes"]}\t{g["strict_successes"]}\n' for a,g in groups.items()))
    return result
