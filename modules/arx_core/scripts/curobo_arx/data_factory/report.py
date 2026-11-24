"""One-page coverage/throughput report and deterministic raw-video selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from .common import VERSION,digest,read,write
from .ledger import Ledger


def union_seconds(intervals):
    end=None;total=0.
    for start,stop in sorted(intervals):
        if stop<start:raise ValueError('Invalid GPU process interval')
        if end is None or start>end:
            total+=stop-start;end=stop
        elif stop>end:
            total+=stop-end;end=stop
    return total


def gpu_hours(campaign,end):
    intervals={}
    reg=read(campaign/'registration.json')
    campaigns=[Path(e['campaign']) for e in reg.get('lineage',[])]+[campaign]
    for epoch in campaigns:
        state=read(epoch/'state.json');epoch_end=min(end,state.get('finished_unix',end))
        for path in (epoch/'workers').glob('*/process.json'):
            meta=read(path);stop=path.parent/'STOP'
            if not meta.get('gpu_uuid'):continue  # Official SDG candidate workers are CPU-only.
            intervals.setdefault(meta['gpu_uuid'],[]).append((path.stat().st_mtime,min(epoch_end,stop.stat().st_mtime) if stop.exists() else epoch_end))
        for path in (epoch/'group_registrations').glob('*.process.json'):
            launch=read(path.with_name(path.name.removesuffix('.process.json')+'.json'))
            shutdown=epoch/'groups'/launch['group_id']/'shutdown_receipt.json'
            stop=min(epoch_end,read(shutdown)['wall']) if shutdown.exists() else epoch_end
            intervals.setdefault(launch['gpu_uuid'],[]).append((path.stat().st_mtime,stop))
    return sum(union_seconds(values) for values in intervals.values())/3600


def report(campaign, output=None):
    campaign=Path(campaign)
    output=Path(output) if output else campaign
    output.mkdir(parents=True,exist_ok=True)
    reg=read(campaign/'registration.json')
    ledger=Ledger(campaign/'ledger.sqlite3')
    rows=ledger.rows()
    goals=[dict(r) for r in ledger.db.execute('SELECT object_id,id,record FROM goals')]
    exhausted={r['object_id']:r['reason'] for r in ledger.db.execute('SELECT object_id,reason FROM exhausted_objects')}
    pool_events=[json.loads(r[0]) for r in ledger.db.execute("SELECT value FROM events WHERE event='goal_pool'")]
    ledger.close()
    # Older registrations did not project rejection totals into ledger events.
    # Derive that missing field from their durable, hash-bound pool receipts;
    # never alter the old database or silently report a zero.
    missing={e['pool_sha256']:e for e in pool_events if 'fixed_G_planning_rejections' not in e}
    if missing:
        for path in (campaign/'groups').glob('*/blocks/*/qualified_pool.json'):
            pool=read(path);event=missing.get(digest(pool))
            if event is not None:
                event['fixed_G_planning_rejections']=len(pool.get('planning_pool_rejections',[]))
    success=[r for r in rows if r['state']=='committed']
    inherited=set(reg.get('resume',{}).get('inherited_attempt_ids',[]))
    per_object=[]
    quota=reg['quotas']
    for obj in reg['catalog']['objects']:
        name=obj['object_id']
        applicable=obj.get('collection_enabled',True) and (obj['available'] if 'collection_selection' in reg['catalog'] else True)
        attempts=[r for r in rows if r['object_id']==name]
        passed=[r for r in attempts if r['state']=='committed']
        bins=Counter((r['goal_id'],r['distance']) for r in passed)
        full=sum(all(bins[(g['id'],d)]>=quota['successes_per_bin'] for d in quota['distances_mm']) for g in goals if g['object_id']==name)
        distance={str(d):sum(r['distance']==d for r in passed) for d in quota['distances_mm']}
        reasons=Counter(r['outcome'] for r in attempts if r['state'] not in ('committed','claimed','executing','writing'))
        per_goal=[sum(min(bins[(g['id'],d)],quota['successes_per_bin']) for d in quota['distances_mm']) for g in goals if g['object_id']==name]
        contribution=sum(sorted(per_goal,reverse=True)[:quota['goals_per_object']])
        placements=obj.get('collection_placements',[])
        goal_records={g['id']:json.loads(g['record']) for g in goals if g['object_id']==name}
        expected=[p['placement_id'] for p in placements] if placements else [None]
        coverage=[];contribution=0
        for ident in expected:
            scoped=[g for g,r in goal_records.items() if r.get('placement_id')==ident]
            full_scoped=sum(all(bins[(g,d)]>=quota['successes_per_bin'] for d in quota['distances_mm']) for g in scoped)
            per_scoped=[sum(min(bins[(g,d)],quota['successes_per_bin']) for d in quota['distances_mm']) for g in scoped]
            contribution+=sum(sorted(per_scoped,reverse=True)[:quota['goals_per_object']])
            coverage.append({'placement_id':ident,'qualified_goals':len(scoped),
                'physical_attempts':sum(r['physical'] for r in attempts if r['goal_id'] in scoped),
                'successful_demonstrations':sum(r['goal_id'] in scoped for r in passed),
                'quota_complete':full_scoped>=quota['goals_per_object']})
        per_object.append({'object_id':name,'available':obj['available'],'split':obj['split'],'quota_contribution':contribution,
            'collection_enabled':obj.get('collection_enabled',True),'quota_applicable':applicable,
            'collection_exclusion_reason':obj.get('collection_exclusion_reason'),
            'fixed_G_planning_rejections':sum(e.get('fixed_G_planning_rejections',0) for e in pool_events if e['object_id']==name),
            'candidate_goals':sum(g['object_id']==name for g in goals),'successful_goals':len({r['goal_id'] for r in passed}),
            'fully_covered_goals':full,'successful_starts_by_distance':distance,'successes':len(passed),
            'physical_attempts':sum(r['physical'] for r in attempts),'dispatches':len(attempts),
            'planning_rejections':sum(r['state']=='rejected' for r in attempts),'failures':sum(r['state']=='failed' for r in attempts),
            'faults_or_unknown':sum(r['state'] in ('fault','interrupted','recovered_interrupted') for r in attempts),
            'registered_placements':len(expected) if applicable else 0,
            'physically_attempted_placements':sum(c['physical_attempts']>0 for c in coverage) if applicable else 0,
            'successful_placements':sum(c['successful_demonstrations']>0 for c in coverage) if applicable else 0,
            'placement_coverage':coverage if applicable else [],
            'visual_rejected':sum(r['state']=='visual_rejected' for r in attempts),
            'quota_complete':all(c['quota_complete'] for c in coverage) if applicable else None,'issues':obj['issues'],
            'major_gap':obj.get('collection_exclusion_reason') or '; '.join(obj['issues']) or exhausted.get(name) or (reasons.most_common(1)[0][0] if reasons else 'awaiting production')})
    launch=read(campaign/'launch.json') if (campaign/'launch.json').exists() else {}
    end=read(campaign/'state.json').get('finished_unix') if (campaign/'state.json').exists() else None
    elapsed=max(0.,(end or time.time())-reg.get('campaign_started_unix',launch.get('started_unix',time.time())))
    hours=elapsed/3600
    process_gpu_hours=gpu_hours(campaign,end or time.time())
    covered=sum(bool(p['successes']) for p in per_object)
    satisfied=sum(p['quota_contribution'] for p in per_object)
    state=read(campaign/'state.json')
    infrastructure_stops=sum(e.get('status')=='fault' for e in reg.get('lineage',[]))+int(state['status']=='fault')
    result={'version':VERSION,'successful_demonstrations':len(success),'objects_with_data':covered,'catalog_objects':len(per_object),
        'collection_objects':sum(o['quota_applicable'] for o in per_object),
        'quota_fraction':satisfied/quota['target_episodes'],'quota_complete':all(o['quota_complete'] for o in per_object if o['quota_applicable']),
        'task_ended':state['status'] in ('budget_exhausted','quota_complete','stopped','fault'),
        'status':state['status'],'wall_hours':hours,'demonstrations_per_wall_hour':len(success)/hours if hours else 0.,
        'gpu_hours':process_gpu_hours,'gpu_hours_scope':'Union of registered process intervals per GPU, including startup and resident compute; not exclusive/full-utilization device hours',
        'counts':dict(Counter(r['state'] for r in rows)),'physical_attempts':sum(r['physical'] for r in rows),
        'registered_object_placements':sum(o['registered_placements'] for o in per_object),
        'physically_attempted_object_placements':sum(o['physically_attempted_placements'] for o in per_object),
        'successful_object_placements':sum(o['successful_placements'] for o in per_object),
        'visual_rejected':sum(r['state']=='visual_rejected' for r in rows),
        'inherited_successful_demonstrations':sum(r['id'] in inherited for r in success),
        'new_successful_demonstrations':sum(r['id'] not in inherited for r in success),
        'infrastructure_stops':infrastructure_stops,
        'retained_interrupted_attempts':sum(r['state']=='recovered_interrupted' for r in rows),
        'unknown_physical_outcomes':sum(r['physical'] and r['state'] in ('interrupted','recovered_interrupted') for r in rows),
        'planning_rejections':sum(r['state']=='rejected' for r in rows),'record_or_infrastructure_faults':sum(r['state']=='fault' for r in rows),
        'fixed_G_planning_rejections':sum(o['fixed_G_planning_rejections'] for o in per_object),
        'active_or_unknown':sum(r['state'] in ('claimed','executing','writing','interrupted') for r in rows),'objects':per_object}
    write(output/'summary.json',result)
    selections=select_videos(rows)
    write(output/'video_selection.json',selections)
    lines=['# ARX 自动示教生产', '',
        f"**{len(success)} 条成功示教｜{covered}/{result['collection_objects']} 个采集物体有数据｜G/H 配额 {result['quota_fraction']:.2%}｜{result['demonstrations_per_wall_hour']:.2f} 条/墙钟小时**",'',
        f"状态：{state['status']}；任务已结束：{result['task_ended']}；配额完成：{result['quota_complete']}。",'',
        f"承接成功 {result['inherited_successful_demonstrations']}；本次新增 {result['new_successful_demonstrations']}；历史基础设施停机 {infrastructure_stops}；保留中断 {result['retained_interrupted_attempts']}（其中物理结果未知 {result['unknown_physical_outcomes']}）。",'',
        f"物理尝试 {result['physical_attempts']}；运动前规划拒绝 {result['planning_rejections']}；已标记故障尝试 {result['record_or_infrastructure_faults']}；当前在途或未承接中断 {result['active_or_unknown']}。",'',
        f"固定 G 预筛拒绝 {result['fixed_G_planning_rejections']}（未下发运动，不计作已执行抓取）。",'',
        '| 物体 | G 候选/有成功/满额 | 成功起点 '+('/'.join(map(str,quota['distances_mm'])))+'mm | 合格轨迹 | 物理尝试/派发 | 主要缺口 |',
        '|---|---|---|---:|---|---|']
    for obj in per_object:
        b='/'.join(str(obj['successful_starts_by_distance'][str(d)]) for d in quota['distances_mm'])
        lines.append(f"| {obj['object_id']} | {obj['candidate_goals']}/{obj['successful_goals']}/{obj['fully_covered_goals']} | {b} | {obj['successes']} | {obj['physical_attempts']}/{obj['dispatches']} | {obj['major_gap']} |")
    lines+=['',f"墙钟 {hours:.3f} 小时；GPU 进程时段合计 {result['gpu_hours']:.3f} 小时（同卡重叠时段合并，包含启动与常驻计算，不代表全卡独占或满负载）。预热、感知、规划、物理执行、写盘均计入墙钟。",'',
        '抽样视频见 [video_selection.json](video_selection.json)。不足 6 个可录制案例时明确留缺；不能复制旧视频补数。', '',
        '原始数据含 G、H、实际入口、腕部 RGB-D、相机标定、实测关节/末端状态、实际命令、接触、阶段和保持高度。旧 S02 结果独立保留。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def select_videos(rows):
    eligible=[]
    for row in rows:
        if row['state'] in ('claimed','executing','writing','interrupted'):continue
        if not row['episode'] or not (Path(row['episode'])/'samples.jsonl').is_file():continue
        root=Path(row['episode'])
        complete=(root/'episode.json').exists() and read(root/'episode.json').get('write_complete')
        if not complete and not (root.parent.parent/'shutdown_receipt.json').exists():continue
        eligible.append(row)
    selected=[]
    def direction(row):
        root=Path(row['episode']);meta=root/'episode.json';selection=root/'selected_grasp.json'
        if meta.exists():goal=read(meta)['goal']['grasp_world']
        elif selection.exists():goal=read(selection)['grasp_world']
        else:return 'unknown'
        axis=max(range(3),key=lambda i:abs(goal[i][2]))
        return 'xyz'[axis]+('+' if goal[axis][2]>=0 else '-')
    def take(predicate):
        candidates=[r for r in eligible if r['id'] not in {x['id'] for x in selected} and predicate(r)]
        if candidates:selected.append(candidates[0])
    take(lambda r:r['object_id'].startswith('egad_'))
    take(lambda r:r['object_id'].startswith('egad_') and r['object_id'] not in {x['object_id'] for x in selected})
    take(lambda r:r['state'] in ('failed','fault','rejected'))
    take(lambda r:direction(r)!='unknown' and direction(r) not in {direction(x) for x in selected})
    for distance in (10,40,60,80):
        if len(selected)>=6:break
        take(lambda r:r['distance']==distance)
    for row in eligible:
        if len(selected)>=6:break
        take(lambda r:r['id']==row['id'])
    return {'selection_rule':'Chronologically first two distinct EGAD shapes, first available failure, then deficient distance bins, then chronological fill; no score cherry-picking',
        'selected':[{k:r[k] for k in ('id','object_id','goal_id','distance','state','episode')} for r in selected],
        'missing_slots':6-len(selected),'irregular_object_requirement_met':len({r['object_id'] for r in selected if r['object_id'].startswith('egad_')})>=2,
        'different_approach_directions':sorted({direction(r) for r in selected}),
        'failure_included_if_available':not any(r['state'] in ('failed','fault','rejected') for r in eligible) or any(r['state'] in ('failed','fault','rejected') for r in selected)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    print(json.dumps(report(a.campaign,a.out),ensure_ascii=False))
