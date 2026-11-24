"""Coverage tables for the SDG stage, separately from geometric proposal counts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write
from scripts.curobo_arx.data_factory.report import report as base_report


def report(campaign):
    campaign=Path(campaign).resolve();base=base_report(campaign);reg=read(campaign/'registration.json')
    db=sqlite3.connect('file:'+str(campaign/'ledger.sqlite3')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    attempts=[dict(r) for r in db.execute('SELECT * FROM attempts')]
    goals=[dict(r) for r in db.execute('SELECT * FROM goals')];db.close()
    accepted=[r for r in attempts if r['state']=='committed']
    stats=[];groups=Counter();perturbed=Counter()
    for r in accepted:
        meta=read(Path(r['episode'])/'episode.json');h=meta['sampled_entry']
        delta=np.asarray(h['local_translation_m'])+[0,0,r['distance']/1000]
        p=bool(np.linalg.norm(delta)>1e-8 or np.linalg.norm(h['local_rotation_vector_rad'])>1e-8)
        groups[(r['object_id'],meta['goal']['group'])]+=1;perturbed[r['object_id']]+=p
    for obj in base['objects']:
        rr=[r for r in accepted if r['object_id']==obj['object_id']]
        stats.append({'object_id':obj['object_id'],'category':next(o for o in reg['catalog']['objects'] if o['object_id']==obj['object_id']).get('category','EGAD' if obj['object_id'].startswith('egad_') else 'historical_everyday'),
            'collection_enabled':obj['collection_enabled'],
            'successes':len(rr),'successful_G':len({r['goal_id'] for r in rr}),
            'surface_direction_roll_groups':sum(k[0]==obj['object_id'] for k in groups),
            'perturbed_successes':perturbed[obj['object_id']],
            'success_10mm':obj['successful_starts_by_distance']['10'],'success_40mm':obj['successful_starts_by_distance']['40'],
            'success_60mm':obj['successful_starts_by_distance']['60'],'success_80mm':obj['successful_starts_by_distance']['80'],
            'physical_attempts':obj['physical_attempts'],'major_gap':obj['major_gap']})
        stats[-1].update(registered_placements=obj['registered_placements'],
            physically_attempted_placements=obj['physically_attempted_placements'],successful_placements=obj['successful_placements'],
            visual_rejected=obj['visual_rejected'])
    bank=read(Path(reg['sdg']['candidate_bank'])/'manifest.json')
    result={'stage':'sdg_stage1','status':base['status'],'candidate_source':'Official NVIDIA SDG antipodal '+reg['sdg']['extension_version'],
        'objects':base['collection_objects'],'scene_catalog_objects':len(reg['catalog']['objects']),'objects_with_success':base['objects_with_data'],
        'offline_geometric_candidates':sum(r.get('raw_candidates',0) for r in bank['objects']),
        'nominal_ARX_qualified_G':len(goals),'full_ARX_successful_demonstrations':len(accepted),
        'independent_local_gripper_only_demonstrations':0,'VLA_policy_evaluations':0,
        'successful_G':len({(r['object_id'],r['goal_id']) for r in accepted}),
        'perturbed_successes':sum(perturbed.values()),'physical_attempts':base['physical_attempts'],
        'physical_failures':sum(r['physical'] and r['state']=='failed' for r in attempts),
        'pre_motion_rejections':sum(r['state']=='rejected' for r in attempts),
        'quota_complete':base['quota_complete'],'old166_included_as_new':False}
    result.update(registered_object_placements=base['registered_object_placements'],
        physically_attempted_object_placements=base['physically_attempted_object_placements'],
        successful_object_placements=base['successful_object_placements'],visual_rejected=base['visual_rejected'])
    write(campaign/'sdg_summary.json',result)
    with (campaign/'sdg_coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    text=['# SDG 第一阶段生产','',f"状态：{result['status']}；成功示教 **{len(accepted)}** 条；{result['objects_with_success']}/{result['objects']} 个物体有数据；成功 G {result['successful_G']}；主动扰动成功 {result['perturbed_successes']}。",'',
        f"官方几何候选 {result['offline_geometric_candidates']}，ARX 规划预筛 G {len(goals)}；这些不计作示教。实际尝试 {base['physical_attempts']}，实际失败 {result['physical_failures']}，运动前拒绝 {result['pre_motion_rejections']}。",'',
        '全部成功须为完整 ARX 的实际接近/闭合/抬升/保持，并通过原物理/数据门控。旧166条独立保留，不计作本轮新增。未训练VLA，未运行在线入口适配。', '',
        '覆盖表见 sdg_coverage.csv；详细原始统计见 REPORT.md。训练导出与 D_requested/D_actual 起点索引由 sdg_stage1.export 生成。']
    (campaign/'SDG_REPORT.md').write_text('\n'.join(text)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');a=p.parse_args();print(report(a.campaign))
