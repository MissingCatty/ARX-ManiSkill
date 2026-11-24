"""Re-evaluate saved physical traces from raw contact points and object heights."""
from pathlib import Path
import json

import numpy as np

from audit import HERE, read, save

ROOT=HERE/'runs/physical3_20261123_02'


def main():
    summary=read(ROOT/'summary.json');rows=[]
    for entry in read(ROOT/'registration.json')['schedule']:
        out=ROOT/'cases'/entry['id'];result=read(out/'result.json')
        origin=read(out/'execution_origin.json')['object_initial_z_m']
        v1_bad=0;v2_bad=0;bilateral=set();hold=[];tip_events=0;max_depth=0.;release=None
        for line in (out/'execution.jsonl').read_text().splitlines():
            sample=json.loads(line);eef=np.asarray(sample['eef_world']);stage=sample['stage'];frame_tip=[]
            assert np.isfinite(eef).all() and abs(np.linalg.det(eef[:3,:3])-1)<1e-5
            for event in sample['contacts']:
                link=event['a'] if 'right_link' in event['a'] or 'left_link' in event['a'] else event['b']
                other=event['b'] if link==event['a'] else event['a']
                finger=next((n for n in ['right_link27','right_link28'] if n in link),None)
                original=bool(finger and 'target' in other and stage!='observe')
                assert original==event['original_allowed'],(entry['id'],link,other)
                allowed=original
                if original and stage=='close':bilateral.add(finger)
                if finger and 'table-workspace' in other and stage in ['approach','close','lift']:
                    points=[p for p in event['points'] if p['impulse_ns']>1e-8]
                    depth=max([max(0.,-p['separation_m']) for p in points],default=0.)
                    local=[(np.asarray(p['world'])-eef[:3,3])@eef[:3,:3] for p in points]
                    top=all(abs(p['normal'][2])>.95 and abs(p['world'][2])<=.0031 for p in points)
                    distal=all(p[2]>=.034970-1e-6 for p in local)
                    allowed=bool(points) and top and distal and depth<=.002
                    if allowed:frame_tip.append((event,depth))
                if not original:v1_bad+=1
                if not allowed:v2_bad+=1
                # Release depth is checked at the frame level below.
                if stage!='lift':assert allowed==event['allowed']
            depth=max([d for _,d in frame_tip],default=0.)
            if stage=='close':release=depth
            if stage=='lift':
                ceiling=0. if release is None else release
                if depth>ceiling+.0001:
                    v2_bad+=len(frame_tip)
                    for event,_ in frame_tip:assert not event['allowed']
                else:
                    for event,_ in frame_tip:assert event['allowed']
                release=min(ceiling,depth)
            tip_events+=sum(event['allowed'] for event,_ in frame_tip);max_depth=max(max_depth,depth)
            if stage=='hold':hold.append(float(sample['object_pose'][2])-origin)
        stages=result.get('stages',{})
        required=['pregrasp','approach','close','lift','hold']
        settled=all(k in stages and stages[k]['settled'] and stages[k]['joint_error_rad']<.005 for k in required)
        height_ok=bool(hold) and min(hold)>=.01 and len(hold)/120>=3
        v2=bool(settled and height_ok and len(bilateral)==2 and v2_bad==0)
        v1=bool(v2 and v1_bad==0)
        assert v2==result['success_v2'] and v1==result['success_v1']
        assert v2_bad==result['v2_forbidden_events'] and v1_bad==result['v1_forbidden_events']
        evaluation={'passed':True,'independent_success_v2':v2,'independent_success_v1':v1,
                    'continuous_hold_samples':len(hold),'minimum_hold_lift_m':min(hold) if hold else None,
                    'hold_seconds':len(hold)/120,'bilateral_fingers':sorted(bilateral),
                    'v1_forbidden_events':v1_bad,'v2_forbidden_events':v2_bad,
                    'allowed_tip_table_events':tip_events,'maximum_allowed_contact_depth_m':max_depth,
                    'all_required_stages_settled':settled}
        save(out/'independent_evaluation.json',evaluation);rows.append(evaluation)
    save(ROOT/'independent_verification.json',{'passed':True,'completed':len(rows),'registered':3,
         'success_v2':sum(r['independent_success_v2'] for r in rows),
         'success_v1':sum(r['independent_success_v1'] for r in rows),'raw_contact_points_rechecked':True})
    print('Independent physical verification:',len(rows),'cases')


if __name__=='__main__':main()
