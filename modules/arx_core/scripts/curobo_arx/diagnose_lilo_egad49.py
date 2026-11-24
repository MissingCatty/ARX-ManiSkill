"""Read existing ARX logs to diagnose a stopped LiLo-VLA benchmark; no motion."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from report_lilo_egad49 import classify,read


def diagnose(campaign):
    state=read(campaign/'state.json')
    results=[];first=[];times=[];clipped=components=0
    adapter=Path(read(campaign/'policy_health.json')['adapter'])
    stats=read(adapter/'dataset_statistics.json')['arx_bottle_local']
    for scenario in read(campaign/'scenarios.json'):
        trial=campaign/'trials'/scenario['name']
        if not (trial/'result.json').exists():continue
        result=read(trial/'result.json');category,error=classify(trial,result)
        results.append({'egad_id':scenario['egad_id'],'success':result['success'],
            'category':category,'error':error,'trial':str(trial)})
        for file in trial.glob('local_*_prediction.json'):
            prediction=read(file);times.append(prediction['inference_seconds'])
            pose=prediction['normalized_state'][:6]
            clipped+=sum(abs(v)>=1-1e-8 for v in pose);components+=len(pose)
        file=trial/'local_0000_prediction.json'
        if not file.exists():continue
        prediction=read(file);observation=read(trial/'lilo_observations/0000/observation.json')
        delta=np.asarray(prediction['actions'][0][:3])*.05
        row={'egad_id':scenario['egad_id'],'raw_state':prediction['raw_state'],
            'normalized_state':prediction['normalized_state'],
            'predicted_delta_position_m':delta.tolist(),
            'predicted_gripper_open':prediction['actions'][0][6]>=.5}
        execution=trial/'local_0000_execution.json'
        if execution.exists():
            end=read(execution)['final']
            before=np.asarray(observation['snapshot']['link_world']['right_eef_link'])[:3,3]
            after=np.asarray(end['link_world']['right_eef_link'])[:3,3]
            row['actual_delta_position_m']=(after-before).tolist()
            row['actual_goal_position_error_m']=float(np.linalg.norm(after-before-delta))
        first.append(row)
    errors=[r['actual_goal_position_error_m'] for r in first if 'actual_goal_position_error_m' in r]
    summary={'scope':'Stopped user-requested benchmark prefix, read-only original ARX logs; no completed-49 claim',
        'state':state,'completed_cases':len(results),'successes':sum(r['success'] for r in results),
        'failure_categories':dict(Counter(r['category'] for r in results)),
        'policy_first_frames':len(first),
        'initial_ee_minus_object_z_range_m':[min(r['raw_state'][2] for r in first),max(r['raw_state'][2] for r in first)],
        'training_ee_minus_object_z_range_m':[stats['proprio']['min'][2],stats['proprio']['max'][2]],
        'training_normalization_ee_minus_object_z_q01_q99_m':[stats['proprio']['q01'][2],stats['proprio']['q99'][2]],
        'first_z_above_training_max':sum(r['raw_state'][2]>stats['proprio']['max'][2] for r in first),
        'first_z_clipped_to_one':sum(r['normalized_state'][2]==1 for r in first),
        'first_action_upward':sum(r['predicted_delta_position_m'][2]>0 for r in first),
        'first_action_close':sum(not r['predicted_gripper_open'] for r in first),
        'first_action_actual_goal_position_error_median_m':statistics.median(errors),
        'saved_model_calls':len(times),'model_forward_median_s':statistics.median(times),
        'normalized_pose_components_at_clip_boundary_fraction':clipped/components,
        'training_transitions':stats['num_transitions'],'training_trajectories':stats['num_trajectories'],
        'rows':results,'first_action_rows':first}
    output=campaign/'diagnostics';output.mkdir(exist_ok=True)
    (output/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'completed_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    return {k:v for k,v in summary.items() if k not in ('rows','first_action_rows')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(diagnose(arguments.campaign.resolve()),indent=2))
