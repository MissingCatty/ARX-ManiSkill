"""Measured local tracking errors versus accumulated teacher-path drift."""
import argparse
from pathlib import Path
import numpy as np
from training.nearfield_arx.contract import decode,pose_error
from scripts.curobo_arx.data_factory.common import read,write


def reference_step(trial,index,previous):
    """Feedback-oracle tokens are current-state actions, not source increments."""
    if 'C_waypoints' in trial:
        return np.asarray(trial['C_waypoints'][index])
    return decode('C',previous,trial['C_tokens'][index])


def diagnose(campaign,out):
    reg=read(campaign/'registration.json');results=[]
    for file in (campaign/'groups').glob('*/results.json'):results+=read(file)['trials']
    rows=[]
    for trial in results:
        if trial['method'] not in ('C','D') or not trial['attempts']:continue
        root=Path(trial['attempts'][0]['episode']);seed=reg['seeds'][trial['seed_id']]
        if not (root/'actual_entry.json').exists():continue
        expected=np.asarray(seed['anchor_world']);entry=read(root/'actual_entry.json')['snapshot']['link_world']['right_eef_link']
        events=read(root/'interpreter_events.json') if (root/'interpreter_events.json').exists() else []
        targets=[e for e in events if e['kind']=='target_request']
        steps=[]
        if trial['method']=='C':
            moving=[r for r in read(root/'requests.json') if r['label']=='approach' or r['label'].startswith('c_step_')]
            for i,(request,token) in enumerate(zip(moving,trial['C_tokens'])):
                expected=reference_step(trial,i,expected);label=request['label']
                if not (root/(label+'_execution.json')).exists():continue
                receipt=read(root/(label+'_execution.json'));actual=receipt['final']['link_world']['right_eef_link']
                goal=targets[i]['goal_world'];settling=read(root/(label+'_settling.json'))
                steps.append({'index':i+1,'label':label,'local_tracking':pose_error(actual,goal),
                    'accumulated_vs_source':pose_error(actual,expected),'settling':settling,
                    'requested_world':goal,'actual_world':actual,'teacher_reference_world':expected.tolist()})
        row={'trial_id':trial['trial_id'],'method':trial['method'],'source_entry_error':pose_error(entry,seed['anchor_world']),
            'source_Gstar_world':trial['source_Gstar_world'],'steps':steps,'task_success':trial['success'],'outcome':trial['outcome'],
            'reference_basis':'recorded_next_waypoint' if 'C_waypoints' in trial else 'integrated_source_increments'}
        for phase in ('approach','close'):
            path=root/(phase+'_execution.json')
            if path.exists():row[phase+'_vs_Gstar']=pose_error(read(path)['final']['link_world']['right_eef_link'],trial['source_Gstar_world'])
        rows.append(row)
    write(out,{'scope':'Teacher-forced interpreter diagnostics; no learned policy or causal success-rate estimate',
        'trials':rows});return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=diagnose(a.campaign,a.out);print([(x['trial_id'],x.get('approach_vs_Gstar')) for x in r])
