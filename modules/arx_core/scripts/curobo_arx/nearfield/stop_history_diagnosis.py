"""Read-only raw ARX replay of the unchanged measured stationary criterion."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha
from .advisor32_rules import StopSample,settle


def diagnose(root):
    root=Path(root).resolve();rows=[];evidence={}
    for result_path in sorted(root.glob('groups/*/trials/*/trial_result.json')):
        result=read(result_path)
        if result['outcome']!='stationary_wait_10s_timeout':continue
        ep=Path(result['attempts'][0]['episode']);requests=read(ep/'requests.json')
        label=requests[-1]['label'];execution_path=ep/(label+'_execution.json');execution=read(execution_path)
        raw_path=ep/'samples.jsonl';raw=[json.loads(x) for x in raw_path.read_text().splitlines()]
        samples=[]
        for row in raw:
            s=row['snapshot'];idx=[s['joint_names'].index(n) for n in s['right_arm']]
            samples.append(StopSample(float(s['sim_time']),tuple(np.asarray(s['link_world']['right_eef_link'])[:3,3]),
                tuple(np.asarray(s['joint_positions'])[idx]),tuple(np.asarray(s['joint_velocities'])[idx])))
        final=execution['final']['sim_time'];post=[s for s in samples if s.sim_time>final+1e-6]
        # Counterfactual evaluation only: old outcomes and dispatches stay untouched.
        old=settle(post,active=False,fault=None,elapsed_wall_s=9.9)
        continuous=settle(samples[-30:],active=False,fault=None,elapsed_wall_s=9.9)
        rows.append({'trial_id':result['trial_id'],'last_command':label,
            'after_completion_samples':len(post),'after_completion_span_s':post[-1].sim_time-post[0].sim_time if post else 0,
            'reset_history_result':old,'continuous_recorded_tail_result':continuous,
            'counterfactual_only_no_reclassification':True})
        for p in (result_path,execution_path,raw_path,ep/'requests.json'):evidence[str(p)]=sha(p)
    return {'new_physical_trials':0,'new_neural_calls':0,'frozen_settle_source_sha256':sha(Path(__file__).with_name('advisor32_rules.py')),
        'confirmed_history_starvation':any(not r['reset_history_result']['window_applicable'] and r['continuous_recorded_tail_result']['pass'] for r in rows),
        'rows':rows,'evidence':evidence,'old_failures_retained':True,'note':'Replays actual recorded states with current-controller-idle assumed after the completed action; no success claim or simulated retry.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();target=Path(a.root)/'stop_history_diagnosis.json'
    if target.exists():raise FileExistsError('Preserve previous diagnostic')
    result=diagnose(a.root);write(target,result);print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))
