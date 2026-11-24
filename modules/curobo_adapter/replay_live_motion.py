"""Six bounded offline IK queries on the three recorded VLM motion requests."""
from pathlib import Path
import json
import time

import numpy as np

from adapter import ARXPlanner, HERE, digest, save
from planner_cli import build_frozen_models
from probe import capture

ROOT=HERE/'research/20261117_05/motion_replay'


def main():
    spec=json.loads((ROOT/'registration.json').read_text())
    for path,h in spec['hashes'].items():assert digest(Path(path))==h,path
    records=[];started=time.monotonic()
    for job in spec['jobs']:
        assert time.monotonic()-started<300
        source=Path(job['request']);request=json.loads(source.read_text());state=request['state']
        assert request['closed'] is False
        out=ROOT/job['id'];out.mkdir(exist_ok=False)
        build_frozen_models(state,out/'input')
        planner=ARXPlanner(out/'input',out/'model')
        solver=planner.planner.ik_solver;native=solver._get_result;phases=[]
        def observe(opt,metrics,n,ret,*a,**kw):
            result=native(opt,metrics,n,ret,*a,**kw)
            phases.append(capture(opt,metrics,result,solver.config));return result
        solver._get_result=observe
        try:
            for name,goal in [('measured_zero_motion',state['links']['right_eef_link']),('recorded_VLM_motion',request['world_goal'])]:
                phases.clear();goal=np.asarray(goal)
                row={'job':job['id'],'kind':name,'goal':goal.tolist(),'live_action_accepted':job['live_action_accepted']}
                with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                planner.planner.reset_seed()
                q,result=planner.ik(goal,planner.measured(state))
                row.update(accepted=q is not None,q=None if q is None else q.tolist(),native_phases=phases.copy())
                with (out/'calls.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                final=phases[-1]
                records.append({'job':job['id'],'kind':name,'accepted':q is not None,
                    'live_action_accepted':job['live_action_accepted'],
                    'pose_converged_seeds':sum(final['pose_converged']),
                    'feasible_seeds':sum(final['feasible']),
                    'pose_converged_constraint_violations':final['pose_converged_constraint_violations'],
                    'best_position_error_m':final['best_pose_position_m'],
                    'best_orientation_error_rad':final['best_pose_rotation_rad']})
                print(records[-1],flush=True)
        finally:
            del solver._get_result;planner.planner.destroy()
    assert len(records)==6
    save(ROOT/'results.json',{'records':records,'native_ik_queries':6,'physical_trials':0,'cloud_requests':0,
                            'wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
