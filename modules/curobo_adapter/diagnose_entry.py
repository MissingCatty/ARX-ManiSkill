"""Bounded archived H-entry checks with frozen ARX models and native candidates."""
from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from adapter import ARXPlanner, HERE, digest, save, flush_planning_artifacts
from probe import capture
from entry_selector import select_pregrasp, decode_candidate

ROOT = HERE / "research/20261117_02"


def run_case(case: dict, spec: dict) -> dict:
    out=ROOT/case['id'];out.mkdir(exist_ok=False)
    trial=Path(spec['source_campaign'])/'trials'/case['id']
    state=json.loads((trial/'state.json').read_text())
    rows=np.load(trial/'postprocessed_grasps.npy',allow_pickle=False)
    rows=rows[np.argsort(-rows[:,0],kind='stable')]
    indices=case['indices']
    save(out/'inputs.json',{'state_sha256':digest(trial/'state.json'),
         'postprocessed_sha256':digest(trial/'postprocessed_grasps.npy'),
         'indices':indices,'physical_trials':0,'scene_changes':0})
    planner=ARXPlanner(trial,out/'model');solver=planner.planner.ik_solver
    native=solver._get_result;phases=[]
    def observe(opt,metrics,n,ret,*args,**kwargs):
        result=native(opt,metrics,n,ret,*args,**kwargs)
        phases.append(capture(opt,metrics,result,solver.config));return result
    solver._get_result=observe
    start=planner.measured(state);solutions={};records=[]
    requests=(out/'requests.jsonl').open('x');calls=(out/'calls.jsonl').open('x')
    def query(index,kind,world,current):
        phases.clear();identity=len(records)
        requests.write(json.dumps({'request_id':identity,'index':index,'kind':kind,
            'world_goal':world.tolist(),'current_q':current.tolist()})+'\n');requests.flush()
        q,result=planner.ik(world,current)
        assert phases
        record={'request_id':identity,'index':index,'kind':kind,'accepted':q is not None,
                'q':None if q is None else q.tolist(),'native_phases':phases.copy(),
                'world_goal':world.tolist()}
        calls.write(json.dumps(record,allow_nan=False)+'\n');calls.flush();records.append(record)
        return q
    began=time.monotonic()
    try:
        for index in indices:
            h=decode_candidate(rows[index],state['camera_world_optical'])[1]
            q=query(index,'H_from_measured_start',h,start)
            solutions[index]=None if q is None else q.clone()
        # Positive controls retain their original selected candidate index.
        # Excluded indices in their rows are absent from the precomputed map.
        try:
            plan=select_pregrasp(planner,state,rows,np.asarray(state['camera_world_optical']),
                                solutions=solutions,max_path_attempts=spec['maximum_path_attempts_per_case'])
        except RuntimeError as exc:
            if str(exc)!='Trajectory optimizer rejected all attempts; no PRM-only fallback':raise
            plan={'success':False,'reason':'entry_trajectory_optimizer_rejected',
                  'path_attempts':None,'other_H_paths_not_exhausted':True,
                  'planning_scope':'pregrasp_entry_only','full_grasp_validated':False}
        save(out/'entry_plan.json',plan)
        verified=None;future=None
        if plan['success']:
            selected=plan['selected'];h=np.asarray(selected['pregrasp_world'])
            end=torch.tensor(plan['paths']['pregrasp']['q'][-1],dtype=torch.float32,device='cuda')
            kin=planner.planner.compute_kinematics(planner.js(end)).tool_poses
            p=kin.position.detach().cpu().numpy().reshape(-1,3)[0]
            q=kin.quaternion.detach().cpu().numpy().reshape(-1,4)[0]
            expected=planner.invbase@h
            pe=float(np.linalg.norm(p-expected[:3,3]));re=float(Rotation.from_matrix(
                Rotation.from_quat([*q[1:],q[0]]).as_matrix()@expected[:3,:3].T).magnitude())
            verified={'position_error_m':pe,'rotation_error_rad':re,
                      'collision_valid':bool(planner.checker.validate(end[None,None]).all())}
            assert pe<.002 and re<.03 and verified['collision_valid'],verified
            index=selected['index'];g=decode_candidate(rows[index],state['camera_world_optical'])[0]
            from_start=query(index,'G_from_measured_start_diagnostic_only',g,start)
            from_h=query(index,'G_from_projected_H_diagnostic_only',g,
                         torch.tensor(selected['q_pre'],dtype=torch.float32,device='cuda'))
            future={'G_from_measured_start_passed':from_start is not None,
                    'G_from_projected_H_passed':from_h is not None,
                    'projected_H_is_not_measured_execution':True}
        h_records=[r for r in records if r['kind']=='H_from_measured_start']
        result={'id':case['id'],'control':case['control'],'H_queries':len(h_records),
                'H_ik_accepted':sum(r['accepted'] for r in h_records),
                'entry_trajectory_valid':bool(plan['success']),
                'entry_failure_reason':plan.get('reason'),
                'selected_index':plan.get('selected',{}).get('index'),
                'path_attempts':plan['path_attempts'],'endpoint_verification':verified,
                'future_G_diagnostic':future,'native_queries':len(records),
                'wall_seconds_after_initialization':time.monotonic()-began,
                'full_grasp_validated':False,'physical_trials':0,'cloud_requests':0,
                'frozen_model_scene_byte_identical':True}
        save(out/'result.json',result);print('DONE',case['id'],json.dumps(result),flush=True)
        return result
    finally:
        requests.close();calls.close()
        try:flush_planning_artifacts()
        finally:
            del solver._get_result
            planner.planner.destroy()


def main() -> None:
    spec=json.loads((ROOT/'scope.json').read_text());started=time.monotonic();results=[]
    for case in spec['cases']:
        assert time.monotonic()-started<spec['maximum_wall_seconds']
        for p,h in {**spec['protected_hashes'],**spec['source_hashes']}.items():assert digest(Path(p))==h,p
        assert not subprocess.check_output(['git','-C',str(HERE/'source'),'diff','--name-only'],text=True).strip()
        memory=int(subprocess.check_output(['nvidia-smi','-i',spec['gpu_uuid'],
             '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        assert memory>=8192,'Resource preflight failed; stop before next case'
        consumed=sum(len(p.read_text().splitlines()) for p in ROOT.rglob('requests.jsonl'))
        assert consumed+len(case['indices'])+2<=spec['maximum_native_ik_calls']
        print('START',case['id'],'H goals',len(case['indices']),flush=True)
        results.append(run_case(case,spec));gc.collect();torch.cuda.empty_cache()
        save(ROOT/'progress.json',{'completed_cases':len(results),'results':results})
    save(ROOT/'entry_results.json',{'results':results,'physical_trials':0,
         'H_queries':sum(r['H_queries'] for r in results),
         'H_ik_accepted':sum(r['H_ik_accepted'] for r in results),
         'new_failed_cases_with_entry':sum(r['entry_trajectory_valid'] for r in results if not r['control']),
         'native_queries':sum(r['native_queries'] for r in results)})


if __name__=='__main__':main()
