"""Optimize a PRM seed and independently validate the ARX command trajectory."""
import json,time,hashlib,os
from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np
import torch
from curobo.types import JointState

_artifact_writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='arx-plan-evidence')
_artifact_futures=[]
_reports={}

def flush_planning_artifacts():
    """Wait for ordered evidence writes and propagate every storage error."""
    while _artifact_futures:_artifact_futures.pop(0).result()

def get_optimization_report(out,label):
    """Read the completed in-memory report while evidence is being persisted."""
    destination=out/f'{label}_optimization.json'
    if destination in _reports:return copy.deepcopy(_reports[destination])
    return json.loads(destination.read_text())

def optimize_path(planner,checker,path,names,limits,minimum_duration,out,label):
    policy=os.environ.get('GRASP_TRAJOPT_POLICY','historical')
    assert policy in ('interactive','first_valid','historical'),f'Unknown trajectory policy: {policy}'
    report={'method':'PRM seed + L-BFGS B-spline trajectory optimization','stage':label,'success':False,'attempts':[],
            'policy':policy,'phase_seconds':{'solve':0.,'validation':0.},
            'timing_scope':'trajectory optimization and original dense checks; prior IK/PRM and final evidence flush are separate',
            'artifact_persistence':'ordered asynchronous writes; flushed before run completion' if policy=='interactive' else 'synchronous'}
    destination=out/f'{label}_optimization.json'
    def write(path,value):
        if policy=='interactive':_artifact_futures.append(_artifact_writer.submit(path.write_text,value))
        else:path.write_text(value)
    def save():
        _reports[destination]=copy.deepcopy(report)
        write(destination,json.dumps(report,indent=2))
    started=time.monotonic()
    try:
        seed_path=path.detach().cpu().numpy()
        write(out/f'{label}_prm_seed.json',json.dumps(seed_path.tolist()))
        report['prm_seed_sha256']=hashlib.sha256(seed_path.tobytes()).hexdigest()
        horizon=planner.trajopt_solver.action_horizon
        seed=torch.nn.functional.interpolate(path.T[None],size=horizon,mode='linear',align_corners=True).transpose(1,2)
        # Explicit PRM seeds receive no random perturbation inside cuRobo. One
        # copy is sufficient for the first attempt; retain both historical
        # attempts if that solution fails ANY of the original acceptance gates.
        base_seed=seed[:,None]
        js=lambda q:JointState.from_position(q[None],joint_names=names)
        attempts=[(100,planner.trajopt_solver.config.num_seeds,3),(500,planner.trajopt_solver.config.num_seeds,3)]
        if policy in ('interactive','first_valid'):attempts.insert(0,(100,1,0))
        if policy=='interactive':attempts.insert(0,(25,1,0))
        for iterations,num_seeds,finetunes in attempts:
            attempt={'initial_iters':iterations,'num_seeds':num_seeds,'finetune_attempts':finetunes}
            report['attempts'].append(attempt);save()
            print('TRAJOPT_ATTEMPT',label,json.dumps(attempt),flush=True)
            seed=base_seed.repeat(1,num_seeds,1,1).contiguous()
            solve_started=time.monotonic()
            result=planner.trajopt_solver.solve_cspace(js(path[-1]),js(path[0]),seed_traj=seed,num_seeds=num_seeds,initial_iters=iterations,finetune_attempts=finetunes,finetune_dt_scale=.75)
            solved=bool(result.success.all())
            elapsed=time.monotonic()-solve_started
            attempt.update(solver_success=solved,solve_seconds=elapsed)
            report['phase_seconds']['solve']+=elapsed
            try:
                candidate=result.get_interpolated_plan().reorder(names).position.reshape(-1,len(names))
                attempt['candidate_max_limit_exceed_rad']=float(torch.maximum(limits[:,0]-candidate,candidate-limits[:,1]).clamp_min(0).max())
                attempt['candidate_goal_error_rad']=float((candidate[-1]-path[-1]).abs().max())
            except Exception as diagnostic_error:
                attempt['diagnostic_error']=str(diagnostic_error)
            save()
            if not solved:continue
            validation_started=time.monotonic()
            try:
                q,times,checks=_validate_trajectory(result,checker,path,names,limits,minimum_duration)
            except AssertionError as exc:
                attempt['validation_error']=str(exc)
                elapsed=time.monotonic()-validation_started
                attempt['validation_seconds']=elapsed;report['phase_seconds']['validation']+=elapsed
                save();continue
            elapsed=time.monotonic()-validation_started
            attempt.update(validation_seconds=elapsed,accepted=True,**checks)
            report['phase_seconds']['validation']+=elapsed
            report.update(success=True,**checks)
            return q,times
        raise RuntimeError('Trajectory optimizer rejected all attempts; no PRM-only fallback')
    except Exception as exc:
        report['error']=str(exc);raise
    finally:
        report['planning_wall_seconds']=time.monotonic()-started
        report['trajectory_optimization_wall_seconds']=report['planning_wall_seconds'];save()


def _validate_trajectory(result,checker,path,names,limits,minimum_duration):
    """Apply the unchanged endpoint, limit, dense collision and playback gates."""
    trajectory=result.get_interpolated_plan().reorder(names)
    q=trajectory.position.reshape(-1,len(names)).detach().clone()
    assert len(q)>1 and torch.isfinite(q).all(),'Invalid optimized positions'
    dt=float(torch.as_tensor(trajectory.dt).flatten()[0]);assert np.isfinite(dt) and dt>0
    violation=float(torch.maximum(limits[:,0]-q,q-limits[:,1]).clamp_min(0).max())
    report=dict(samples=len(q),native_dt_s=dt,max_limit_exceed_rad=violation,
                  start_error_rad=float((q[0]-path[0]).abs().max()),goal_error_rad=float((q[-1]-path[-1]).abs().max()))
    assert violation<=1e-6,'Optimized trajectory exceeds joint position limits'
    assert report['start_error_rad']<.005 and report['goal_error_rad']<.005,'Optimized endpoint mismatch'
    a=torch.arange(10,device=q.device)/10
    dense=torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,len(names)),q[-1:]])
    invalid=sum(int((~checker.validate(chunk[:,None])).sum()) for chunk in dense.split(64))
    report.update(dense_samples=len(dense),invalid_samples=invalid)
    assert invalid==0,'Optimized path failed dense scene/limit validation'
    maxima={}
    for field in ('velocity','acceleration','jerk'):
        v=getattr(trajectory,field)
        assert v is not None and torch.isfinite(v).all(),f'Invalid optimized {field}'
        maxima[field]=float(v.abs().max())
    # Preserve cuRobo's uniform time grid, stretching only if needed for controlled playback.
    native_duration=dt*(len(q)-1)
    scale=max(1.,minimum_duration/native_duration,maxima['velocity']/1.5,
              (maxima['acceleration']/5.)**.5,(maxima['jerk']/100.)**(1/3))
    # Also bound finite differences of the actual sampled position command.
    fd=q
    for order,bound in ((1,1.5),(2,5.),(3,100.)):
        fd=torch.diff(fd,dim=0)/dt
        peak=float(fd.abs().max()) if fd.numel() else 0.
        scale=max(scale,(peak/bound)**(1/order))
    duration=native_duration*scale
    assert duration<=120,'Optimized trajectory too long for execution'
    report.update(native_derivative_maxima=maxima,time_scale=scale,execution_duration_s=duration,
                  execution_derivative_maxima={k:v/scale**i for i,(k,v) in enumerate(maxima.items(),1)})
    return q,np.arange(len(q))*dt*scale,report
