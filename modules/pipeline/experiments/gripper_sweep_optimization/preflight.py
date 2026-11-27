"""Original native FK/mask comparisons, including an interior-only obstacle."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from arx_project.layout import read_json,write_json
from modules.curobo.dual_arm import accelerated_adapter as native
from modules.pipeline.experiments.dual_arm_grasp.benchmark_acceleration import serial_opening
from modules.curobo.gripper_sweep.adapter import batch_sweep,narrowing_canonical


def compare(pool,canonical,root):
    reference=serial_opening(pool,root/'serial',canonical)
    captured=[];original=native.validate_sphere_batch
    def capture(checker,spheres,q):
        captured.append(spheres.detach().cpu().numpy().copy())
        return original(checker,spheres,q)
    with patch.object(native,'validate_sphere_batch',capture):result=batch_sweep(pool,root/'batch',canonical)
    invalid=np.flatnonzero(~np.asarray(reference['valid']));first=int(invalid[0]) if len(invalid) else None
    if result.get('first_invalid_sample')!=first:raise RuntimeError('Opening first-invalid mismatch')
    error=0.
    if captured:
        error=float(np.abs(reference['spheres']-captured[0]).max())
        if (error>2e-7 or not np.array_equal(reference['spheres'][...,3],captured[0][...,3])
            or reference['valid']!=result['valid_per_sample']):raise RuntimeError('Opening native sphere/mask mismatch')
    elif first!=0:raise RuntimeError('Batch omitted noninitial samples')
    proof={'passed':True,'samples':len(reference['valid']),'maximum_sphere_error_m':error,
        'radii_exactly_equal':True,'first_invalid_sample':first,'serial_seconds':reference['wall_seconds'],
        'batch_seconds':result['wall_seconds'],'valid_per_sample':reference['valid']}
    write_json(root/'equivalence.json',proof)
    return proof,reference['spheres']


def run(planner,warmup,root):
    state=read_json(warmup/'state.json');reports=[];left=None
    for side in ['right','left']:
        canonical=planner.model.bind(state,side,root/side/'arm')
        observed={n:canonical['qpos'][canonical['joint_names'].index(n)] for n in ['right_joint27','right_joint28']}
        for name in observed:canonical['qpos'][canonical['joint_names'].index(name)]=.044
        canonical['planned_opening']={'target_per_finger_m':.044,'observed_solver_finger_positions_m':observed}
        proof,spheres=compare(planner.pool,canonical,root/side);reports.append({'case':side,**proof})
        if side=='left':left=(canonical,spheres)
    for side,width in [('right',.06),('left',.04)]:
        canonical=planner.model.bind(state,side,root/('narrow_'+side)/'arm')
        canonical=narrowing_canonical(canonical,width)
        proof,_=compare(planner.pool,canonical,root/('narrow_'+side));reports.append({'case':'narrow_'+side,**proof})
    canonical,spheres=left;mid=spheres[len(spheres)//2,0,:,:3];ends=spheres[[0,-1],0]
    margins=np.array([(np.linalg.norm(ends[...,:3]-p,axis=-1)-ends[...,3]).min() for p in mid])
    best=int(np.argmax(margins))
    if margins[best]<=.001:raise RuntimeError('No valid endpoint-clear interior collision fixture')
    obstructed=deepcopy(canonical);body=np.asarray(canonical['links']['body4'])
    point=mid[best]@body[:3,:3].T+body[:3,3]
    obstructed['scene_cuboids']['opening_probe']={'center':point.tolist(),'dims':[.001]*3}
    proof,_=compare(planner.pool,obstructed,root/'interior_obstacle')
    if proof['first_invalid_sample'] in [None,0,proof['samples']-1]:raise RuntimeError('Interior collision not exercised')
    reports.append({'case':'interior_obstacle',**proof})
    write_json(root/'acceptance.json',{'passed':True,'reports':reports,'physical_grasps':0,'simulation_instances':0,
        'same_samples_native_bounds_self_world_target_parked_arm':True})
    return reports
