"""Prospective SE(3) sampling and conservative ARX finger sweep calculations.

Pure CPU geometry. No generated observation or physical rollout is represented
by these poses. The existing robot/collision configuration remains unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor35_diagnose import OLD, Evidence, dump, mesh_point_transform


def symmetric_pair(grasp, lateral_m, rotation_rad, axial_m, axial_jitter_m=0.):
    """Specify opposites in LABEL coordinates, then E = G @ inv(residual).

    Positive approach distance is kept. Only lateral/rotation and centered axial
    jitter have zero mean; absolute approach residual Z cannot be forced to zero.
    """
    g=np.asarray(grasp,dtype=float)
    xy=np.asarray(lateral_m,dtype=float);r=np.asarray(rotation_rad,dtype=float)
    if g.shape!=(4,4) or xy.shape!=(2,) or r.shape!=(3,):raise ValueError('Invalid pose shape')
    if axial_m-abs(axial_jitter_m)<=0:raise ValueError('Pair crosses the grasp plane')
    if not np.isfinite(np.r_[g.ravel(),xy,r,axial_m,axial_jitter_m]).all():raise ValueError('Nonfinite pair')
    result=[]
    for sign in (1,-1):
        residual=np.eye(4);residual[:3,:3]=Rotation.from_rotvec(sign*r).as_matrix()
        residual[:3,3]=np.r_[sign*xy,axial_m+sign*axial_jitter_m]
        entry=g@np.linalg.inv(residual)
        result.append({'sign':sign,'entry':entry,'residual':residual})
    return result


def separated_sweep(current, dp, dr, fingers_local, object_world, clearance_m=.010, samples=61):
    """Separating plane normal is -EE_Z at H, fixed during Step1.

    Sample a translation-in-plane/rotation sweep; subtract a rotational Lipschitz
    bound for between-sample motion. This is an explicit Cartesian finger-mesh
    certificate, not a cuRobo path, collision mesh, full-arm, or execution proof.
    """
    e=np.asarray(current);dp=np.asarray(dp).copy();dp[2]=0.;dr=np.asarray(dr)
    f=np.asarray(fingers_local);obj=np.asarray(object_world)
    n=-e[:3,2];h=float((obj@n).max());gaps=[]
    for t in np.linspace(0,1,samples):
        rot=e[:3,:3]@Rotation.from_rotvec(t*dr).as_matrix()
        position=e[:3,3]+e[:3,:3]@(t*dp)
        gaps.append(float(((f@rot.T+position)@n).min()-h))
    radius=float(np.linalg.norm(f,axis=1).max())
    between=radius*float(np.linalg.norm(dr))/(2*(samples-1))
    lower=min(gaps)-between
    return {'normal_world':n.tolist(),'initial_gap_mm':gaps[0]*1000,
        'sampled_minimum_gap_mm':min(gaps)*1000,'between_sample_bound_mm':between*1000,
        'certified_visual_finger_gap_mm':lower*1000,'required_outward_shift_mm':max(0.,clearance_m-lower)*1000,
        'clearance_mm':clearance_m*1000,'samples':samples,'full_robot_collision_checked':False,
        'measured_execution':False}


def main(out):
    from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
    from training.nearfield_arx.capture_review import finger_geometry
    from .advisor32_rules import points_in_frame,parse_prediction
    out=Path(out).resolve();target=out/'two_step_sweep_design.json'
    if target.exists():raise FileExistsError('Preserve design calculations')
    ev=Evidence();ev.pin(__file__);reg=ev.read(OLD/'registration.json')
    first=ev.read(out/'first_inputs.json')['requests'];trials={r['trial_id']:r for r in reg['seed_trials']}
    fingers=finger_geometry(Path(reg['learned_policy']['robot_usd']));meshes={};results=[]
    for d in first:
        req=ev.read(d['request_file']);row=ev.read(req['observation']);s=row['snapshot'];E=np.asarray(s['link_world']['right_eef_link'])
        if req['asset'] not in meshes:meshes[req['asset']]=composed_mesh(req['asset'])[0]
        world=mesh_point_transform(meshes[req['asset']].vertices,s['objects']['test_object']['world'])
        local=np.concatenate([points_in_frame(v,s['link_world'][name],E) for name,(v,f) in fingers.items()])
        dp,dr=parse_prediction('D',d['prediction'])
        result=separated_sweep(E,dp,dr,local,world)
        ep=Path(d['request_file']).parents[1];snap=ev.read(ep/'capture_snapshot.json')
        G=np.asarray(s['objects']['test_object']['world'])@np.linalg.inv(np.asarray(snap['objects']['test_object']['world']))@np.asarray(trials[d['trial_id']]['source_Gstar_world'])
        residual=np.linalg.inv(E)@G
        results.append({'trial_id':d['trial_id'],**result,
            'offline_reference_residual_mm_deg':np.r_[residual[:3,3]*1000,Rotation.from_matrix(residual[:3,:3]).as_rotvec(degrees=True)].tolist()})
    # Algebraic checks use actual ARX trial transforms, with no robot setter.
    errors=[]
    for trial in trials.values():
        G=np.asarray(trial['source_Gstar_world'])
        pair=symmetric_pair(G,[.020,-.010],np.deg2rad([5.,-5.,15.]),.050,.005)
        recovered=[np.linalg.inv(p['entry'])@G for p in pair]
        for p,r in zip(pair,recovered):
            errors.append(float(abs(p['residual']-r).max()))
        assert np.allclose(recovered[0][:2,3]+recovered[1][:2,3],0,atol=1e-12)
        assert np.allclose(sum(Rotation.from_matrix(r[:3,:3]).as_rotvec() for r in recovered),0,atol=1e-12)
        assert np.isclose(sum(r[2,3] for r in recovered),.100)
    dump(target,{'first_motion_designs':results,'algebra_actual_ARX_reference_poses':len(trials),
        'maximum_residual_reconstruction_error':max(errors),'new_motion_goals_executed':0,
        'scope':'Counterfactual lateral-only Cartesian sweep of frozen visual finger meshes; requires full frozen collision-model path checks before execution.',
        'evidence_files':ev.files})
    print('CPU geometry design complete: '+str(target))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
