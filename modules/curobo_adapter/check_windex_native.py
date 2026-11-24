"""Frozen native collision validation and pose-tolerance collision certificate."""
from pathlib import Path
import json
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from adapter import HERE, ARXPlanner, NAMES, save
from probe import capture
from windex_reachability import AnalyticARX, URDF
from audit_collision_cover import link_frames
from audit_runtime_colliders import runtime_geometry

ROOT=HERE/'research/20261117_06'


def main():
    spec=json.loads((ROOT/'native_registration.json').read_text())
    for name,h in spec['hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    data=json.loads((ROOT/'analytic_results.json').read_text());source=Path(spec['input'])
    state=json.loads((source/'input/state.json').read_text());cfg=yaml.safe_load((source/'model/arx.yml').read_text())['robot_cfg']['kinematics'];bounds=np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES]);target=np.asarray(data['target_in_base']);R=Rotation.from_matrix(target[:3,:3]).as_matrix();target[:3,:3]=R
    actual=np.asarray([state['qpos'][state['joint_names'].index(n)] for n in NAMES]);analytic=AnalyticARX();queries=[{'kind':'measured_control','q':actual.tolist()}]
    queries += [{'kind':'exact_analytic_root','q':row['q']} for row in data['roots']]
    rng=np.random.default_rng(171107)
    for i in range(24):
        dp=rng.normal(size=3);dp=dp/np.linalg.norm(dp)*.00195
        dr=rng.normal(size=3);dr=dr/np.linalg.norm(dr)*.029
        perturbed=target.copy();perturbed[:3,3]+=dp;perturbed[:3,:3]=Rotation.from_rotvec(dr).as_matrix()@R
        roots=analytic.roots(perturbed,bounds)
        for row in roots['roots']:
            queries.append({'kind':'inside_original_pose_tolerance','q':row['q']})
    assert len(queries)<=spec['maximum_witness_checks']
    planner=ARXPlanner(source/'input',ROOT/'native_model')
    try:
        q=torch.tensor([x['q'] for x in queries],device='cuda',dtype=torch.float32)
        kin=planner.checker.get_kinematics(q[:,None]);spheres=kin.robot_spheres.detach().cpu().numpy().reshape(len(queries),-1,4)
        valid=planner.checker.validate(q[:,None]).detach().cpu().numpy().reshape(-1)
        collision=planner.checker.get_collision_constraint(kin).detach().cpu().numpy().reshape(len(queries),-1)
        bound=planner.checker.get_bound(q[:,None]).detach().cpu().numpy().reshape(len(queries),-1)
        positions=kin.tool_poses.position.detach().cpu().numpy().reshape(len(queries),-1,3)[:,0]
        quats=kin.tool_poses.quaternion.detach().cpu().numpy().reshape(len(queries),-1,4)[:,0]
        rows=[];max_sphere_error=0.
        for i,query in enumerate(queries):
            values=dict(cfg['lock_joints']);values.update(dict(zip(NAMES,np.asarray(query['q'],dtype=np.float32).astype(float))));frames=link_frames(values);own=[];links=[]
            for link in cfg['collision_link_names']:
                for s in cfg['collision_spheres'][link]:
                    own.append([*(frames[link][:3,:3]@np.asarray(s['center'])+frames[link][:3,3]),s['radius']]);links.append(link)
            max_sphere_error=max(max_sphere_error,float(np.max(np.abs(np.asarray(own)-spheres[i]))))
            assert max_sphere_error<2e-6
            p_error=float(np.linalg.norm(positions[i]-target[:3,3]));quat=quats[i];rot=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix();r_error=float(Rotation.from_matrix(rot@R.T).magnitude())
            row={**query,'native_valid':bool(valid[i]),'bound_cost':float(bound[i].sum()),
                 'position_error_m':p_error,'orientation_error_rad':r_error,
                 'colliding_spheres':np.flatnonzero(collision[i]>0).tolist(),
                 'colliding_links':sorted({links[j] for j in np.flatnonzero(collision[i]>0)}),
                 'collision_cost':float(collision[i].sum())}
            if query['kind']!='measured_control':
                assert p_error<.002 and r_error<.03 and bound[i].sum()==0
            rows.append(row)
        assert rows[0]['native_valid']
        assert all(not row['native_valid'] for row in rows[1:])
        np.savez(ROOT/'native_witness_arrays.npz',q=q.cpu().numpy(),spheres=spheres,collision=collision,bound=bound,valid=valid)
        save(ROOT/'native_witnesses.json',{'records':rows,'native_checker_queries':len(rows),'independent_sphere_crosscheck_max_error_m':max_sphere_error,'physical_trials':0})
        # Preserve the native result stages for a measured-start query and a
        # hypothetical root-start query. These are model queries, never motion.
        solver=planner.planner.ik_solver;native=solver._get_result;phases=[]
        def observe(opt,metrics,n,ret,*a,**kw):
            result=native(opt,metrics,n,ret,*a,**kw);phases.append(capture(opt,metrics,result,solver.config));return result
        solver._get_result=observe
        try:
            goal=np.asarray(json.loads((source/'calls.jsonl').read_text().splitlines()[-1])['goal'])
            records=[]
            for name,start in [('actual_measured_start',actual),('hypothetical_exact_root_start',np.asarray(data['roots'][0]['q']))]:
                phases.clear();record={'kind':name,'q_start':start.tolist(),'goal':goal.tolist()}
                with (ROOT/'native_requests.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                planner.planner.reset_seed();answer,result=planner.ik(goal,torch.tensor(start,device='cuda',dtype=torch.float32))
                record.update(accepted=answer is not None,native_phases=phases.copy())
                with (ROOT/'native_calls.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                records.append({'kind':name,'accepted':answer is not None})
            assert all(not r['accepted'] for r in records)
        finally:del solver._get_result
        save(ROOT/'native_summary.json',{'native_witness_checks':len(rows),'kinematic_pose_qualified':len(rows)-1,'collision_free_target_witnesses':sum(r['native_valid'] for r in rows[1:]),'measured_control_passed':True,'native_ik_queries':len(records),'native_ik':records,'physical_trials':0,'cloud_requests':0})
    finally:planner.planner.destroy()
    # A material point belonging to a distal rigid link has the same EEF-local
    # coordinates for every arm branch. Its table SDF is 1-Lipschitz. Thus
    # penetration > translation_limit + 2*sin(angle_limit/2)*lever + slack
    # proves collision for the entire accepted pose region, not just samples.
    values=dict(cfg['lock_joints']);values.update(dict(zip(NAMES,data['roots'][0]['q'])));frames=link_frames(values)
    scene=json.loads((source/'model/scene.json').read_text());proofs=[]
    spheres_proof=[];offset=0
    for link in cfg['collision_link_names']:
        for sphere in cfg['collision_spheres'][link]:
            index=offset;offset+=1
            if link not in ('right_link26','right_link27','right_link28'):continue
            point=frames[link][:3,:3]@sphere['center']+frames[link][:3,3];local=R.T@(point-target[:3,3]);budget=.002+2*np.sin(.03/2)*np.linalg.norm(local)+.0001
            box=scene['cuboid']['table'];bp=box['pose'];br=Rotation.from_quat([*bp[4:],bp[3]]).as_matrix();d=np.abs((point-bp[:3])@br)-np.asarray(box['dims'])/2;sdf=np.linalg.norm(np.maximum(d,0))+min(d.max(),0)
            guarantee=float(sphere['radius']-sdf-budget)
            if guarantee>0:spheres_proof.append({'link':link,'sphere_index':index,'center_eef':local.tolist(),'radius_m':sphere['radius'],'nominal_overlap_m':float(sphere['radius']-sdf),'maximum_displacement_m':float(budget),'guaranteed_collision_margin_m':guarantee})
    geometry=runtime_geometry(HERE.parent/'maniskill/assets/arx')
    for link in ('right_link26','right_link27','right_link28'):
        for part,vertices in enumerate(geometry[link]):
            points=vertices@frames[link][:3,:3].T+frames[link][:3,3];local=(points-target[:3,3])@R;budget=.002+2*np.sin(.03/2)*np.linalg.norm(local,axis=1)+.0001
            box=scene['cuboid']['table'];bp=box['pose'];br=Rotation.from_quat([*bp[4:],bp[3]]).as_matrix();d=np.abs((points-bp[:3])@br)-np.asarray(box['dims'])/2;sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0);guarantee=-sdf-budget
            for i in np.flatnonzero(guarantee>0):proofs.append({'link':link,'part':part,'source_vertex_link':vertices[i].tolist(),'source_vertex_eef':local[i].tolist(),'nominal_penetration_m':float(-sdf[i]),'maximum_displacement_m':float(budget[i]),'guaranteed_penetration_m':float(guarantee[i])})
    assert spheres_proof and proofs
    witness=proofs[0];invariance_error=0.
    for q in rng.uniform(bounds[:,0],bounds[:,1],size=(128,6)):
        values=dict(cfg['lock_joints']);values.update(dict(zip(NAMES,q)));f=link_frames(values);eef,_=analytic.reference.fk(q);point=f[witness['link']][:3,:3]@witness['source_vertex_link']+f[witness['link']][:3,3];local=eef[:3,:3].T@(point-eef[:3,3]);invariance_error=max(invariance_error,float(np.max(np.abs(local-witness['source_vertex_eef']))))
    assert invariance_error<1e-12
    save(ROOT/'tolerance_certificate.json',{'position_tolerance_m':.002,'orientation_tolerance_rad':.03,'conservative_arithmetic_slack_m':.0001,
        'method':'Rigid EEF-local sphere/vertex; displacement bound eps_p+2 sin(eps_R/2)*norm(local); 1-Lipschitz signed distance to frozen tabletop box',
        'requires_measured_finger_state_unchanged':True,'source_vertex_branch_invariance_max_m':invariance_error,
        'sphere_certificates':spheres_proof,'native_source_vertex_certificates':proofs,
        'native_source_scope':'Existing exported convex-piece source geometry; no PhysX runtime or contact simulation was invoked',
        'conclusion':'Every EEF pose within the original motion acceptance tolerances intersects the frozen table with this open gripper; endpoint is collision-infeasible regardless of IK branch or path',
        'physical_trials':0})
    print('PROVED target collision across full original pose-tolerance region',len(spheres_proof),'spheres',len(proofs),'source-part vertices',flush=True)


if __name__=='__main__':main()
