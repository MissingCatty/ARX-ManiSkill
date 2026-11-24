"""UNREGISTERED tilted ring poses; fixed CAD quality proxy, no model/property edits."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sphere_refinement import PAD_CENTER_EEF

PROXY=Path(__file__).resolve().parents[2]/'runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.npz'
TARGET_CLEARANCE_M=.003
INNER_CENTER_OFFSET_M={1.:.000774802761625486,-1.:.004346792740823248}

def proposals(rows,camera,fit):
    if not fit.get('accepted'):return []
    meta=json.loads(PROXY.with_suffix('.json').read_text());assert hashlib.sha256(PROXY.read_bytes()).hexdigest()==meta['data_sha256']
    with np.load(PROXY) as f:parts={n:f[n].copy() for n in f.files}
    radius_cover=meta['sample_cover_radius_m'];required=TARGET_CLEARANCE_M+radius_cover
    center=np.asarray(fit['center_world']);a,b=fit['ellipse_radii_m'];minor=fit['minor_radius_m'];yaw=fit['yaw_rad']
    c,s=np.cos(yaw),np.sin(yaw);ellipse_rotation=np.array([[c,-s],[s,c]])
    seed=None
    for index,row in enumerate(rows):
        if not np.isfinite(row).all():continue
        rg=camera[:3,:3]@row[4:13].reshape(3,3);contact=camera[:3,:3]@row[13:16]+camera[:3,3]+row[3]*rg[:,0]
        if rg[2,0]<-.35 and np.linalg.norm(contact-center)<.065:seed=(index,row,contact,rg[:,0]);break
    if seed is None:return []
    index,row,seed_contact,seed_approach=seed;xy=ellipse_rotation.T@(seed_contact[:2]-center[:2]);theta0=np.arctan2(xy[1]/b,xy[0]/a)
    angles=[theta0+i*np.pi/6 for i in (0,1,-1,2,-2,3,-3,4,-4,5,-5,6)]+[0.,np.pi/2,np.pi,3*np.pi/2]
    angles.sort(key=lambda theta:abs(np.arctan2(np.sin(theta-theta0),np.cos(theta-theta0))))
    parameter=np.arange(1440)*2*np.pi/1440;curve=center[:2]+np.column_stack([a*np.cos(parameter),b*np.sin(parameter)])@ellipse_rotation.T
    tree=cKDTree(curve);curve_error=max(a,b)*np.pi/1440
    def swept_clearance(points,rotation,translation):
        world=points@rotation.T+translation;rho=np.linalg.norm(world[:,:2]-center[:2],axis=1)
        dz=np.maximum.reduce([world[:,2]-center[2],center[2]-(world[:,2]+.08),np.zeros(len(world))])
        radial_lb=np.maximum.reduce([min(a,b)-rho,rho-max(a,b),np.zeros(len(rho))])
        lower=np.maximum(dz,radial_lb)-minor;mask=lower<required
        if mask.any():
            d,_=tree.query(world[mask,:2],workers=1);d=np.maximum(0,d-curve_error)
            lower[mask]=np.sqrt(d*d+dz[mask]*dz[mask])-minor
        return float(lower.min())
    seed_pitch=np.rad2deg(np.arcsin(np.clip(-seed_approach[2],0,1)))
    pitches=sorted([75.,60.,45.,30.],key=lambda value:abs(value-seed_pitch))
    candidates=[]
    for theta in angles:
        tube=center.copy();tube[:2]+=ellipse_rotation@np.array([a*np.cos(theta),b*np.sin(theta)])
        n=ellipse_rotation@np.array([np.cos(theta)/a,np.sin(theta)/b]);n/=np.linalg.norm(n)
        support=float(n@(tube[:2]-center[:2]))
        for sign in (1.,-1.):
            desired=support-INNER_CENTER_OFFSET_M[sign]
            if not 0<desired<.04399999976158142:continue
            points=np.concatenate([part+desired*np.asarray(meta['motion_axis_by_link'][name]) for name,part in parts.items()])
            closing=np.r_[n*sign,0.];tangent=np.cross(closing,[0,0,-1.]);directions=sorted([1.,-1.],key=lambda value:-float((value*tangent)@seed_approach))
            for pitch in pitches:
                for direction in directions:
                    approach=np.cos(np.deg2rad(pitch))*direction*tangent+np.array([0,0,-np.sin(np.deg2rad(pitch))])
                    rotation=np.column_stack([np.cross(closing,approach),closing,approach])
                    for extra_height in (0.,.003,.006,.009,.012):
                        pad=tube.copy();pad[2]+=minor*.5+extra_height
                        grasp=np.eye(4);grasp[:3,:3]=rotation;grasp[:3,3]=pad-rotation@PAD_CENTER_EEF
                        clearance=swept_clearance(points,rotation,grasp[:3,3])
                        if clearance<required:continue
                        pre=grasp.copy();pre[2,3]+=.08
                        lift=grasp.copy();lift[2,3]+=.12
                        candidates.append({'index':index,'network_row':row.tolist(),'score':float(row[0]),'width_m':float(row[1]),'depth_m':float(row[3]),
                            'grasp_world':grasp.tolist(),'pregrasp_world':pre.tolist(),'lift_world':lift.tolist(),'target_contact_world':pad.tolist(),
                            'research_change':{'hypothesis':'H014_ring_tilt','method':'Tilted local ring pinch with intended vertical entry',
                                'ring_angle_rad':float(theta),'closing_sign':sign,'approach_down_pitch_deg':pitch,'approach_tangent_sign':direction,
                                'extra_robot_goal_height_m':extra_height,'tube_center_world':tube.tolist(),'preshape_finger_joint_m':desired,
                                'pad_center_eef_m_frozen':PAD_CENTER_EEF.tolist(),'minimum_swept_CAD_proxy_clearance_m':clearance,
                                'required_proxy_point_clearance_m':required,'object_selection_clearance_m':TARGET_CLEARANCE_M,'proxy_sample_cover_radius_m':radius_cover,
                                'proxy_scope':'Extra fixedCAD sampled-volume quality proxy against observed torus and intended vertical sweep; not the cuRobo/PhysX model, no property edits, no actual contact/feasibility guarantee',
                                'network_score_scope':'Original neural seed only, not the refined pose','network_seed_contact_world':seed_contact.tolist(),
                                'sampled_tracking_delta_rad':.003,'robot_configuration_changed':False}})
    return candidates

def select(rows,camera,fit,start,ik,path,closed_checker,tracking_valid,out,started):
    """Normal bounded policy search; all fixed planner/checker functions supplied."""
    import json,time
    candidates=proposals(rows,camera,fit)
    report={'method':'H014 tilted ring proposals','proposals':len(candidates),'examined':0,'rejected':{},'selected':False}
    def save(): (out/'ring_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    def reject(key):report['rejected'][key]=report['rejected'].get(key,0)+1;save()
    for index,proposal in enumerate(candidates):
        if time.monotonic()-started>180:report['timeout']=True;break
        report['examined']=index+1
        qg=ik(np.asarray(proposal['grasp_world']),start)
        if qg is None:reject('grasp_ik');continue
        if not closed_checker.validate(qg[None,None]).all() or not tracking_valid(qg,'ring_grasp_goal'):reject('closed_grasp_goal');continue
        qp=ik(np.asarray(proposal['pregrasp_world']),qg);ql=ik(np.asarray(proposal['lift_world']),qg)
        if qp is None or ql is None:reject('pre_or_lift_ik');continue
        if not closed_checker.validate(ql[None,None]).all() or not tracking_valid(ql,'ring_lift_goal'):reject('closed_lift_goal');continue
        route=path(start,qp)
        if route is None:reject('pregrasp_route');continue
        if path(qp,qg) is None:reject('approach_route');continue
        proposal.update(q_pre=qp.tolist(),q_grasp=qg.tolist(),q_lift=ql.tolist(),table_clearance_raise_m=None)
        report.update(selected=True,selected_proposal=index);save()
        return proposal,route
    save();return None,None
