"""UNREGISTERED ring-tube proposals; only poses, no scene/configuration writes."""
import numpy as np

from sphere_refinement import PAD_CENTER_EEF

PRESHAPE_FINGER_M = .030


def proposals(rows, camera, fit):
    if not fit.get('accepted'): return []
    center=np.asarray(fit['center_world']);a,b=fit['ellipse_radii_m'];minor=fit['minor_radius_m'];yaw=fit['yaw_rad']
    c,s=np.cos(yaw),np.sin(yaw);ellipse_rotation=np.array([[c,-s],[s,c]])
    seed=None
    for index,row in enumerate(rows):
        if not np.isfinite(row).all():continue
        rg=camera[:3,:3]@row[4:13].reshape(3,3)
        contact=camera[:3,:3]@row[13:16]+camera[:3,3]+row[3]*rg[:,0]
        if rg[2,0]<-.35 and np.linalg.norm(contact-center)<.065:
            seed=(index,row,contact);break
    if seed is None:return []
    index,row,seed_contact=seed;local=ellipse_rotation.T@(seed_contact[:2]-center[:2]);theta0=np.arctan2(local[1]/b,local[0]/a)
    candidates=[]
    angles=[theta0+i*np.pi/6 for i in (0,1,-1,2,-2,3,-3,4,-4,5,-5,6)]
    angles.extend([0.,np.pi/2,np.pi,3*np.pi/2])
    angles=sorted(angles,key=lambda theta:abs(np.arctan2(np.sin(theta-theta0),np.cos(theta-theta0))))
    # Measured cross-sections of the unchanged CAD at original pad z=.0375.
    # This is an additional proposal proxy, not replacement collision geometry.
    pad_sections=[(-.0137213,.0017213,.0021776+PRESHAPE_FINGER_M,.0065160+PRESHAPE_FINGER_M),
                  (-.0137213,.0017213,-.0029441-PRESHAPE_FINGER_M,.0013945-PRESHAPE_FINGER_M)]
    for theta in angles:
        tube=center.copy();tube[:2]+=ellipse_rotation@np.array([a*np.cos(theta),b*np.sin(theta)])
        normal=ellipse_rotation@np.array([np.cos(theta)/a,np.sin(theta)/b]);normal/=np.linalg.norm(normal)
        for sign in (1.,-1.):
            closing=np.r_[normal*sign,0.];approach=np.array([0.,0.,-1.]);rotation=np.column_stack([np.cross(closing,approach),closing,approach])
            footprints=[]
            for x0,x1,y0,y1 in pad_sections:
                corners=np.array([[x,y] for x in (x0,x1) for y in (y0,y1)])-PAD_CENTER_EEF[:2]
                footprints.append(tube[:2]+corners@rotation[:2,:2].T)
            radii=[np.linalg.norm(corners-center[:2],axis=1) for corners in footprints]
            inner=int(np.argmin([r.mean() for r in radii]));outer=1-inner
            hole_clearance=float(min(a,b)-minor-radii[inner].max())
            exterior_clearance=float(radii[outer].min()-max(a,b)-minor)
            if min(hole_clearance,exterior_clearance)<.003:continue
            for extra_height in (0.,.003,.006,.009,.012):
                pad=tube.copy();pad[2]+=minor*.5+extra_height
                grasp=np.eye(4);grasp[:3,:3]=rotation;grasp[:3,3]=pad-rotation@PAD_CENTER_EEF
                pre=grasp.copy();pre[:3,3]-=.08*approach
                lift=grasp.copy();lift[2,3]+=.12
                candidates.append({'index':index,'network_row':row.tolist(),'score':float(row[0]),'width_m':float(row[1]),'depth_m':float(row[3]),
                    'grasp_world':grasp.tolist(),'pregrasp_world':pre.tolist(),'lift_world':lift.tolist(),'target_contact_world':pad.tolist(),
                    'research_change':{'hypothesis':'H013_ring','method':'Observed ellipse-ring local tube pinch, one finger in hole and one outside',
                        'ring_angle_rad':float(theta),'closing_sign':sign,'extra_robot_goal_height_m':extra_height,'tube_center_world':tube.tolist(),
                        'preshape_finger_joint_m':PRESHAPE_FINGER_M,'pad_center_eef_m_frozen':PAD_CENTER_EEF.tolist(),
                        'pad_section_hole_proxy_clearance_m':hole_clearance,'pad_section_exterior_proxy_clearance_m':exterior_clearance,
                        'proxy_scope':'Pad-plane cross-section proxy only, not full target collision/contact proof',
                        'network_score_scope':'Score belongs to original neural seed, not the refined pose','network_seed_contact_world':seed_contact.tolist(),
                        'sampled_tracking_delta_rad':.003,'robot_configuration_changed':False}})
    return candidates


def select(rows,camera,fit,start,ik,path,closed_checker,tracking_valid,out,started):
    """Normal bounded policy search; all fixed planner/checker functions supplied."""
    import json,time
    candidates=proposals(rows,camera,fit)
    report={'method':'H013 ring-tube proposals','proposals':len(candidates),'examined':0,'rejected':{},'selected':False}
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
