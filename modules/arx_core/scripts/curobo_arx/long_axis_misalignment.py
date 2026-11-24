"""UNREGISTERED H018 proposal draft: correct long-axis closing on RGB-D shapes.

No simulator pose/dimensions or outcome are used in this geometry module.
Unchanged planners/checkers are passed in by a separately registered executor.
"""
import copy,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from sphere_refinement import PAD_CENTER_EEF


def condition(folder,selected):
    folder=Path(folder);read=lambda n:json.loads((folder/n).read_text())
    params=read('camera/params.json');k=params['intrinsics']
    depth=np.asarray(Image.open(folder/'camera/depth.png'),dtype=float)*params['depth_scale']
    mask=binary_erosion(np.asarray(Image.open(folder/'camera/mask.png'))>0,iterations=2)&(depth>0)
    y,x=np.where(mask);z=depth[y,x]
    if len(z)<150:return {'eligible':False,'reason':'Insufficient original masked RGB-D'}
    points=np.c_[(x-k['ppx'])*z/k['fx'],(y-k['ppy'])*z/k['fy'],z]
    camera=np.asarray(read('camera_world_optical.json'));points=points@camera[:3,:3].T+camera[:3,3]
    values,vectors=np.linalg.eigh(np.cov(points.T));major=vectors[:,-1]
    norm=float(np.linalg.norm(major[:2]))
    if norm<1e-9:return {'eligible':False,'reason':'Observed major axis vertical'}
    horizontal=np.r_[major[:2]/norm,0.]
    length=float(np.diff(np.quantile(points@horizontal,[.01,.99]))[0])
    ratio=float(values[-1]/values[-2])
    dot=abs(float(np.asarray(selected['grasp_world'])[:3,1]@horizontal))
    eligible=bool(abs(major[2])<=.35 and .10<=length<=.25 and ratio>=2. and dot>.85)
    return {'eligible':eligible,'major_world':major.tolist(),'horizontal_axis_world':horizontal.tolist(),
            'observed_length_m':length,'principal_eigenvalue_ratio':ratio,
            'old_selected_closing_abs_dot_axis':dot,
            'scope':'Observed shape/orientation proposal condition only, not grasp feasibility. Existing target isolation/calibration unchanged.'}


def proposals(selected,shape):
    axis=np.asarray(shape['horizontal_axis_world'])
    contact=np.asarray(selected['target_contact_world'])
    old_rotation=np.asarray(selected['grasp_world'])[:3,:3]
    preferred_direction=1. if float(axis@old_rotation[:,2])>=0 else -1.
    preferred_closing=1. if float(np.cross([0.,0.,1.],axis)@old_rotation[:,1])>=0 else -1.
    for pitch_deg in (45.,30.,60.):
        angle=np.deg2rad(pitch_deg)
        for direction in (preferred_direction,-preferred_direction):
            approach=direction*np.cos(angle)*axis+np.array([0.,0.,-np.sin(angle)])
            for flip in (preferred_closing,-preferred_closing):
                closing=flip*np.cross([0.,0.,1.],axis)
                rotation=np.column_stack([np.cross(closing,approach),closing,approach])
                assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10) and np.linalg.det(rotation)>0
                for height in (0.,.003,.006,.009,.012):
                    world=np.eye(4);world[:3,:3]=rotation
                    pad_target=contact+np.array([0.,0.,height])
                    world[:3,3]=pad_target-rotation@PAD_CENTER_EEF
                    pre=world.copy();pre[2,3]+=.08
                    lift=world.copy();lift[2,3]+=.12
                    yield world,pre,lift,{'hypothesis':'H018_cross_major_axis',
                        'method':'Perpendicular closing on observed elongated shape; preserve neural contact and frozen pad calibration',
                        'approach_down_pitch_deg':pitch_deg,'axis_approach_direction':direction,
                        'closing_flip':flip,'world_z_pose_search_offset_m':height,
                        'pad_target_world':pad_target.tolist(),'pad_center_eef_m_frozen':PAD_CENTER_EEF.tolist(),
                        'sampled_tracking_delta_rad':.003,'robot_configuration_changed':False,
                        'scope':'Geometry-inspired proposal, not learned rescore, exact paper reproduction or physical proof'}


def select(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=180.):
    """Additional proposal search; every original model/criterion stays fixed."""
    started=time.monotonic();goals=list(proposals(selected,shape))
    rejected={k:0 for k in ('grasp_ik','closed_goal','tracking_grasp','pre_lift_ik','closed_lift','tracking_lift','pre_route','approach_route')}
    report={'method':'H018 perpendicular-major proposal search','proposals':len(goals),'examined':0,
            'selected':False,'rejected':rejected,'time_cap_s':max_seconds,'shape':shape}
    def save():(out/'axis_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    save()
    for world,pre,lift,change in goals:
        if time.monotonic()-started>=max_seconds:report['timeout']=True;break
        report['examined']+=1
        qg=ik(world,start)
        if qg is None:rejected['grasp_ik']+=1;save();continue
        if not closed_checker.validate(qg[None,None]).all():rejected['closed_goal']+=1;save();continue
        if not tracking_valid(qg,'axis_grasp_goal'):rejected['tracking_grasp']+=1;save();continue
        qp=ik(pre,qg);ql=ik(lift,qg)
        if qp is None or ql is None:rejected['pre_lift_ik']+=1;save();continue
        if not closed_checker.validate(ql[None,None]).all():rejected['closed_lift']+=1;save();continue
        if not tracking_valid(ql,'axis_lift_goal'):rejected['tracking_lift']+=1;save();continue
        route=path(start,qp)
        if route is None:rejected['pre_route']+=1;save();continue
        if path(qp,qg) is None:rejected['approach_route']+=1;save();continue
        refined=copy.deepcopy(selected)
        refined.update(grasp_world=world.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),
                       q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),research_change=change)
        report.update(selected=True,elapsed_s=time.monotonic()-started);save()
        return refined,route
    report.update(elapsed_s=time.monotonic()-started);save()
    return None,None
