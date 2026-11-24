"""H022 ordinary pre/lift waypoint targets on observed body-grasp candidates.

No robot calibration, configuration, model, physics or criterion modifications.
"""
import copy,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from axis_body_center import select as select_previous,condition
from long_axis_misalignment import proposals
from palm_target_quality import PalmTargetQuality
from sphere_refinement import PAD_CENTER_EEF


def select(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=180.):
    retained,retained_route=select_previous(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds)
    report={'hypothesis':'H022_body_waypoints','selected':False,'queries':[], 'rejected':{k:0 for k in ('palm_quality','grasp_ik','closed_grasp','tracking_grasp','pre_ik','lift_ik','closed_lift','tracking_lift','pre_route','approach_route')}}
    def save():(out/'waypoint_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    save()
    if retained is None or retained.get('research_change',{}).get('hypothesis')=='H021_body_center':
        report['reason']='No retained complete proposal or retained H021 body candidate has priority';save();return retained,retained_route
    trigger=condition(out,retained,shape);report['trigger']=trigger;save()
    if not trigger['eligible']:return retained,retained_route
    quality=PalmTargetQuality(out);axis=np.asarray(shape['horizontal_axis_world']);source=np.asarray(selected['target_contact_world']);started=time.monotonic()
    report.update(time_cap_s=max_seconds,examined=0)
    for base_index,(original,unused_pre,unused_lift,base_change) in enumerate(proposals(selected,shape)):
        for fraction in (0.,-.125,.125):
            if time.monotonic()-started>=max_seconds:report.update(timeout=True,elapsed_s=time.monotonic()-started);save();return retained,retained_route
            report['examined']+=1;shift=trigger['observed_axis_midpoint_m']+fraction*trigger['observed_axis_length_m']-float(source@axis)
            world=original.copy();world[:3,3]+=shift*axis;palm=quality.evaluate(world)
            if not palm['pass']:report['rejected']['palm_quality']+=1;save();continue
            qg=ik(world,start.detach().clone())
            if qg is None:report['rejected']['grasp_ik']+=1;save();continue
            if not closed_checker.validate(qg[None,None]).all():report['rejected']['closed_grasp']+=1;save();continue
            if not tracking_valid(qg,'waypoint_grasp_goal'):report['rejected']['tracking_grasp']+=1;save();continue
            pad=world[:3,3]+world[:3,:3]@PAD_CENTER_EEF
            rotated=world[:3,:3]@Rotation.from_euler('y',base_change['approach_down_pitch_deg']-30.,degrees=True).as_matrix()
            assert np.allclose(rotated[:,1],world[:3,1])
            qp=None;item={'base_pose_index':base_index,'axial_body_fraction':fraction,'pre_queries':[],'lift_queries':[]};report['queries'].append(item)
            for label,height,rotation in [('pre_up8',.08,world[:3,:3]),('pre_up5',.05,world[:3,:3]),('pre_up3',.03,world[:3,:3]),('pre_up8_approach30',.08,rotated)]:
                if time.monotonic()-started>=max_seconds:break
                pre=world.copy();pre[:3,:3]=rotation;pre[:3,3]=pad+np.array([0.,0.,height])-rotation@PAD_CENTER_EEF
                qp=ik(pre,qg.detach().clone());item['pre_queries'].append({'label':label,'admitted':qp is not None})
                if qp is not None:pre_label=label;break
                report['rejected']['pre_ik']+=1;save()
            if qp is None:continue
            for orientation,rotation in [('original_orientation',world[:3,:3]),('approach30',rotated)]:
                for height in (.12,.09,.08):
                    if time.monotonic()-started>=max_seconds:break
                    lift=world.copy();lift[:3,:3]=rotation;lift[:3,3]=pad+np.array([0.,0.,height])-rotation@PAD_CENTER_EEF
                    ql=ik(lift,qg.detach().clone());query={'orientation':orientation,'pad_lift_height_m':height,'original_IK_admitted':ql is not None};item['lift_queries'].append(query)
                    if ql is None:report['rejected']['lift_ik']+=1;save();continue
                    if not closed_checker.validate(ql[None,None]).all():report['rejected']['closed_lift']+=1;save();continue
                    if not tracking_valid(ql,'waypoint_lift_goal'):report['rejected']['tracking_lift']+=1;save();continue
                    route=path(start,qp)
                    if route is None:report['rejected']['pre_route']+=1;save();continue
                    if path(qp,qg) is None:report['rejected']['approach_route']+=1;save();continue
                    change=copy.deepcopy(base_change);change.update(hypothesis='H022_body_waypoints',method='Retained observed body contact plus alternative ordinary pre/lift motion targets; fixed world closing axis/PAD calibration',axial_body_fraction=fraction,axial_contact_target_shift_m=shift,source_neural_contact_world=selected['target_contact_world'],pad_target_world=pad.tolist(),palm_target_quality=palm,body_trigger=trigger,pre_waypoint_label=pre_label,lift_orientation_label=orientation,commanded_pad_lift_height_m=height,commanded_lift_height_m=height,EEF_origin_lift_height_m=float(lift[2,3]-world[2,3]),physical_success_threshold_unchanged_m=.06,physical_hold_duration_unchanged_s=3.,robot_configuration_changed=False)
                    refined=copy.deepcopy(selected);refined.update(grasp_world=world.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),research_change=change)
                    report.update(selected=True,selected_change=change,elapsed_s=time.monotonic()-started);save();return refined,route
    report['elapsed_s']=time.monotonic()-started;save();return retained,retained_route
