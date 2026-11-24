"""Experimental H021 observed axial-body contact proposals under frozen models.

Normal quality is an exploration trigger only, NOT an antipodal/force certificate.
All original safety, IK, collision and tracking checks stay authoritative.
"""
import copy,json,time
import numpy as np
from axis_palm_quality import select as select_previous
from contact_normal_quality import ObservedContactQuality
from palm_target_quality import PalmTargetQuality
from long_axis_misalignment import proposals


def condition(folder,selected,shape):
    quality=ObservedContactQuality(folder);axis=np.asarray(shape['horizontal_axis_world'])
    extent=np.quantile(quality.palm.points@axis,[.01,.99]);centre=float(extent.mean());length=float(extent[1]-extent[0])
    contact=np.asarray(selected['target_contact_world']);fraction=2*(float(contact@axis)-centre)/length
    normal=quality.evaluate_normals(np.asarray(selected['grasp_world']))
    eligible=abs(fraction)>.65 and normal['visible_tip_ROI_points']>=100 and normal['median_abs_surface_normal_dot_closing']<.8
    return {'eligible':bool(eligible),'observed_axis_midpoint_m':centre,'observed_axis_length_m':length,'original_neural_axial_fraction':fraction,'normal_diagnostic':normal,'scope':'Near-end contact and weak visible normal alignment trigger alternate ordinary EEF contact targets, not physical force/model changes or contact proof.'}


def select(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=180.):
    retained,retained_route=select_previous(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds)
    if retained is None:
        report={'selected':False,'reason':'Retained H020 lacks complete candidate; no trigger assessed'}
        (out/'body_candidate_diagnostics.json').write_text(json.dumps(report,indent=2));return retained,retained_route
    trigger=condition(out,retained,shape)
    report={'hypothesis':'H021_body_center','selected':False,'trigger':trigger,'phases':[], 'queries':[], 'rejected':{k:0 for k in ('palm_quality','grasp_ik','closed_grasp','tracking_grasp','pre_ik','lift_ik','closed_lift','tracking_lift','pre_route','approach_route')}}
    def save():(out/'body_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    save()
    if not trigger['eligible']:return retained,retained_route
    quality=PalmTargetQuality(out);axis=np.asarray(shape['horizontal_axis_world']);source=np.asarray(selected['target_contact_world'])
    goals=[]
    for world,pre,lift,change in proposals(selected,shape):
        for fraction in (0.,-.125,.125):
            shift=trigger['observed_axis_midpoint_m']+fraction*trigger['observed_axis_length_m']-float(source@axis)
            goal=world.copy();goal[:3,3]+=shift*axis
            approach=goal.copy();approach[2,3]+=.08
            goals.append((goal,approach,copy.deepcopy(change),fraction,shift))
    report['proposals']=len(goals)
    for phase,heights in [('original_12cm',(.12,)),('adaptive_endpoint',(.12,.10,.09,.085,.08))]:
        started=time.monotonic();stage={'name':phase,'examined':0,'time_cap_s':max_seconds};report['phases'].append(stage)
        for index,(world,pre,change,fraction,shift) in enumerate(goals):
            if time.monotonic()-started>=max_seconds:stage['timeout']=True;break
            stage['examined']+=1;palm=quality.evaluate(world)
            if not palm['pass']:report['rejected']['palm_quality']+=1;save();continue
            qg=ik(world,start.detach().clone())
            if qg is None:report['rejected']['grasp_ik']+=1;save();continue
            if not closed_checker.validate(qg[None,None]).all():report['rejected']['closed_grasp']+=1;save();continue
            if not tracking_valid(qg,'body_grasp_goal'):report['rejected']['tracking_grasp']+=1;save();continue
            qp=ik(pre,qg.detach().clone())
            if qp is None:report['rejected']['pre_ik']+=1;save();continue
            for height in heights:
                if time.monotonic()-started>=max_seconds:stage['timeout']=True;break
                lift=world.copy();lift[2,3]+=height
                item={'phase':phase,'pose_index':index,'lift_height_m':height};report['queries'].append(item)
                ql=ik(lift,qg.detach().clone());item['original_IK_admitted']=ql is not None
                if ql is None:report['rejected']['lift_ik']+=1;save();continue
                if not closed_checker.validate(ql[None,None]).all():report['rejected']['closed_lift']+=1;save();continue
                if not tracking_valid(ql,'body_lift_goal'):report['rejected']['tracking_lift']+=1;save();continue
                route=path(start,qp)
                if route is None:report['rejected']['pre_route']+=1;save();continue
                if path(qp,qg) is None:report['rejected']['approach_route']+=1;save();continue
                change.update(hypothesis='H021_body_center',method='Observed axial body-centre contact relocation after near-end/weak-visible-normal trigger; retained H018 orientation/height/pre family',axial_body_fraction=fraction,axial_contact_target_shift_m=shift,source_neural_contact_world=selected['target_contact_world'],pad_target_world=(np.asarray(change['pad_target_world'])+shift*axis).tolist(),palm_target_quality=palm,body_trigger=trigger,commanded_lift_height_m=height,physical_success_threshold_unchanged_m=.06,physical_hold_duration_unchanged_s=3.)
                refined=copy.deepcopy(selected);refined.update(grasp_world=world.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),research_change=change)
                stage['elapsed_s']=time.monotonic()-started;report.update(selected=True,selected_phase=phase,selected_pose=index,selected_change=change);save();return refined,route
        stage['elapsed_s']=time.monotonic()-started;save()
    return retained,retained_route
