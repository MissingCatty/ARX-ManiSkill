"""UNREGISTERED H020: original pose family plus extra palm-target quality."""
import copy,json,time
import numpy as np
from long_axis_misalignment import proposals
from palm_target_quality import PalmTargetQuality


def select(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=180.):
    quality=PalmTargetQuality(out);goals=list(proposals(selected,shape));cache={}
    rejected={k:0 for k in ('palm_quality','grasp_ik','closed_grasp','tracking_grasp','pre_ik','lift_ik','closed_lift','tracking_lift','pre_route','approach_route')}
    report={'hypothesis':'H020_palm_quality','proposals':len(goals),'selected':False,
            'quality_required_observed_gap_m':.010,'rejected':rejected,'phases':[],'queries':[],
            'scope':'Extra target-aware grasp proposal filter only. Original models, margins, contacts, limits and evaluator unchanged.'}
    def save():(out/'palm_quality_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    save()
    for phase,heights in [('original_12cm',(.12,)),('adaptive_endpoint',(.12,.10,.09,.085,.08))]:
        started=time.monotonic();stage={'name':phase,'examined':0,'time_cap_s':max_seconds};report['phases'].append(stage)
        for index,(world,pre,unused_lift,change) in enumerate(goals):
            if time.monotonic()-started>=max_seconds:stage['timeout']=True;break
            stage['examined']+=1
            if index not in cache:cache[index]=quality.evaluate(world)
            evidence=cache[index]
            if not evidence['pass']:rejected['palm_quality']+=1;save();continue
            qg=ik(world,start)
            if qg is None:rejected['grasp_ik']+=1;save();continue
            if not closed_checker.validate(qg[None,None]).all():rejected['closed_grasp']+=1;save();continue
            if not tracking_valid(qg,'quality_grasp_goal'):rejected['tracking_grasp']+=1;save();continue
            qp=ik(pre,qg)
            if qp is None:rejected['pre_ik']+=1;save();continue
            for height in heights:
                if time.monotonic()-started>=max_seconds:stage['timeout']=True;break
                lift=world.copy();lift[2,3]+=height
                query={'phase':phase,'pose_index':index,'commanded_lift_height_m':height,'palm_quality':evidence};report['queries'].append(query)
                ql=ik(lift,qg);query['original_IK_admitted']=ql is not None
                if ql is None:rejected['lift_ik']+=1;save();continue
                query['q_lift']=ql.tolist();query['original_closed_goal_valid']=bool(closed_checker.validate(ql[None,None]).all())
                if not query['original_closed_goal_valid']:rejected['closed_lift']+=1;save();continue
                query['original_tracking_goal_valid']=tracking_valid(ql,'quality_lift_goal')
                if not query['original_tracking_goal_valid']:rejected['tracking_lift']+=1;save();continue
                route=path(start,qp)
                if route is None:rejected['pre_route']+=1;save();continue
                if path(qp,qg) is None:rejected['approach_route']+=1;save();continue
                refined=copy.deepcopy(selected);new_change=copy.deepcopy(change)
                new_change.update(hypothesis='H020_palm_quality',
                    method='Retained transverse-axis pose/height family with additional frozen palm-CAD versus observed-target quality filter and original/adaptive lift goals',
                    commanded_lift_height_m=height,source_neural_contact_world=selected['target_contact_world'],
                    palm_target_quality=evidence,physical_success_threshold_unchanged_m=.06,
                    physical_hold_duration_unchanged_s=3.,robot_configuration_changed=False,
                    scope='Perception/proposal quality only. No scene/robot/collision-model configuration or criterion change; original contact safety still authoritative.')
                refined.update(grasp_world=world.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),
                               q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),research_change=new_change)
                stage['elapsed_s']=time.monotonic()-started
                report.update(selected=True,selected_phase=phase,selected_pose=index,
                              selected_palm_quality=evidence,commanded_lift_height_m=height,
                              quality_cache={str(k):v for k,v in cache.items()});save()
                return refined,route
        stage['elapsed_s']=time.monotonic()-started;save()
    report['quality_cache']={str(k):v for k,v in cache.items()};save()
    return None,None
