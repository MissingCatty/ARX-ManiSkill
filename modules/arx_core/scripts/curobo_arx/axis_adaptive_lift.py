"""UNREGISTERED H019 lift-target search, original H018 selection has priority.

Changes normal commanded endpoint positions only. Robot/world models,
limits, margins, convergence and the actual >6cm/3s evaluator stay fixed.
"""
import copy,json,time
import numpy as np
from long_axis_misalignment import select as select_original,proposals


def select(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=180.):
    refined,route=select_original(selected,shape,start,ik,path,closed_checker,tracking_valid,out,max_seconds=max_seconds)
    if refined is not None:
        return refined,route
    started=time.monotonic();goals=list(proposals(selected,shape))
    rejected={k:0 for k in ('grasp_ik','closed_grasp','tracking_grasp','pre_ik','lift_ik','closed_lift','tracking_lift','pre_route','approach_route')}
    report={'hypothesis':'H019_adaptive_lift','proposals':len(goals),'examined':0,'selected':False,
            'lift_height_search_m':[.12,.10,.09,.085,.08],
            'fixed_physical_success_threshold_m':.06,'fixed_hold_duration_s':3.,
            'scope':'Commanded endpoint goal search only; original evaluator/model/physics unchanged. Original H018complete-selection tried first.',
            'rejected':rejected,'queries':[],'time_cap_s':max_seconds}
    def save():(out/'adaptive_lift_candidate_diagnostics.json').write_text(json.dumps(report,indent=2))
    save()
    for index,(world,pre,unused_lift,change) in enumerate(goals):
        if time.monotonic()-started>=max_seconds:report['timeout']=True;break
        report['examined']+=1
        qg=ik(world,start)
        if qg is None:rejected['grasp_ik']+=1;save();continue
        if not closed_checker.validate(qg[None,None]).all():rejected['closed_grasp']+=1;save();continue
        if not tracking_valid(qg,'adaptive_grasp_goal'):rejected['tracking_grasp']+=1;save();continue
        qp=ik(pre,qg)
        if qp is None:rejected['pre_ik']+=1;save();continue
        for height in (.12,.10,.09,.085,.08):
            if time.monotonic()-started>=max_seconds:report['timeout']=True;break
            lift=world.copy();lift[2,3]+=height
            query={'pose_index':index,'lift_height_m':height};report['queries'].append(query)
            ql=ik(lift,qg);query['original_IK_admitted']=ql is not None
            if ql is None:rejected['lift_ik']+=1;save();continue
            query['q_lift']=ql.tolist();query['original_closed_goal_valid']=bool(closed_checker.validate(ql[None,None]).all())
            if not query['original_closed_goal_valid']:rejected['closed_lift']+=1;save();continue
            query['original_tracking_goal_valid']=tracking_valid(ql,'adaptive_lift_goal')
            if not query['original_tracking_goal_valid']:rejected['tracking_lift']+=1;save();continue
            route=path(start,qp)
            if route is None:rejected['pre_route']+=1;save();continue
            if path(qp,qg) is None:rejected['approach_route']+=1;save();continue
            refined=copy.deepcopy(selected);new_change=copy.deepcopy(change)
            new_change.update(hypothesis='H019_adaptive_lift',
                method='Retained perpendicular-major grasp, choose highest admissible commanded lift endpoint in the declared per-pose grid',
                commanded_lift_height_m=height,original_commanded_lift_height_m=.12,
                physical_success_threshold_unchanged_m=.06,physical_hold_duration_unchanged_s=3.,
                scope='Same original NNcontact/PAD, robot/world/limits/margins/evaluator; endpoint change is an action, not a configuration edit. Not global highest feasible lift proof.')
            refined.update(grasp_world=world.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),
                           q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),research_change=new_change)
            report.update(selected=True,selected_pose=index,commanded_lift_height_m=height,
                          elapsed_s=time.monotonic()-started);save()
            return refined,route
    report.update(elapsed_s=time.monotonic()-started);save()
    return None,None
