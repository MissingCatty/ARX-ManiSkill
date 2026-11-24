"""Use native postprocessed grasp poses; apply only robot feasibility checks."""
import numpy as np

# Existing frozen mapping from the ZeroGrasp frame to the authored ARX EEF.
ARX_AXES=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])


def decode_candidate(row,camera):
    """Coordinate conversion only: no target-center or height-dependent changes."""
    row=np.asarray(row);camera=np.asarray(camera)
    rotation=camera[:3,:3]@row[4:13].reshape(3,3)
    surface=camera[:3,:3]@row[13:16]+camera[:3,3]
    contact=surface+row[3]*rotation[:,0]
    grasp=np.eye(4);grasp[:3,:3]=rotation@ARX_AXES
    grasp[:3,3]=contact-.040*rotation[:,0]
    pregrasp=grasp.copy();pregrasp[:3,3]-=.08*rotation[:,0]
    lift=grasp.copy();lift[2,3]+=.12
    return grasp,pregrasp,lift,contact


def select_postprocessed(planner,state,ordered,camera):
    start=planner.measured(state)
    rejected={name:0 for name in ('grasp_ik','closed_collision','pregrasp_ik','lift_ik',
                                 'pregrasp_path','approach_path','optimizer')}
    for index,row in enumerate(ordered):
        grasp,pre,lift,contact=decode_candidate(row,camera)
        qg=planner.ik(grasp,start)
        if qg is None:
            rejected['grasp_ik']+=1;continue
        if not planner.closed_checker.validate(qg[None,None]).all():
            rejected['closed_collision']+=1;continue
        qp=planner.ik(pre,qg)
        if qp is None:
            rejected['pregrasp_ik']+=1;continue
        ql=planner.ik(lift,qg)
        if ql is None:
            rejected['lift_ik']+=1;continue
        reach=planner.path(start,qp)
        if reach is None:
            rejected['pregrasp_path']+=1;continue
        approach=planner.path(qp,qg)
        if approach is None:
            rejected['approach_path']+=1;continue
        selected=dict(index=index,score=float(row[0]),network_row=row.tolist(),
            grasp_world=grasp.tolist(),pregrasp_world=pre.tolist(),lift_world=lift.tolist(),
            target_contact_world=contact.tolist(),q_grasp=qg.tolist(),q_pre=qp.tolist(),q_lift=ql.tolist(),
            source='ZeroGrasp postprocessed pose; frozen coordinate conversion only',
            geometric_prefilters=False,height_adjustment_m=0.,additional_orientation_search=False)
        try:
            paths=dict(pregrasp=planner.optimize('pregrasp',reach,6),
                       approach=planner.optimize('approach',approach,5))
        except RuntimeError:
            rejected['optimizer']+=1;raise
        return dict(success=True,selected=selected,paths=paths,rejected=rejected,
                    geometric_prefilters=False)
    return dict(success=False,reason='no_executable_postprocessed_candidate',count=len(ordered),
                rejected=rejected,geometric_prefilters=False)
