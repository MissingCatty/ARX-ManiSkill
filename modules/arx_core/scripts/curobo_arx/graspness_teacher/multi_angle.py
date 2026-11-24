"""Opt-in decoding of multiple learned rolls, with unchanged GSNet weights.

Each roll retains its own predicted best insertion depth and aperture. This
changes proposals only; original ARX, point-cloud collision and image gates
still determine admission. The original pool remains an exact prefix.
"""
import numpy as np


def decode(end_points, rotation, *, angles=4, maximum_width=.1):
    import torch
    outputs=[]
    for i in range(len(end_points['point_clouds'])):
        score=end_points['grasp_score_pred'][i].float()
        width=end_points['grasp_width_pred'][i].float()
        if score.ndim!=3 or score.shape!=width.shape or not 1<=angles<=score.shape[1]:
            raise ValueError('Unexpected GSNet angle/depth heads or angle count')
        count,num_angles,num_depth=score.shape
        if num_angles!=12 or num_depth!=4:
            raise ValueError('Expected the pinned 12-angle, 4-depth GSNet model')
        best,depth=score.max(-1)
        rolls=torch.argsort(best,dim=-1,descending=True,stable=True)[:,:angles]
        rows=[]
        for rank in range(angles):
            roll=rolls[:,rank]
            depth_index=depth.gather(1,roll[:,None]).flatten()
            index=roll*num_depth+depth_index
            value=score.reshape(count,-1).gather(1,index[:,None])
            aperture=(1.2*width/10.).reshape(count,-1).gather(1,index[:,None])
            aperture=torch.clamp(aperture,min=0.,max=maximum_width)
            angle=roll*np.pi/12
            insertion=((depth_index+1)*.01)[:,None]
            orientation=rotation(-end_points['grasp_top_view_xyz'][i].float(),angle).reshape(count,9)
            rows.append(torch.cat([value,aperture,.02*torch.ones_like(value),insertion,
                orientation,end_points['xyz_graspable'][i].float(),-torch.ones_like(value)],dim=-1))
        outputs.append(torch.cat(rows,dim=0))
    return outputs


def merge_pools(baseline, expanded, maximum):
    """Keep original success opportunities; append distinct learned hypotheses."""
    import copy
    from scripts.curobo_arx.data_factory.geometry import grasp_equal
    result=copy.deepcopy(baseline)
    if maximum<len(baseline['candidates']):
        raise ValueError('Do not truncate the original candidate prefix')
    seen={g['goal_id'] for g in baseline['candidates']}
    for goal in expanded['candidates']:
        if (goal['goal_id'] not in seen and len(result['candidates'])<maximum
                and not any(grasp_equal(goal,old) for old in result['candidates'])):
            result['candidates'].append(copy.deepcopy(goal));seen.add(goal['goal_id'])
    result['proposal_extension']={'version':'learned_rolls_v1',
        'original_prefix_ids':[g['goal_id'] for g in baseline['candidates']],
        'added_candidates':len(result['candidates'])-len(baseline['candidates']),
        'constraints_and_calibration_changed':False}
    return result
