"""Learned alternative approach directions under the unchanged GSNet model."""
import math
import torch
from .multi_angle import decode


def select_views(scores, templates, count, minimum_degrees):
    if scores.ndim!=3 or templates.shape!=(scores.shape[-1],3) or not 1<=count<=len(templates):
        raise ValueError('Invalid learned view scores/templates or branch count')
    if not 0<minimum_degrees<180:raise ValueError('Require a positive approach separation below 180 degrees')
    available=torch.ones_like(scores,dtype=torch.bool);selected=[]
    for rank in range(count):
        masked=scores.masked_fill(~available,float('-inf'))
        value,index=masked.max(-1)
        if not torch.isfinite(value).all():raise ValueError('Insufficient separated approach hypotheses')
        selected.append(index)
        chosen=templates[index]
        cosine=torch.einsum('bnc,vc->bnv',chosen,templates)
        available &= cosine<=math.cos(math.radians(minimum_degrees))
    return selected


def proposals(net,end_points,crop_inputs,*,views,angles,minimum_degrees):
    from loss_utils import generate_grasp_views,GRASP_MAX_WIDTH
    from label_generation import batch_viewpoint_params_to_matrix
    if net.training:raise RuntimeError('Alternative views require frozen evaluation-mode weights')
    xyz,features=crop_inputs
    templates=generate_grasp_views(end_points['view_score'].shape[-1]).to(features.device)
    choices=select_views(end_points['view_score'],templates,views,minimum_degrees)
    if not torch.equal(choices[0],end_points['grasp_top_view_inds']):
        raise RuntimeError('Primary learned view differs from pinned network selection')
    original=decode(end_points,batch_viewpoint_params_to_matrix,angles=angles,maximum_width=GRASP_MAX_WIDTH)
    rows=[list([p]) for p in original]
    with torch.no_grad():
        for choice in choices[1:]:
            direction=templates[choice]
            flat=direction.reshape(-1,3)
            rotation=batch_viewpoint_params_to_matrix(-flat,torch.zeros(len(flat),device=flat.device)).reshape(*choice.shape,3,3)
            temporary={**end_points,'grasp_top_view_xyz':direction,'grasp_top_view_rot':rotation,
                'grasp_top_view_inds':choice}
            output=net.swad(net.crop(xyz,features,rotation),temporary)
            decoded=decode(output,batch_viewpoint_params_to_matrix,angles=angles,maximum_width=GRASP_MAX_WIDTH)
            for i,item in enumerate(decoded):rows[i].append(item)
    return [torch.cat(parts,dim=0) for parts in rows]
