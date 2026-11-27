"""Compare every resident native kinematics tensor with a full original bind."""
from copy import deepcopy

from arx_project.layout import read_json,write_json
from modules.graspgenx.native_postprocessing.width_planning import target_state


def snapshot(parameters):
    import torch
    result={};seen=set()
    def visit(value,path):
        if isinstance(value,torch.Tensor):
            result[path]=value.detach().cpu().clone()
        elif id(value) not in seen:
            seen.add(id(value))
            if isinstance(value,dict):
                for name,child in value.items():visit(child,path+'/'+str(name))
            elif isinstance(value,(list,tuple)):
                for index,child in enumerate(value):visit(child,path+'/'+str(index))
            elif type(value).__module__.startswith('curobo.') and hasattr(value,'__dict__'):
                visit(vars(value),path)
    for index,value in enumerate(parameters):visit(value,str(index))
    return result


def run(planner,warmup,root):
    import torch
    pool=planner.pool;flags=pool.width_refresh_enabled,pool.sphere_reuse_enabled
    source=read_json(warmup/'state.json');reports=[]
    try:
        for side in ['right','left']:
            canonical=planner.model.bind(source,side,root/side/'arm')
            for width in [.088,.060,.040]:
                state=target_state(canonical,width)
                pool.width_refresh_enabled=False;pool.sphere_reuse_enabled=False
                pool.bind(root/side/str(width)/'original',state,include_target=False)
                expected=snapshot(pool.parameters['open'])
                pool.width_refresh_enabled=True
                pool.bind(root/side/str(width)/'refresh_start',target_state(state,.088),include_target=False)
                pool.bind(root/side/str(width)/'refresh_end',state,include_target=False)
                actual=snapshot(pool.parameters['open'])
                if set(expected)!=set(actual) or not all(torch.equal(expected[k],actual[k]) for k in expected):
                    raise RuntimeError('Lock-only update changed resident native kinematics tensors')
                reports.append({'arm':side,'width_m':width,'tensors_checked':len(expected),'all_bitwise_equal':True})
        changed=deepcopy(state);changed['scene_cuboids']['table']['center'][0]+=.01
        before=pool.width_only_binder.full_binds
        pool.bind(root/'changed_world',changed,include_target=False)
        if pool.width_only_binder.full_binds!=before+1:raise RuntimeError('Changed world was reused')
        write_json(root/'acceptance.json',{'passed':True,'reports':reports,
            'all_native_tensors_bitwise_equal':True,'changed_world_forced_update':True,'physical_grasps':0})
    finally:pool.width_refresh_enabled,pool.sphere_reuse_enabled=flags
    return reports
