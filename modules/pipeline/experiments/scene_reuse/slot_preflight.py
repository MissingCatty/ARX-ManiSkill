"""Hot arm switches, target toggles and cross-request invalidation checks."""
from pathlib import Path
import numpy as np

from arx_project.layout import now, read_json, write_json
from modules.graspgenx.native_postprocessing.width_planning import target_state
from .preflight import cold_meshes


def hot_switches(planner, cases, root, legacy, slots):
    import torch
    pool=planner.pool; cpu=planner.cpu_geometry_reuse; gpu=planner.gpu_geometry_reuse
    reports=[]
    sequence=[('right',True),('left',True),('right',False),('left',False),('right',True)]
    for case in cases:
        if case['id'] not in ['M0062_S4_clutter_keychain','M0066_S4_clutter_mug','M0078_S4_clutter_teapot']: continue
        source=read_json(Path(case['source_directory'])/'state.json'); values={}; traces={}
        for mode in ['control','cached']:
            cold_meshes(planner); pool.memory_bindings.scenes=legacy if mode=='control' else slots
            cpu.begin(None); gpu.begin(True); pool.memory_updates_enabled=True; pool.memory_bindings.begin()
            rows=[]
            try:
                for i,(arm,include) in enumerate(sequence):
                    work=root/case['id']/mode/str(i)
                    state=target_state(planner.model.bind(source,arm,work/'physical_arm'),.088)
                    context=pool.bind(work/'state',state,include_target=include)
                    validator=pool.contact_view(context,state)
                    q=context.measured(state)
                    qs=q[None]+torch.linspace(-.02,.02,32,device=q.device)[:,None]
                    qs=qs.clamp(context.limits[:,0]+1e-4,context.limits[:,1]-1e-4)
                    spheres=context.checker.get_kinematics(qs[:,None]).robot_spheres.detach().clone()
                    cost,gradient=context.checker.get_collision_vector(spheres)
                    rows.append({'valid':context.checker.validate(qs[:,None]).cpu().numpy().copy(),
                        'local':validator.validate(qs,stage='final_approach').copy(),
                        'cost':cost.cpu().numpy().copy(),'gradient':gradient.cpu().numpy().copy()})
                values[mode]=rows
            finally:
                traces[mode]=pool.memory_bindings.finish(); pool.memory_updates_enabled=False
                cpu.finish(); gpu.finish()
        if not all(np.array_equal(a[k],b[k]) for a,b in zip(values['control'],values['cached']) for k in a):
            raise RuntimeError('Fixed slots changed hot-switch collision results')
        if traces['cached']['scenes']['geometry_replacements']!=3 or not traces['cached']['scenes']['slot_replacements']:
            raise RuntimeError('Hot switches still rebuilt whole scenes')
        reports.append({'case':case['id'],'steps':len(sequence),'states_per_step':32,
            'exact_native_and_local_results':True,'scenes':{m:t['scenes'] for m,t in traces.items()}})
    if len(reports)!=3: raise RuntimeError('Missing original capacity-problem fixtures')
    # Continue the last unchanged scene without clearing the native buffers.
    pool.memory_bindings.scenes=slots; cpu.begin(None); gpu.begin(True)
    pool.memory_updates_enabled=True; pool.memory_bindings.begin()
    try:
        context=pool.bind(root/'next_request',state,include_target=True); pool.contact_view(context,state)
    finally:
        reused=pool.memory_bindings.finish(); pool.memory_updates_enabled=False; cpu.finish(); gpu.finish()
    if reused['scenes']['geometry_replacements']!=0: raise RuntimeError('Unchanged cross-request scene was not reused')
    context.checker.scene_model.enable_obstacle('table',False)
    cpu.begin(None); gpu.begin(True); pool.memory_updates_enabled=True; pool.memory_bindings.begin()
    try:
        context=pool.bind(root/'after_external_change',state,include_target=True); pool.contact_view(context,state)
    finally:
        restored=pool.memory_bindings.finish(); pool.memory_updates_enabled=False; cpu.finish(); gpu.finish()
    if restored['scenes']['request_boundary_invalidations']<1 or restored['scenes']['geometry_replacements']<1:
        raise RuntimeError('External scene mutation was silently reused')
    pool.memory_bindings.scenes=legacy
    result={'at':now(),'passed':True,'hot_switches':reports,
            'unchanged_cross_request':reused['scenes'],'external_mutation':restored['scenes']}
    write_json(root/'acceptance.json',result)
    return result
