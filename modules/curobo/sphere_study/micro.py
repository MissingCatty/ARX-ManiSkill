"""Fixed joint inputs for separating count cost from changed search decisions."""
from copy import deepcopy
import time

import numpy as np

from arx_project.layout import ROOT,read_json,write_json,sha256
from modules.graspgenx.native_postprocessing.width_planning import target_state


def measure(functions,repeats=20):
    import torch
    keys=list(functions);graphs={};outputs={};eager={k:[] for k in keys};events={k:[] for k in keys}
    for key,function in functions.items():
        for _ in range(3):outputs[key]=function()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):outputs[key]=function()
        graphs[key]=graph
        for _ in range(3):graph.replay()
    torch.cuda.synchronize()
    for repeat in range(5):
        for key in keys[repeat%len(keys):]+keys[:repeat%len(keys)]:
            began=time.perf_counter();eager_output=functions[key]();torch.cuda.synchronize()
            eager[key].append((time.perf_counter()-began)*1000)
    for repeat in range(repeats):
        for key in keys[repeat%len(keys):]+keys[:repeat%len(keys)]:
            start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record();graphs[key].replay();end.record();events[key].append((start,end))
    torch.cuda.synchronize()
    return {str(key):{'graph_cuda_median_ms':float(np.median([a.elapsed_time(b) for a,b in events[key]])),
        'graph_cuda_p90_ms':float(np.percentile([a.elapsed_time(b) for a,b in events[key]],90)),
        'eager_wall_median_ms':float(np.median(eager[key])),'graph_repeats':repeats,'eager_repeats':5}
        for key in keys}


def run(planner,directory,source,ordinal):
    import torch
    directory.mkdir(parents=True,exist_ok=True)
    source=ROOT/source;original=read_json(source/'plan_result.json');selected=original['selected']
    state=read_json(source/'state.json');arm=selected['arm']
    canonical=planner.model.bind(state,arm,directory/'physical_arm')
    canonical=target_state(canonical,2*selected['open_finger_m'])
    context=planner.pool.bind(directory/'bound',canonical,include_target=False)
    names=['right_joint'+str(i) for i in range(21,27)]
    limits=np.asarray([canonical['joint_limits'][canonical['joint_names'].index(n)] for n in names])
    path=np.concatenate([np.asarray(original['paths'][stage]['q']) for stage in ['pregrasp','approach']])
    actual=path[np.linspace(0,len(path)-1,96,dtype=int)]
    rng=np.random.default_rng(20261127+ordinal)
    random=rng.uniform(limits[:,0]+1e-4,limits[:,1]-1e-4,(160,6))
    values=np.concatenate([actual,random]).astype(np.float32)
    np.save(directory/'fixed_joints.npy',values)
    import yaml
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    scene=read_json(context.out/'scene.json')
    config=yaml.load(context.model.read_text(),Loader=yaml.CSafeLoader)
    checkers={}
    for sphere_count in [871,512,256]:
        cfg=deepcopy(config)
        profile=read_json(planner.pool.campaign_root/'profiles'/str(sphere_count)/'profile.json')
        cfg['robot_cfg']['kinematics']['collision_spheres']=profile['collision_spheres']
        checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=cfg,scene_model=scene,n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
        checker.self_collision_cost=None
        if int(checker.kinematics.kinematics_config.total_spheres)!=sphere_count:
            raise RuntimeError('Fixed-work checker sphere count mismatch')
        checkers[sphere_count]=checker
    rows=[]
    for count in [1,32,256]:
        q=torch.as_tensor(values[:count,None],device='cuda',dtype=torch.float32)
        for checker in checkers.values():checker.setup_batch_tensors(count,1)
        spheres={n:c.get_kinematics(q).robot_spheres.detach().clone() for n,c in checkers.items()}
        rows.append({'joint_states':count,
            'FK':measure({n:(lambda c=c:c.get_kinematics(q).robot_spheres) for n,c in checkers.items()}),
            'world_collision':measure({n:(lambda n=n,c=c:c.get_collision_constraint(spheres[n])) for n,c in checkers.items()}),
            'complete_validation':measure({n:(lambda c=c:c.validate(q)) for n,c in checkers.items()})})
    q=torch.as_tensor(values[:,None],device='cuda',dtype=torch.float32)
    masks={str(n):c.validate(q).reshape(-1).cpu().tolist() for n,c in checkers.items()}
    write_json(directory/'result.json',{'passed':True,'sphere_counts':list(checkers),
        'arm':arm,'source':str(source.relative_to(ROOT)),'fixed_joint_sha256':sha256(directory/'fixed_joints.npy'),
        'states':len(values),'source_path_states':96,'random_limit_valid_states':160,
        'native_world_clear':masks,'self_collision_enabled':False,
        'measurements':rows,'no_IK_or_path_search_in_fixed_work_comparison':True,
        'all_counts_interleaved_on_one_GPU':True,
        'graph_time_scope':'CUDA graph replay elapsed; eager wall timing separately recorded'})
    return {'success':True,'sphere_counts':list(checkers),'states':len(values),
        'micro_result':str((directory/'result.json').relative_to(ROOT))}
