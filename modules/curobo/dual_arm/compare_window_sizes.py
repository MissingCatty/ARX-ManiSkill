"""Adjacent AB/BA timing of two window policies on one resident model pool."""
import argparse
from pathlib import Path
import os
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np

from arx_project.layout import ROOT,now,read_json,sha256,write_json,ensure_run_path


def run(root,parent):
    if root.exists():
        raise ValueError('Use a fresh comparison directory')
    accepted = read_json(parent/'acceptance.json')
    if not accepted['passed']:
        raise ValueError('Parent planning replay must pass')
    registration = read_json(parent/'registration.json')
    case = next(row for row in registration['cases'] if row['id'] == 'M0010_S1_open_boxed_food')
    source = Path(case['source_directory'])
    root.mkdir(parents=True)
    inputs = [parent/'acceptance.json',source/'state.json',source/'candidates/candidate_pool.json',
        parent/'warmup/state.json',parent/'warmup/closed_state.json']
    sources = [Path(p) for p in registration['source_hashes']]
    sources.append(Path(__file__).resolve())
    write_json(root/'registration.json',{'at':now(),'user_authorization':'先优化交替',
        'parent':str(parent),'gpu':registration['gpu'],'case':case,
        'physical_grasps':0,'simulation_instances':0,'planning_requests':4,
        'order':[['fixed','expanded'],['expanded','fixed']],
        'first_window':16,'subsequent_windows':{'fixed':16,'expanded':64},
        'source_hashes':{str(p):sha256(p) for p in sources},
        'input_hashes':{str(p):sha256(p) for p in inputs}})
    os.environ.update(CUDA_VISIBLE_DEVICES=registration['gpu'],GRASP_TRAJOPT_POLICY='interactive',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    import torch
    from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
    from modules.curobo.dual_arm.placement_service import create
    from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner
    from modules.curobo.scene_cache import mesh_stores,prune_inactive_meshes
    torch.set_num_threads(4)
    free,_ = torch.cuda.mem_get_info()
    if free < 10000*1024**2:
        raise RuntimeError('Insufficient CUDA memory for adjacent planning comparison')
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'free_mib':free/1024**2})
    started = time.perf_counter()
    original = create(SimpleNamespace(root=root/'resident',warmup_trial=parent/'warmup'))
    planners = {name:InterleavedPlacementPlanner(original.pool,original.model,original.root)
                for name in ['fixed','expanded']}
    planners['fixed'].subsequent_candidate_batch_size = 16
    planners['expanded'].subsequent_candidate_batch_size = 64
    write_json(root/'startup.json',{'at':now(),'seconds':time.perf_counter()-started,
        'resident_loads':original.pool.loads,'all_four_requests_share_model_pool':True})
    expected = read_json(parent/'plans/graspgenx'/case['id']/'interleaved/plan_result.json')['selected']
    payload = read_json(source/'candidates/candidate_pool.json')
    rows = []
    try:
        for repeat,modes in enumerate([['fixed','expanded'],['expanded','fixed']]):
            for mode in modes:
                directory = root/'plans'/str(repeat)/mode
                directory.mkdir(parents=True)
                shutil.copy2(source/'state.json',directory/'state.json')
                descriptor = publish_candidate_pool(directory,payload)
                torch.manual_seed(20261126);np.random.seed(20261126)
                for context in original.pool.contexts.values():context.planner.reset_seed()
                original.pool.pure_solver.reset_seed()
                write_json(root/'progress.json',{'at':now(),'phase':'planning',
                    'completed_requests':len(rows),'total_requests':4,'mode':mode,'repeat':repeat,'results':rows})
                print('PLAN_BEGIN',repeat,mode,flush=True)
                torch.cuda.synchronize();started = time.perf_counter()
                result = planners[mode].request(directory,candidate_transport=descriptor)
                torch.cuda.synchronize();seconds = time.perf_counter()-started
                finish_candidate_transport(directory)
                if (not result['success'] or any(result['selected'][k] != expected[k] for k in ['arm','index'])
                        or not np.array_equal(result['selected']['contact_pose_world'],expected['contact_pose_world'])):
                    raise RuntimeError('Window-size comparison changed the selected executable grasp')
                schedule = read_json(directory/'interleaved_search.json')['schedule']
                row = {'repeat':repeat,'mode':mode,'seconds':seconds,'windows':len(schedule),
                    'candidate_checks':sum(r['candidate_checks'] for r in schedule),
                    'selected_arm':result['selected']['arm'],'selected_index':result['selected']['index']}
                rows.append(row)
                print('PLAN_END',row,flush=True)
                prune_inactive_meshes(mesh_stores(original),torch.cuda.synchronize)
        medians = {mode:float(np.median([r['seconds'] for r in rows if r['mode'] == mode]))
                   for mode in planners}
        write_json(root/'acceptance.json',{'at':now(),'passed':True,'results':rows,'median_seconds':medians,
            'physical_grasps':0,'simulation_instances':0,'same_selected_arm_index_and_pose':True})
        write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_requests':4,'total_requests':4,'results':rows})
    finally:
        close_owned_transports()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if root.exists():
        parser.error('Comparison directory already exists; keep its existing records')
    try:
        run(root,args.parent.resolve())
    except BaseException:
        if root.exists():write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'physical_grasps':0})
        raise
