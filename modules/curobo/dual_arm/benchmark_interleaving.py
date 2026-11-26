"""Compare only arm-search scheduling on frozen V6 inputs; no simulation."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json


POLICY = ROOT/'configs/experiments/dual_arm_interleaving.json'


def prepare(root,source,gpu,reference=None):
    if root.exists():
        raise ValueError('Use a fresh replay registration')
    if not read_json(source/'acceptance.json')['passed']:
        raise ValueError('V6 source results were not accepted')
    policy = read_json(POLICY)
    root.mkdir(parents=True)
    warmup = source/'methods/zerograsp/preflight/head_warmup'
    hashes = {}
    for name in ['state.json','closed_state.json']:
        src = warmup/name
        hashes[str(src)] = sha256(src)
        (root/'warmup').mkdir(exist_ok=True)
        shutil.copy2(src,root/'warmup'/name)
    cases = []
    for method,case in policy['validation']['cases']:
        src = source/'methods'/method/'trials'/case
        for name in ['state.json','candidates/candidate_pool.json','plan_result.json']:
            hashes[str(src/name)] = sha256(src/name)
        cases.append({'method':method,'id':case,'source_directory':str(src)})
    if reference is not None:
        reference = Path(reference).resolve()
        accepted = read_json(reference/'acceptance.json')
        prior = read_json(reference/'registration.json')
        if not accepted['passed'] or prior['gpu'] != gpu or Path(prior['source_run']) != source:
            raise ValueError('Baseline reference must be an accepted replay on the same GPU and inputs')
        if {(r['method'],r['id']) for r in accepted['paired_cases']} != {(r['method'],r['id']) for r in cases}:
            raise ValueError('Baseline reference must contain the same cases')
        if any(prior['input_hashes'].get(name) != digest for name,digest in hashes.items()):
            raise ValueError('Baseline reference inputs differ')
        hashes[str(reference/'acceptance.json')] = sha256(reference/'acceptance.json')
        for case in cases:
            p = reference/'plans'/case['method']/case['id']/'original/plan_result.json'
            hashes[str(p)] = sha256(p)
    sources = [POLICY,ROOT/'configs/modules.json',ROOT/'configs/experiments/active_experiment.json',
        ROOT/'configs/policies/fingertip_table.json',ROOT/'modules/curobo/arx_adapter.py',
        ROOT/'modules/maniskill/resident_planner.py',ROOT/'modules/maniskill/fast_resident_planner.py']
    for directory in ['modules/curobo/dual_arm','modules/curobo/candidate_io','modules/curobo/fingertip_contact']:
        sources += sorted((ROOT/directory).glob('*.py'))
    write_json(root/'registration.json',{'at':now(),'user_authorization':'先优化交替',
        'source_run':str(source),'gpu':gpu,'minimum_free_mib':10000,
        'cases':cases,'candidate_batch_size':policy['candidate_batch_size'],
        'subsequent_candidate_batch_size':policy['subsequent_candidate_batch_size'],
        'reference_run':str(reference) if reference is not None else None,
        'physical_grasp_budget':0,'simulation_instances':0,'model_inference_calls':0,
        'planning_requests':len(cases)*(1 if reference is not None else 2),'automatic_retries':False,
        'candidate_transport':'shared_memory_once_per_request',
        'source_hashes':{str(p):sha256(p) for p in sources},'input_hashes':hashes,
        'comparison':'existing_placement_planner_vs_same_planner_with_interleaved_search',
        'selection':'posthoc engineering fixtures: slow tail, short case, median case, long both-arm search and planning failure'})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_requests':0,
        'total_requests':len(cases)*(1 if reference is not None else 2)})


def run(root):
    reg = read_json(root/'registration.json')
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Replay already started; preserve existing results')
    for filename,digest in {**reg['source_hashes'],**reg['input_hashes']}.items():
        if sha256(filename) != digest:
            raise RuntimeError('Registered replay source/input changed: '+filename)
    os.environ.update(CUDA_VISIBLE_DEVICES=reg['gpu'],GRASP_TRAJOPT_POLICY='interactive',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    import torch
    from arx_project.artifacts import publish_candidate_pool,finish_candidate_transport,close_owned_transports
    from modules.curobo.dual_arm.placement_service import create
    from modules.curobo.dual_arm.interleaved_adapter import InterleavedPlacementPlanner
    from modules.curobo.scene_cache import mesh_stores,prune_inactive_meshes
    torch.set_num_threads(4)
    available,total = torch.cuda.mem_get_info()
    if available/1024**2 < reg['minimum_free_mib']:
        raise RuntimeError('Insufficient free CUDA memory for planning replay')
    probe = torch.zeros(1,device='cuda');torch.cuda.synchronize()
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu':reg['gpu'],
        'free_before_start_mib':available/1024**2,'gpu_name':torch.cuda.get_device_name()})
    startup = time.perf_counter()
    original = create(SimpleNamespace(root=root/'resident',warmup_trial=root/'warmup'))
    interleaved = InterleavedPlacementPlanner(original.pool,original.model,original.root)
    interleaved.candidate_batch_size = reg['candidate_batch_size']
    interleaved.subsequent_candidate_batch_size = reg['subsequent_candidate_batch_size']
    write_json(root/'startup.json',{'at':now(),'seconds':time.perf_counter()-startup,
        'model_loads':original.pool.loads,'both_variants_share_resident_models':True})
    planners = {'original':original,'interleaved':interleaved}
    results,completed = [],0
    try:
        for ordinal,case in enumerate(reg['cases']):
            pair = {}
            modes = ['interleaved','original'] if ordinal%2 == 0 else ['original','interleaved']
            if reg['reference_run']:
                reference = Path(reg['reference_run'])
                previous = next(row for row in read_json(reference/'acceptance.json')['paired_cases']
                    if row['method'] == case['method'] and row['id'] == case['id'])
                pair['original'] = {'seconds':previous['original_seconds'],
                    'candidate_checks':previous['original_checks'],
                    'result':read_json(reference/'plans'/case['method']/case['id']/'original/plan_result.json')}
                modes = ['interleaved']
            for mode in modes:
                free,_ = torch.cuda.mem_get_info()
                if free < 1536*1024**2:
                    raise RuntimeError('Insufficient CUDA reserve before next replay request')
                directory = root/'plans'/case['method']/case['id']/mode
                directory.mkdir(parents=True)
                source = Path(case['source_directory'])
                shutil.copy2(source/'state.json',directory/'state.json')
                payload = read_json(source/'candidates/candidate_pool.json')
                descriptor = publish_candidate_pool(directory,payload)
                torch.manual_seed(20261126);np.random.seed(20261126)
                for context in original.pool.contexts.values():
                    context.planner.reset_seed()
                if original.pool.pure_solver is not None:
                    original.pool.pure_solver.reset_seed()
                print('PLAN_BEGIN',case['method'],case['id'],mode,flush=True)
                write_json(root/'progress.json',{'at':now(),'phase':'planning','completed_requests':completed,
                    'total_requests':reg['planning_requests'],'current':{**case,'mode':mode},'results':results})
                torch.cuda.synchronize();started = time.perf_counter()
                result = planners[mode].request(directory,candidate_transport=descriptor)
                torch.cuda.synchronize();seconds = time.perf_counter()-started
                finish_candidate_transport(directory)
                io = read_json(directory/'candidate_io.json')
                if io['source'] != 'shared_memory' or io['decode_count'] != 1 or io['full_arm_pool_copies'] != 0:
                    raise RuntimeError('Replay did not preserve the candidate-I/O contract')
                arm_checks = {}
                for side in ['right','left']:
                    path = directory/'arms'/side/'plan_result.json'
                    arm_checks[side] = len(read_json(path).get('candidate_events',[])) if path.exists() else 0
                pair[mode] = {'seconds':seconds,'result':result,'candidate_checks':arm_checks}
                completed += 1
                print('PLAN_END',case['id'],mode,round(seconds,4),'success',result['success'],
                      'arm',result.get('selected',{}).get('arm'),'index',result.get('selected',{}).get('index'),
                      'candidate_checks',arm_checks,flush=True)
                prune_inactive_meshes(mesh_stores(original),torch.cuda.synchronize)
            a,b = [pair[m]['result'] for m in ['original','interleaved']]
            passed = a['success'] == b['success']
            if a['success'] and b['success']:
                passed &= all(a['selected'][key] == b['selected'][key] for key in ['arm','index'])
                passed &= np.array_equal(a['selected']['contact_pose_world'],b['selected']['contact_pose_world'])
            row = {**case,'passed':bool(passed),'original_seconds':pair['original']['seconds'],
                'interleaved_seconds':pair['interleaved']['seconds'],
                'speedup':pair['original']['seconds']/pair['interleaved']['seconds'],
                'original_checks':pair['original']['candidate_checks'],'interleaved_checks':pair['interleaved']['candidate_checks'],
                'original_arm':a.get('selected',{}).get('arm'),'interleaved_arm':b.get('selected',{}).get('arm'),
                'original_index':a.get('selected',{}).get('index'),'interleaved_index':b.get('selected',{}).get('index')}
            results.append(row)
            write_json(root/'progress.json',{'at':now(),'phase':'comparison','completed_requests':completed,
                'total_requests':reg['planning_requests'],'results':results})
            if not passed:
                raise RuntimeError('Paired replay changed the executable candidate or selected arm')
        for filename,digest in reg['input_hashes'].items():
            if sha256(filename) != digest:
                raise RuntimeError('Replay modified a source input')
        write_json(root/'acceptance.json',{'at':now(),'passed':True,'paired_cases':results,
            'physical_grasps':0,'simulation_instances':0,'model_inference_calls':0,
            'resident_loads':original.pool.loads,'same_gpu':reg['gpu'],'shared_memory':True,
            'baseline_reference_run':reg['reference_run'],
            'same_selected_arm_index_and_pose':True})
        write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_requests':completed,
            'total_requests':reg['planning_requests'],'results':results})
    finally:
        close_owned_transports()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--gpu')
    parser.add_argument('--reference',type=Path)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if args.stage == 'prepare':
        prepare(root,args.source.resolve(),args.gpu,args.reference)
    else:
        try:
            run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'physical_grasps':0})
            raise


if __name__ == '__main__':
    main()
