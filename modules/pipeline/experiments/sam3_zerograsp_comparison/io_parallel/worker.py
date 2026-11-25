"""One independently admitted simulator process per method."""
import argparse
import os
from pathlib import Path
import shutil
import time
import traceback

from arx_project.layout import ROOT,now,sha256,write_json
from arx_project.artifacts import atomic_json as write_json
from arx_project.json_io import read_json
from modules.pipeline.experiments.dual_arm_grasp.run import imports
from ..cohort import check_pair
from ..replay import restore
from ..run import NORMAL
from .campaign import guard
from .preparation import Preparation
from .services import Services
from .continuation import execution_pairs, verify_inheritance


def run(root):
    reg=read_json(root/'registration.json');campaign=ROOT/reg['campaign'];guard(campaign)
    pairs = execution_pairs(reg)
    if (root/'started.json').exists() or any((root/'claims'/(p['id']+'.json')).exists() for _,p in pairs):
        raise RuntimeError('Worker already started; no automatic retry')
    verify_inheritance(campaign, read_json(campaign/'registration.json'), full=True)
    roles=read_json(root/'preflight/resources.json')['gpu_roles'];reg['gpu_roles']=roles
    os.environ['CUDA_VISIBLE_DEVICES']=roles['simulation'];imports()
    from modules.pipeline.experiments.repaired_grasp_benchmark.simulation import install
    executor,evaluator,_=install(root)
    write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'gpu_roles':roles})
    services=None
    try:
        pair=next(p for p in reg['cohort']['pairs'] if not p.get('common_precondition_failure'))
        frozen=check_pair(pair);warm=root/'preflight/head_warmup'
        for name in pair['input_hashes']:
            target=warm/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(frozen/name,target)
        write_json(warm/'closed_state.json',read_json(warm/'state.json'))
        env=executor.make(pair['entry']['model_id'],scene_spec=pair['entry']['scene'])
        try:
            import torch
            env.reset(seed=pair['entry']['seed'])
            for _ in range(80):env._step_action(torch.tensor(env.initial_q[None],device=env.device))
            restore(env,executor,warm,read_json(warm/'state.json'))
        finally:env.close()
        services=Services(root,reg);services.start()
        if reg['method']=='graspgenx':
            from repair_contract import make_candidates
            make_candidates(warm,background=True)
        options={} if pair['workspace']['preferred_arm'] is None else {'arm':pair['workspace']['preferred_arm']}
        response=services.request('curobo_dual_arm',warm,**options)
        write_json(root/'preflight/native_full_chain.json',response)
        write_json(root/'preflight/all_ready.json',{'at':now(),'passed':True,'physical_grasps':0,
            'method':reg['method'],'warm_planning_success':response['result']['success']})
        from modules.pipeline.experiments.dual_arm_grasp.run import gpu_inventory
        free=gpu_inventory()
        if min(free[gpu] for gpu in set(roles.values())) < reg['configuration']['resource_floors_mib']['minimum_after_warmup']:
            raise RuntimeError('Insufficient remaining memory after method warmup')
        if (campaign/'ABORT').exists():raise RuntimeError('Campaign aborted during warmup')
        write_json(root/'preflight/release.json',{'at':now(),'passed':True,'gpu_roles':roles,'free_mib':free})
        for number,pair in pairs:
            guard(campaign);check_pair(pair)
            if shutil.disk_usage(root).free < 10*1024**3:raise RuntimeError('Insufficient disk space')
            free=gpu_inventory()
            if min(free[gpu] for gpu in set(roles.values())) < reg['configuration']['resource_floors_mib']['minimum_after_warmup']:
                raise RuntimeError('Insufficient GPU headroom before claiming next scene')
            claim=root/'claims'/(pair['id']+'.json');out=root/'trials'/pair['id']
            if claim.exists() or out.exists():raise RuntimeError('Slot already claimed')
            write_json(claim,{'state':'claimed','at':now(),'attempt':1,'scene':pair['id'],'method':reg['method']})
            write_json(root/'current_trial.json',{'at':now(),'number':number+1,'total':74,'scene':pair['id']})

            def worker(script,directory,gpu,log,**flags):
                if script!='planner.py' or not (flags.get('approach') or flags.get('lift')):
                    raise RuntimeError('Unexpected fallback in paired repeat')
                try:
                    if (campaign/'ABORT').exists():raise RuntimeError('Concurrent campaign aborted')
                    response=services.request('curobo_dual_arm',Path(directory),**flags)
                    write_json(log,response);return response['returncode']
                except BaseException:
                    write_json(Path(directory)/'infrastructure_error.json',{'at':now(),'error':traceback.format_exc()});raise

            executor.worker=worker
            preparation=Preparation(reg['method'],pair,services,executor);begin=time.perf_counter()
            print('BEGIN_IO74',reg['method'],number+1,pair['id'],flush=True)
            executor.trial(pair['entry'],out,roles[reg['method']],roles['curobo_dual_arm'],
                'head',prepare_observation=preparation.prepare)
            end=time.perf_counter();evaluator.verify_trial(out);result=read_json(out/'result.json')
            allowed=NORMAL|{'rayst3r_empty_pointcloud','graspgenx_no_postprocessed_candidate'}
            if result['reason'] not in allowed or (out/'infrastructure_error.json').exists():
                raise RuntimeError('Execution/resource/record fault: '+result['reason'])
            if not read_json(out/'independent_evaluation.json')['passed']:raise RuntimeError('Independent evaluation failed')
            if result.get('planning_success') and not read_json(out/'prepared_plan_execution_gate.json')['passed']:
                raise RuntimeError('Prepared plan handoff failed')
            if result.get('physical_attempt'):
                for name in ['execution.jsonl','execution.mp4','head.mp4','stages.json','contact_entities.json','paired_restore_verification.json']:
                    if not (out/name).is_file() or not (out/name).stat().st_size:raise RuntimeError('Missing evidence: '+name)
            timings={**preparation.timings,'trial_wall_seconds':end-begin}
            if preparation.generation_started is not None:
                timings['generation_to_trial_return_seconds']=end-preparation.generation_started
            if result.get('physical_attempt') and preparation.returned is not None:
                timings['execution_with_replanning_seconds']=end-preparation.returned
            write_json(out/'latency.json',{'method':reg['method'],'common_precondition_failure':bool(pair.get('common_precondition_failure')),
                'timings':timings,'physical_attempt':bool(result.get('physical_attempt'))})
            write_json(claim,{'state':'complete','at':now(),'scene':pair['id'],'method':reg['method'],
                'result_sha256':sha256(out/'result.json'),'latency_sha256':sha256(out/'latency.json')})
            from .metrics import summarize
            summarize(campaign,'running')
            print('RESULT_IO74',reg['method'],number+1,pair['id'],result['reason'],flush=True)
        completed=sum(read_json(path).get('state')=='complete' for path in (root/'claims').glob('*.json'))
        write_json(root/'complete.json',{'at':now(),'completed':completed,
            'inherited_interrupted':len(reg.get('continuation',{}).get('inherited_interrupted_case_ids',[])),
            'authorized_pending_schedule_exhausted':True})
    finally:
        if services is not None:services.stop()
        else:write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':True,'models_started':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    try:run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()})
        if not (root/'services_stopped.json').exists() and not list((root/'resident').glob('*/service.log')):
            write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':True,'models_started':False})
        raise


if __name__=='__main__':main()
