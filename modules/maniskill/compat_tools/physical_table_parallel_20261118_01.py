"""Resume the same 400 slots with independent GPU backends for the two arms."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import textwrap
import time
import types

CODE=Path('/home/zhaoyechong/projects/ARX-ManiSkill/modules/maniskill')
SELF=Path(__file__).resolve()
POLICY=CODE/'tmp/physical_table_pair_20261118_01.py'
sys.path.insert(0,str(CODE))
GPU0='GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169'
GPU1='GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e'
GPU2='GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'


def policy():
    spec=importlib.util.spec_from_file_location('physical_table_unchanged_policy',POLICY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def configure(reg):
    import inspect
    import short_lift_bench as base
    if 'graspcl' not in reg['required_components']:
        original=base.verify
        tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
        replaced=[]
        class OptionalUnusedBank(ast.NodeTransformer):
            def visit_Assert(self,node):
                if 'graspcl_model/model/score_cl/online_data.npz' in ast.unparse(node):
                    replaced.append(True)
                    return ast.If(test=ast.parse("'graspcl' in reg['required_components']",mode='eval').body,body=[node],orelse=[])
                return self.generic_visit(node)
        tree=ast.fix_missing_locations(OptionalUnusedBank().visit(tree));assert len(replaced)==1
        scope=dict(original.__globals__);exec(compile(tree,str(SELF)+':optional_unused_bank','exec'),scope)
        generated=scope[original.__name__]
        base.verify=types.FunctionType(generated.__code__,original.__globals__,original.__name__,original.__defaults__)
    module=policy();base=module.configure_benchmark()
    base.__file__=str(SELF)
    base.core.batch.BACKEND=reg['runtime_backend_gpu']
    base.core.batch.SCENES=reg['runtime_scene_gpus']
    if 'graspcl' not in reg['required_components']:
        original_backend=base.Backends
        class NativeBackend(original_backend):
            def start(self):
                super().start()
                self.config['sockets']['graspcl']=str(Path(self.config['sockets']['zerograsp']).with_name('graspcl.sock'))
                self.config['local_adapters']={'graspcl':'Native candidate adapter only; no GraspCL process or model'}
                save(self.campaign/'resident/backend_config.json',self.config)
        base.Backends=NativeBackend
        from zerograsp_short_lift_bench import validate_single
        def validate(out,backends,registration):
            validate_single(out,backends,registration)
            receipt=read(out/'startup_validation.json')
            receipt.update(collision_policy=module.PROTOCOL,extra_closed_table_margin_m=0.,
                           nominal_closed_precheck=False,collision_models_and_table_margin_unchanged=None)
            save(out/'startup_validation.json',receipt)
        base.existing.validate=validate
    return base


def prepare(out,previous):
    import graspcl_pipeline as common
    from graspcl_execute import guard
    old=read(previous/'registration.json');guard(old)
    clock=read(previous/'suite_timing.json');assert clock['all_owned_services_exited']
    assert (previous/'parallel_handoff_requested.json').exists()
    carried={}
    for arm,_,_ in old['arms']:
        complete={}
        for f in (previous/arm/'claims').glob('*.json'):
            claim=read(f);assert claim['state']=='complete',('Unknown slot must not be replayed',f)
            trial=previous/arm/'trials'/f.stem
            assert common.sha(trial/'result.json')==claim['result_sha256']
            assert read(trial/'independent_evaluation.json')['passed']
            complete[f.stem]=claim
        assert {p.name for p in (previous/arm/'trials').iterdir()}==set(complete)
        carried[arm]=complete
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(previous/'source',out/'source')
    shutil.copy2(SELF,out/'source/parallel_runtime.py')
    shutil.copy2(CODE/'zerograsp_short_lift_bench.py',out/'source/zerograsp_short_lift_bench.py')
    shutil.copy2(previous/'report_baseline.json',out/'report_baseline.json')
    receipt={'created_at':datetime.now().astimezone().isoformat(),'previous':str(previous),
        'user_authorized_runtime_change':True,'algorithm_changed':False,'same_collision_and_success_policy':old['collision_policy'],
        'total_budget':400,'prior_claims_inherited_without_replay':{a:len(v) for a,v in carried.items()},
        'inherited_result_sha256':{a:{k:v['result_sha256'] for k,v in claims.items()} for a,claims in carried.items()},
        'prior_epoch_timing':clock,'prior_services_exited':True,'unknown_slots':[]}
    save(out/'continuation.json',receipt)
    reg=deepcopy(old)
    reg['source_hashes'].update({str(SELF):common.sha(SELF),str(CODE/'zerograsp_short_lift_bench.py'):common.sha(CODE/'zerograsp_short_lift_bench.py'),str(previous/'registration.json'):common.sha(previous/'registration.json'),str(out/'continuation.json'):common.sha(out/'continuation.json')})
    reg.update(runtime_change='Two concurrent single-scene arms with independent resident neural/planner backends',
        parallel_suites={'fixed_zerograsp':str(out/'zerograsp'),'fixed_graspcl':str(out/'graspcl')},
        inherited_completed=sum(len(v) for v in carried.values()),source_campaign=str(previous))
    save(out/'registration.json',reg)
    for group,arm,profile,backend,scene,required in [
        ('zerograsp','fixed_zerograsp','zerograsp_200',GPU1,GPU1,['zerograsp','curobo']),
        ('graspcl','fixed_graspcl','graspcl_200',GPU2,GPU0,['zerograsp','graspcl','curobo'])]:
        child=out/group;child.mkdir();(child/'source').symlink_to(out/'source',target_is_directory=True)
        c=deepcopy(reg);c.update(arms=[[arm,profile,'fixed_orientation']],total_slots=200,
            attempt_budget_by_arm={arm:200},required_components=required,
            runtime_backend_gpu=backend,runtime_scene_gpus=[scene],scene_workers=1,
            backend_lock=str(child/'backend.lock'),inherited_completed=len(carried[arm]),
            source_evaluation_results_carried=len(carried[arm]),prior_test_results_carried=len(carried[arm]),
            component_lifecycle='One preload/warmup per owned component before the first newly claimed slot in this resumed runtime epoch; no per-case reload or cold fallback.')
        save(child/'registration.json',c)
        shutil.copy2(out/'report_baseline.json',child/'report_baseline.json')
        target=child/arm;target.mkdir()
        for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(target/folder).mkdir()
        for name in ['reference/common_initial_conditions.json','storage_policy.json']:
            shutil.copy2(previous/arm/name,target/name)
        (target/'resident').symlink_to(child/'resident',target_is_directory=True)
        a=deepcopy(c);a.update(suite=str(child),experiment_arm=arm,pipeline_profile=profile,
            orientation_mode='fixed_orientation',attempt_budget=200,
            common_reset_source=str(target/'reference/common_initial_conditions.json'))
        a['source_hashes'][str(target/'reference/common_initial_conditions.json')]=common.sha(target/'reference/common_initial_conditions.json')
        save(target/'registration.json',a)
        for identity,claim in carried[arm].items():
            (target/'trials'/identity).symlink_to(previous/arm/'trials'/identity,target_is_directory=True)
            save(target/'claims'/(identity+'.json'),claim)
            for suffix in ['.json','_binding.json']:
                p=previous/arm/'entries'/(identity+suffix)
                if p.exists():shutil.copy2(p,target/'entries'/p.name)
        import short_lift_bench as baseline
        baseline.existing.summarize(target,'prepared')
        baseline.update_suite(child,'prepared')
        (out/arm).symlink_to(target,target_is_directory=True)
    aggregate(out,{})
    print('PARALLEL CONTINUATION REGISTERED',receipt['prior_claims_inherited_without_replay'],flush=True)


def aggregate(out,cache):
    module=policy();reg=read(out/'registration.json');states={};clocks={}
    for arm,child in reg['parallel_suites'].items():
        child=Path(child);states[arm]=read(child/'progress.json')
        clocks[arm]=read(child/'suite_timing.json') if (child/'suite_timing.json').exists() else {}
    phase='complete' if all(v['phase']=='complete' for v in states.values()) else 'halted' if any(v['phase']=='halted' for v in states.values()) else 'running'
    progress={'phase':phase,'registered':400,'completed':sum(v['completed'] for v in states.values()),
        'active_round':'parallel','rounds':{arm:next(iter(v['rounds'].values())) for arm,v in states.items()},
        'updated_at':datetime.now().astimezone().isoformat()}
    save(out/'progress.json',progress)
    all_exited=all(c.get('all_owned_services_exited') for c in clocks.values())
    save(out/'suite_timing.json',{'phase':phase,'parallel_epochs':clocks,'all_owned_services_exited':all_exited,
        'inherited_runtime_epoch':read(out/'continuation.json')['prior_epoch_timing'],
        'scope':'Current arm clocks cover their resumed work only; inherited cases and startup/shutdown belong to the preserved prior runtime epoch.'})
    if phase=='complete' and all_exited:
        checks={arm:read(Path(path)/'final_verification.json') for arm,path in reg['parallel_suites'].items()}
        assert all(v['passed'] for v in checks.values())
        save(out/'final_verification.json',{'passed':True,'terminal_results':400,'arms':checks,
            'same_400_slot_budget':True,'inherited_slots_not_replayed':True,'all_owned_services_exited':True})
    metrics=module.update_report(out,cache)
    return phase,all_exited,metrics


def watch(out):
    cache={}
    while True:
        try:phase,exited,metrics=aggregate(out,cache)
        except json.JSONDecodeError:
            time.sleep(1);continue
        print(json.dumps({'phase':phase,'completed':metrics['completed'],'arms':metrics['arms']},ensure_ascii=False),flush=True)
        if phase in ['complete','halted'] and exited:return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute','watch'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,args.previous.resolve())
    elif args.stage=='watch':watch(out)
    else:
        registration=read(out/'registration.json') if args.stage=='run' else read(Path(read(args.entry)['campaign_registration']))
        base=configure(registration)
        if args.stage=='run':base.run(out)
        else:base.execute(out,args.entry.resolve())
