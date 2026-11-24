"""Fresh TCP43 demonstrations followed by the frozen 160-scene paired benchmark."""
from arx_project.layout import canonical_path
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
sys.path.insert(0,str(CODE.parent/'reachability/工具'))
import 运行选定高度抓取对照 as legacy
from 公共 import read,save,sha,now
from graspcl_示教兼容 import self_test,validate_trial

PRIOR=CODE/'outputs/arx_selected_height_pair_20261120_01'
SOURCE_INPUTS=CODE/'benchmark_inputs/arx_native_post_20261117_01'
GROUPS=['collection','fixed_zerograsp','fixed_graspcl']
GPU='GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
LAYOUTS=['S1_open','S2_edge','S3_sparse','S4_clutter']


def imports():legacy.imports()


def prepare(root):
    imports()
    from graspcl_execute import guard
    previous=read(PRIOR/'registration.json');guard(previous)
    assert read(PRIOR/'progress.json')['completed']==400
    assert read(PRIOR/'运行结束.json')['all_owned_services_exited']
    initial=read(PRIOR/'高度初态/common_initial_conditions.json')
    old=read(SOURCE_INPUTS/'manifest.json')
    by_model={}
    for entry in old['schedule']:by_model.setdefault(entry['model_id'],[]).append(entry)
    assert len(by_model)==40
    collection=[]
    for number,(model,entries) in enumerate(by_model.items()):
        preferred=LAYOUTS[number%4]
        choices=[e for e in entries if e['layout'] in LAYOUTS]
        choices.sort(key=lambda e:e['layout']!=preferred)
        admitted=[read(SOURCE_INPUTS/'admission'/e['id']/'admission.json') for e in choices]
        record=next(a for a in admitted if a['passed'])
        collection.append(deepcopy(record['entry']))
    tests=[e for e in previous['schedule'] if e['layout']!='S5_occluded']
    assert len(tests)==160 and Counter(e['layout'] for e in tests)==dict.fromkeys(LAYOUTS,40)
    source_models={a['model_id'] for e in collection for a in e['scene']['actors']}
    test_models=set(previous['test_object_ids'])
    assert not source_models&test_models
    asset=CODE/'assets/data/assets/mani_skill2_ycb/models'
    intersections={}
    for mesh in ['collision.ply','textured.obj']:
        a={sha(asset/m/mesh) for m in source_models};b={sha(asset/m/mesh) for m in test_models}
        intersections[mesh]=sorted(a&b)
    assert not any(intersections.values())
    root.mkdir(parents=True,exist_ok=False);(root/'source').mkdir()
    save(root/'离线接口检查.json',self_test())
    save(root/'隔离核验.json',{'passed':True,'source_models':sorted(source_models),'test_models':sorted(test_models),
        'model_intersection':[],'mesh_intersections':intersections,'semantic_category_disjoint_asserted':False,
        'test_case_ids':[e['id'] for e in tests],'excluded_occluded_ids':[e['id'] for e in previous['schedule'] if e['layout']=='S5_occluded']})
    (root/'高度初态').symlink_to(PRIOR/'高度初态',target_is_directory=True)
    source_inputs=root/'collection_inputs';(source_inputs/'declared_resets').mkdir(parents=True)
    save(source_inputs/'common_initial_conditions.json',initial)
    files=[SELF,CODE/'graspcl_示教兼容.py',Path(legacy.__file__),SOURCE_INPUTS/'manifest.json',
        PRIOR/'registration.json',PRIOR/'运行结束.json',root/'隔离核验.json',root/'离线接口检查.json']
    for directory in [CODE/'arx_pipeline',CODE.parent/'research_dependencies/graspcl_model_public/grasp_learning_cl']:
        files+=list(directory.rglob('*.py'))
    base=deepcopy(previous)
    base.update(created_at=now(),suite=str(root),selected_lift_m=.452,
        schedule=collection+tests,arms=[(g,'zerograsp_200' if g!='fixed_graspcl' else 'graspcl_200','fixed_orientation') for g in GROUPS],
        bank=str(root/'bank_tcp43'),source_feedback_ids_in_memory_order=[],
        source_collection_budget=40,test_scene_count=160,test_slot_budget=320,total_slots=360,
        test_object_ids=sorted(test_models),excluded_old_object_ids=sorted(source_models),
        required_components=['zerograsp','curobo'],backend_lock=str(root/'backend.lock'),
        source_campaign=str(PRIOR),old_results_carried=0,prior_test_results_carried=0,
        user_authorization='Fix demonstration compatibility and evaluate the original 160 non-S5 scenes; both prior pipelines retained.',
        fresh_demonstration_policy='Forty independent development objects, one visibility-admitted non-S5 recipe each; TCP43, joint4=.452, >=1cm for 3s and bilateral contacts. No test outcomes enter memory.',
        template_policy='Fresh TCP43 contacts and embeddings only; no historical ARX positives, negatives, or templates are imported.',
        source_object_ids=sorted(source_models),excluded_occluded_ids=[e['id'] for e in previous['schedule'] if e['layout']=='S5_occluded'],
        physical_slot_budget=360,automatic_grasp_retries=False,required_components_by_phase={'collection':['zerograsp','curobo'],'test':['zerograsp','graspcl','curobo']},
        source_raw_evidence_retained=True,bank_frozen_before_test_claims=True,test_feedback_updates=False,
        experimental_variable='Fresh matching GraspCL bank; both test arms use the same fixed 45.2cm height and the existing physical-table, TCP43 and short-lift execution. No seven-DOF motion.')
    for g in GROUPS:
        folder=root/g
        for name in ['trials','claims','entries','backend_requests','reference','workers','resource_wait']:(folder/name).mkdir(parents=True)
        (folder/'resident').symlink_to(root/'resident',target_is_directory=True)
        shutil.copy2(PRIOR/'fixed_zerograsp/storage_policy.json',folder/'storage_policy.json')
        save(folder/'reference/common_initial_conditions.json',initial)
        child=deepcopy(base);is_source=g=='collection';schedule=collection if is_source else tests
        child.update(schedule=schedule,attempt_budget=len(schedule),total_slots=len(schedule),layouts=LAYOUTS,
            experiment_arm=g,pipeline_profile='graspcl_200' if g=='fixed_graspcl' else 'zerograsp_200',
            orientation_mode='fixed_orientation',partition='collection' if is_source else 'test',
            required_components=['zerograsp','curobo'] if is_source else ['zerograsp','graspcl','curobo'],
            test_object_ids=sorted(source_models if is_source else test_models),
            excluded_old_object_ids=sorted(test_models if is_source else source_models),
            source_feedback_ids_in_memory_order=[],inputs=str(source_inputs if is_source else PRIOR/'高度初态'),
            common_reset_source=str(folder/'reference/common_initial_conditions.json'))
        for row in schedule:
            slot=row['id']
            if is_source:
                e=deepcopy(row);original=read(SOURCE_INPUTS/'resets'/slot/'initial_conditions.json')
                assert original['initial_q']==read(SOURCE_INPUTS/'common_initial_conditions.json')['initial_q']
                ref=folder/'reference'/slot;ref.mkdir();original['initial_q']=deepcopy(initial['initial_q'])
                save(ref/'initial_conditions.json',original)
                warm=read(Path(base['warmup_trial'])/'state.json');old_lift=warm['qpos'][warm['joint_names'].index('joint4')]
                e['observation']['goal_world'][2][3]+=.452-old_lift
                admission=ref/'admission.json';save(admission,{'entry':{'observation':e['observation'],'scene':e['scene']}})
                e.update(historical_trial=str(ref),source_admission=str(admission),admission_failed=False)
                declared={'spawn':e['scene'],'base_pose':initial['base_pose'],'initial_q':initial['initial_q']}
                save(source_inputs/'declared_resets'/slot/'declared_reset.json',declared)
                files += [SOURCE_INPUTS/'admission'/slot/'admission.json',SOURCE_INPUTS/'resets'/slot/'initial_conditions.json',admission,ref/'initial_conditions.json']
            else:
                e=read(PRIOR/g/'entries'/(slot+'.json'))
                files += [PRIOR/g/'entries'/(slot+'.json')]
            e.update(campaign_registration=str(folder/'registration.json'),pipeline_profile=child['pipeline_profile'],orientation_mode='fixed_orientation',experiment_arm=g)
            ep=folder/'entries'/(slot+'.json');save(ep,e)
            evidence={str(Path(e['source_admission'])):sha(e['source_admission']),
                str(Path(e['historical_trial'])/'initial_conditions.json'):sha(Path(e['historical_trial'])/'initial_conditions.json')}
            save(folder/'entries'/(slot+'_binding.json'),{'entry_sha256':sha(ep),'source_evidence_sha256':evidence})
        save(folder/'registration.json',child)
    files+=list(source_inputs.rglob('*.json'))
    sources={**previous['source_hashes'],**{str(p.resolve()):sha(p) for p in set(files)}}
    for folder in [root,*[root/g for g in GROUPS]]:
        r=base if folder==root else read(folder/'registration.json');r['source_hashes']=sources
        save(folder/'registration.json',r)
    for p in [SELF,CODE/'graspcl_示教兼容.py',Path(legacy.__file__)]:shutil.copy2(p,root/'source'/p.name)
    save(root/'collection_inputs/登记.json',{'source':str(SOURCE_INPUTS),'source_scenes':40,'selection':'One source object each, preferring cyclic S1..S4 visibility-admitted recipes without grasp outcome screening.','layouts':dict(Counter(e['layout'] for e in collection))})
    progress(root,'prepared')
    print('PREPARED 40 independent collection slots + 2x160 non-S5 tests',flush=True)


def progress(root,phase):
    groups={}
    for g in GROUPS:
        rows=[];unknown=0
        for p in (root/g/'claims').glob('*.json'):
            claim=read(p)
            if claim['state']=='complete':rows.append(read(root/g/'trials'/p.stem/'result.json'))
            else:unknown+=1
        groups[g]={'registered':40 if g=='collection' else 160,'completed':len(rows),
            'planning_passes':sum(bool(r['planning_success']) for r in rows),'successes':sum(bool(r['success']) for r in rows),
            'physical_attempts':sum(bool(r['physical_attempt']) for r in rows),'in_progress_or_unknown':unknown,
            'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success']))}
    save(root/'progress.json',{'phase':phase,'source_collection':groups.pop('collection'),'groups':groups,
        'test_completed':sum(g['completed'] for g in groups.values()),'test_registered':320,'updated_at':now()})


def build_bank(root):
    imports();sys.path.insert(0,str(CODE.parent/'research_dependencies/graspcl_model_public'))
    import graspcl_pipeline as common
    import graspcl_unseen_bank as bank_tools
    from unittest.mock import patch
    from graspcl_execute import guard
    source=root/'collection';reg=read(source/'registration.json');guard(reg)
    assert all(read(source/'claims'/(e['id']+'.json'))['state']=='complete' for e in reg['schedule'])
    bank=root/'bank_tcp43';bank_tools.extract(bank);common.seed();module=common.load_model(bank)
    assert not len(module.grasp_scoring_module_cl.X_on)
    assert not module.grasp_recall_module_object._db and not module.grasp_recall_module_patch._db
    records=[];excluded=[]
    for e in reg['schedule']:
        d=source/'trials'/e['id'];r=read(d/'result.json');evaluation=read(d/'independent_evaluation.json');assert evaluation['passed']
        label=1 if r['success'] and evaluation.get('independently_successful') else None
        if (label is None and r['physical_attempt'] and r['reason'] in ['no_bilateral_target_contact','insufficient_hold_height']
                and r.get('stages',{}).get('close',{}).get('settled') and not read(d/'forbidden_contacts.json')):label=0
        if label is None:excluded.append({'id':e['id'],'reason':r['reason']});continue
        original_read=common.read;plan=deepcopy(read(d/'plan_result.json'));selected=plan['selected']
        candidate=read(d/'graspcl/candidate_pool.json')['candidates'][selected['index']]
        selected['network_row']=candidate['metadata']['source_postprocessed_row']
        def legacy_read(path):return plan if Path(path)==d/'plan_result.json' else original_read(path)
        try:
            with patch.object(common,'TCP_OFFSET',.043),patch.object(common,'read',legacy_read):
                grasp=common.commanded_grasp(d,{**e,'label':label,'original_reason':r['reason']})
        except RuntimeError as error:
            excluded.append({'id':e['id'],'reason':str(error)});continue
        grasp['metadata'].update(contact_storage_tcp_m=.043,source_height_m=.452,
            source_evaluation_protocol=reg['evaluation_protocol'],source_minimum_hold_lift_m=.01,
            source_required_hold_seconds=3,source_network_depth_m=candidate['metadata']['source_network_depth_m'])
        eef=np.array(selected['grasp_world']);pose=np.array(grasp['pose'])
        assert np.allclose(pose[:3,3]-.043*pose[:3,2],eef[:3,3],atol=1e-9)
        state=read(d/'state.json')
        assert abs(state['qpos'][state['joint_names'].index('joint4')]-.452)<.001
        scene,target,_=common.clouds(d);encoded=module.encoder.predict(scene,[grasp])
        module.grasp_scoring_module_cl.add_datapoint(encoded)
        if label==1:
            module.grasp_recall_module_object.add_demonstration(target,scene,deepcopy(encoded),'unknown','arx_tcp43_hold10')
            module.grasp_recall_module_patch.add_demonstration(target,scene,deepcopy(encoded),'unknown','arx_tcp43_hold10',debug=False)
        record={'id':e['id'],'label':label,'trial':str(d),'grasp':grasp,'input_sha256':common.inputs(d),
            'result_sha256':sha(d/'result.json'),'evaluation_sha256':sha(d/'independent_evaluation.json')}
        common.save(bank/'demonstrations'/(e['id']+'.json'),record);records.append(record)
        print('BANK',e['id'],label,flush=True)
    assert sum(r['label']==1 for r in records)>=12,'Fewer than twelve clean positive demonstrations; tests will not start'
    module.feedback_counts={'compat_tcp43_hold10':{'labels':len(records),'demonstrations':sum(r['label']==1 for r in records)}}
    module.save(bank/'model')
    save(bank/'bank_complete.json',{'positive_demonstrations':sum(r['label']==1 for r in records),
        'negative_feedback':sum(r['label']==0 for r in records),'online_points':len(records),
        'object_templates':len(module.grasp_recall_module_object._db),'patch_templates':len(module.grasp_recall_module_patch._db),
        'source_ids':[r['id'] for r in records],'contact_storage_tcp_m':.043,'planner_recall_contact_shift_m':0.,
        'model_sha256':{str(p):sha(p) for p in (bank/'model').rglob('*') if p.is_file()}})
    save(bank/'excluded_collection_cases.json',excluded)
    save(bank/'provenance.json',{'passed':True,'source_budget':40,'clean_labels':len(records),'historical_arx_feedback_imported':0,
        'source_test_model_and_mesh_overlap':0,'current_test_observations_or_outcomes_used':0,
        'all_contacts_authored_tcp43':True,'all_labels_use_hold10_3seconds':True,
        'source_ids':[r['id'] for r in records],'raw_source_images_clouds_and_execution_retained':True})


def serve(arguments):
    imports();sys.path.insert(0,str(CODE.parent/'research_dependencies/graspcl_model_public'))
    import short_lift_services as services
    from independent_services import InputGuard
    from graspcl_示教兼容 import install
    if arguments[arguments.index('--kind')+1]=='curobo':legacy.policy().install_collision_policy();install()
    class PhaseGuard(InputGuard):
        def check(self,directory,kind,**flags):
            path=Path(directory).resolve()
            if path.parent.name!='trials':return super().check(directory,kind,**flags)
            assert path.parent.parent.parent==self.campaign and path.parent.parent.name in GROUPS
            child=read(path.parent.parent/'registration.json')
            saved=(self.reg,self.allowed,self.excluded,self.entries)
            try:
                self.reg=child;self.allowed=set(child['test_object_ids']);self.excluded=set(child['excluded_old_object_ids'])
                self.entries={e['id']:e for e in child['schedule']}
                return super().check(directory,kind,**flags)
            finally:self.reg,self.allowed,self.excluded,self.entries=saved
    services.InputGuard=PhaseGuard;sys.argv=[str(SELF),*arguments];services.main()


def simulator(root,address,owner):
    imports();import short_lift_bench as bench
    original=bench.execute
    def checked(directory,entry):
        original(directory,entry);reg=read(read(entry)['campaign_registration']);validate_trial(directory,reg)
    bench.execute=checked
    legacy.simulator(root,address,owner)


def run(root):
    imports()
    from graspcl_execute import guard
    from benchmark_lifecycle import mark_ready
    from arx_service_readiness import wait_ready
    import short_lift_bench as bench
    reg=read(root/'registration.json');guard(reg)
    resident=root/'resident';resident.mkdir();sockets=Path('/tmp')/('arxcompat_'+sha(root/'registration.json')[:10]);sockets.mkdir(mode=0o700)
    streams=[];children={};sims={};config={'sockets':{}};began=time.perf_counter();failed=False
    def launch(kind,policy_registration=None):
        python=canonical_path(CODE.parent/{'zerograsp':'tmp/arx-fullchain/env/bin/python','curobo':'curobo/.venv/bin/python','graspcl':'tmp/arx-graspcl-env/bin/python'}[kind])
        address=sockets/(kind+'.sock');config['sockets'][kind]=str(address)
        command=['bash','-c','source "$1"; shift; exec "$@"','compat-service',str(CODE.parent/'arx_core/scripts/curobo_arx/environment.sh'),
            str(python),str(SELF),'serve','--kind',kind,'--socket',str(address),'--root',str(resident),'--warmup-trial',reg['warmup_trial']]
        if kind=='graspcl':command+=['--bank',str(root/'bank_tcp43'),'--policy-registration',str(policy_registration)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if kind=='graspcl' else GPU,ARX_NEURAL_GPU=GPU,ARX_PLANNER_GPU=GPU,ARX_SIM_GPU=GPU,
            LILO_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',GRASP_TRAJOPT_POLICY='interactive');env.pop('PYTHONPATH',None)
        stream=(resident/(kind+'.log')).open('x');streams.append(stream)
        children[kind]=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        health=wait_ready(address,resident/(kind+'_ready.json'),children[kind],kind,timeout=1800)
        print('READY',kind,children[kind].pid,health['startup_seconds'],flush=True)
    def run_group(g,phase):
        folder=root/g;group=read(folder/'registration.json')
        for e in group['schedule']:
            if (root/'STOP.json').exists():return
            guard(group)
            if shutil.disk_usage(root).free<1073741824:raise RuntimeError('Registered 1 GiB storage floor reached')
            slot=e['id'];claim=folder/'claims'/(slot+'.json');reserved=claim.with_suffix('.reserved');trial=folder/'trials'/slot
            assert not claim.exists() and not reserved.exists() and not trial.exists()
            with reserved.open('x') as f:f.write(now()+'\n')
            c={'state':'claimed','id':slot,'group':g,'claimed_at':now()};save(claim,c)
            try:
                entry=folder/'entries'/(slot+'.json');details=read(entry)
                if details.get('admission_failed'):
                    trial.mkdir();save(trial/'result.json',{**e,'success':False,'planning_success':False,'physical_attempt':False,'reason':'independent_scene_admission_failed'})
                    bench.placeholder(trial,'Original registered initialization failure; no replacement')
                    save(trial/'independent_evaluation.json',{'passed':True,'independently_successful':False,'physical_attempt':False,'admission_failed':True})
                else:
                    worker,address=sims['source' if g=='collection' else g];assert worker.poll() is None
                    response=legacy.rpc(address,{'command':'execute','output':str(trial),'entry':str(entry),
                        'log':str(folder/'entries'/(slot+'.log'))},timeout=1800)
                    if response['status']!='ok':raise RuntimeError(response['error'])
                    assert read(trial/'independent_evaluation.json')['passed']
                guard(group);result=read(trial/'result.json');c.update(state='complete',finished_at=now(),result_sha256=sha(trial/'result.json'));save(claim,c)
                with (root/'summary.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);progress(root,phase)
                print('RESULT',g,slot,result['success'],result['reason'],flush=True)
            except BaseException:
                save(root/'STOP.json',{'time':now(),'group':g,'slot':slot,'error':traceback.format_exc(),'automatic_replay':False});raise
    def start_sim(name):
        address=sockets/(name+'.sock');env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,MANISKILL_GPU_UUID=GPU,ARX_SIM_GPU=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        stream=(resident/(name+'_simulation.log')).open('x');streams.append(stream)
        command=['bash',str(CODE/'run.sh'),str(SELF),'simulator','--output',str(root),'--socket',str(address),'--owner-pid',str(os.getpid())]
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);sims[name]=(child,address)
        deadline=time.monotonic()+1800
        while not address.exists():
            assert child.poll() is None,'Simulation preload failed'
            if time.monotonic()>deadline:raise TimeoutError('Simulation preload exceeded budget')
            time.sleep(1)
        assert legacy.rpc(address,{'command':'health'})['runtime_import_count']==1
    try:
        progress(root,'starting_collection_components')
        launch('zerograsp');launch('curobo')
        # The native adapter dispatches on this configured name without making
        # a GraspCL RPC; no scorer is loaded during demonstration collection.
        source_config=deepcopy(config);config['sockets']['graspcl']=str(sockets/'graspcl.sock')
        save(resident/'backend_config.json',config)
        mark_ready(resident,['zerograsp','curobo'],source_config,children);shutil.copy2(resident/'all_ready.json',resident/'collection_all_ready.json')
        start_sim('source');progress(root,'collecting_matching_demonstrations');run_group('collection','collecting_matching_demonstrations')
        progress(root,'building_tcp43_bank')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(CODE.parent/'research_dependencies/graspcl_model_public')+':'+str(CODE),OMP_NUM_THREADS='4')
        with (root/'bank_build.log').open('x') as log:
            subprocess.run([str(canonical_path(CODE.parent/'tmp/arx-graspcl-env/bin/python')),str(SELF),'build-bank','--output',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        bank=read(root/'bank_tcp43/bank_complete.json');assert read(root/'bank_tcp43/provenance.json')['passed']
        assert bank['contact_storage_tcp_m']==.043 and bank['planner_recall_contact_shift_m']==0.
        for g in ['fixed_zerograsp','fixed_graspcl']:
            path=root/g/'registration.json';r=read(path);r['source_feedback_ids_in_memory_order']=bank['source_ids']
            r['source_hashes'].update(bank['model_sha256']);r['source_hashes'][str(root/'bank_tcp43/bank_complete.json')]=sha(root/'bank_tcp43/bank_complete.json')
            save(path,r)
        save(root/'bank_frozen_before_tests.json',{'time':now(),'bank_complete_sha256':sha(root/'bank_tcp43/bank_complete.json'),
            'positive':bank['positive_demonstrations'],'negative':bank['negative_feedback'],'test_claims':0})
        launch('graspcl',root/'fixed_graspcl/registration.json');save(resident/'backend_config.json',config)
        mark_ready(resident,['zerograsp','graspcl','curobo'],config,children)
        # Retain the already warm renderer process for the first test arm.
        sims['fixed_zerograsp']=sims.pop('source');start_sim('fixed_graspcl')
        save(root/'test_started.json',{'time':now(),'all_components_ready':True,'registered_slots':320,'bank_frozen':True})
        progress(root,'testing_160_scene_pair')
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures=[workers.submit(run_group,g,'testing_160_scene_pair') for g in ['fixed_zerograsp','fixed_graspcl']]
            for f in futures:f.result()
        for g in ['fixed_zerograsp','fixed_graspcl']:
            assert len(list((root/g/'claims').glob('*.json')))==160
            guard(read(root/g/'registration.json'))
        progress(root,'complete')
    except BaseException:
        failed=True;save(root/'控制器故障.json',{'time':now(),'error':traceback.format_exc()});progress(root,'halted');raise
    finally:
        exits={}
        for name,(child,address) in {**{k:(v,Path(config['sockets'][k])) for k,v in children.items()},**sims}.items():
            if child.poll() is None:
                try:legacy.rpc(address,{'command':'shutdown'},timeout=10);child.wait(timeout=20)
                except Exception:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            exits[name]={'pid':child.pid,'returncode':child.returncode,'exited':child.poll() is not None}
        for stream in streams:stream.close()
        save(root/'运行结束.json',{'time':now(),'failed':failed,'all_owned_services_exited':all(v['exited'] for v in exits.values()),'services':exits,'wall_seconds':time.perf_counter()-began})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='serve':serve(sys.argv[2:])
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','build-bank','simulator'])
        p.add_argument('--output',required=True,type=Path);p.add_argument('--socket',type=Path);p.add_argument('--owner-pid',type=int)
        a=p.parse_args();root=a.output.resolve()
        if a.stage=='prepare':prepare(root)
        elif a.stage=='run':run(root)
        elif a.stage=='build-bank':build_bank(root)
        else:simulator(root,a.socket,a.owner_pid)
