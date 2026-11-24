"""New object/mesh/scene/seed-disjoint ARX test set; admission has no grasping."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
import graspcl_pipeline as common

CODE=Path(__file__).resolve().parent
OLD=CODE/'benchmark_inputs/arx_native_post_20261117_01'
ASSETS=CODE/'assets/data/assets/mani_skill2_ycb'
LAYOUTS=['S1_open','S2_edge','S3_sparse','S4_clutter','S5_occluded']
GPUS=['GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169','GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e']


def now():return datetime.now().astimezone().isoformat()


def family(model):
    code=int(model.split('_')[0].split('-')[0])
    if code==2:return 'packaging'
    if code in [28,29,32]:return 'vessels_utensils'
    if code in [44,51,52,59]:return 'tools'
    if code in [53,54,63]:return 'balls'
    if code in [36,70,71]:return 'blocks'
    if code==65:return 'cups'
    return 'toys'


def prepare(inputs):
    old=common.read(OLD/'manifest.json')
    used={e['model_id'] for e in old['schedule']};oldseeds={e['seed'] for e in old['schedule']}
    db=common.read(ASSETS/'info_pick_v0.json')
    for entry in old['schedule']:
        source=common.read(OLD/'admission'/entry['id']/'admission.json')
        used.update(a['model_id'] for a in source['entry'].get('scene',{}).get('actors',[]))
    new=sorted(set(db)-used)
    assert len(new)==38 and len(used)==40
    trainhash={kind:{common.sha(ASSETS/'models'/m/kind) for m in used if (ASSETS/'models'/m/kind).exists()}
               for kind in ['collision.ply','textured.obj']}
    catalog=[];asset_hashes={}
    for m in new:
        directory=ASSETS/'models'/m
        for kind in trainhash:
            assert (directory/kind).exists(),(m,kind)
            assert common.sha(directory/kind) not in trainhash[kind],('shared geometry',m,kind)
        for path in directory.rglob('*'):
            if path.is_file():asset_hashes[str(path)]=common.sha(path)
        info=db[m];scale=info.get('scales',[1.])[0]
        catalog.append({'model_id':m,'family':family(m),'scale':scale,'density':info['density'],
            'bbox_m':((np.asarray(info['bbox']['max'])-info['bbox']['min'])*scale).tolist(),
            'collision_sha256':common.sha(directory/'collision.ply'),'visual_sha256':common.sha(directory/'textured.obj')})
    rng=np.random.default_rng(1711174300);raw=[];occupied={s+v*1009 for s in oldseeds for v in range(12)}
    for layout in LAYOUTS:
        # Every unused model appears in each layout; two extra models come from
        # the least represented families, decided before seeing any scene.
        pool=list(new);counts=Counter(family(m) for m in pool)
        for _ in range(2):
            minimum=min(counts.values());groups=sorted(k for k,v in counts.items() if v==minimum)
            group=str(rng.choice(groups));m=str(rng.choice([m for m in new if family(m)==group]))
            pool.append(m);counts[group]+=1
        rng.shuffle(pool)
        for model in pool:
            while True:
                seed=int(rng.integers(1_000_000_000,2_000_000_000))
                keys={seed+v*1009 for v in range(12)}
                if not (keys&occupied):occupied.update(keys);break
            raw.append({'model_id':model,'family':family(model),'layout':layout,'seed':seed})
    pending=list(raw);schedule=[]
    while pending:
        previous=schedule[-1] if schedule else None
        options=[e for e in pending if previous is None or (e['layout']!=previous['layout'] and e['model_id']!=previous['model_id'] and e['family']!=previous['family'])]
        if not options:options=[e for e in pending if previous is None or e['layout']!=previous['layout']] or pending
        sizes=Counter(e['layout'] for e in pending)
        chosen=max(options,key=lambda e:sizes[e['layout']]+float(rng.random()))
        pending.remove(chosen);schedule.append(chosen)
    for index,e in enumerate(schedule):e['id']=f'I{index+1:04d}_{e["layout"]}_{e["model_id"]}'
    assert len(schedule)==200 and len({e['id'] for e in schedule})==200
    assert Counter(e['layout'] for e in schedule)=={layout:40 for layout in LAYOUTS}
    inputs.mkdir(parents=True,exist_ok=False)
    for name in ['admission','admission_claims','admission_workers','declared_resets']:(inputs/name).mkdir()
    shutil.copytree(OLD/'frozen_bank',inputs/'frozen_bank')
    complete=common.read(inputs/'frozen_bank/bank_complete.json')
    complete['model_sha256']={str(p):common.sha(p) for p in (inputs/'frozen_bank/model').rglob('*') if p.is_file()}
    common.save(inputs/'frozen_bank/bank_complete.json',complete)
    for name in ['common_initial_conditions.json','storage_policy.json']:shutil.copy2(OLD/name,inputs/name)
    bank_ids=set(complete['source_ids'])
    assert not ({e['id'] for e in schedule}&bank_ids)
    excluded_hashes={kind:sorted(values) for kind,values in trainhash.items()}
    plan={'created_at':now(),'seed_master':1711174300,'schedule':schedule,'layouts':LAYOUTS,
        'registered':200,'test_object_ids':new,'excluded_old_object_ids':sorted(used),
        'excluded_old_scene_ids':[e['id'] for e in old['schedule']],
        'excluded_old_seed_values':sorted(oldseeds),'old_mesh_hashes':excluded_hashes,
        'test_catalog':catalog,'asset_sha256':asset_hashes,'bank':str(inputs/'frozen_bank'),
        'bank_source':str(OLD/'frozen_bank'),'source_feedback_ids_in_memory_order':complete['source_ids'],
        'warmup_trial':old['warmup_trial'],'success':old['success'],'camera':old['camera'],
        'table_height_m':old['table_height_m'],'recording_camera':old['recording_camera'],
        'admission_protocol':'Only unchanged reset/stability, declared wrist observation, visibility, physical 20-40% occlusion and duplicate checks. No grasp candidates, grasp IK, learned scores, grasp motions or outcome screening. Max 12 fixed-seed variants. Failed admission remains an explicit failure in both arms, never replaced.',
        'observation_protocol':old['observation_protocol'],
        'model_policy':'All 38 installed YCB model IDs outside the old 40, including size/part variants. Test targets and distractors both use only these 38. No mesh-content duplicate with old collision or visual meshes.',
        'independence_scope':'Disjoint from local ARX demonstration/feedback and previous 200-case benchmark at object-model, geometry, scene and seed level. Upstream pretrained checkpoints are kept fixed; absence from their original training corpora is not asserted.',
        'source_hashes':{str(CODE/'independent_inputs.py'):common.sha(CODE/'independent_inputs.py')},
        'physics_source_hashes':old['source_hashes'],'online_updates_during_evaluation':False,
        'physical_attempts_during_admission':0}
    common.save(inputs/'preparation.json',plan)
    common.save(inputs/'admission_progress.json',{'phase':'prepared','completed':0,'passed':0,'registered':200})
    print('INDEPENDENT INPUTS REGISTERED',inputs,'38 novel models; 200 new scenes',flush=True)


def install_sampler(plan):
    sys.path.insert(0,str(CODE/'arx_pipeline'))
    import baseline
    models=plan['test_object_ids'];db=baseline.DB
    def sample(entry,variant):
        rng=np.random.default_rng(entry['seed']+variant*1009)
        yaw=float(rng.uniform(-np.pi,np.pi));center=np.array([rng.uniform(-.48,-.32),rng.uniform(-.14,.14)])
        if entry['layout']=='S2_edge':
            raw=baseline.actor_spec('target',entry['model_id'],[0,0],yaw);lo,hi=baseline.bounds(raw)
            center=np.array([baseline.TABLE_MIN[0]+.020+(hi[0]-lo[0])/2,rng.uniform(-.28,-.18)])
        actors=[baseline.actor_spec('target',entry['model_id'],center,yaw)]
        layout=entry['layout'];count=int(rng.integers(2,4)) if layout=='S3_sparse' else int(rng.integers(4,7)) if layout=='S4_clutter' else int(rng.integers(2,5)) if layout=='S5_occluded' else 0
        used={entry['model_id']}
        for i in range(count):
            candidates=[m for m in models if m not in used]
            if layout=='S5_occluded' and i==0:
                candidates=[m for m in candidates if np.ptp(baseline.corners(db[m]),axis=0)[2]>=.075]
            model=str(rng.choice(candidates));used.add(model)
            for _ in range(300):
                xy=center+rng.uniform([-.22,-.34],[.36,.34])
                spec=baseline.actor_spec(f'distractor_{i:02d}',model,xy,float(rng.uniform(-np.pi,np.pi)))
                if baseline.separate(spec,actors,gap=.05 if layout=='S3_sparse' else .012):break
            else:raise RuntimeError('Could not place held-out nonoverlapping distractors')
            actors.append(spec)
        assert all(a['model_id'] in models for a in actors)
        return {'actors':actors,'layout':layout,'variant':variant}
    def duplicate(first,second):
        a={x['model_id']:x for x in first['actors']};b={x['model_id']:x for x in second['actors']}
        if set(a)!=set(b):return False
        for model,x in a.items():
            p=np.array(x['pose']);q=np.array(b[model]['pose'])
            if np.linalg.norm(p[:2]-q[:2])>=.03:return False
            symmetric=model.split('_')[0].split('-')[0] in ['053','054','063','065']
            if not symmetric:
                r1=Rotation.from_quat([*p[4:],p[3]]);r2=Rotation.from_quat([*q[4:],q[3]])
                if (r1.inv()*r2).magnitude()>=np.deg2rad(20):return False
        return True
    baseline.sample_scene=sample;baseline.near_duplicate=duplicate
    return baseline


def admit_variant(inputs,entry_path,out,variant):
    plan=common.read(inputs/'preparation.json');baseline=install_sampler(plan)
    baseline.admit(common.read(entry_path),out,max_variants=1,start_variant=variant)


def admit_one(inputs,entry,env):
    out=inputs/'admission'/entry['id'];out.mkdir(exist_ok=False)
    prior=[]
    for p in (inputs/'admission').glob('*/admission.json'):
        r=common.read(p)
        if r['passed'] and r['entry']['model_id']==entry['model_id']:prior.append(r['entry']['scene'])
    bound=deepcopy(entry);bound['avoid_scenes']=prior;common.save(out/'entry.json',bound)
    failures=[]
    for variant in range(12):
        dest=out/f'variant_{variant:02d}'
        command=['bash',str(CODE/'run.sh'),str(Path(__file__).resolve()),'admit_variant','--inputs',str(inputs),
            '--entry',str(out/'entry.json'),'--output',str(dest),'--variant',str(variant)]
        with (out/f'variant_{variant:02d}.log').open('x') as stream:
            child=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        if child.returncode or not (dest/'admission.json').exists():
            raise RuntimeError('Admission infrastructure failed; no physical attempt: '+entry['id'])
        receipt=common.read(dest/'admission.json')
        for error in receipt.get('rejected_variants',[]):
            if any(w in error['error'].lower() for w in ['out of memory','vk_error','resource descriptor','supported physical device']):
                raise RuntimeError('GPU/render infrastructure failure: '+str(error))
        if receipt['passed']:
            for name in ['wrist.png','overview.png','observation_probe_state.json']:shutil.copy2(dest/name,out/name)
            receipt['prior_variants']=failures
            # This is a declared new reset, explicitly not an old execution receipt.
            initial=common.read(inputs/'common_initial_conditions.json')
            common.save(inputs/'declared_resets'/entry['id']/'declared_reset.json',{
                'receipt_kind':'new_declared_reset_not_historical_execution','spawn':receipt['entry']['scene'],
                'base_pose':initial['base_pose'],'initial_q':initial['initial_q'],
                'scene_admission_sha256':common.sha(dest/'admission.json')})
            common.save(out/'admission.json',receipt);return True
        failures+=receipt.get('rejected_variants',[])
    common.save(out/'admission.json',{'passed':False,'entry':{**entry,'admission_failed':True},'rejected_variants':failures})
    return False


def generate(inputs):
    plan=common.read(inputs/'preparation.json');mutex=threading.Lock();failed=threading.Event();active=set();claimed=set()
    if list((inputs/'admission_claims').glob('*.json')):raise RuntimeError('Admission generation already started; preserve state')
    def progress():
        rows=[common.read(p) for p in (inputs/'admission').glob('*/admission.json')]
        common.save(inputs/'admission_progress.json',{'phase':'halted' if failed.is_set() else 'generating',
            'completed':len(rows),'passed':sum(r['passed'] for r in rows),'registered':200,'active':sorted(active),'updated_at':now()})
    def worker(index,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MANISKILL_GPU_UUID=gpu,ARX_SIM_GPU=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        try:
            while not failed.is_set():
                entry=None
                with mutex:
                    options=[e for e in plan['schedule'] if e['id'] not in claimed and e['model_id'] not in active]
                    if options:
                        entry=options[0];claimed.add(entry['id']);active.add(entry['model_id'])
                        common.save(inputs/'admission_claims'/(entry['id']+'.json'),{'state':'initialization_only','worker':index,'started_at':now(),'physical_attempts':0})
                        progress()
                    elif len(claimed)==200:break
                if entry is None:failed.wait(.5);continue
                passed=admit_one(inputs,entry,env)
                with mutex:
                    active.remove(entry['model_id']);common.save(inputs/'admission_claims'/(entry['id']+'.json'),{
                        'state':'complete','worker':index,'passed':passed,'completed_at':now(),'physical_attempts':0});progress()
                print('ADMISSION',entry['id'],passed,flush=True)
        except BaseException:
            failed.set();common.save(inputs/'admission_workers'/(str(index)+'_halt.json'),{'error':traceback.format_exc(),'at':now()})
            with mutex:progress()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,gpu) for i,gpu in enumerate(GPUS)]
        for future in futures:future.result()
    if failed.is_set():raise RuntimeError('Dataset initialization halted; no grasp attempted')
    receipts=[common.read(inputs/'admission'/e['id']/'admission.json') for e in plan['schedule']]
    assert len(receipts)==200
    allowed=set(plan['test_object_ids']);old=set(plan['excluded_old_object_ids'])
    assert not (allowed&old)
    for r in receipts:
        if r['passed']:assert all(a['model_id'] in allowed for a in r['entry']['scene']['actors'])
    manifest=deepcopy(plan)
    manifest.update(frozen_at=now(),admitted=sum(r['passed'] for r in receipts),admission_failed=sum(not r['passed'] for r in receipts),
        admission_sha256={str(p):common.sha(p) for p in (inputs/'admission').glob('*/admission.json')},
        declared_reset_sha256={str(p):common.sha(p) for p in (inputs/'declared_resets').glob('*/declared_reset.json')},
        input_sha256={str(p):common.sha(p) for p in (inputs/'frozen_bank').rglob('*') if p.is_file()})
    for name in ['common_initial_conditions.json','storage_policy.json','preparation.json']:
        manifest['input_sha256'][str(inputs/name)]=common.sha(inputs/name)
    common.save(inputs/'manifest.json',manifest)
    common.save(inputs/'independence_preflight.json',{'passed':True,'scene_id_overlap':0,'object_model_overlap':0,
        'collision_mesh_overlap':0,'visual_mesh_overlap':0,'old_seed_overlap':0,'test_scenes':200,
        'test_model_count':38,'targets_and_distractors_disjoint':True,'physical_attempts':0})
    common.save(inputs/'admission_progress.json',{'phase':'complete','completed':200,'passed':manifest['admitted'],'registered':200,'updated_at':now()})
    print('FROZEN INDEPENDENT INPUTS',manifest['admitted'],'admitted; remaining stay failures',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','generate','admit_variant'])
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--entry',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--variant',type=int,default=0)
    args=parser.parse_args();inputs=args.inputs.resolve()
    if args.stage=='prepare':prepare(inputs)
    elif args.stage=='generate':generate(inputs)
    else:admit_variant(inputs,args.entry.resolve(),args.output.resolve(),args.variant)
