"""Freeze a 50-scene memory / 100-scene category-disjoint ARX experiment."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import threading
import traceback

import numpy as np
import graspcl_pipeline as common
import independent_inputs as original

CODE = Path(__file__).resolve().parent
LAYOUTS = ['S1_open', 'S2_edge', 'S3_sparse', 'S4_clutter']
SOURCE_CODES = {3, 4, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18,
                53, 54, 55, 56, 57, 58, 63, 72}


def category(model):
    code = int(model.split('_')[0].split('-')[0])
    if code in {3, 4, 8, 9}: return '食品纸盒'
    if 11 <= code <= 18: return '水果'
    if code in {53, 54, 55, 56, 57, 58, 63}: return '球类（含弹珠）'
    if code == 72: return '玩具飞机（所有部件变体）'
    if code in {2, 5, 7, 10}: return '食品金属罐'
    if code in {6, 21, 22}: return '瓶装容器'
    if code in {19, 24, 25, 28, 29, 65}: return '餐具容器（杯碗盘壶及盖）'
    if code in {30, 31, 32, 33}: return '餐具工具'
    if code == 26: return '海绵'
    if code == 35: return '电钻'
    if code == 37: return '剪刀'
    if code == 38: return '挂锁'
    if code == 40: return '记号笔'
    if code == 42: return '扳手'
    if code in {43, 44}: return '螺丝刀'
    if code == 48: return '锤子'
    if code in {50, 51, 52}: return '夹具（所有尺寸）'
    if code == 59: return '链条'
    if code in {36, 61, 70}: return '积木块（木块及泡沫块）'
    if code == 62: return '骰子'
    if code == 71: return '插孔板'
    if code == 73: return '乐高积木（所有部件变体）'
    if code == 77: return '魔方'
    raise ValueError('Unreviewed semantic category: ' + model)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    db = common.read(original.ASSETS / 'info_pick_v0.json')
    source = sorted(m for m in db if int(m.split('_')[0].split('-')[0]) in SOURCE_CODES)
    test = sorted(set(db) - set(source))
    assert len(source) == 25 and len(test) == 53
    catalog, hashes = {}, {}
    for model in sorted(db):
        directory = original.ASSETS / 'models' / model
        catalog[model] = {'category': category(model), 'partition': 'experience' if model in source else 'test'}
        for mesh in ['collision.ply', 'textured.obj']:
            catalog[model][mesh + '_sha256'] = common.sha(directory / mesh)
        for path in directory.rglob('*'):
            if path.is_file(): hashes[str(path)] = common.sha(path)
    intersections = {'model': sorted(set(source) & set(test)),
        'semantic_category': sorted({category(m) for m in source} & {category(m) for m in test})}
    for mesh in ['collision.ply', 'textured.obj']:
        intersections[mesh] = sorted({catalog[m][mesh + '_sha256'] for m in source} &
                                    {catalog[m][mesh + '_sha256'] for m in test})
    assert not any(intersections.values()), intersections
    common.save(root / '语义类别映射.json', catalog)
    common.save(root / 'isolation_preflight.json', {'passed': True, 'intersections': intersections,
        'experience_models': source, 'test_models': test, 'checks_include_all_distractors': True,
        'upstream_pretraining_category_unseen_asserted': False})
    old = common.read(original.OLD / 'manifest.json')
    rng = np.random.default_rng(2026111950100)
    occupied = set()
    for group, models, count in [('experience', source, 50), ('test', test, 100)]:
        directory = root / group
        directory.mkdir()
        for name in ['admission', 'admission_claims', 'admission_workers', 'declared_resets']:
            (directory / name).mkdir()
        pool = list(models)
        rng.shuffle(pool)
        targets = (pool * 2)[:count]
        rng.shuffle(targets)
        schedule = []
        for i, model in enumerate(targets):
            layout = LAYOUTS[i % 4]
            while True:
                seed = int(rng.integers(2_100_000_000, 3_100_000_000))
                variants = {seed + v * 1009 for v in range(12)}
                if not variants & occupied:
                    occupied.update(variants)
                    break
            schedule.append({'id': f'{"E" if group == "experience" else "T"}{i+1:04d}_{layout}_{model}',
                'model_id': model, 'family': category(model), 'layout': layout, 'seed': seed})
        assert len(schedule) == count
        for name in ['common_initial_conditions.json', 'storage_policy.json']:
            shutil.copy2(original.OLD / name, directory / name)
        plan = {k: deepcopy(old[k]) for k in ['success', 'camera', 'table_height_m', 'recording_camera', 'observation_protocol']}
        plan.update(created_at=original.now(), schedule=schedule, layouts=LAYOUTS, registered=count,
            test_object_ids=models, excluded_old_object_ids=test if group == 'experience' else source,
            source_feedback_ids_in_memory_order=[], warmup_trial=old['warmup_trial'],
            asset_sha256={p:h for p,h in hashes.items() if any('/models/'+m+'/' in p for m in models)},
            source_hashes={str(Path(__file__).resolve()):common.sha(__file__)},
            physics_source_hashes=old['source_hashes'], physical_attempts_during_admission=0,
            independence_scope='Category and model/mesh disjoint between new experience and test; pretrained corpus not certified unseen.',
            partition=group, frozen_semantic_map=str(root / '语义类别映射.json'),
            maximum_distractor_occlusion_fraction=.40,
            admission_protocol='No grasp/IK screening; up to 12 fixed reset/view/stability variants, never replace after execution.')
        common.save(directory / 'preparation.json', plan)
        common.save(directory / 'admission_progress.json', {'phase':'prepared','completed':0,'registered':count})
    common.save(root / 'split_frozen.json', {'created_at':original.now(), 'master_seed':2026111950100,
        'experience_scenes':50, 'test_scenes':100, 'test_layout_counts':dict(Counter(e['layout'] for e in common.read(root/'test/preparation.json')['schedule'])),
        'maximum_target_model_repetitions_per_partition':2,
        'sha256':{str(p):common.sha(p) for p in [root/'语义类别映射.json', root/'isolation_preflight.json',
            root/'experience/preparation.json',root/'test/preparation.json']}})


def generate(directory, gpus):
    # Reuse the original admission orchestration, routing only its child entry
    # to the visibility cap below. No grasp prediction participates in admission.
    original.__file__ = str(Path(__file__).resolve())
    plan = common.read(directory / 'preparation.json')
    total = len(plan['schedule'])
    mutex, failed = threading.Lock(), threading.Event()
    active, claimed = set(), set()
    if list((directory/'admission_claims').glob('*.json')):
        raise RuntimeError('Admission already started; preserve receipts')
    def progress():
        rows = [common.read(p) for p in (directory/'admission').glob('*/admission.json')]
        common.save(directory/'admission_progress.json', {'phase':'halted' if failed.is_set() else 'generating',
            'completed':len(rows),'passed':sum(r['passed'] for r in rows),'registered':total,'active':sorted(active)})
    def worker(index, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MANISKILL_GPU_UUID=gpu, ARX_SIM_GPU=gpu,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        try:
            while not failed.is_set():
                with mutex:
                    options = [e for e in plan['schedule'] if e['id'] not in claimed and e['model_id'] not in active]
                    entry = options[0] if options else None
                    if entry:
                        claimed.add(entry['id']); active.add(entry['model_id'])
                        common.save(directory/'admission_claims'/(entry['id']+'.json'), {'state':'initialization_only','worker':index})
                    elif len(claimed) == total: break
                if entry is None:
                    failed.wait(.5); continue
                passed = original.admit_one(directory, entry, env)
                with mutex:
                    active.remove(entry['model_id'])
                    common.save(directory/'admission_claims'/(entry['id']+'.json'), {'state':'complete','worker':index,'passed':passed})
                    progress()
                print('ADMISSION', plan['partition'], entry['id'], passed, flush=True)
        except BaseException:
            failed.set()
            common.save(directory/'admission_workers'/(str(index)+'_halt.json'), {'error':traceback.format_exc()})
            with mutex: progress()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker,i,gpu) for i,gpu in enumerate(gpus)]
        for future in futures: future.result()
    if failed.is_set(): raise RuntimeError('Scene admission infrastructure halted')
    rows = [common.read(directory/'admission'/e['id']/'admission.json') for e in plan['schedule']]
    allowed = set(plan['test_object_ids'])
    for row in rows:
        if row['passed']:
            assert {a['model_id'] for a in row['entry']['scene']['actors']} <= allowed
    manifest = deepcopy(plan)
    manifest.update(frozen_at=original.now(), admitted=sum(r['passed'] for r in rows), admission_failed=sum(not r['passed'] for r in rows),
        admission_sha256={str(p):common.sha(p) for p in (directory/'admission').glob('*/admission.json')},
        declared_reset_sha256={str(p):common.sha(p) for p in (directory/'declared_resets').glob('*/declared_reset.json')},
        input_sha256={str(directory/n):common.sha(directory/n) for n in ['preparation.json','common_initial_conditions.json','storage_policy.json']})
    common.save(directory/'manifest.json', manifest)
    common.save(directory/'admission_progress.json', {'phase':'complete','completed':total,'registered':total,'passed':manifest['admitted']})


def admit_variant(inputs,entry_path,out,variant):
    plan=common.read(inputs/'preparation.json')
    baseline=original.install_sampler(plan)
    visibility=baseline.visibility
    def limited_visibility(*args,**kwargs):
        result=visibility(*args,**kwargs)
        result['maximum_distractor_occlusion_fraction']=.40
        result['admitted_view']=bool(result['admitted_view'] and result['occlusion_fraction']<=.40)
        return result
    baseline.visibility=limited_visibility
    baseline.admit(common.read(entry_path),out,max_variants=1,start_variant=variant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','generate','admit_variant'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+')
    parser.add_argument('--entry',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--variant',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.inputs.resolve())
    elif args.stage=='generate': generate(args.inputs.resolve(),args.gpus)
    else: admit_variant(args.inputs.resolve(),args.entry.resolve(),args.output.resolve(),args.variant)
