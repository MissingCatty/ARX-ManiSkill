"""Epoch 4: carry 137 scenes and retain the renderer in resident simulators."""
from pathlib import Path
import argparse
import shutil
import time

import graspcl_unseen_resident_resume_20261119 as resident
from graspcl_execute import guard

common=resident.common
bench=resident.bench
CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch04'
SIMULATOR=CODE/'graspcl_unseen_simulator_pinned_20261119.py'
ERROR_WORDS=['out of memory','vk_error','resource descriptor','supported physical device']


def infrastructure(row):
    return any(any(w in e['error'].lower() for w in ERROR_WORDS) for e in row.get('rejected_variants',[]))


def prepare(root):
    reg=common.read(root/'registration.json');guard(reg)
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    assert not any((root/p).exists() for p in bench.PHASES)
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    inputs=Path(reg['inputs']);proofs={}
    for partition in ['experience','test']:
        directory=inputs/partition
        shutil.copytree(directory/'admission_claims',epoch/'original_admission_claims'/partition)
        complete={str(p):common.sha(p) for p in (directory/'admission').glob('*/admission.json')}
        assert len(complete)==(50 if partition=='experience' else 87)
        proofs[partition]={'completed_admission_sha256':complete,
            'frozen_preparation_sha256':common.sha(directory/'preparation.json')}
    paths=[SELF,SIMULATOR,Path(resident.__file__),CODE/'graspcl_unseen_simulator_20261119.py',
        CODE/'graspcl_unseen_resume_20261119.py',CODE/'arx_runtime_imports.py',
        CODE/'source/mani_skill/envs/tasks/mobile_manipulation/__init__.py']
    validation=root/'debug/runtime_scope_verification.json'
    render_validation=root/'debug/render_lifetime_kept.json'
    assert common.read(validation)['passed'] and common.read(render_validation)['passed']
    common.save(epoch/'registration.json',{'created_at':bench.core.now(),'runtime_epoch':4,
        'original_registration_sha256':common.sha(root/'registration.json'),
        'source_sha256':{str(p):common.sha(p) for p in paths},'retained_scene_admission':proofs,
        'validation':str(validation),'validation_sha256':common.sha(validation),
        'renderer_validation':str(render_validation),'renderer_validation_sha256':common.sha(render_validation),
        'change':'Hold one renderer context after the first original environment creates it; keep resident simulation modules; do not change actual episode behavior.',
        'completed_initializations_retained':137,'claimed_grasp_slots':0,'total_grasp_slots_unchanged':350,
        'infrastructure_receipts':'Preserved; not geometric rejections. Retry same seed/variant only in a new initialization directory.',
        'component_startup_wait_seconds':1800,'per_request_budgets_unchanged':True})
    for path in paths[:6]:shutil.copy2(path,epoch/path.name)
    common.save(epoch/'progress.json',{'phase':'prepared','retained_admissions':137,'physical_grasps_claimed':0})


def admit_one(directory,entry,env,epoch):
    out=directory/'admission'/entry['id'];out.mkdir(exist_ok=True)
    binding=out/'entry.json'
    if not binding.exists():
        prior=[]
        for p in (directory/'admission').glob('*/admission.json'):
            row=common.read(p)
            if row['passed'] and row['entry']['model_id']==entry['model_id']:prior.append(row['entry']['scene'])
        common.save(binding,{**entry,'avoid_scenes':prior})
    assert all(common.read(binding)[k]==entry[k] for k in ['id','model_id','layout','seed'])
    failures=[];index=bench.GPUS.index(env['CUDA_VISIBLE_DEVICES'])
    for variant in range(12):
        files=sorted(out.glob(f'variant_{variant:02d}*/admission.json'))
        valid=[]
        for path in files:
            row=common.read(path)
            if not infrastructure(row):valid.append((path,row))
        if valid:
            assert len({row['passed'] for _,row in valid})==1
            path,receipt=valid[0];destination=path.parent
        else:
            destination=out/f'variant_{variant:02d}_{EPOCH}'
            assert not destination.exists(),'Do not replay initialization in this epoch'
            request={'command':'admit','inputs':str(directory),'entry':str(binding),'output':str(destination),
                'variant':variant,'log':str(out/f'variant_{variant:02d}_{EPOCH}.log')}
            start=time.monotonic()
            response=resident.POOL.request(index,request,timeout=600)
            common.save(epoch/'initialization_requests'/(entry['id']+f'_v{variant:02d}.json'),{
                'request':request,'response':response,'wall_seconds':time.monotonic()-start,'physical_grasp':False,
                'prior_infrastructure_receipts_preserved':[str(p) for p in files]})
            receipt=common.read(destination/'admission.json')
        if infrastructure(receipt):raise RuntimeError('Renderer infrastructure failure preserved: '+entry['id'])
        if receipt['passed']:
            resident.previous.commit_admission(directory,entry,destination,receipt,failures)
            return True
        failures.extend(receipt.get('rejected_variants',[]))
    common.save(out/'admission.json',{'passed':False,'entry':{**entry,'admission_failed':True},'rejected_variants':failures})
    return False


def run(root):
    receipt=common.read(root/EPOCH/'registration.json')
    assert common.sha(receipt['renderer_validation'])==receipt['renderer_validation_sha256']
    resident.EPOCH=EPOCH;resident.SIMULATOR=SIMULATOR;resident.admit_one=admit_one
    resident.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())
