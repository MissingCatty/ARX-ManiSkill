"""Checked epoch-6 continuation of the same untouched 350 grasp slots."""
from pathlib import Path
import argparse
import hashlib
import shutil

import graspcl_unseen_ready_resume_20261119 as driver

common=driver.common
bench=driver.bench
CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch06'


def prepare(root):
    reg=common.read(root/'registration.json');driver.guard(reg)
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    assert not any((root/name).exists() for name in bench.PHASES)
    assert common.read(root/'continuation_epoch05/model_services_stopped.json')['all_owned_model_services_exited']
    assert common.read(root/'continuation_epoch05/simulation_stopped.json')['all_owned_simulators_exited']
    prior=common.read(root/'continuation_epoch05/registration.json')
    inputs=Path(reg['inputs'])
    retained={}
    for partition,total in [('experience',50),('test',100)]:
        directory=inputs/partition
        manifest=common.read(directory/'manifest.json')
        assert manifest['admitted']==total and manifest['admission_failed']==0
        progress=common.read(directory/'admission_progress.json');assert progress['phase']=='complete'
        records={str(p):common.sha(p) for p in (directory/'admission').glob('*/admission.json')}
        assert len(records)==total
        retained[partition]={'completed_admission_sha256':records,
            'frozen_preparation_sha256':common.sha(directory/'preparation.json')}
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    receipt=dict(prior)
    receipt.update(created_at=bench.core.now(),runtime_epoch=6,retained_scene_admission=retained,
        previous_startup_halt_sha256=common.sha(root/'continuation_epoch05/halt.json'),
        startup_preflight='Run the exact completed-scene carry-over interface before any model launch or grasp claim.',
        archived_failed_prestart_registration_sha256=prior['archived_failed_prestart_registration_sha256'])
    receipt['source_sha256']=dict(prior['source_sha256'])
    receipt['source_sha256'][str(SELF)]=common.sha(SELF)
    common.save(epoch/'registration.json',receipt)
    for path in receipt['source_sha256']:
        p=Path(path)
        if p.parent==CODE:shutil.copy2(p,epoch/p.name)
    # These are the exact two calls that failed in epoch 5. Both partitions
    # already have sealed manifests, so this only verifies retained evidence.
    driver.resident.previous.EPOCH=EPOCH
    before={partition:common.sha(inputs/partition/'manifest.json') for partition in retained}
    for partition in ['experience','test']:
        driver.resident.previous.resume_admission(root,partition)
    assert before=={partition:common.sha(inputs/partition/'manifest.json') for partition in retained}
    common.save(epoch/'startup_preflight.json',{'passed':True,'retained_initializations':150,
        'exact_carry_over_interface_passed':True,'manifests_unchanged':True,'physical_grasp_claims':0,
        'source_sha256':receipt['source_sha256']})
    common.save(epoch/'progress.json',{'phase':'prepared','retained_initializations':150,'physical_grasp_claims':0})
    print('STARTUP PREFLIGHT PASSED: 150 retained scenes; 0 grasp claims',flush=True)


def run(root):
    preflight=common.read(root/EPOCH/'startup_preflight.json');assert preflight['passed']
    for path,digest in preflight['source_sha256'].items():assert common.sha(path)==digest
    driver.EPOCH=EPOCH
    driver.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())
