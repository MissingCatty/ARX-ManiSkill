"""Epoch 5: verified service handshake and one model pair for all 350 slots."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import socket
import time

import graspcl_unseen_resident_resume_20261119 as resident
import graspcl_unseen_shared_backends_20261119 as shared
from graspcl_execute import guard
from resident_backend import rpc

common=resident.common
bench=resident.bench
CODE=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch05'
SIMULATOR=CODE/'graspcl_unseen_simulator_pinned_20261119.py'
ORIGINAL_PREPARE=bench.prepare_group
ORIGINAL_AGGREGATE=resident.aggregate


def aggregate(root,phase=None):
    ORIGINAL_AGGREGATE(root,phase)
    value=common.read(root/'progress.json')
    value['runtime_epoch']=5
    value['historical_halts_retained']=['epoch01','epoch02','epoch03','epoch04']
    if shared.MANAGER is not None:
        value['model_services']={name:{'pid':record['process'].pid,'alive':record['process'].poll() is None,
            'kind':record['kind']} for name,record in shared.MANAGER.all_processes.items()}
    common.save(root/'progress.json',value)


def prepare(root):
    reg=common.read(root/'registration.json');guard(reg)
    assert not list(root.glob('*/fixed_*/claims/*.json'))
    inputs=Path(reg['inputs'])
    manifests={}
    for partition in ['experience','test']:
        directory=inputs/partition
        progress=common.read(directory/'admission_progress.json')
        assert progress['phase']=='complete' and progress['passed']==(50 if partition=='experience' else 100)
        manifests[str(directory/'manifest.json')]=common.sha(directory/'manifest.json')
    collection=root/'collection'
    assert collection.exists()
    assert not list((collection/'fixed_zerograsp/trials').iterdir())
    assert common.read(collection/'suite_timing.json')['all_owned_services_exited']
    assert common.read(root/'continuation_epoch04/simulation_stopped.json')['all_owned_simulators_exited']
    assert not any((root/name).exists() for name in bench.PHASES[1:])
    proof=root/'debug/startup_readiness_verification.json'
    assert common.read(proof)['passed']
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    old_registration_hash=common.sha(collection/'registration.json')
    collection.rename(epoch/'prestart_collection_epoch04')
    # The archive must refer to its own historical service records, not to the
    # fresh collection directory created by the resumed controller.
    link=epoch/'prestart_collection_epoch04/fixed_zerograsp/resident'
    original_link=str(link.readlink());link.unlink();link.symlink_to('../resident',target_is_directory=True)
    cleaned=[]
    for i in range(2):
        key=hashlib.sha256((str(root)+str(i)).encode()).hexdigest()[:12]
        path=CODE/'tmp'/('sim3_'+key+'.sock')
        if path.exists():
            try:
                rpc(path,{'command':'health'},timeout=1)
            except (ConnectionRefusedError,FileNotFoundError):
                path.unlink();cleaned.append(str(path))
            else:raise RuntimeError('An owned simulator is still running; preserve it')
    paths=[SELF,Path(shared.__file__),CODE/'graspcl_unseen_shared_services_20261119.py',
        CODE/'arx_service_readiness.py',Path(resident.__file__),SIMULATOR,
        CODE/'graspcl_unseen_simulator_20261119.py',CODE/'graspcl_unseen_resume_20261119.py',
        CODE/'arx_runtime_imports.py',CODE/'source/mani_skill/envs/tasks/mobile_manipulation/__init__.py']
    common.save(epoch/'registration.json',{'created_at':bench.core.now(),'runtime_epoch':5,
        'original_registration_sha256':common.sha(root/'registration.json'),
        'source_sha256':{str(p):common.sha(p) for p in paths},'frozen_initialization_manifest_sha256':manifests,
        'validation':str(root/'debug/runtime_scope_verification.json'),
        'validation_sha256':common.sha(root/'debug/runtime_scope_verification.json'),
        'readiness_validation':str(proof),'readiness_validation_sha256':common.sha(proof),
        'retained_completed_initializations':150,'claimed_grasp_slots':0,'unchanged_total_slots':350,
        'model_gpu_by_component':{'zerograsp':shared.NEURAL_GPU,'curobo':shared.PLANNER_GPU,'graspcl':'CPU'},
        'simulator_gpus':bench.GPUS,'model_lifecycle':'One warmed ZeroGrasp and cuRobo pair for collection and all three test arms; B/C each load their own frozen scoring bank once before their 100 slots.',
        'startup_handshake':'Wait for published ready PID plus successful health RPC. Socket path existence alone never admits a service.',
        'algorithm_robot_camera_collision_evaluator_unchanged':True,
        'archived_failed_prestart_registration_sha256':old_registration_hash,
        'archived_arm_original_resident_symlink':original_link,'stale_owned_simulator_sockets_removed':cleaned})
    for path in paths[:9]:shutil.copy2(path,epoch/path.name)
    common.save(epoch/'progress.json',{'phase':'prepared','scene_initializations':150,'physical_grasps_claimed':0})


class Campaign(resident.Campaign):
    def resources(self,worker,gpu):
        while not self.failed.is_set():
            if not super().resources(worker,gpu):return False
            raw=bench.core.batch.subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            free={line.split(',')[0].strip():int(line.split(',')[1]) for line in raw.splitlines()}
            if free.get(shared.NEURAL_GPU,0)>=self.reg['gpu_backend_free_mib_floor']:return True
            self.progress(worker,phase='waiting_for_neural_gpu_memory',free_mib=free)
            self.failed.wait(15)
        return False


def prepare_group(*args,**kwargs):
    group,arm=ORIGINAL_PREPARE(*args,**kwargs)
    for directory in [group,arm]:
        path=directory/'registration.json';reg=common.read(path)
        reg['model_gpu_by_component']={'zerograsp':shared.NEURAL_GPU,'curobo':shared.PLANNER_GPU,'graspcl':'CPU'}
        reg['shared_model_lifecycle']='One common neural/planner pair for all 350 slots; phase-specific frozen scoring bank.'
        common.save(path,reg)
    # The preserved controller performs its own startup-budget adapter before
    # calling this hook. Install the corrected backend only after that adapter.
    bench.Backends=shared.Backends
    return group,arm


def run(root):
    receipt=common.read(root/EPOCH/'registration.json')
    for path,digest in receipt['frozen_initialization_manifest_sha256'].items():assert common.sha(path)==digest
    assert common.sha(receipt['readiness_validation'])==receipt['readiness_validation_sha256']
    assert common.read(receipt['readiness_validation'])['passed']
    shared.MANAGER=shared.Manager(root,EPOCH)
    resident.EPOCH=EPOCH;resident.SIMULATOR=SIMULATOR;resident.Campaign=Campaign
    resident.aggregate=aggregate;bench.prepare_group=prepare_group
    bench.GPU0=shared.PLANNER_GPU
    success=False
    try:
        resident.run(root)
        success=True
        aggregate(root,'finalizing_services')
    finally:
        shared.MANAGER.stop()
        stopped=common.read(root/EPOCH/'model_services_stopped.json')
        for name in bench.PHASES:
            timing=root/name/'suite_timing.json'
            if timing.exists():
                value=common.read(timing)
                value['model_pair_shared_until_suite_shutdown']=True
                value['all_owned_services_exited']=stopped['all_owned_model_services_exited']
                common.save(timing,value)
        if success:
            final=common.read(root/'final_verification.json')
            final.update(runtime_epoch=5,all_services_exited=stopped['all_owned_model_services_exited'],
                zerograsp_model_loads=stopped['zero_model_loads'],curobo_pool_loads=stopped['curobo_pool_loads'])
            common.save(root/'final_verification.json',final)
            aggregate(root,'complete')
        else:aggregate(root,'halted')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())
