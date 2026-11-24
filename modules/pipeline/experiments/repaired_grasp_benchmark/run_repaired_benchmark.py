"""Run the repaired pipeline on the full, unchanged frozen 200-scene dataset."""
import argparse
import fcntl
import importlib.util
import inspect
from itertools import permutations
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import types

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
LEGACY=HERE.parent/'rayst3r_graspgenx'
BASE=HERE.parent/'diverse_grasp_benchmark'
sys.path[:0]=[str(BASE),str(LEGACY)]
import run_benchmark as baseline
import run_frozen_benchmark as frozen
import run_suite as executor
from common import read,save,sha,now
from repair_contract import VERSION,CONTACT_OFFSET_M,ORIGIN_BACK_M,TCP_M,descriptor,make_candidates
from presentation_report import render as render_presentation

DEFAULT=HERE/'runs/geometry200_refined871_20261124_01'


def inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    return [{'index':int(a),'uuid':b.strip(),'free_mib':int(c)}
        for a,b,c in (line.split(',') for line in output.strip().splitlines())]


def choose_gpus():
    choices=[]
    # Keep 20/14GiB for the neural components. Planner/simulation use 10GiB at
    # startup and must still have 2GiB free after all components warm up.
    for ray,sam,plan in permutations(inventory(),3):
        if ray['free_mib']>=20000 and sam['free_mib']>=14000 and plan['free_mib']>=10000:
            choices.append((min(ray['free_mib']-20000,sam['free_mib']-14000,plan['free_mib']-10000),ray,sam,plan))
    assert choices,'Need three available GPUs for resident completion, segmentation/grasping, and planning/simulation'
    _,ray,sam,plan=max(choices,key=lambda c:c[0])
    return {'rayst3r':ray['uuid'],'sam3':sam['uuid'],'graspgenx':sam['uuid'],'curobo':plan['uuid']},\
        {ray['uuid']:20000,sam['uuid']:14000,plan['uuid']:10000}


def report(root,summary):
    render_presentation(root)
    doc=PROJECT/'docs/diverse_grasp_benchmark/抓取测试结果.md'
    text=doc.read_text()
    intro=('本轮使用修复后的链路：物理接触偏移1毫米；GraspGenX虚拟夹爪原点后移100毫米，'
           '条件与输出同步换算；实际TCP仍为43毫米。固定45.2厘米、原200个独立采样场景、'
           '头部SAM3→RaySt3R→原生GraspGenX→cuRobo六轴，成功标准不变。\n\n')
    text=text.replace('\n\n','\n\n'+intro,1)
    doc.write_text(text);(root/'report.md').write_text(text)


def prepare(root):
    sys.path.insert(0,str(HERE.parent/'current_configuration'))
    from configuration import load_current, CONFIG
    current=load_current(verify_dataset=True)
    config=read(PROJECT/'configs/scene_dataset.json')
    assert config['grasp_tests_authorized'] and config['grasp_test_budget']==200
    dataset=Path(config['dataset'])
    evidence=frozen.verify_dataset(dataset)
    gpus,floors=choose_gpus()
    baseline.prepare(root)
    save(root/'gripper_descriptor.json',descriptor())
    reg=read(root/'registration.json')
    reg.update(active_configuration_id=current['profile_id'],active_configuration_sha256=sha(CONFIG),
        collision_profile_id=current['collision_profile_id'])
    reg['source_hashes'].update(current['configuration_hashes'])
    reg['source_hashes'][str(CONFIG.resolve())]=sha(CONFIG)
    reg.update(protocol=VERSION,repair_contract=VERSION,user_authorization=
        '2026-11-23: fix contact physics and GraspGenX gripper-frame adaptation, clear old test artifacts and rerun the same 200 scenes',
        construction_independent_of_ik=True,all_frozen_scenes_retained=True,posthoc_IK_used_for_selection=False,
        dataset_freeze_sha256=evidence['freeze_sha256'],gpus=gpus,minimum_free_mib=floors,
        minimum_free_after_warmup_mib=2048,robot_and_scene_physics='Original assets and drives; explicit contact_offset .001m',
        physical_changes={'contact_offset_m':CONTACT_OFFSET_M,'rest_offset_m':0.,'drive_and_material_changes':False},
        model_frame_changes={'origin_behind_eef_m':ORIGIN_BACK_M,'actual_tcp_m':TCP_M,'change_pointcloud_coordinates':False},
        runtime_entry=str(Path(__file__)),same_frozen_scene_schedule=True,old_results_carried=0)
    for file in [*HERE.glob('*.py'),BASE/'presentation_report.py',dataset/'frozen_dataset.json',
                 dataset/'sampling_region.json',root/'gripper_descriptor.json']:
        reg['source_hashes'][str(file.resolve())]=sha(file)
    assert [e['id'] for e in reg['schedule']]==evidence['scene_ids']
    save(root/'registration.json',reg)
    save(root/'preflight/frozen_dataset.json',evidence)
    for file in HERE.glob('*.py'):shutil.copy2(file,root/'source'/file.name)
    baseline.aggregate(root,'registered');report(root,read(root/'progress.json'))
    print('REGISTERED',root,gpus,flush=True)


class CheckedServices(executor.Services):
    def start(self):
        super().start()
        free={v['uuid']:v['free_mib'] for v in inventory()}
        minimum=self.reg['minimum_free_after_warmup_mib']
        save(self.root/'warm_memory.json',{'free_mib':free,'minimum_free_mib':minimum,'at':now()})
        assert all(free[gpu]>=minimum for gpu in set(self.reg['gpus'].values())), 'Insufficient memory after component warmup'


def run(root):
    sys.path.insert(0,str(HERE.parent/'current_configuration'))
    from configuration import load_current, CONFIG
    current=load_current(verify_dataset=True)
    reg=read(root/'registration.json')
    assert reg.get('active_configuration_id')==current['profile_id'], 'Retired experiment registration'
    assert reg.get('active_configuration_sha256')==sha(CONFIG), 'Active experiment configuration changed'
    assert reg.get('collision_profile_id')==current['collision_profile_id'], 'Retired collision model'
    assert reg['repair_contract']==VERSION and reg['old_results_carried']==0
    evidence=frozen.verify_dataset(Path(reg['scene_dataset']))
    assert evidence['freeze_sha256']==reg['dataset_freeze_sha256']
    assert [r['id'] for r in reg['schedule']]==evidence['scene_ids']
    save(root/'preflight/frozen_dataset_before_run.json',evidence)
    spec=importlib.util.spec_from_file_location('simulation',HERE/'simulation.py')
    module=importlib.util.module_from_spec(spec);sys.modules['simulation']=module;spec.loader.exec_module(module)
    executor.HERE=HERE
    executor.GPUS=reg['gpus']
    executor.Services=CheckedServices
    executor.make_candidates=make_candidates
    executor.aggregate=baseline.aggregate
    executor.report=report
    source=inspect.getsource(executor.run)
    for old,new in [("'/160'","'/200'"),
        ("floors = {GPUS['rayst3r']: 20000, GPUS['sam3']: 14000, GPUS['curobo']: 12000}",
         "floors = reg['minimum_free_mib']")]:
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    scope=dict(executor.__dict__);exec(compile(source,str(__file__)+':repaired_executor','exec'),scope)
    execute=types.FunctionType(scope['run'].__code__,executor.__dict__,'run')
    try:
        execute(root)
        save(root/'frozen_dataset_after_run.json',frozen.verify_dataset(Path(reg['scene_dataset'])))
        save(root/'completion_verification.json',{'passed':True,'at':now(),
            'frozen_scenes_unchanged':True,'completed':read(root/'progress.json')['completed'],
            'registered':len(reg['schedule']),'repair_contract':VERSION})
    except BaseException:
        if not (root/'STOP').exists():(root/'STOP').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=DEFAULT)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='prepare':prepare(root)
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(root)
