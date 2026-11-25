"""Register precisely the accepted cohort, admit resources, run two workers."""
import argparse
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json
from arx_project.artifacts import atomic_json as write_json
from arx_project.json_io import read_json
from modules.pipeline.experiments.dual_arm_grasp.run import allocation_probes, gpu_inventory
from modules.pipeline.experiments.current_configuration.configuration import load_current
from .contracts import METHODS, source_succession

HERE = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/experiments/sam3_zerograsp_io74_parallel.json'


def prepare(root):
    if root.exists():raise RuntimeError('Use a fresh repeat directory')
    config = read_json(CONFIG)
    source = ROOT/config['source_comparison']
    previous = read_json(source/'registration.json')
    accepted = read_json(source/'acceptance.json')
    if (not accepted['passed'] or not accepted['task_complete']
            or not read_json(source/'services_stopped.json')['all_owned_exited']):
        raise RuntimeError('Previous comparison is not complete and accepted')
    cohort = previous['cohort']
    if (cohort['pair_count'] != 74 or len({p['id'] for p in cohort['pairs']}) != 74
            or sum(bool(p.get('common_precondition_failure')) for p in cohort['pairs']) != 5):
        raise RuntimeError('The authorized cohort must remain the same 74 cases')
    activation = ROOT/config['candidate_io_activation']
    succession = source_succession(previous, activation)
    current = load_current(verify_dataset=True)
    sources = {filename:sha256(filename) for filename in previous['source_hashes']}
    for filename in current['configuration_hashes']:sources[filename] = sha256(filename)
    for folder in [HERE,ROOT/'arx_project',ROOT/'modules/curobo/candidate_io']:
        for path in folder.glob('*.py'):sources[str(path)] = sha256(path)
    sources[str(CONFIG)] = sha256(CONFIG)
    baseline = read_json(ROOT/previous['baseline']/'registration.json')
    weights = {**baseline['weights'],**previous['weights']}
    for filename, record in weights.items():
        if Path(filename).stat().st_size != record['size'] or sha256(filename) != record['sha256']:
            raise RuntimeError('Frozen model weights changed: '+filename)
    root.mkdir(parents=True)
    for name in ['preflight','methods','source_registered']: (root/name).mkdir()
    shutil.copy2(ROOT/'website/内容/迭代/sam3-graspgenx-v6.md',root/'previous_website_page.md')
    for filename in sources:
        target = root/'source_registered'/Path(filename).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(filename,target)
    write_json(root/'software_succession.json', {'at':now(),'approved_candidate_IO_changes':succession,
        'activation_receipt_sha256':sha256(activation/'activation_registration.json'),
        'physics_and_model_parameters_changed':False})
    registration = {'at':now(),'version':'V6','run_type':config['run_type'],
        'user_authorization':config['user_authorization'],'configuration':config,'cohort':cohort,
        'source_comparison':config['source_comparison'],
        'source_registration_sha256':sha256(source/'registration.json'),
        'source_acceptance_sha256':sha256(source/'acceptance.json'),
        'baseline':previous['baseline'],'source_hashes':sources,'weights':weights,
        'same_success_criteria':previous['same_success_criteria'],
        'physical_slot_budget':138,'physical_slots_per_method':69,'automatic_retries':False,
        'maximum_attempts_per_case':1,'methods':list(METHODS),
        'component_modules':previous['component_modules']}
    write_json(root/'registration.json',registration)
    for method in METHODS:
        target = root/'methods'/method;target.mkdir()
        for name in ['preflight','resident','claims','trials']:(target/name).mkdir()
        for name in ['gripper_descriptor.json','object_catalog.json','common_initial_conditions.json']:
            shutil.copy2(source/name,target/name)
        components = ['rayst3r','graspgenx','curobo_dual_arm'] if method=='graspgenx' else ['zerograsp','curobo_dual_arm']
        write_json(target/'registration.json',{**registration,'method':method,
            'campaign':str(root.relative_to(ROOT)),'physical_slot_budget':69,
            'required_components':components})
    write_json(root/'progress.json',{'phase':'registered','registered_pairs':74,'completed_pairs':0})
    print('REGISTERED_PARALLEL_74',root,flush=True)


def guard(root):
    root = Path(root)
    if (root/'STOP').exists() or (root/'ABORT').exists():raise RuntimeError('Campaign is stopped')
    reg = read_json(root/'registration.json')
    for name, digest in reg['source_hashes'].items():
        if sha256(name)!=digest:raise RuntimeError('Registered source changed: '+name)
    old = ROOT/reg['source_comparison']
    if (sha256(old/'registration.json')!=reg['source_registration_sha256']
            or sha256(old/'acceptance.json')!=reg['source_acceptance_sha256']):
        raise RuntimeError('Predecessor evidence changed')
    from .continuation import verify_inheritance
    verify_inheritance(root, reg)
    return reg


def method_resource_choices(free, floors, method, unavailable=()):
    free={gpu:memory for gpu,memory in free.items() if gpu not in unavailable}
    if method=='zerograsp':
        return [(free[gpu]-floors['zerograsp_single_device'],
            {'zerograsp':gpu,'curobo_dual_arm':gpu,'simulation':gpu})
            for gpu in sorted(free,key=lambda g:(free[g],g),reverse=True)
            if free[gpu]>=floors['zerograsp_single_device']]
    if method!='graspgenx':raise ValueError('Unknown method')
    choices=[(min(free[ray]-floors['rayst3r'],free[execution]-floors['graspgenx_execution_device']),ray,execution)
        for ray,execution in permutations(free,2)
        if free[ray]>=floors['rayst3r'] and free[execution]>=floors['graspgenx_execution_device']]
    return [(score,{'rayst3r':ray,'graspgenx':execution,'curobo_dual_arm':execution,'simulation':execution})
        for score,ray,execution in sorted(choices,reverse=True)]


def placement_still_fits(free,floors,method,roles):
    if method=='zerograsp':return free[roles['zerograsp']]>=floors['zerograsp_single_device']
    return (free[roles['rayst3r']]>=floors['rayst3r']
        and free[roles['simulation']]>=floors['graspgenx_execution_device'])


def occupied_devices(assignments, active_methods):
    """Reserve admitted devices immediately, including model-loading time."""
    return {gpu for method in active_methods for gpu in assignments[method]['gpu_roles'].values()}


def run(root):
    reg = guard(root)
    if (root/'started.json').exists():raise RuntimeError('Never restart an executed campaign')
    floors = reg['configuration']['resource_floors_mib']
    failed_devices={}
    pending={method for method in METHODS if not (root/'methods'/method/'complete.json').exists()}
    finished=set(METHODS)-pending;assignments={}
    children,streams={},[]
    try:
        while pending or any(c.poll() is None for c in children.values()):
            guard(root)
            for method,child in children.items():
                if child.poll() not in [None,0]:raise RuntimeError('Method stopped: '+method)
                if child.poll()==0:
                    if not (root/'methods'/method/'complete.json').exists():raise RuntimeError('Worker exited without completion')
                    finished.add(method)
            for method in METHODS:
                if method not in pending:continue
                free=gpu_inventory()
                active=[m for m,c in children.items() if c.poll() is None]
                reserved=occupied_devices(assignments,active)
                unavailable={gpu for gpu,stamp in failed_devices.items() if time.monotonic()-stamp<60}|reserved
                choices=method_resource_choices(free,floors,method,unavailable)
                selected=None
                for _,roles in choices:
                    devices=set(roles.values())
                    if devices&unavailable:continue
                    probes=allocation_probes(sorted(devices))
                    write_json(root/'preflight/device_probes.json',{'at':now(),'method':method,'probes':probes})
                    for gpu,record in probes.items():
                        if not record['passed']:failed_devices[gpu]=time.monotonic();unavailable.add(gpu)
                    if all(p['passed'] for p in probes.values()) and placement_still_fits(gpu_inventory(),floors,method,roles):
                        selected=(roles,probes);break
                if selected is None:continue
                roles,probes=selected;folder=root/'methods'/method
                evidence={'at':now(),'method':method,'gpu_roles':roles,'probes':probes,'free_mib':gpu_inventory()}
                write_json(folder/'preflight/resources.json',evidence);assignments[method]=evidence
                write_json(root/'preflight/resources.json',{'at':now(),'scheduling':'independent_first_ready','per_method':assignments})
                if not (root/'started.json').exists():write_json(root/'started.json',{'at':now(),'pid':os.getpid(),'scheduling':'independent_first_ready'})
                stream=(folder/'controller.log').open('x');streams.append(stream)
                command=[str(ROOT/'environments/simulation/bin/python'),'-u','-m',
                    'modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.worker','--root',str(folder)]
                children[method]=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                pending.remove(method)
                print('DISPATCHED_IO74',method,roles,flush=True)
            free=gpu_inventory()
            write_json(root/'preflight/resource_wait.json',{'at':now(),'state':'waiting_for_resources' if pending else 'all_methods_dispatched',
                'free_mib':free,'floors_mib':floors,'waiting_methods':sorted(pending),
                'active_methods':[m for m,c in children.items() if c.poll() is None],
                'reserved_by_active_methods':{m:sorted(set(assignments[m]['gpu_roles'].values()))
                    for m,c in children.items() if c.poll() is None},
                'temporarily_unavailable_cuda_devices':[gpu for gpu,stamp in failed_devices.items() if time.monotonic()-stamp<60]})
            from .metrics import summarize
            summarize(root)
            time.sleep(1)
        if len(finished)<2:
            for method,child in children.items():
                if child.poll()!=0 or not (root/'methods'/method/'complete.json').exists():raise RuntimeError('Method did not complete')
        from .metrics import summarize
        summarize(root,'complete')
    except BaseException:
        write_json(root/'ABORT',{'at':now(),'error':traceback.format_exc()})
        for child in children.values():
            if child.poll() is None:
                import signal
                child.send_signal(signal.SIGINT)
        for child in children.values():
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        for stream in streams:stream.close()
        stopped = {m:read_json(root/'methods'/m/'services_stopped.json')
            if (root/'methods'/m/'services_stopped.json').exists() else {'all_owned_exited':m not in children,'models_started':False} for m in METHODS}
        write_json(root/'services_stopped.json',{'at':now(),
            'all_owned_exited':all(v['all_owned_exited'] for v in stopped.values()),'methods':stopped})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','continue','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path)
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root);return
    if args.stage=='continue':
        if args.predecessor is None:parser.error('--predecessor is required')
        from .continuation import prepare as prepare_continuation
        prepare_continuation(root, ensure_run_path(args.predecessor));return
    import fcntl
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()})
            if not (root/'started.json').exists():
                write_json(root/'services_stopped.json',{'at':now(),'all_owned_exited':True,'models_started':False})
            from .metrics import summarize
            summarize(root,'stopped');raise


if __name__=='__main__':main()
