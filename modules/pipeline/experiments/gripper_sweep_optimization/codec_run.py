"""Register a new phase while preserving the original paired replay runner."""
import argparse
from pathlib import Path
from unittest.mock import patch
import traceback

import yaml
from arx_project.layout import ROOT,now,write_json,sha256,ensure_run_path
from . import run as base
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import raw


def prepare(root,gpu,previous):
    proof=raw(previous/'paired_review.json')
    if not proof['passed'] or not proof['timing_gate_passed'] or not raw(previous/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Sweep phase is not accepted and stopped')
    config=ROOT/'configs/experiments/gripper_search_codec.json'
    with patch.object(base,'CONFIG',config):base.prepare(root,gpu)
    reg=raw(root/'registration.json')
    for path in [previous/'registration.json',previous/'acceptance.json',previous/'paired_review.json',previous/'services_stopped.json']:
        reg['input_sha256'][str(path)]=sha256(path)
    path=ROOT/'modules/curobo/indexed_planning/binding_serialization.py'
    target=root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    __import__('shutil').copy2(path,target);reg['source_sha256'][str(path)]=sha256(path)
    reg.update(previous_phase=str(previous.relative_to(ROOT)),worker_module='modules.pipeline.experiments.gripper_sweep_optimization.codec_worker',
        interpretation={'legacy':'batched_opening_narrowing_original_YAML','batched':'batched_opening_narrowing_C_YAML'},
        candidate_optimization='same private bind_inputs with CSafeLoader/CSafeDumper; original native binding/math unchanged')
    write_json(root/'registration.json',reg)


def run(root):
    original=base.subprocess.Popen
    def start(command,*args,**kwargs):
        old='modules.pipeline.experiments.gripper_sweep_optimization.worker'
        if isinstance(command,(list,tuple)) and old in command:
            command=list(command)
            command[command.index(old)]='modules.pipeline.experiments.gripper_sweep_optimization.codec_worker'
        return original(command,*args,**kwargs)
    with patch.object(base.subprocess,'Popen',start):base.run(root)
    reg=raw(root/'registration.json');pairs=[]
    for case in reg['cases']:
        for repeat in [1,2]:
            a,b=[root/'plans'/case['id']/(mode+'_'+str(repeat)) for mode in ['legacy','batched']]
            left={p.relative_to(a):p for p in (a/'arms').rglob('arx.yml')}
            right={p.relative_to(b):p for p in (b/'arms').rglob('arx.yml')}
            if set(left)!=set(right):raise RuntimeError('Binding model paths/count changed')
            for path in left:
                if yaml.safe_load(left[path].read_text())!=yaml.safe_load(right[path].read_text()):
                    raise RuntimeError('Binding YAML values changed')
            pairs.append({'case':case['id'],'repeat':repeat,'models_equal':len(left)})
    write_json(root/'codec_model_review.json',{'at':now(),'passed':True,'pairs':pairs,
        'actual_models_with_original_joint_locks_spheres_limits_and_transforms_equal':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--gpu')
    args=p.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.gpu,ensure_run_path(args.previous))
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
