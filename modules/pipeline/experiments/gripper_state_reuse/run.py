"""Reuse the accepted paired replay, adding complete source and model review."""
import argparse
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import yaml

from arx_project.layout import ROOT,now,read_json,write_json,sha256,ensure_run_path
from modules.pipeline.experiments.gripper_sweep_optimization import run as base


def prepare(root,gpu,phase,previous):
    if not read_json(previous/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Previous models are still running')
    previous_reg=read_json(previous/'registration.json')
    proof=read_json(previous/('reuse_review.json' if phase=='geometry' else 'paired_review.json'))
    if not proof['passed']:raise RuntimeError('Previous record/trajectory review is not accepted')
    cfg=ROOT/f'configs/experiments/gripper_state_reuse_{phase}.json'
    with patch.object(base,'CONFIG',cfg):base.prepare(root,gpu)
    reg=read_json(root/'registration.json')
    sources={ROOT/'modules/curobo/indexed_planning/binding_serialization.py'}
    sources.update((ROOT/'modules/pipeline/experiments/gripper_state_reuse').glob('*.py'))
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);reg['source_sha256'][str(path)]=sha256(path)
    for path in [previous/'registration.json',previous/'services_stopped.json',
                 previous/('reuse_review.json' if phase=='geometry' else 'paired_review.json')]:
        reg['input_sha256'][str(path)]=sha256(path)
    reg.update(reuse_phase=phase,previous_phase=str(previous.relative_to(ROOT)),
        accepted_width_refresh=bool(phase=='geometry' and
            (previous_reg.get('accepted_width_refresh',False) if previous_reg.get('reuse_phase')=='geometry' else proof['kept'])),
        sweep_strategy=read_json(cfg).get('sweep_strategy','memo'),
        same_warm_batched_grippers_and_C_YAML_in_both_modes=True,
        new_campaign_planning_budget=80,minimum_total_saving_fraction=.05,
        worker_module='modules.pipeline.experiments.gripper_state_reuse.worker')
    write_json(root/'registration.json',reg)
    (root/'research.md').write_text('# 夹爪状态与场景复用\n\n## Goal\n降低完整规划请求耗时。\n\n'
        '## Success Metric\n累计至少省5%、中位下降、P90不增5%，全部抓法/终点/路径逐位一致。\n\n'
        '## Constraints\n- max_iterations:1\n- pause_every:never\n- keep_policy:score_improvement\n'
        '- noise_runs:2\n- 20 ABBA/BAAB requests,0physical\n'
        '- per-request1800s deadline; resource10000/1536MiB; no automatic retries\n'
        '- guard: original .5mm samples,871 spheres,TCP43,V2,32seeds,and exact trajectory retention\n\n'
        f'## Current Approach\n{phase}\n\n## Search Space\nOne optional reuse change.\n\n'
        '## History\n| Iteration | Change | Result | Status |\n| --- | --- | --- | --- |\n')


def run(root):
    original=base.subprocess.Popen
    def start(command,*args,**kwargs):
        old='modules.pipeline.experiments.gripper_sweep_optimization.worker'
        if isinstance(command,(list,tuple)) and old in command:
            command=list(command);command[command.index(old)]='modules.pipeline.experiments.gripper_state_reuse.worker'
        return original(command,*args,**kwargs)
    with patch.object(base.subprocess,'Popen',start):base.run(root)
    reg=read_json(root/'registration.json');pairs=[]
    for case in reg['cases']:
        for repeat in [1,2]:
            a,b=[root/'plans'/case['id']/(m+'_'+str(repeat)) for m in ['legacy','batched']]
            left={p.relative_to(a):p for p in (a/'arms').rglob('arx.yml')}
            right={p.relative_to(b):p for p in (b/'arms').rglob('arx.yml')}
            if set(left)!=set(right):raise RuntimeError('Actual model paths changed')
            for path in left:
                if yaml.load(left[path].read_text(),Loader=yaml.CSafeLoader)!=yaml.load(right[path].read_text(),Loader=yaml.CSafeLoader):
                    raise RuntimeError('Actual model configuration values differ')
            pairs.append({'case':case['id'],'repeat':repeat,'models_equal':len(left)})
    write_json(root/'codec_model_review.json',{'at':now(),'passed':True,'pairs':pairs})
    proof=read_json(root/'paired_review.json')
    kept=(proof['timing_gate_passed'] and proof['total_saving_fraction']>=.05
          and proof['all_endpoints_bitwise_equal'] and proof['all_paths_bitwise_equal'])
    write_json(root/'reuse_review.json',{'at':now(),'passed':True,'kept':kept,
        'record_integrity':True,'timing_gate_passed':proof['timing_gate_passed'],
        'minimum_saving_gate_passed':proof['total_saving_fraction']>=.05,
        'all_endpoints_and_paths_bitwise_equal':proof['all_endpoints_bitwise_equal'] and proof['all_paths_bitwise_equal'],
        'total_saving_fraction':proof['total_saving_fraction'],'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu')
    parser.add_argument('--phase',choices=['binding','geometry']);parser.add_argument('--previous',type=Path)
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.gpu,args.phase,ensure_run_path(args.previous))
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
