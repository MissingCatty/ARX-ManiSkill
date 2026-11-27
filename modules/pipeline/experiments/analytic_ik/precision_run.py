"""Validate FP64 analytic algebra and a resident FP32-command collision API."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from modules.curobo.analytic_ik.executable import ExecutableAnalyticIK
from modules.curobo.analytic_ik.session import AnalyticBatchSession
from modules.curobo.analytic_ik.collision import BranchCollisionChecker
from modules.curobo.indexed_planning.geometric.kinematics import ArmChain
from .prepare import guard
from .run import independent, locked_width_equivalence
from .timing import measure, summarize


def pure_validation(root, cfg):
    rows=[]
    for arm in ['right','left']:
        data=np.load(root/'random'/arm/'states.npz',allow_pickle=False)
        solver=ExecutableAnalyticIK(arm,device='cuda:0')
        targets=torch.as_tensor(data['targets'],device='cuda',dtype=torch.float64)
        answer=solver.solve(targets)
        q=answer['q'].cpu().numpy();mask=answer['valid'].cpu().numpy()
        proof=independent(data['targets'],q,mask,ArmChain(arm))
        recovered=np.where(mask,np.linalg.norm(q-data['q'][:,None],axis=-1),np.inf).min(-1)<1e-5
        if not recovered.all():raise ValueError('Precision variant lost a known legal branch')
        ref=read_json(root/'random'/arm/'EAIK_reference.json');expected=0;matched=0
        for i, row in enumerate(ref['answers']):
            for value,pe,re in zip(row['q'],row['position_errors_m'],row['rotation_errors_rad']):
                if pe>1e-7 or re>1e-7:continue
                expected+=1
                if (np.linalg.norm(q[i][mask[i]]-value,axis=-1)<1e-5).any():matched+=1
        if expected!=matched:raise ValueError('An exact EAIK branch was lost after FP32 conversion')
        batch=targets[:cfg['batch_capacity']].clone()
        timing=measure({'analytic_256x8':lambda:solver.solve(batch)},
            cfg['GPU_graph_repeats'],cfg['GPU_eager_repeats'])
        rows.append({'arm':arm,'targets':len(targets),'known_branches_recovered':int(recovered.sum()),
            'exact_EAIK_reference_branches':expected,'exact_EAIK_branches_matched':matched,
            'targets_with_valid_branch':int(mask.any(-1).sum()),
            'fallback_required':int(answer['needs_fallback'].sum().item()),
            'independent_FK':proof,'timing':timing})
        write_json(root/'GPU_pure_analytic.json',{'at':now(),'passed':True,'arms':rows})
    return rows


def check_session(session, solver, engine, targets, widths, raw):
    n=int(raw['input_valid'].sum())
    result=session.device_query(targets[:n],widths[:n])
    expected=solver.solve(targets)
    checked=engine.check(expected['q'],expected['valid'],widths)
    comparisons={'q':expected['q'],'valid_ik':expected['valid'],
        'valid_world':checked['valid_world_only'],'valid_collision':checked['valid_world_and_self'],
        'needs_fallback':expected['needs_fallback'],'valid_input':expected['valid_input']}
    for key,value in comparisons.items():
        if not torch.equal(result[key],value[:n]):raise ValueError('Captured API changed output '+key)
    if not bool(result['self_collision_checked'].all()):raise ValueError('Self-collision receipt missing')
    snapshot={key:value.clone() for key,value in result.items()}
    session.device_query(targets[:1],widths[:1]*.5)
    for key in result:
        if not torch.equal(result[key],snapshot[key]):raise ValueError('Next query overwrote owned output '+key)
    empty=session.device_query(targets[:0],widths[:0])
    if any(len(value) for value in empty.values()):raise ValueError('Empty input returned a padded row')
    bad_widths=widths[:1].clone();bad_widths[0]=float('nan')
    bad=session.device_query(targets[:1],bad_widths)
    if bool(bad['valid_collision'].any()):raise ValueError('Invalid opening width passed collision gate')
    bad_target=targets[:1].clone();bad_target[0,0,0]=float('nan')
    bad=session.device_query(bad_target,widths[:1])
    if bool(bad['valid_input'].any()) or bool(bad['valid_ik'].any()):
        raise ValueError('Malformed target passed captured API')
    return {'passed':True,'graph_matches_eager':True,'owned_outputs':True,'empty_input':True,
        'invalid_width_rejected':True,'invalid_target_rejected':True,'padding_hidden':True}


def run(root):
    reg=guard(root);cfg=reg['config']
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Do not retry a claimed run')
    write_json(root/'started.json',{'at':now(),'GPU_uuid':reg['gpu']})
    torch.set_num_threads(1)
    free=torch.cuda.mem_get_info()[0]/1024**2
    if free<cfg['startup_free_mib']:raise RuntimeError('Original startup memory gate unavailable')
    probe=torch.arange(1024,device='cuda',dtype=torch.float32)
    if float((probe*2).sum().item())!=1047552.:raise RuntimeError('CUDA arithmetic probe failed')
    write_json(root/'resource_probe.json',{'at':now(),'passed':True,'free_mib':free,
        'GPU_uuid':reg['gpu'],'device':torch.cuda.get_device_name(0)})
    pure=pure_validation(root,cfg);rows=[];chain=ArmChain()
    for item in reg['cases']:
        guard(root)
        if torch.cuda.mem_get_info()[0]/1024**2<cfg['minimum_request_free_mib']:
            raise RuntimeError('Original per-batch memory gate unavailable')
        directory=ROOT/item['directory'];claim=directory/'claimed.json'
        if claim.exists():raise ValueError('Collision case already claimed')
        write_json(claim,{'at':now(),'state':'claimed'})
        write_json(root/'progress.json',{'at':now(),'phase':'GPU_collision','current':item,
            'completed_collision_cases':len(rows),'collision_cases':len(reg['cases'])})
        raw=np.load(directory/'targets.npz',allow_pickle=False)
        targets=torch.as_tensor(raw['targets'],device='cuda',dtype=torch.float64)
        widths=torch.as_tensor(raw['widths'],device='cuda',dtype=torch.float32)
        real=torch.as_tensor(raw['input_valid'],device='cuda')
        config=yaml.load((directory/'arx.yml').read_text(),Loader=yaml.CSafeLoader)
        scene=read_json(directory/'scene.json')
        solver=ExecutableAnalyticIK(device='cuda:0');engine=BranchCollisionChecker(config,scene)
        equivalence=locked_width_equivalence(engine,config,chain)
        began=time.perf_counter()
        session=AnalyticBatchSession(config,scene,include_self=True)
        setup_seconds=time.perf_counter()-began
        contract=check_session(session,solver,engine,targets,widths,raw)
        first=solver.solve(targets)
        values=first['q'].clone();mask=(first['valid']&real[:,None]).clone()
        checked=engine.check(values,mask,widths)
        proof=independent(raw['targets'],values.cpu().numpy(),mask.cpu().numpy(),chain)
        def complete(with_self):
            answer=solver.solve(targets)
            return answer,engine.check(answer['q'],answer['valid']&real[:,None],widths,
                include_self=with_self)
        functions={'analytic_256x8':lambda:solver.solve(targets),
            'native_FK_world_2048':lambda:engine.check(values,mask,widths,include_self=False),
            'native_FK_world_self_2048':lambda:engine.check(values,mask,widths),
            'analytic_and_FK_world':lambda:complete(False),
            'analytic_and_FK_world_self':lambda:complete(True)}
        timing=measure(functions,cfg['GPU_graph_repeats'],cfg['GPU_eager_repeats'])
        n=int(real.sum().item());roundtrip=[]
        for _ in range(cfg['GPU_eager_repeats']):
            began=time.perf_counter();session.host_query(raw['targets'][:n],raw['widths'][:n])
            torch.cuda.synchronize();roundtrip.append((time.perf_counter()-began)*1000)
        np.savez_compressed(directory/'branches.npz',q=values.cpu().numpy(),valid=mask.cpu().numpy(),
            world_clear=checked['world_clear'].cpu().numpy(),self_clear=checked['self_clear'].cpu().numpy(),
            valid_world_only=checked['valid_world_only'].cpu().numpy(),
            valid_world_and_self=checked['valid_world_and_self'].cpu().numpy(),
            fallback_required=(first['needs_fallback']&real).cpu().numpy())
        row={**item,'passed':True,'timing':timing,'initialization_and_graph_capture_seconds':setup_seconds,
            'host_API_with_upload_owned_outputs_and_download':summarize(roundtrip),
            'analytic_targets_with_legal_branch':int(mask.any(-1).sum()),
            'valid_analytic_branches':int(mask.sum()),
            'world_clear_branches':int(checked['valid_world_only'].sum()),
            'world_and_self_clear_branches':int(checked['valid_world_and_self'].sum()),
            'fallback_targets':int((first['needs_fallback']&real).sum()),
            'native_sphere_count':871,'native_FK_measured_coordinates':8,'planning_arm_DOF':6,
            'width_equivalence':equivalence,'session_contract':contract,'independent_FK':proof,
            'V2_table_narrow_phase_not_applied_to_these_native_masks':True,
            'source_active_self_collision_setting':False,'diagnostic_world_self_mask_also_available':True,
            'branch_receipt_sha256':sha256(directory/'branches.npz')}
        write_json(directory/'metrics.json',row);rows.append(row)
        write_json(claim,{'at':now(),'state':'complete','metrics_sha256':sha256(directory/'metrics.json')})
        print('PRECISE_BATCH',len(rows),item['case'],item['arm'],'IK_ms',
            round(timing['analytic_256x8']['graph_device']['median_ms'],4),'world_self_ms',
            round(timing['analytic_and_FK_world_self']['graph_device']['median_ms'],4),flush=True)
        del session,solver,engine,first,checked,functions
        torch.cuda.synchronize()
    guard(root)
    write_json(root/'acceptance.json',{'at':now(),'task_complete':True,'record_integrity':True,
        'collision_cases':len(rows),'pure_analytic':pure,'cases':rows,
        'planning_requests':0,'physical_grasps':0,'simulation_instances':0,'production_defaults_changed':False})
    write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_collision_cases':len(rows),
        'collision_cases':len(reg['cases'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=ensure_run_path(args.root)
    try:run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False});raise
