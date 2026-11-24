"""Supplement a stopped ARX slot; preserve its absent native terminal receipt."""
import argparse
import json
from pathlib import Path
import types
import xml.etree.ElementTree as ET
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
from .advisor31_campaign import exited
from .advisor33_boundary import RANGE_ERROR,range_rejection
from .advisor33_audit import audit_policy,bound_stop,image_binding
from .runtime import audit_actions
from .capture import audit_episode
from .report import video


def record(previous,out):
    previous=Path(previous).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve supplemental records')
    reg=read(previous/'registration.json');state=read(previous/'state.json');summary=read(previous/'summary.json')
    if state['status']!='fault' or state['unconfirmed_world'] or not (previous/'STOP').is_file():
        raise RuntimeError('Predecessor must be stopped with confirmed World absence')
    idle=exited(previous);verify_files({**reg['execution_source_files'],**reg['dependencies']})
    claims=summary['incomplete_accounting']
    if len(claims)!=1 or not claims[0]['physical']:raise RuntimeError('Expected one consumed physical interruption')
    claim=claims[0];ident=claim['trial_id'];trial=next(t for t in reg['seed_trials'] if t['trial_id']==ident)
    ep=Path(claim['episode']);group=ep.parents[3];folder=ep.parent.parent
    if claim['state']!='executing' or claim['outcome'] is not None:
        raise RuntimeError('Original native ledger changed')
    if (folder/'trial_result.json').exists() or (ep/'physical_receipt.json').exists():
        raise RuntimeError('Do not replace a native terminal result')
    fault=read(previous/'workers/planner/FAULT.json')
    if fault['ok'] is not False or not fault['error'].rstrip().endswith('ValueError: '+RANGE_ERROR):
        raise RuntimeError('Unrelated planner fault')
    paths=list((previous/'workers/planner/requests').glob('*_'+fault['id']+'.done'))
    if len(paths)!=1:raise RuntimeError('Missing unique failed planning request')
    request=read(paths[0]);response=previous/'workers/planner/responses'/(fault['id']+'.json')
    if read(response)!=fault or Path(request['output'])!=ep or request['action']!='plan':
        raise RuntimeError('Failed planner request binding differs')
    shutdown=read(group/'shutdown_receipt.json')
    if not shutdown['all_trajectories_idle'] or not shutdown['raw_writer_drained_and_closed']:
        raise RuntimeError('Execution or writer is not confirmed idle')
    requests=read(ep/'requests.json');labels=[r['label'] for r in requests]
    if request['label'] in labels:raise RuntimeError('Failed planning request already dispatched')
    for suffix in ('_command.json','_execution.json'):
        if {p.name.removesuffix(suffix) for p in ep.glob('*'+suffix)}!=set(labels):
            raise RuntimeError('Missing or orphan occurred action record')
    actions=audit_actions(ep,requests);policy=audit_policy(ep);capture=audit_episode(ep)
    raw=[json.loads(x) for x in (ep/'samples.jsonl').read_text().splitlines()]
    for p in ep.glob('*_waypoint_stop.json'):bound_stop(read(p),raw)
    for row in policy['requests']:
        p=ep/'policy'/(row['request_id']+'_request.json');req=read(p);rep=read(p.with_name(req['request_id']+'_reply.json'))
        image_binding(reg,req,rep)
        if req['registration_sha256']!=sha(previous/'registration.json'):raise RuntimeError('Wrong policy registration')
    mode='model_closed' if request.get('phase')=='lift' else 'model'
    model=previous/'workers/planner/models'/mode/'arx.urdf'
    joints={x.attrib['name']:x for x in ET.parse(model).getroot().findall('joint')}
    names=['right_joint27','right_joint28'];limits={n:tuple(float(joints[n].find('limit').attrib[k]) for k in ('lower','upper')) for n in names}
    updater=types.SimpleNamespace(names=names,limits=limits)
    boundary=range_rejection(ValueError(RANGE_ERROR),request,{'open':(None,None,updater,None),'closed':(None,None,updater,None)})
    sequence=[(n if n in ('pregrasp','close','lift','hold') else 'approach',n) for n in labels]
    physical=evaluate(ep,sequence=sequence)
    if physical['pass']:raise RuntimeError('Unexpected complete physical success')
    frozen=list(folder.glob('blocks/*/frozen_*.json'))
    if not frozen or any(not read(p)['pass'] for p in frozen):raise RuntimeError('Pre-execution frozen configuration failed')
    evidence={str(p):sha(p) for p in ep.rglob('*') if p.is_file()}
    for p in [*frozen,*paths,response,model,previous/'registration.json',previous/'STOP',previous/'summary.json',
              previous/'advisor33_verification.json',previous/'workers/planner/FAULT.json',group/'FAULT.json',group/'shutdown_receipt.json']:
        evidence[str(p)]=sha(p)
    detail={k:trial.get(k) for k in ('trial_id','phase','object_id','condition','vision_arm','comparison_arm','repeat','target_delta_world_m')}
    occurred=set(labels)
    flags={'H_reached':'state_generation' in occurred,'first_D_motion':'d_refine_001' in occurred,
           'second_D_motion':'d_refine_002' in occurred,'close':'close' in occurred,'lift':'lift' in occurred,
           'hold':'hold' in occurred,'neural_inference':bool(policy['model_calls']),'full_stale_two_pairs':False,
           'position_window':any(read(p)['origin']=='settled_by_position_window' for p in ep.glob('*_waypoint_stop.json'))}
    detail.update(classification='interrupted_unknown',outcome='infrastructure_interrupted: '+boundary['rejected'],
        task_success=False,physical_attempt=True,record_integrity=True,native_terminal_present=False,
        record_integrity_scope='Occurred action/request/capture prefix verified; native terminal and final physical receipt remain absent',
        source_campaign=str(previous),source_episode=str(ep),event_applicability=flags,model_calls=policy['model_calls'],
        controller_calls=policy['policy_calls'],stale_pairs=0,capture=capture,physical_prefix=physical,
        measured_gripper_rejection=boundary,failed_planning_label=request['label'],retry=False,
        final_frozen_fingerprint_present=(folder/'frozen_after.json').exists())
    out.mkdir();detail['video']=video(ep,out/'videos'/(ident+'.mp4'))
    value={'previous':str(previous),'detail':detail,'original_claim':claim,'evidence':evidence,'old_execution_exit':idle,
        'new_physical_trials':0,'new_neural_calls':0,'native_records_or_ledger_rewritten':False,
        'corrected_consumed_slots':len(summary['trials'])+1,'corrected_unexecuted':len(summary['unexecuted']),
        'previous_report_misclassified_started_slot_as_unexecuted':True}
    write(out/'interruption_record.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');p.add_argument('out');a=p.parse_args()
    value=record(a.previous,a.out);print(value['detail']['outcome'])
