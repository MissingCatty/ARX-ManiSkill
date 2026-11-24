"""Verify actual entry coverage, chronology and provenance after ARX exit."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sqlite3
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files
from training.nearfield_arx.export import state_time
from training.nearfield_arx.contract import pose_error
from .prepare_v2 import predecessors
from .runtime import audit_actions
from .capture import audit_episode
from .closure import validate_reply
from .verify_validation import reset_signature


def entry_matches_raw(row,entry):
    # Recording.add adds only storage paths to its own copy. The saved live
    # observation must match every original field, including rendered state.
    added={'rgb_path','depth_path','rgb_relative','depth_relative','capture_files'}
    if {k:v for k,v in row.items() if k not in added}!=entry:
        raise ValueError('Entry observation differs from its original raw row')
    return True


def verify(campaign):
    campaign=Path(campaign).resolve()
    if (campaign/'postrun_verification.json').exists():raise FileExistsError('Preserve completed verification')
    reg=read(campaign/'registration.json');summary=read(campaign/'summary.json')
    idle=predecessors([campaign]);verify_files({**reg['execution_source_files'],**reg['dependencies']})
    registered={t['trial_id']:t for t in reg['seed_trials']}
    seen=set();details=[];coverage=Counter();pins={}
    for trial in summary['trials']:
        ident=trial['trial_id']
        if ident in seen or ident not in registered:raise ValueError('Unexpected/repeated trial')
        seen.add(ident);spec=registered[ident]
        for key in ('seed_id','G0_world','entry_world','offset_goal_world','source_split','placement','perturbation'):
            if trial[key]!=spec[key]:raise ValueError('Executed slot differs from registration')
        ledger=campaign/'trial_ledgers'/(ident+'.sqlite3')
        db=sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:attempts=[dict(row) for row in db.execute('SELECT * FROM attempts ORDER BY started,id')]
        finally:db.close()
        if attempts!=trial['attempts'] or len(attempts)>1:raise ValueError('Ledger/terminal record mismatch')
        item={'trial_id':ident,'object_id':trial['object_id'],'split':trial['source_split'],
            'physical_success':trial['success'],'admitted':trial['admitted'],'outcome':trial['outcome'],
            'motion_dispatched':bool(trial['physical_dispatches']),'record_integrity_pass':True}
        flags={k:False for k in ('entry_motion','fresh_entry','teacher_motion','GRASP','closure_gate','close','lift','hold')}
        if not attempts:
            if trial['success'] or trial['admitted'] or trial['physical_dispatches']:
                raise ValueError('Outcome credited without an attempt')
            item.update(event_applicability=flags,task_stage='goal_qualification_rejected');details.append(item);continue
        a=attempts[0];episode=Path(a['episode']);physical=read(episode/'physical_receipt.json')
        old=Path(reg['seeds'][trial['seed_id']]['episode'])/'initial_conditions.json'
        if reset_signature(read(episode/'initial_conditions.json'))!=reset_signature(read(old)):
            raise ValueError('Historical initial reset changed')
        if trial['success']!=physical['pass'] or trial['admitted']!=(a['state']=='committed'):
            raise ValueError('Physical result/admission mismatch')
        requests=read(episode/'requests.json') if (episode/'requests.json').exists() else []
        actions=audit_actions(episode,requests)
        if a['physical'] and not actions['executed']:raise ValueError('Physical ledger lacks execution')
        events=read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
        event_counts=Counter(e['kind'] for e in events)
        samples_path=episode/'samples.jsonl';raw=[]
        if samples_path.exists():raw=[json.loads(line) for line in samples_path.read_text().splitlines()]
        capture=audit_episode(episode) if raw else {'applicable':False,'frames':0}
        if trial['admitted']:
            q=read(episode/'quality_receipt.json');receipt=json.loads(a['receipt']) if isinstance(a['receipt'],str) else a['receipt']
            if not q['pass']:raise ValueError('Admitted data lacks quality pass')
            verify_files({str(episode/n):h for n,h in q['files'].items()})
            verify_files({str(episode/'quality_receipt.json'):receipt['quality_sha256'],str(episode/'physical_receipt.json'):receipt['physical_sha256']})
        occurred=set(actions['executed']);flags.update(entry_motion='state_generation' in occurred,
            fresh_entry=bool(event_counts['physical_entry_stopped']),
            teacher_motion=any(k.startswith(('c_step_','d_refine_')) for k in occurred),GRASP=bool(event_counts['grasp_request']),
            close='close' in occurred,lift='lift' in occurred,hold='hold' in occurred)
        entry=[e for e in events if e['kind']=='physical_entry_stopped']
        if entry:
            if len(entry)!=1 or not flags['entry_motion']:raise ValueError('Fresh entry lacks a unique actual arrival')
            event=entry[0];index=event['raw_index'];row=raw[index]
            generation=read(episode/'state_generation_execution.json')
            if state_time(row)<=generation['final']['sim_time']+1e-6:raise ValueError('Entry image precedes state-generation completion')
            if event['registered_G0_world']!=trial['G0_world'] or event['registered_H_world']!=trial['entry_world']:
                raise ValueError('Entry/prior binding changed')
            entry_matches_raw(row,read(episode/'physical_entry_observation.json'))
            target=generation['final']['objects']['test_object']['path']
            if any(c['num_contacts']>0 and target in c['a']+c['b'] for c in generation['contacts']):
                raise ValueError('Positive entry generated by target contact')
            start=read(episode/'capture_snapshot.json')['objects']['test_object']['world']
            drift=pose_error(generation['final']['objects']['test_object']['world'],start)
            if drift['translation_m']>.001 or drift['rotation_rad']>.01:raise ValueError('Positive entry moved the object')
            speed=dict(zip(row['snapshot']['joint_names'],row['snapshot']['joint_velocities']))
            if max(abs(speed[n]) for n in row['snapshot']['right_arm'])>.01:raise ValueError('Positive entry was moving')
            error=pose_error(row['snapshot']['link_world']['right_eef_link'],trial['source_Gstar_world'])
            item.update(actual_entry_to_source_Gstar=error,entry_rgb=str(episode/row['rgb_relative']),
                entry_raw_index=index,entry_rgb_sha256=sha(episode/row['rgb_relative']),
                registered_rotation_degrees=float(np.linalg.norm(trial['perturbation']['rotation_Gstar_axes_degrees'])))
            traces=read(episode/'decision_traces.json')
            if any(t['raw_index']<index for t in traces):raise ValueError('State generation has positive decision labels')
        if flags['teacher_motion'] and not flags['fresh_entry']:raise ValueError('Teacher motion lacks fresh physical entry')
        if flags['close']:
            gate=read(episode/'closure_gate_optimization.json');binding=gate['binding']
            reply=read(episode/'closure_model_validation.json');previous=read(episode/(binding['previous_command_label']+'_command.json'))
            if event_counts['grasp_request']!=1 or not validate_reply(binding,reply,previous):
                raise ValueError('Close lacks original explicit-GRASP validation')
            verify_files(binding['files']);flags['closure_gate']=True
        if physical['pass'] and not all(flags.values()):raise ValueError('Success lacks a required event')
        if requests:
            chronology=read(episode.parent.parent/'chronological_evaluation.json')
            if chronology['original_receipt_sha256']!=sha(episode/'physical_receipt.json') or chronology['result']['pass']!=physical['pass']:
                raise ValueError('Independent chronological evaluation mismatch')
        frozen=list((episode.parent.parent/'blocks').glob('*/frozen_*.json'))
        if not frozen:raise ValueError('Live frozen-property evidence missing')
        for f in frozen:
            result=read(f)
            if not result['pass'] or result['mismatches'] or not result['additional_assets']['pass']:
                raise ValueError('Frozen robot/scene properties changed')
            pins[str(f)]=sha(f)
        coverage.update({k:int(v) for k,v in flags.items()})
        item.update(event_applicability=flags,capture=capture,actions=actions,
            historical_reset_unchanged=True,hold_min_rise_m=physical.get('hold_min_rise_m'))
        details.append(item)
        for name in ('physical_receipt.json','quality_receipt.json','requests.json','interpreter_events.json',
                     'physical_entry_observation.json','decision_traces.json','samples.jsonl'):
            f=episode/name
            if f.exists():pins[str(f)]=sha(f)
    result={'pass':not summary['incomplete_accounting'],'record_integrity_pass':not summary['incomplete_accounting'],
        'scheduled':len(registered),'terminal_slots':len(details),'physical_attempts':summary['actual_attempts'],
        'physical_successes':sum(d['physical_success'] for d in details),'admitted':sum(d['admitted'] for d in details),
        'unexecuted':summary['unexecuted'],'unknown_or_incomplete':summary['incomplete_accounting'],
        'all_scheduled_finished':len(details)==len(registered),'event_coverage_trials':dict(coverage),'trials':details,
        'evidence':pins,'trained_policy_trials':0,'verification_source_sha256':sha(__file__)}
    write(campaign/'final_idle_verification.json',{'pass':True,'proof':idle,'all_registered_sources_rechecked':True})
    write(campaign/'postrun_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');a=p.parse_args();r=verify(a.campaign)
    print({k:v for k,v in r.items() if k not in ('trials','evidence','unexecuted','unknown_or_incomplete')})
