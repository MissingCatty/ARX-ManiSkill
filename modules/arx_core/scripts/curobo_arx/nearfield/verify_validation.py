"""Offline post-run verification of the bounded held-out ARX recapture."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sqlite3

from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from .capture import audit_episode
from .closure import validate_reply
from .prepare_v2 import predecessors
from .runtime import audit_actions


def reset_signature(value):
    # Clone identity and simulation clock are not scene initial conditions.
    return {k:({name:{field:(re.sub(r'/World/Envs/env_\d+/', '/World/Envs/ENV/', item)
                            if field=='path' else item) for field,item in obj.items()}
                for name,obj in v.items()} if k=='objects' else v)
            for k,v in value.items() if k!='reset_sim_time'}


def verify(campaign):
    campaign=Path(campaign).resolve()
    if (campaign/'postrun_verification.json').exists():
        raise FileExistsError('Preserve previous verification')
    reg=read(campaign/'registration.json');summary=read(campaign/'summary.json')
    if read(campaign/'state.json')['status']!='complete':
        raise RuntimeError('Execution is not complete')
    idle=predecessors([campaign])
    pins={**reg['execution_source_files'],**reg['dependencies']}
    # Immutable independent reads; this never queries a simulator or GPU.
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda item:verify_files(dict([item])),pins.items()))
    selected=read(campaign/'source_preflight.json')['selected']
    source_by_id={s['source_episode_id']:s for s in selected}
    trials={t['trial_id']:t for t in summary['trials']}
    if (len(selected)!=reg['authorized_validation_slots'] or
        len({r['source_episode_id'] for r in selected})!=len(selected) or
        {t['source_episode_id'] for t in trials.values()}!={r['source_episode_id'] for r in selected if r['eligible']} or
        set(trials)!={t['trial_id'] for t in reg['seed_trials']} or
        summary['incomplete_accounting'] or summary['unexecuted']):
        raise RuntimeError('Declared, scheduled and terminal trial accounting disagree')
    if reg['training_authorized_here'] or reg['formal_48_trials_authorized_here']:
        raise RuntimeError('Unexpected campaign scope')
    details=[];coverage=Counter();frozen=[];evidence={}
    for trial in trials.values():
        ledger=campaign/'trial_ledgers'/(trial['trial_id']+'.sqlite3')
        db=sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:rows=[dict(row) for row in db.execute('SELECT * FROM attempts ORDER BY started,id')]
        finally:db.close()
        if len(rows)!=1 or len(trial['attempts'])!=1 or rows[0]!=trial['attempts'][0]:
            raise RuntimeError('No-retry ledger differs from terminal trial record')
        attempt=rows[0];episode=Path(attempt['episode'])
        old_reset=Path(source_by_id[trial['source_episode_id']]['source_episode'])/'initial_conditions.json'
        if reset_signature(read(episode/'initial_conditions.json'))!=reset_signature(read(old_reset)):
            raise RuntimeError('Historical commanded reset conditions changed')
        if not trial['completed'] or not trial['block']['idle']:
            raise RuntimeError('Completed/idle trial record missing')
        physical=read(episode/'physical_receipt.json')
        quality=read(episode/'quality_receipt.json')
        receipt=json.loads(attempt['receipt']) if isinstance(attempt['receipt'],str) else attempt['receipt']
        if trial['admitted']:
            if not quality['pass'] or not receipt:
                raise RuntimeError('Admitted episode lacks a passing quality/ledger receipt')
            verify_files({str(episode/'physical_receipt.json'):receipt['physical_sha256'],
                          str(episode/'quality_receipt.json'):receipt['quality_sha256']})
        # Failed physical tasks deliberately have no admission receipt. Their
        # occurred requests, actions and captures are still checked below.
        verify_files({str(episode/name):value for name,value in quality.get('files',{}).items()})
        if physical['pass']!=trial['success'] or (attempt['state']=='committed')!=trial['admitted']:
            raise RuntimeError('Task result and admission disagree with ledger')
        requests=read(episode/'requests.json');actions=audit_actions(episode,requests)
        if not set(physical['executed_stages'])<=set(actions['executed']):
            raise RuntimeError('Occurred stage lacks action evidence')
        capture=audit_episode(episode)
        events=Counter(e['kind'] for e in read(episode/'interpreter_events.json'))
        close='close' in actions['executed'];lift='lift' in actions['executed'];hold='hold' in actions['executed']
        if close:
            gate=read(episode/'closure_gate_optimization.json');binding=gate['binding']
            reply=read(episode/'closure_model_validation.json')
            previous=read(episode/(binding['previous_command_label']+'_command.json'))
            if events['grasp_request']!=1 or not binding['pass'] or not validate_reply(binding,reply,previous):
                raise RuntimeError('Closure lacks a unique explicit GRASP and bound model check')
            verify_files(binding['files'])
        if physical['pass'] and not (close and lift and hold):
            raise RuntimeError('Successful task is missing required executed stages')
        chronology=read(episode.parent.parent/'chronological_evaluation.json')
        if (chronology['original_receipt_sha256']!=sha(episode/'physical_receipt.json') or
            chronology['result']['pass']!=physical['pass']):
            raise RuntimeError('Independent chronological evaluation disagrees')
        checks=list((episode.parent.parent/'blocks').glob('*/frozen_*.json'))
        if not checks:raise RuntimeError('Missing frozen live-properties evidence')
        for path in checks:
            check=read(path)
            if not check['pass'] or check['mismatches'] or not check['additional_assets']['pass']:
                raise RuntimeError('Frozen robot/scene properties failed')
            frozen.append({'file':str(path),'sha256':sha(path),'trial_id':trial['trial_id']})
        video=read(campaign/'videos'/(trial['trial_id']+'.json'))
        if not video['decode_pass']:raise RuntimeError('Video failed to decode')
        verify_files({video['path']:video['sha256']})
        applicability={'STEP':events['target_request']>0,'GRASP':bool(events['grasp_request']),
            'closure_gate':close,'close':close,'lift':lift,'hold':hold}
        coverage.update({k:int(v) for k,v in applicability.items()})
        details.append({'trial_id':trial['trial_id'],'source_episode_id':trial['source_episode_id'],
            'episode_id':episode.name,'object_id':trial['object_id'],'physical_success':physical['pass'],
            'admitted':trial['admitted'],'record_integrity_pass':True,'actions':actions,'capture':capture,
            'historical_commanded_reset_unchanged':True,
            'historical_reset_file':str(old_reset),'historical_reset_sha256':sha(old_reset),
            'admission_ledger_hash_applicable':bool(trial['admitted']),
            'events':dict(events),'event_applicability':applicability,
            'hold_min_rise_m':physical.get('hold_min_rise_m'),
            'recorded_hold_duration_s':physical.get('recorded_hold_duration_s'),
            'bilateral_closure':physical.get('bilateral_closure'),'video':video})
        for name in ('physical_receipt.json','quality_receipt.json','requests.json','interpreter_events.json'):
            evidence[str(episode/name)]=sha(episode/name)
    rejected=[r for r in selected if not r['eligible']]
    result={'pass':True,'declared_sources':len(selected),'scheduled':len(trials),
        'physically_attempted':summary['actual_attempts'],'physical_successes':sum(d['physical_success'] for d in details),
        'admitted':sum(d['admitted'] for d in details),'grasp_failures':sum(not d['physical_success'] for d in details),
        'preflight_unexecuted':rejected,'unknown_outcomes':0,'incomplete_records':0,
        'raw_synchronized_frames':sum(d['capture']['frames'] for d in details),
        'actual_commands':sum(len(d['actions']['executed']) for d in details),
        'event_applicability':{'scheduled_physical_trials':len(details),
            'all_motion_events_not_applicable_to_preflight_rejections':len(rejected)},
        'key_event_coverage_trials':dict(coverage),'trials':details,'frozen_checks':frozen,
        'unchanged_registered_file_count':len(pins),'evidence':evidence,
        'verification_source_sha256':sha(__file__),'trained_policy_trials':0,'formal_comparison_trials':0}
    write(campaign/'final_idle_verification.json',{'pass':True,'proof':idle,
        'execution_and_dependency_hashes_unchanged':True,'training_steps':0})
    write(campaign/'postrun_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args()
    r=verify(a.campaign)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trials','frozen_checks','evidence','preflight_unexecuted')}))
