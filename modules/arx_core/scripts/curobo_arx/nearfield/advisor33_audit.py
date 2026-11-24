"""Independent raw-action/stop/physical evaluation for advisor33."""
from pathlib import Path
import json
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files,digest
from scripts.curobo_arx.data_factory.recording import physical as original_physical
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate as original_evaluate
from training.nearfield_arx.learned_policy import audit_policy as base_audit,read_bound_reply
from .runtime import audit_actions
from .advisor32_rules import StopSample,settle,local_admission


def record_quality(episode):
    """Benchmark raw-record admission; no legacy G0-conditioned training codec."""
    from PIL import Image
    from .capture import audit_episode
    ep=Path(episode);meta=read(ep/'episode.json');rows=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()]
    if not meta['write_complete'] or meta['samples']!=len(rows) or len(rows)<9:raise RuntimeError('Incomplete synchronized record')
    capture=audit_episode(ep);requests=read(ep/'requests.json');ids={r['id'] for r in requests};content=[];files={}
    audit_actions(ep,requests);audit_policy(ep)
    from scipy.spatial.transform import Rotation
    for i,row in enumerate(rows):
        if any(row[k]!=meta[k] for k in ('env_id','attempt_id','object_id')):raise RuntimeError('Mixed episode capture')
        if row['camera_owner_root']!=meta['environment_root'] or row['camera_role']!='right_wrist':raise RuntimeError('Wrong camera owner/role')
        if abs(row['frame_time']-row['capture_sim_time'])>1e-4:raise RuntimeError('Stale frame/state')
        if meta['visual_geometry_required'] and not row['visual_geometry']['pass']:raise RuntimeError('Invalid visual frame')
        cmd=row.get('teacher_command') or {}
        if cmd and cmd['id'] not in ids:raise RuntimeError('Unregistered captured command')
        if 'state' in row:raise RuntimeError('Unexpected legacy goal-conditioned state in zero-prior record')
        for key in ('rgb_relative','depth_relative'):
            p=ep/row[key];files[row[key]]=sha(p)
        with Image.open(ep/row['rgb_relative']) as im:im.verify()
        content.append([files[row['rgb_relative']],row['snapshot']['link_world']['right_eef_link'],row.get('stage'),cmd.get('gripper_target')])
        if i:
            prior=rows[i-1]
            if not .19<=row['frame_time']-prior['frame_time']<=.21:raise RuntimeError('Missing consecutive capture')
            a=np.array(prior['snapshot']['link_world']['right_eef_link']);b=np.array(row['snapshot']['link_world']['right_eef_link'])
            action=np.r_[(b[:3,3]-a[:3,3])/.05,Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).as_rotvec()/.5]
            if not np.isfinite(action).all() or np.max(np.abs(action))>1+1e-6:raise RuntimeError('Measured motion exceeds original action representation')
    for p in ep.rglob('*'):
        if p.is_file() and p.name not in ('quality_receipt.json',) and p.suffix in ('.json','.jsonl','.png','.npy','.npz'):
            files[str(p.relative_to(ep))]=sha(p)
    result={'pass':True,'samples':len(rows),'capture':capture,'content_id':digest(content),'files':files,
        'physical_receipt_sha256':sha(ep/'physical_receipt.json'),'training_export':False,
        'scope':'Benchmark synchronized raw action/state/image integrity, no G0 state or training labels generated'}
    write(ep/'quality_receipt.json',result);return result


def bound_stop(proof,raw):
    verify_files(proof['files'])
    for sample in proof['samples']:
        matches=[r['snapshot'] for r in raw if abs(r['snapshot']['sim_time']-sample['sim_time'])<1e-8]
        if not matches:raise RuntimeError('Stop sample missing from synchronized raw recording')
        s=matches[-1];idx=[s['joint_names'].index(n) for n in s['right_arm']]
        for actual,expected in ((np.array(s['link_world']['right_eef_link'])[:3,3],sample['position']),
                (np.array(s['joint_positions'])[idx],sample['joints']),
                (np.array(s['joint_velocities'])[idx],sample['velocities'])):
            if not np.allclose(actual,expected,atol=1e-12,rtol=0):raise RuntimeError('Altered measured stop sample')
    expected=settle([StopSample(**x) for x in proof['samples']],active=proof['controller_active'],fault=None,elapsed_wall_s=proof['wait_elapsed_wall_s'])
    if expected['pass']!=proof['pass'] or expected['origin']!=proof['origin']:raise RuntimeError('Stop evidence does not reproduce')
    return expected


def image_binding(reg,req,rep):
    actual=[str(Path(req['observation']).with_name(req['request_id']+'_'+n+'.png')) for n in ('rgb','depth')]
    if not rep['model_inference']:expected=[]
    elif req['vision_arm']=='full_stale':expected=reg['full_stale_donors'][req['object_id']]['images_by_plane'][str(req['motions_completed'])]
    else:expected=actual
    if rep['visual_control']['images']!=expected:raise RuntimeError('Unregistered images supplied to frozen model')
    if rep['model_inference'] and rep['checkpoint']!=reg['learned_policy']['checkpoints'][req['method']]:raise RuntimeError('Model checkpoint changed')
    prompt=read(Path(req['observation']).with_name(req['request_id']+'_input.json'))
    if prompt['images']!=(expected or actual):raise RuntimeError('Model prompt differs from registered intervention')
    return True


def audit_partial(trial_dir):
    """Preserve one known in-flight action; never excuse missing completed evidence."""
    folder=Path(trial_dir);timeout=read(folder/'ISOLATED_TIMEOUT.json');verify_files(timeout['files'])
    if not timeout['known_partial_record'] or timeout['outcome']!='interrupted_unknown':raise RuntimeError('Unknown timeout record')
    ep=Path(timeout['episode']);requests=read(ep/'requests.json');labels={r['label'] for r in requests}
    commands={p.name.removesuffix('_command.json') for p in ep.glob('*_command.json')}
    executions={p.name.removesuffix('_execution.json') for p in ep.glob('*_execution.json')}
    if labels!=commands or executions-labels:raise RuntimeError('Orphan/missing action command at timeout')
    missing=labels-executions
    if len(missing)>1 or (missing and next(iter(missing))!=requests[-1]['label']):raise RuntimeError('Earlier completed action receipt missing')
    audit_actions(ep,[r for r in requests if r['label'] in executions])
    ledger=read(ep/'policy_requests.json') if (ep/'policy_requests.json').exists() else []
    ids={r['request_id'] for r in ledger};present={p.name.removesuffix('_request.json') for p in (ep/'policy').glob('*_request.json')}
    if ids!=present:raise RuntimeError('Incomplete policy request ledger at timeout')
    pending=[]
    for row in ledger:
        p=Path(row['request']);req=read(p);verify_files(req['files']);reply=p.with_name(req['request_id']+'_reply.json')
        if reply.exists():read_bound_reply(p,reply)
        else:pending.append(req['request_id'])
    if len(pending)>1 or (pending and pending[0]!=ledger[-1]['request_id']):raise RuntimeError('Earlier policy reply missing')
    return {'pass':True,'known_unfinished_actions':sorted(missing),'known_pending_policy':pending,'success':False,'retry':False}


def audit_policy(episode):
    episode=Path(episode);result=base_audit(episode);events=read(episode/'interpreter_events.json')
    for row in result['requests']:
        p=episode/'policy'/(row['request_id']+'_request.json');req,rep=read_bound_reply(p,p.with_name(row['request_id']+'_reply.json'))
        if req['method']=='D' and req['vision_arm']!='g0' and 'G0_world' in req['inputs']:
            raise RuntimeError('Coarse prior in zero-prior D request')
        for event in (e for e in events if e.get('policy_request_id')==req['request_id']):
            if event['kind']=='target_request':
                if rep['action']['kind']!='move' or not np.allclose(event['goal_world'],rep['action']['goal_world'],atol=1e-12,rtol=0):
                    raise RuntimeError('Executed target differs from recorded decision')
                if not local_admission(req['inputs']['eef_world'],event['goal_world'])['pass']:
                    raise RuntimeError('Executed local movement exceeds bounds')
            elif event['kind']=='grasp_request':
                if rep['action']['kind']!='close' or not rep['finger_box']['pass']:
                    raise RuntimeError('Closure without visual admission')
        if req['mode']=='final_gate' and rep['model_inference']:raise RuntimeError('Extra inference after motion budget')
        if rep['model_inference'] and req['vision_arm']=='full_stale' and not rep['visual_control']['replaced']:
            raise RuntimeError('Full-stale decision reverted to a fresh view')
        if req['vision_arm']=='g0' and rep['model_inference']:raise RuntimeError('Geometry baseline called model')
    result['model_calls']=sum(read(episode/'policy'/(r['request_id']+'_reply.json'))['model_inference'] for r in result['requests'])
    return result


def physical(episode, persist=True):
    ep=Path(episode);audit_actions(ep,read(ep/'requests.json'))
    audit_policy(ep)
    if persist:
        original=original_physical(ep);write(ep/'strict_physical_receipt.json',original)
    sequence=read(ep/'execution_roles.json')['sequence'];strict=original_evaluate(ep,sequence=sequence)
    raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
    accepted=[];stops=[]
    for path in ep.glob('*_waypoint_stop.json'):
        proof=read(path);bound_stop(proof,raw)
        if proof['pass']:accepted.append(proof['label'])
        stops.append(proof['origin'])
    closure=False
    if (ep/'close_execution.json').exists():
        gate=read(ep/'closure_gate_optimization.json');binding=gate['binding'];cert=read(ep/'closure_model_validation.json')
        verify_files(binding['files'])
        bound_stop(binding['arrival'],raw)
        closure=(gate['pass'] and cert['pass'] and cert['binding_sha256']==digest(binding)
                 and gate['maximum_joint_drift_rad']<=.0005 and binding['visual_gate']['pass'])
        if read(ep/'close_command.json')['position']!=[binding['hold_q'],binding['hold_q']]:raise RuntimeError('Wrong closure hold command')
    exempt=[s for s in strict['executed_stages'] if s not in strict['converged_stages'] and s in accepted
            and (s in ('state_generation','close') or s.startswith(('c_step_','d_refine_')))]
    issues=[i for i in strict.get('issues',[]) if not(isinstance(i,dict) and i.get('stage') in exempt and i.get('error')=='Original ARX convergence gate rejected execution')]
    planning=strict.get('planning',{})
    passed=bool(not issues and not strict['missing_stages'] and closure and strict['bilateral_closure']
        and strict['all_recorded_hold_samples_above_6cm'] and not strict['forbidden_contact_events']
        and planning.get('dense_invalid_samples')==0 and planning.get('max_limit_exceed_rad',float('inf'))<=1e-6
        and len(strict['converged_stages'])+len(exempt)==len(sequence) and not strict.get('reported_error'))
    result={**strict,'pass':passed,'score':float(passed),'issues':issues,'strict_original_success':strict['pass'],
        'strict_original_issues':strict['issues'],'independent_closure_pass':closure,
        'certified_stationary_tracking_exemptions':exempt,'stop_origins':stops,
        'outcome':'success' if passed else 'failed_execution_or_planning'}
    if persist:write(ep/'physical_receipt.json',result)
    return result
