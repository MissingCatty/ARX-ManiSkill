"""Read-only, shared ARX single-episode checks; no dataset statistics or copies."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from observations import relative_state, encode_state, teacher_action, policy_observation
from interaction_contract import validate_episode, STATE, digest, file_sha

LOCAL_STAGES=('approach','close','lift','hold')


def transition_label(first,second,include_command_starts=False):
    """Label the measured future interval, including a command starting in it.

    The first observation stays untouched. A newly issued gripper command is
    the label for that interval, rather than the preceding idle/open command.
    Command-onset precision is bounded by the actual recording interval.
    """
    current=first.get('teacher_command') or {};following=second.get('teacher_command') or {}
    starts=bool(include_command_starts and second.get('stage') in LOCAL_STAGES
        and following.get('id') and following.get('id')!=current.get('id'))
    stage=second.get('stage') if starts else first.get('stage')
    if stage not in LOCAL_STAGES:return None
    command=following if starts else current
    if include_command_starts and 'gripper_target' not in command:
        raise ValueError('A local action label requires its actual recorded gripper command')
    labeled_first={**first,'teacher_command':command}
    return stage,teacher_action(labeled_first,second),command,starts


def object_identity(source):
    names=[]
    for filename,key in (('scenario.json','asset'),('active_asset.json','name')):
        path=Path(source)/filename
        if path.exists():names.append(json.loads(path.read_text())[key])
    if len(set(names))>1:raise ValueError('Conflicting object identity: '+str(source))
    return names[0] if names else 'bbq_sauce_bottle'


def recording_identity(rows):
    # Ignore transport paths/wall clocks/request UUIDs; retain measured temporal
    # sequence and image content. Copying the same recording cannot add episodes.
    payload=[]
    for row in rows:
        command=row.get('teacher_command') or {}
        payload.append({'frame_time':row['frame_time'],'capture_sim_time':row['capture_sim_time'],
            'state':row['state'],'stage':row.get('stage'),'camera_role':row.get('camera_role'),
            'gripper_target':command.get('gripper_target'), 'rgb_sha256':file_sha(row['rgb_path']),
            'ee_world':row['snapshot']['link_world']['right_eef_link']})
    return digest(payload)


def inspect_episode(source, *, state_convention=STATE, require_entry=True,
                    include_command_starts=True, exclude_gated_dwell_starts=True,
                    instruction=None, image_sink=None, require_contract=False):
    source=Path(source)
    resolved=None
    if (source/'resolved_interaction.json').exists():
        resolved=validate_episode(source,json.loads((source/'resolved_interaction.json').read_text())['contract'])
        if state_convention != STATE:raise ValueError('State convention conflicts with recorded interaction contract')
        require_entry=include_command_starts=exclude_gated_dwell_starts=True
    elif require_contract:raise ValueError('New collection requires a recorded interaction contract')
    verification=json.loads((source/'verification.json').read_text())
    recording=json.loads((source/'teacher_recording.json').read_text())
    if not verification['pass'] or recording.get('error'):
        raise ValueError(f'Failed execution/recording cannot become a successful demonstration: {source}')
    recording_sha=hashlib.sha256((source/'teacher_raw/samples.jsonl').read_bytes()).hexdigest()
    grasp_path=source/'selected_grasp.json'
    grasp=json.loads(grasp_path.read_text())['grasp_world'] if grasp_path.exists() else None
    if state_convention!='object_relative_v1' and grasp is None:
        raise ValueError(f'Explicit selected grasp is required for goal conditioning: {source}')
    entry=None
    if require_entry:
        from egad_evaluate import evaluate
        physical=evaluate(source)
        if not physical['pass']:raise ValueError(f'Actual ARX execution failed unchanged evaluator: {source}')
        certificate=json.loads((source/'handoff_actual_entry.json').read_text())
        if not certificate['pass']:raise ValueError(f'Unverified shared handoff entry: {source}')
        entry=json.loads((source/'handoff_actual_observation/observation.json').read_text())
    rows=[json.loads(line) for line in (source/'teacher_raw/samples.jsonl').read_text().splitlines()]
    if recording.get('samples')!=len(rows):raise ValueError('Recording receipt/raw frame count differs')
    for row in rows:
        if row.get('camera_role')!='right_wrist':raise ValueError('Non-wrist recording')
        if not Path(row['rgb_path']).is_file():raise ValueError('Missing raw wrist image: '+row['rgb_path'])
        with Image.open(row['rgb_path']) as raw_image:raw_image.verify()
        if not np.isfinite(row['capture_sim_time']) or abs(row['frame_time']-row['capture_sim_time'])>1e-4:
            raise ValueError('Recording frame/state time mismatch')
        if not np.isfinite(np.asarray(row['state'])).all():raise ValueError('Nonfinite recorded state')
        if not np.allclose(relative_state(row['snapshot']),row['state'],atol=1e-7,rtol=0):
            raise ValueError('Recorded state differs from measured ARX state')
        encode_state(row['snapshot'],state_convention,grasp)
    times_raw=np.asarray([row['frame_time'] for row in rows])
    if not np.isfinite(times_raw).all() or (np.diff(times_raw)<=0).any():
        raise ValueError(f'Duplicate or nonmonotonic wrist frame keys: {source}')
    items=[];actions=[];states=[];rejected=[];first_approach=None
    for i,(row,next_row) in enumerate(zip(rows,rows[1:])):
        label=transition_label(row,next_row,include_command_starts)
        if label is None:continue
        stage,action,command,command_starts=label
        if any(r.get('camera_role')!='right_wrist' for r in (row,next_row)):
            raise ValueError(f'Non-wrist camera in local policy data: {source}')
        dt=next_row['frame_time']-row['frame_time']
        if any(not np.isfinite(r['capture_sim_time']) or abs(r['frame_time']-r['capture_sim_time'])>1e-4 for r in (row,next_row)):
            rejected.append({'sample':i,'reason':'frame/state time mismatch'});continue
        if not .19<=dt<=.21:
            rejected.append({'sample':i,'reason':'nonconsecutive frames','dt':dt});continue
        state=relative_state(row['snapshot'])
        if not np.allclose(state,row['state'],atol=1e-7,rtol=0):
            raise ValueError('Recorded ARX state does not match released LiLo state convention')
        if not np.isfinite(state).all():raise ValueError(f'Nonfinite state: {source} row {i}')
        state=encode_state(row['snapshot'],state_convention,grasp)
        if not np.isfinite(action).all() or (np.abs(action[:6])>1).any():
            rejected.append({'sample':i,'reason':'teacher delta exceeds declared action space'});continue
        image,boxes,state=policy_observation(row,state_convention,grasp)
        path=f'{len(items):06d}.png'
        if image_sink is not None:image_sink(path,image)
        items.append({'image':path,'source':row['rgb_path'],
            'frame_time':row['frame_time'],'stage':stage,'observation_stage':row.get('stage'),'distractor_boxes':boxes,
            'raw_sample_index':i,'teacher_command_id':command.get('id'),
            'command_started_during_interval':command_starts,'next_frame_time':next_row['frame_time']})
        if first_approach is None and stage=='approach':first_approach=row
        states.append(state);actions.append(action)
    if len(items)<8:raise ValueError(f'Insufficient valid local interaction frames: {source}')
    onsets={stage:sum(item['command_started_during_interval'] and item['stage']==stage for item in items) for stage in LOCAL_STAGES}
    if include_command_starts and any(count<1 for count in onsets.values()):
        raise ValueError(f'Missing synchronized pre-command transition for a local stage: {source}: {onsets}')
    states=np.asarray(states);actions=np.asarray(actions)
    times=np.array([item['frame_time'] for item in items])
    contiguous=[i for i in range(len(items)-7) if np.allclose(np.diff(times[i:i+8]),.2,atol=1e-4,rtol=0)]
    # Deployment handles the original 3s closure dwell and final hold
    # internally. Keep these real frames as future chunk labels, but do
    # not teach the NN to wait again after that gate already completed.
    starts=[i for i in contiguous if not exclude_gated_dwell_starts
        or items[i]['stage'] in ('approach','lift')
        or items[i]['stage']=='close' and items[i]['command_started_during_interval']]
    if not starts:raise ValueError(f'No contiguous eight-action training chunks: {source}')
    if include_command_starts and not any(items[i]['stage']=='close' and items[i]['command_started_during_interval'] for i in starts):
        raise ValueError(f'No trainable chunk includes the pre-closure observation: {source}')
    if require_entry and next((i for i,item in enumerate(items) if item['stage']=='approach'),None) not in starts:
        raise ValueError(f'Actual handoff observation has no contiguous training chunk: {source}')
    alignment=None
    if entry:
        if first_approach is None:raise ValueError(f'No approach data from actual handoff: {source}')
        a=np.asarray(entry['snapshot']['link_world']['right_eef_link'])
        b=np.asarray(first_approach['snapshot']['link_world']['right_eef_link'])
        alignment={'first_approach_position_delta_m':float(np.linalg.norm(b[:3,3]-a[:3,3])),
            'first_approach_rotation_delta_rad':float(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude()),
            'actual_entry_observation':str(source/'handoff_actual_observation/observation.json')}
        alignment['pass']=alignment['first_approach_position_delta_m']<=.002 and alignment['first_approach_rotation_delta_rad']<=.03
        if not alignment['pass']:raise ValueError(f'Local data starts after leaving the handoff neighborhood: {source}')
    name=object_identity(source)
    display={'rubiks_cube':"rubik's cube",'lemon_01':'lemon'}.get(name,name.replace('_',' '))
    language=instruction or ('pick up the object' if name.startswith('egad_') else 'pick up the '+display)

    if resolved:
        if instruction is not None and instruction != resolved['instruction']:
            raise ValueError('Dataset instruction conflicts with recorded interaction contract')
        language = resolved['instruction']
    episode={'source':str(source),'instruction':language,'object_name':name,
        'selected_grasp_world':grasp,'state_convention':state_convention,'entry_alignment':alignment,
        'chunk_starts':starts,
        'items':items,'rejected':rejected,'physical_verification':verification,
        'source_recording_sha256':recording_sha,
        'interaction_contract_sha256':resolved['contract_sha256'] if resolved else None,
        'episode_sha256':resolved['episode_sha256'] if resolved else None}
    quality={'source':str(source),'object_name':name,'raw_frames':len(rows),'valid_local_transitions':len(items),
        'rejected_local_pairs':len(rejected),'rejection_reasons':{reason:sum(v['reason']==reason for v in rejected) for reason in {v['reason'] for v in rejected}},
        'included_command_start_intervals':sum(item['command_started_during_interval'] for item in items),
        'included_closure_start_intervals':sum(item['command_started_during_interval'] and item['stage']=='close' for item in items),
        'command_start_intervals_by_stage':onsets,
        'excluded_controller_dwell_chunk_starts':len(contiguous)-len(starts),
        'training_chunk_starts_by_stage':{stage:sum(items[i]['stage']==stage for i in starts) for stage in LOCAL_STAGES},
        'unique_frame_keys':len(np.unique(times_raw)),'synchronized_retained_frames':True,
        'finite_states_and_actions':True,'contiguous_eight_action_chunks':len(starts),'entry_alignment':alignment}

    quality.update(pass_check=True, content_sha256=recording_identity(rows), source_recording_sha256=recording_sha)
    return {'episode':episode,'quality':quality,'states':states,'actions':actions}


def main():
    import argparse
    from interaction_contract import write
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-legacy',action='store_true')
    args=parser.parse_args()
    try:
        result=inspect_episode(args.source,require_contract=not args.allow_legacy)
        write(args.output,{'pass':True,**result['quality']})
    except Exception as error:
        write(args.output,{'pass':False,'error_type':type(error).__name__,'error':str(error)})
        raise

if __name__=='__main__':main()
