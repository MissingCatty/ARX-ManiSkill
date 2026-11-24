"""Build local ARX interaction chunks only from physically verified teacher runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from observations import relative_state, encode_state, STATE_CONVENTIONS, teacher_action, ACTION_PERIOD_S, policy_observation
from scipy.spatial.transform import Rotation
from interaction_contract import validate_episode, STATE


from episode_quality import LOCAL_STAGES, transition_label, inspect_episode

def build(sources, output, validation_sources=(), *, state_convention=None,
          dataset_key='arx_bottle_local', instruction=None, object_split=False, require_entry=False,
          include_command_starts=False, exclude_gated_dwell_starts=False, interaction_contract=None):
    if not sources:raise ValueError('At least one physical training episode is required')
    combined=[*sources,*validation_sources]
    if len(set(combined))!=len(combined):
        raise ValueError('Episodes must be unique and training/validation sources must be disjoint')
    recorded_contracts = any((source/'resolved_interaction.json').exists() for source in combined)
    resolved_sources = {}
    if interaction_contract is not None or recorded_contracts:
        if state_convention is not None and state_convention != STATE:
            raise ValueError('State convention conflicts with recorded interaction contract')
        for source in combined:
            path = source/'resolved_interaction.json'
            if not path.exists():
                raise ValueError('Legacy and contracted episodes cannot be mixed or relabeled: '+str(source))
            source_contract = interaction_contract if interaction_contract is not None else json.loads(path.read_text())['contract']
            resolved_sources[source] = validate_episode(source, source_contract)
            if instruction is not None and instruction != resolved_sources[source]['instruction']:
                raise ValueError('Dataset instruction conflicts with recorded interaction contract')
        state_convention = STATE
        include_command_starts = exclude_gated_dwell_starts = object_split = require_entry = True
    state_convention = state_convention or 'object_relative_v1'
    if exclude_gated_dwell_starts and not include_command_starts:
        raise ValueError('Keeping the actual closure-start observation is required before excluding controller dwell starts')
    output.mkdir(parents=True,exist_ok=False)
    episodes=[];all_states=[];all_actions=[]
    training_sources=set(sources)
    identities={}
    for source in combined:
        names=[]
        for filename,key in (('scenario.json','asset'),('active_asset.json','name')):
            path=source/filename
            if path.exists():names.append(json.loads(path.read_text())[key])
        if len(set(names))>1:raise ValueError(f'Conflicting object identity: {source}')
        if object_split and not names:raise ValueError(f'Object identity missing for object-disjoint split: {source}')
        identities[source]=names[0] if names else 'bbq_sauce_bottle'
    train_objects={identities[p] for p in sources};validation_objects={identities[p] for p in validation_sources}
    if object_split and train_objects&validation_objects:
        raise ValueError('Object leakage between training and validation: '+str(sorted(train_objects&validation_objects)))
    quality=[];seen_recordings=set()
    for number,source in enumerate([*sources,*validation_sources]):
        folder=output/f'episode_{number:03d}';folder.mkdir()
        checked=inspect_episode(source,state_convention=state_convention,require_entry=require_entry,
            include_command_starts=include_command_starts,exclude_gated_dwell_starts=exclude_gated_dwell_starts,
            instruction=instruction,image_sink=lambda name,image: Image.fromarray(image).save(folder/name))
        episode=checked['episode'];quality_row=checked['quality']
        for identity in (episode['source_recording_sha256'],quality_row['content_sha256']):
            if identity in seen_recordings:raise ValueError(f'Duplicate physical recording under different episode paths: {source}')
            seen_recordings.add(identity)
        states,actions=checked['states'],checked['actions']
        np.savez_compressed(folder/'trajectory.npz',state=states,action=actions)
        if source in training_sources:
            all_states.extend(states);all_actions.extend(actions)
        episode.update(folder=folder.name,split='train' if source in training_sources else 'validation')
        for item in episode['items']:item['image']=folder.name+'/'+item['image']
        episodes.append(episode);quality.append(quality_row)
    def stats(values):
        a=np.asarray(values)
        return {k:getattr(a,k)(axis=0).tolist() for k in ('mean','std','min','max')}|{
            'q01':np.quantile(a,.01,axis=0).tolist(),'q99':np.quantile(a,.99,axis=0).tolist()}
    action_stats=stats(all_actions);action_stats['mask']=[True]*6+[False]
    state_stats=stats(all_states)
    (output/'dataset_statistics.json').write_text(json.dumps({dataset_key:{
        'action':action_stats,'proprio':state_stats,'num_transitions':len(all_actions),
        'num_trajectories':len(sources)}},indent=2)+'\n')
    (output/'dataset.json').write_text(json.dumps({'episodes':episodes,'camera':'right_wrist RGB only',
        'state_convention':state_convention,'dataset_key':dataset_key,
        'interaction_contracts':{v['contract_sha256']:v['contract'] for v in resolved_sources.values()},
        'action_period_seconds':ACTION_PERIOD_S,'action_chunk_size':8,
        'action_convention':'world delta position / 0.05m, left delta rotation / 0.5rad, 1=open',
        'action_label_source':('Measured EE displacement over synchronized frame interval; command starting during that interval, otherwise current command' if include_command_starts else 'Measured EE displacement between consecutive synchronized frames; gripper command at the first frame'),
        'action_alignment_version':'command_interval_v2' if include_command_starts else 'frame_command_v1',
        'command_onset_precision':'Within the actual 0.2s recording interval; no exact control-start timestamp claim' if include_command_starts else None,
        'exclude_controller_closure_dwell_and_hold_chunk_starts':exclude_gated_dwell_starts,
        'object_disjoint_split_required':object_split,'train_objects':sorted(train_objects),'validation_objects':sorted(validation_objects),
        'scope':('Shared-entry ARX multishape head-only pilot, object-disjoint validation; not a full EGAD49 evaluation.' if object_split else 'ARX local interaction pilot, split by episode; no cross-object generalization claim.')},indent=2)+'\n')
    (output/'quality_report.json').write_text(json.dumps({'grain':'One source episode / one synchronized wrist frame / one next-frame action',
        'primary_key':['source_episode','frame_time'],'episodes':quality,'pass':True,
        'train_only_normalization_statistics':True,'object_disjoint':not bool(train_objects&validation_objects),
        'failures_are_not_training_demonstrations':True,'physical_attempt_denominator':'Stored in collection campaign, including failed attempts'},indent=2)+'\n')
    print(json.dumps({'episodes':len(episodes),'valid_transitions':len(all_actions),'output':str(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--validation-source',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--state-convention',choices=STATE_CONVENTIONS)
    parser.add_argument('--interaction-contract',type=Path,help='Require this recorded contract; contracted data is otherwise auto-detected and never mixed with legacy recordings')
    parser.add_argument('--dataset-key',default='arx_bottle_local')
    parser.add_argument('--instruction')
    parser.add_argument('--object-split',action='store_true')
    parser.add_argument('--require-entry',action='store_true')
    parser.add_argument('--include-command-starts',action='store_true')
    parser.add_argument('--exclude-gated-dwell-starts',action='store_true')
    args=parser.parse_args();build([p.resolve() for p in args.source],args.output.resolve(),
                                 [p.resolve() for p in args.validation_source],state_convention=args.state_convention,
                                 dataset_key=args.dataset_key,instruction=args.instruction,
                                 object_split=args.object_split,require_entry=args.require_entry,
                                 include_command_starts=args.include_command_starts,
                                 exclude_gated_dwell_starts=args.exclude_gated_dwell_starts,interaction_contract=args.interaction_contract)
