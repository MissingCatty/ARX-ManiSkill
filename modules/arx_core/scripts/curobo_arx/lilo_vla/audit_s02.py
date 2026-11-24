"""Audit actual S02 artifacts. Pixel replay and motion replay use their original environments."""
import argparse
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from interaction_contract import read, write
from lilo_vla.trace import make_request, verify_response, sha, rgb_identity, tensor_identity, elapsed_seconds
from observations import policy_observation, action_target
from s01_checks import state_check
from s02_gates import absence, audit_receipt, event_inventory


def audit(directory,pixels=False):
    directory=Path(directory);errors=[];rows=[]
    inventory=event_inventory(directory)
    if inventory['errors']:
        return audit_receipt(inventory['errors'],dict(command_execution='applicable' if inventory['post_handoff_event'] else 'unknown'),event_inventory=inventory,mode='pixel_replay' if pixels else 'motion_replay')
    predictions=sorted(directory.glob('policy_*_prediction.json'))
    if not predictions:
        justified=absence(directory,'policy_input')
        events={k:'not_applicable' if justified else 'unknown' for k in ('observation','policy_input','command_execution')}
        if not justified:errors.append('Missing prediction without a verified pre-policy rejection')
        observation_file=directory/'handoff_actual_observation/observation.json'
        if (directory/'handoff_actual_entry.json').exists() and not observation_file.exists():errors.append('Handoff certificate lacks its observation record')
        if observation_file.exists():
            events['observation']='applicable'
            try:
                observation=read(observation_file)
                make_request(observation,Path(observation['rgb_path']).read_bytes(),[0.]*8,'offline capture identity check','grasp_goal_world_error_v1')
                if abs(observation['capture_sim_time']-observation['frame_time'])>1e-4:raise ValueError('Frame/state mismatch')
            except Exception as exc:errors.append(str(exc))
        return audit_receipt(errors,events,predictions=0,local_commands=0,mode='pixels' if pixels else 'motion')
    goal=read(directory/'selected_grasp.json')
    resolution=read(directory/'resolved_interaction.json')
    adapter=resolution['model_compatibility']
    if pixels:
        import torch
        sys.path.insert(0,str(ROOT/'servers/lilo_vla'))
        from backend import preprocess
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
        processor=PrismaticImageProcessor.from_pretrained(ROOT.parent/'lilo-vla/checkpoints/lilo-vla-openvla-oft',local_files_only=True)
    else:
        from settling import assess_settling
        statistics=read(ROOT/'runs/lilo_P2_comparison_20261028_02/goal_adapter/dataset_statistics.json')['arx_shared_goal_v1']['proprio']
    by_name={}
    for path in predictions:
        prefix=path.name.removesuffix('_prediction.json');prediction=read(path);by_name[path.name]=prediction
        try:
            observation=read(Path(prediction['source_observation']).parent/'observation.json')
            request=read(directory/(prefix+'_request.json'));png=(directory/(prefix+'_request.png')).read_bytes()
            # Re-reads capture bytes/state and checks measured velocities. Its new
            # request ID is discarded; only persisted IDs may link the response.
            made=make_request(observation,png,request['state'],request['instruction'],request['state_convention'])
            if made['request_sha256']!=request['trace']['request_sha256']:raise ValueError('Saved request contents changed')
            if made['capture']!=request['trace']['capture'] or made['observation_id']!=request['trace']['observation_id']:
                raise ValueError('Saved observation/request association differs')
            if request['instruction']!=resolution['instruction'] or request['state_convention']!=resolution['contract']['state_convention']:
                raise ValueError('Request instruction or convention differs from registration')
            if request['state']!=prediction['raw_state']:raise ValueError('Request and inferred state differ')
            verify_response(request['trace'],prediction['trace'])
            if prediction['policy_artifact_sha256']!=adapter['policy_artifact_sha256']:raise ValueError('Loaded policy artifacts changed')
            if pixels:
                image,_,_=policy_observation(observation,request['state_convention'],goal['grasp_world'])
                if rgb_identity(Image.fromarray(image))!=request['trace']['request_rgb']:raise ValueError('Mask/request pixels differ')
                processed=preprocess(Image.open(io.BytesIO(png)).convert('RGB'))
                if rgb_identity(processed)!=prediction['trace']['preprocessed_rgb']:raise ValueError('Model preprocessing differs')
                tensor=processor(processed,return_tensors='pt')['pixel_values'].to(torch.bfloat16)
                if tensor_identity(tensor)!=prediction['trace']['model_inputs']['pixel_values']:raise ValueError('Model forward pixels differ')
            else:
                checked=state_check(observation,prediction,goal,request['state_convention'],statistics)
                if not all(checked[k] for k in ('state_pass','normalization_pass','source_association_pass','frame_within_contract')):
                    raise ValueError('State/normalization/frame mismatch: '+str(checked))
                for a,b in zip(('client_send','server_received','preprocess_start','forward_start','forward_end','server_finished'),
                               ('server_received','preprocess_start','forward_start','forward_end','server_finished','client_received')):
                    if elapsed_seconds(prediction['trace'][a],prediction['trace'][b]) is None:raise ValueError('Cross-host clock domain is not comparable')
            rows.append(dict(prediction=path.name,pass_check=True))
        except Exception as exc:errors.append(path.name+': '+str(exc))
    commands=sorted(directory.glob('local_*_command.json'))
    if not pixels:
        for path in commands+[p for p in (directory/'close_command.json',directory/'hold_command.json') if p.exists()]:
            try:
                label=path.name.removesuffix('_command.json');command=read(path);source=command['trace']['source']
                prediction=by_name[source['prediction_file']];execution=read(directory/(label+'_execution.json'))
                origin='local_%04d'%source['step'];state=read(directory/(origin+'_state.json'))
                if source['executed_policy_action_index']!=0 or source['execute_steps']!=1:raise ValueError('S02 requires K1 first action')
                target,gripper=action_target(state['link_world']['right_eef_link'],prediction['actions'][0])
                if np.max(np.abs(target-np.array(command['world_goal'])))>1e-10 or gripper!=command['gripper_target']:
                    raise ValueError('Direct world/gripper target differs from policy action')
                for key in ('request_id','observation_id','command_id','world_goal'):
                    if command['trace'][key]!=execution['trace'][key]:raise ValueError('Execution association differs: '+key)
                for key in ('request_id','observation_id'):
                    if command['trace'][key]!=prediction['trace'][key]:raise ValueError('Wrong source prediction: '+key)
                q=np.array(command['position'])
                if not execution['records'] or any(not np.array_equal(r['target'],[np.interp(r['time'],command['time_from_start'],q[:,i]) for i in range(q.shape[1])]) for r in execution['records']):
                    raise ValueError('Recorded joint trajectory differs')
                # Recompute even for a legitimately rejected convergence result.
                actual=assess_settling(execution,target);saved=read(directory/(label+'_settling.json'))
                if actual!=saved:raise ValueError('Settling result differs')
                latency=elapsed_seconds(prediction['trace']['capture']['capture_finished'],execution['trace']['first_arm_setpoint_applied'])
                if latency is None:raise ValueError('Observation/execution clock domain differs')
            except Exception as exc:errors.append(path.name+': '+str(exc))
        if not commands and not absence(directory,'command_execution'):errors.append('Missing command without verified pre-command rejection')
    events=dict(observation='applicable',policy_input='applicable',command_execution='applicable' if commands and not pixels else 'not_applicable' if absence(directory,'command_execution') else 'unknown')
    return audit_receipt(errors,events,predictions=len(predictions),local_commands=len(commands),
                checks=rows,mode='pixel_replay' if pixels else 'motion_replay',
                qualification='Artifact consistency only; task success remains the original physical evaluator')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--pixels',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:result=audit(args.trial,args.pixels)
    except Exception as exc:result=audit_receipt([str(exc)],{},mode='pixel_replay' if args.pixels else 'motion_replay')
    write(args.output,result)
    raise SystemExit(0 if result['pass_check'] else 1)


if __name__=='__main__':main()
