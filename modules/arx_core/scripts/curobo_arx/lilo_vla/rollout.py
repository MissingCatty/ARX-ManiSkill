"""Execute real LiLo-VLA local actions through frozen ARX cuRobo kinematics."""
import base64
from collections import deque
import io
import json
import os
import time

import httpx
import numpy as np
from PIL import Image
import torch
from curobo.types import JointState, GoalToolPose
from scipy.spatial.transform import Rotation

from settling import assess_settling
from gpu_policy import verify_health
from .observations import action_target, smooth_segment, policy_observation
from .tracking import residual
from .interaction_contract import runtime as interaction_runtime, validate_resolution, check_health, model_semantics
from .trace import VERSION as TRACE_VERSION, stamp, identity, make_request, verify_response, elapsed_seconds


def run_local(client,out,planner,checker,names,limits,gripper_update,models,
              initial_z,target_path,stage_timing,stage_convergence):
    interaction = interaction_runtime()
    tracing = os.environ.get('ARX_LILO_TRACE') == '1'
    if interaction:validate_resolution(interaction, require_goal=True)
    url=os.environ['ARX_LILO_POLICY_URL'].rstrip('/')
    max_steps=int(os.environ.get('ARX_LILO_MAX_STEPS','60'))
    execute_steps=int(os.environ.get('ARX_LILO_EXECUTE_STEPS','1'))
    if execute_steps not in (1,4,8):
        raise ValueError('ARX_LILO_EXECUTE_STEPS must be 1, 4 or 8')
    max_sim_seconds=float(os.environ.get('ARX_LILO_MAX_SIM_SECONDS','inf'))
    if max_steps<=0 or max_sim_seconds<=0 or np.isnan(max_sim_seconds):
        raise ValueError('Local action/time budgets must be positive')
    instruction=os.environ.get('ARX_LILO_INSTRUCTION','pick up the bbq sauce bottle')
    require_closure=os.environ.get('ARX_LILO_VERIFY_CLOSURE')=='1'
    closure_verified=False
    began=time.monotonic();contacts=[];predictions=[];error=None;hold=None
    pending=deque();action_sources=[];prediction=None;prediction_file=None;policy_requests=0
    state_convention='object_relative_v1'
    pregrasp=json.loads((out/'pregrasp_execution.json').read_text())['records'][-1]
    tracking_bias=np.asarray(pregrasp['target'])-np.asarray(pregrasp['position'])
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
    def local_sim_seconds():
        return sum(value['sim_duration_s'] for label,value in stage_timing.items() if label.startswith('local_'))
    def bad_contacts(events):
        return [e for e in events if e['num_contacts']>0 and not (
            '/World/Robot/' in e['a'] and '/World/Robot/' in e['b']) and not (
            target_path in e['a']+e['b'] and any(f in e['a']+e['b'] for f in ('right_link27','right_link28')))]
    def send(label,q,times,gripper,world_goal):
        command={'joint_names':names,'position':np.asarray(q).tolist(),
            'time_from_start':np.asarray(times).tolist(),'gripper_target':gripper,
            'position_tolerance_rad':.005,'label':label,
            'planner':'LiLo-VLA action + cuRobo IK and dense scene/limit validation'}
        if tracing:
            command['world_goal'] = np.asarray(world_goal).tolist()
            command['trace'] = dict(version=TRACE_VERSION, command_id=identity(),
                request_id=prediction['trace']['request_id'], observation_id=prediction['trace']['observation_id'],
                source=source, command_sent=stamp(), world_goal=command['world_goal'])
        save(label+'_command.json',command)
        result=client.request('trajectory',timeout=300,**command);save(label+'_execution.json',result)
        if tracing:
            trace = result.get('trace',{})
            for key in ('command_id','request_id','observation_id','world_goal'):
                if trace.get(key) != command['trace'][key]:raise ValueError('Execution trace mismatch: '+key)
            capture = prediction['trace']['capture']
            timing = dict(observation_to_arm_seconds=elapsed_seconds(capture['capture_finished'], trace['first_arm_setpoint_applied']),
                observation_to_execution_sim_seconds=trace['first_arm_setpoint_sim_time']-capture['state_sim_time'],
                response_to_command_seconds=elapsed_seconds(prediction['trace']['client_received'], command['trace']['command_sent']),
                command_roundtrip_seconds=elapsed_seconds(command['trace']['command_sent'],stamp()))
            save(label+'_timing.json',timing)
        contacts.extend(result['contacts'])
        assessment=assess_settling(result,world_goal);save(label+'_settling.json',assessment)
        stage_convergence[label]=assessment
        stage_timing[label]={k:result[k] for k in ('wall_duration_s','sim_duration_s')}
        if bad_contacts(result['contacts']):raise RuntimeError(f'Forbidden contact during {label}')
        if not assessment['accepted']:raise RuntimeError(f'Original convergence check rejected {label}')
        return result
    try:
        with httpx.Client(base_url=url,timeout=180,trust_env=False) as http:
            health=http.get('/health');health.raise_for_status();save('lilo_policy.json',health.json())
            verify_health(health.json())
            if interaction:check_health(interaction, health.json())
            if tracing:
                if health.json().get('trace_version') != TRACE_VERSION:raise ValueError('Policy service lacks S02 trace support')
                capabilities = client.request('grasp_lilo_status')
                save('lilo_trace_capabilities.json',capabilities)
                if capabilities.get('trace_version') != TRACE_VERSION or capabilities.get('trajectory_trace_version') != TRACE_VERSION:
                    raise ValueError('Simulator lacks S02 observation/trajectory trace support')
            state_convention=health.json().get('state_convention','object_relative_v1')
            grasp=json.loads((out/'selected_grasp.json').read_text())['grasp_world'] if state_convention!='object_relative_v1' else None
            for step in range(max_steps):
                if local_sim_seconds()>=max_sim_seconds:
                    raise RuntimeError('LiLo policy exhausted its local simulation-time budget')
                if not pending:
                    observation=client.request('grasp_lilo_observation',directory=str(out/'lilo_observations'/f'{step:04d}'), **({'trace':True} if tracing else {}))
                    profile_file=os.environ.get('ARX_LILO_HANDOFF_PROFILE')
                    if interaction:
                        if observation.get('camera_role') != 'right_wrist' or not np.isfinite([observation['capture_sim_time'],observation['frame_time']]).all() or abs(observation['capture_sim_time']-observation['frame_time']) > 1e-4:
                            raise ValueError('Interaction observation camera/time mismatch')
                        if step == 0:
                            entry = json.loads((out/'handoff_actual_observation/observation.json').read_text())
                            a = np.asarray(entry['snapshot']['link_world']['right_eef_link'])
                            b = np.asarray(observation['snapshot']['link_world']['right_eef_link'])
                            position_delta = float(np.linalg.norm(b[:3,3]-a[:3,3]))
                            rotation_delta = float(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude())
                            accepted = position_delta <= .002 and rotation_delta <= .03
                            save('interaction_first_observation.json', dict(pass_check=accepted,
                                position_delta_m=position_delta, rotation_delta_rad=rotation_delta,
                                source_observation=observation['rgb_path'], contract_sha256=interaction['contract_sha256'],
                                qualification='Input consistency only; not learned-policy success qualification'))
                            if not accepted:raise ValueError('First policy observation differs from measured handoff')
                    if step==0 and profile_file:
                        from .handoff_profile import admit,read
                        try:
                            admission=admit(read(profile_file),health.json(),read(out/'scenario.json'),
                                read(out/'selected_grasp.json'),observation,read(out/'handoff_actual_entry.json'),
                                read(out/'setup.json'),instruction)
                        except Exception as rejection:
                            save('handoff_policy_admission.json',{'pass':False,'error':str(rejection),'profile':profile_file})
                            raise
                        save('handoff_policy_admission.json',{**admission,'profile':profile_file})
                    image,boxes,policy_state=policy_observation(observation,state_convention,grasp)
                    buffer=io.BytesIO();Image.fromarray(image).save(buffer,format='PNG')
                    request={'image_png_base64':base64.b64encode(buffer.getvalue()).decode(),
                        'state':policy_state.tolist(),
                        'state_convention':state_convention,'instruction':instruction}
                    if interaction:request['interaction_semantics'] = model_semantics()
                    if tracing:
                        request['trace'] = make_request(observation,buffer.getvalue(),request['state'],instruction,state_convention)
                        (out/f'policy_{len(predictions):04d}_request.png').write_bytes(buffer.getvalue())
                        save(f'policy_{len(predictions):04d}_request.json',{k:v for k,v in request.items() if k!='image_png_base64'})
                    policy_requests+=1
                    response=http.post('/predict',json=request)
                    received=stamp() if tracing else None
                    if tracing:
                        save(f'policy_{len(predictions):04d}_http.json',dict(status_code=response.status_code,
                            received=received,body=response.text))
                    response.raise_for_status();prediction=response.json()
                    if tracing:
                        # Preserve a rejected server response as failure evidence, before validation.
                        save(f'policy_{len(predictions):04d}_response.json',prediction)
                        verify_response(request['trace'],prediction.get('trace',{}))
                        prediction['trace']['client_received'] = received
                    prediction['distractor_boxes']=boxes;prediction['source_observation']=observation['rgb_path']
                    prediction['policy_call_index']=len(predictions)
                    prediction_file=f'policy_{len(predictions):04d}_prediction.json'
                    save(prediction_file,prediction);predictions.append(prediction)
                    actions=np.asarray(prediction['actions'],dtype=float)
                    if actions.shape!=(8,7) or not np.isfinite(actions).all():
                        raise ValueError('Expected a finite 8 x 7 LiLo action chunk')
                    pending.extend(enumerate(actions[:execute_steps]))
                    snap=observation['snapshot']
                else:
                    # Reuse the predicted action sequence, never its old robot
                    # state: each delta is composed with the latest measured EEF.
                    snap=client.request('snapshot')
                action_index,action=pending.popleft()
                label=f'local_{step:04d}'
                source={'step':step,'policy_call_index':prediction['policy_call_index'],
                    'prediction_file':prediction_file,'executed_policy_action_index':action_index,
                    'returned_policy_chunk_shape':[8,7],'execute_steps':execute_steps}
                action_sources.append(source)
                if tracing:source.update(observation_id=prediction['trace']['observation_id'],request_id=prediction['trace']['request_id'])
                save(label+'_prediction.json',{**prediction,**source})
                save(label+'_state.json',snap)
                ee=np.asarray(snap['link_world']['right_eef_link'])
                world_goal,gripper=action_target(ee,action)
                if tracing:
                    save(label+'_action_target.json',dict(**source, measured_ee_world=ee.tolist(),
                        world_goal=world_goal.tolist(), gripper_target=gripper, action=action.tolist(), composed=stamp()))
                if require_closure and closure_verified and gripper<0:
                    raise RuntimeError('Policy reopened the gripper after verified closure')
                positions=dict(zip(snap['joint_names'],snap['joint_positions']))
                # State update only: unchanged frozen finger geometry, transforms and limits.
                gripper_update.apply(models,positions)
                current=torch.tensor([positions[n] for n in names],device='cuda',dtype=torch.float32)
                invbase=np.linalg.inv(np.asarray(snap['link_world']['body4']));goal=invbase@world_goal
                quat=Rotation.from_matrix(goal[:3,:3]).as_quat();quat=np.r_[quat[3],quat[:3]]
                target=GoalToolPose(tool_frames=['right_eef_link'],
                    position=torch.tensor(goal[:3,3],device='cuda',dtype=torch.float32).reshape(1,1,1,1,3),
                    quaternion=torch.tensor(quat,device='cuda',dtype=torch.float32).reshape(1,1,1,1,4))
                ik=planner.ik_solver.solve_pose(target,current_state=JointState.from_position(current[None],joint_names=names),return_seeds=4)
                if not ik.success.any():raise RuntimeError(f'ARX IK rejected LiLo action {step}')
                candidates=ik.solution[ik.success].reshape(-1,len(names))
                candidates=candidates[checker.validate(candidates[:,None]).flatten()]
                if not len(candidates):raise RuntimeError(f'cuRobo collision checking rejected LiLo action {step}')
                bounds=limits.cpu().numpy()
                diagnostics=[];accepted=None
                for candidate in candidates[(candidates-current).square().sum(-1).argsort()]:
                    nominal=candidate.cpu().numpy()
                    pose_error=residual(checker.kinematics,nominal,goal,names)
                    detail={'nominal_ik_error':pose_error,'nominal_joint_goal':nominal.tolist(),
                        'measured_joint_tracking_bias_rad':tracking_bias.tolist()}
                    diagnostics.append(detail)
                    if pose_error['position_error_m']>.002 or pose_error['orientation_error_rad']>.03:
                        detail['rejection']='original IK precision check';continue
                    # Feed forward the preceding measured target-minus-actual bias.
                    # These are motion targets; robot drives/gains remain unchanged.
                    joint_goal=nominal+tracking_bias
                    q,times=smooth_segment(current.cpu().numpy(),joint_goal)
                    if local_sim_seconds()+float(times[-1])>max_sim_seconds:
                        raise RuntimeError('Next LiLo action would exceed the local simulation-time budget')
                    if ((q<bounds[:,0]-1e-6)|(q>bounds[:,1]+1e-6)).any():
                        detail['rejection']='frozen position limits';continue
                    alpha=np.arange(10)/10
                    dense=np.concatenate([(q[:-1,None]*(1-alpha[None,:,None])+q[1:,None]*alpha[None,:,None]).reshape(-1,len(names)),q[-1:]])
                    tensor=torch.tensor(dense,device='cuda',dtype=torch.float32)
                    valid=torch.cat([checker.validate(chunk[:,None]).flatten() for chunk in tensor.split(64)])
                    invalid=(~valid).nonzero().flatten().cpu().tolist()
                    detail.update(dense_samples=len(dense),invalid_samples=len(invalid),invalid_indices=invalid,
                        command_joint_goal=joint_goal.tolist())
                    if invalid:
                        detail['rejection']='dense frozen-scene collision check';continue
                    accepted=(q,times);break
                save(label+'_validation.json',{'candidates':diagnostics,
                    'accepted':accepted is not None,
                    'position_limits_unchanged':True,'velocity_bound':1.5,'acceleration_bound':5.,'jerk_bound':100.,
                    **source})
                if accepted is None:raise RuntimeError(f'No precise collision-free ARX command for LiLo action {step}')
                q,times=accepted
                result=send(label,q,times,gripper,world_goal)
                if require_closure and not closure_verified and gripper>0:
                    # Execute the model's closing decision with the original
                    # three-second closure dwell and bilateral-contact gate.
                    closing=q[-1].copy()
                    save('close_policy_source.json',{'prediction':label+'_prediction.json',
                        'source_action_index':action_index,'arm_target_source':label+'_command.json'})
                    result=send('close',np.stack([closing,closing]),[0.,3.],gripper,world_goal)
                    closing_fingers={f for e in result['contacts'] if e['num_contacts']>0
                        and target_path in e['a']+e['b'] for f in ('right_link27','right_link28')
                        if f in e['a']+e['b']}
                    if len(closing_fingers)!=2:
                        raise RuntimeError('No bilateral target contact after original 3s closure; lift not attempted')
                    closure_verified=True
                last=result['records'][-1]
                tracking_bias=np.asarray(last['target'])-np.asarray(last['position'])
                final=result['final'];height=final['objects']['test_object']['world'][2][3]
                fingers={f for e in contacts if e['num_contacts']>0 and target_path in e['a']+e['b'] for f in ('right_link27','right_link28') if f in e['a']+e['b']}
                print('LILO_STEP',step,'rise_m',height-initial_z,'bilateral',len(fingers)==2,flush=True)
                # The control target includes 1 cm clearance above the unchanged
                # >6 cm physical success gate. Preserve the existing actuator
                # setpoint during hold: replacing it with measured q reintroduces
                # the native load-tracking offset.
                if height>initial_z+.07 and len(fingers)==2:
                    hold_target=q[-1].copy()
                    save('hold_control.json',{'source_policy_command':label,
                        'joint_target':hold_target.tolist(),'lift_control_target_m':.07,
                        'physical_success_minimum_rise_m':.06,
                        'method':'Preserve the preceding collision-checked motion target; unchanged robot drives'})
                    hold=send('hold',np.stack([hold_target,hold_target]),[0.,3.],gripper,world_goal)
                    break
        if hold is None:raise RuntimeError('LiLo policy exhausted its action budget without a verified grasp')
    except Exception as exc:
        error=str(exc);print('LILO_FAILED',error,flush=True)
    heights=[r['objects']['test_object'][2][3] for r in hold['records']] if hold else [initial_z]
    held=min(heights)>initial_z+.06
    result={'controller':'lilo_vla','success':bool(error is None and held and predictions),
        'status':'completed' if error is None and held else 'failed','error':error,
        'object_held':held,'initial_object_z_m':initial_z,'held_object_min_z_m':float(min(heights)),
        'held_object_final_z_m':float(heights[-1]),'lift_m':float(heights[-1]-initial_z),
        'stage_timing':stage_timing,'stage_convergence':stage_convergence,
        'local_policy_calls':len(predictions),'local_policy_requests':policy_requests,
        'local_action_count':sum(k.startswith('local_') for k in stage_timing),
        'execute_steps':execute_steps,'maximum_local_actions':max_steps,
        'maximum_local_sim_seconds':max_sim_seconds if np.isfinite(max_sim_seconds) else None,
        'local_sim_seconds':local_sim_seconds(),'local_action_attempts':len(action_sources),
        'unconsumed_chunk_actions':len(pending),'action_sources':action_sources,
        'instruction':instruction,'original_closure_gate_required':require_closure,
        'state_convention':state_convention,
        'original_closure_gate_passed':closure_verified if require_closure else None,
        'local_policy_wall_seconds':time.monotonic()-began,
        'success_criteria':'Original >6cm lift, >=3s physical hold, bilateral finger contacts, unchanged convergence/contact/limit checks'}
    save('lilo_contacts.json',contacts)
    return result
