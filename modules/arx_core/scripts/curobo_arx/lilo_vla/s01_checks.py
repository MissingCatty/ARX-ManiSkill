"""Read-only S01 checks; pure numerical helpers also used by fault-injection tests."""
import copy
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from observations import encode_state, action_target, policy_observation
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from settling import assess_settling


def state_check(observation, prediction, grasp, convention, stats):
    raw=encode_state(observation['snapshot'],convention,grasp['grasp_world'])
    low,high=np.asarray(stats['q01']),np.asarray(stats['q99'])
    normalized=np.clip(np.where(stats.get('mask',[True]*8),2*(raw-low)/(high-low+1e-8)-1,raw),-1,1)
    error=float(np.max(np.abs(raw-prediction['raw_state'])))
    norm_error=float(np.max(np.abs(normalized-prediction['normalized_state'])))
    frame_error=abs(observation['capture_sim_time']-observation['frame_time'])
    associated=Path(prediction.get('source_observation','')).resolve()==Path(observation['rgb_path']).resolve()
    return dict(raw_max_abs_error=error,normalized_max_abs_error=norm_error,
        state_pass=error<=1e-10,normalization_pass=norm_error<=1e-10,
        source_association_pass=associated,frame_state_difference_s=frame_error,
        frame_within_contract=frame_error<=1e-4)


def action_check(prediction, local, state, command, validation, execution, settling):
    index=local['executed_policy_action_index']
    chunk=np.asarray(prediction['actions'])
    checks={'chunk_and_index':chunk.shape==(8,7) and index==0 and local['execute_steps']==1,
        'prediction_unchanged':local['actions']==prediction['actions'],
        'prediction_reference':local['prediction_file']=='policy_0000_prediction.json' and local['policy_call_index']==0}
    if not checks['chunk_and_index']:
        return {'pass':False,'checks':checks}
    target,grip=action_target(state['link_world']['right_eef_link'],chunk[index])
    checks['gripper']=grip==command['gripper_target']
    candidates=[c for c in validation['candidates'] if c.get('command_joint_goal') is not None and not c.get('rejection')]
    checks['joint_command']=any(np.array_equal(c['command_joint_goal'],command['position'][-1]) for c in candidates)
    checks['execution_target']=bool(execution['records']) and all(np.array_equal(r['target'],[np.interp(r['time'],command['time_from_start'],np.asarray(command['position'])[:,i]) for i in range(len(command['joint_names']))]) for r in execution['records'])
    recomputed=assess_settling(execution,target)
    checks['settling_reconstruction']=all(recomputed[k]==v if isinstance(v,bool) else abs(recomputed[k]-v)<=1e-10 for k,v in settling.items() if k in recomputed)
    return {'pass':all(checks.values()),'checks':checks,'derived_world_target':target.tolist(),
        'qualification':'Checks recorded command/IK certificate linkage and measured execution; does not rerun IK or prove model image consumption.'}


def fault_tests(observation,prediction,grasp,convention,stats,local,state,command,validation,execution,settling):
    results={}
    good=state_check(observation,prediction,grasp,convention,stats)
    results['real_state']=all(good[k] for k in ['state_pass','normalization_pass','source_association_pass','frame_within_contract'])
    changed=copy.deepcopy(grasp);changed['grasp_world'][0][3]+=.01
    results['wrong_goal_detected']=not state_check(observation,prediction,changed,convention,stats)['state_pass']
    changed=copy.deepcopy(observation);changed['frame_time']-=1
    results['wrong_frame_detected']=not state_check(changed,prediction,grasp,convention,stats)['frame_within_contract']
    changed=copy.deepcopy(prediction);changed['source_observation']='wrong.png'
    results['wrong_image_association_detected']=not state_check(observation,changed,grasp,convention,stats)['source_association_pass']
    results['real_command']=action_check(prediction,local,state,command,validation,execution,settling)['pass']
    changed=copy.deepcopy(local);changed['executed_policy_action_index']=1
    results['wrong_index_detected']=not action_check(prediction,changed,state,command,validation,execution,settling)['pass']
    changed=copy.deepcopy(command);changed['position'][-1][0]+=.1
    results['wrong_command_detected']=not action_check(prediction,local,state,changed,validation,execution,settling)['pass']
    return results


def model_check(prediction, expected):
    identity=(prediction.get('dataset_key')=='arx_shared_goal_v1' and
              prediction.get('adaptation_selected_step')==225 and
              prediction.get('checkpoint_revision')=='f54c56709004f6680d2ce8bc9e06d8cc96567ad8')
    hashes=prediction.get('policy_artifact_sha256')
    return identity and (hashes is None or all(hashes.get(n)==expected[n] for n in ['action_head.pt','dataset_statistics.json','adapter_manifest.json']))
