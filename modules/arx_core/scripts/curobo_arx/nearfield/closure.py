"""Record/idle gate for an explicit GRASP, independent of the teacher index."""
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read,sha,digest
from .runtime import skip_arrival


def observation_timing(observation):
    render=float(observation['frame_time']);physics=float(observation['snapshot']['sim_time'])
    if not np.isfinite([render,physics]).all():raise RuntimeError('Invalid closure observation time')
    capture=observation.get('capture')
    if capture is None:
        if physics!=render:raise RuntimeError('Closure observation/state timestamps disagree')
        return {'basis':'legacy same-clock capture','clock_difference_s':0.}
    from .capture import VERSION,reference_seconds
    if capture['version']!=VERSION:raise RuntimeError('Unknown synchronized capture version')
    if abs(reference_seconds(capture['reference_time'])-render)>1e-9:
        raise RuntimeError('Closure timestamp differs from actual render reference')
    if capture['physics_time_before_render']!=capture['physics_time_after_render']:
        raise RuntimeError('Closure capture advanced physics')
    for t in (capture['physics_time_after_render'],capture['after_render']['sim_time'],capture['pose_sync_state']['sim_time']):
        if not np.isfinite(t) or abs(t-physics)>1e-9:
            raise RuntimeError('Closure capture belongs to a different physics state')
    # The existing camera/recording gate already permits this 100us clock
    # representation tolerance. A 0.2s old frame remains far outside it.
    if abs(physics-render)>1e-4:raise RuntimeError('Closure observation/state timestamps disagree')
    return {'basis':'ReferenceTime and frozen same-tick physics capture',
        'clock_difference_s':physics-render,'existing_capture_tolerance_s':1e-4}


def bind(root, snapshot, observation, command, desired, *, active, fault,
         request_id, configuration_identity, sample_seconds=.2):
    root=Path(root)
    if active or fault:raise RuntimeError('Closure requires confirmed idle healthy controller')
    if command is None:raise RuntimeError('Closure lacks a preceding accepted command')
    age=snapshot['sim_time']-observation['frame_time']
    if age<-.00001 or age>sample_seconds+.01:raise RuntimeError('Closure observation is stale')
    timing=observation_timing(observation)
    if not observation.get('visual_geometry',{}).get('pass'):
        return {'pass':False,'rejected':'current_visual_geometry_rejected'}
    label=command['label'];paths={s:root/(label+s+'.json') for s in ('_command','_execution','_settling')}
    for path in paths.values():
        if not path.is_file():raise RuntimeError('Closure missing prior evidence: '+str(path))
    stored=read(paths['_command'])
    if stored!=command:raise RuntimeError('Closure command differs from actual accepted target')
    requests=read(root/'requests.json')
    if not requests or requests[-1]['id']!=command['id'] or requests[-1]['label']!=label:
        raise RuntimeError('Closure previous command is not the last dispatched request')
    if any(r['label']=='close' for r in requests):raise RuntimeError('Duplicate GRASP/closure')
    execution=read(paths['_execution']);settling=read(paths['_settling'])
    if not settling.get('accepted'):return {'pass':False,'rejected':'preceding_command_not_settled'}
    fresh=skip_arrival(execution,desired,desired,snapshot,command)
    if not fresh['accepted']:return {'pass':False,'rejected':'fresh_arrival_rejected','arrival':fresh}
    if command['joint_names']!=snapshot['right_arm']:raise RuntimeError('Closure joint order changed')
    q=np.asarray(command['position'][-1],dtype=float)
    limits=np.array([snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in snapshot['right_arm']])
    if not np.isfinite(q).all() or np.max(np.maximum(limits[:,0]-q,q-limits[:,1]))>1e-6:
        raise RuntimeError('Closure hold target violates frozen limits')
    return {'pass':True,'grasp_request_id':request_id,'attempt_id':observation['attempt_id'],
        'observation_time_s':observation['frame_time'],'observation_sha256':digest(observation),
        'observation_timing':timing,
        'configuration_identity':configuration_identity,'previous_command_id':command['id'],
        'previous_command_label':label,'hold_joint_names':command['joint_names'],
        'hold_q':q.tolist(),'snapshot_sha256':digest(snapshot),'arrival':fresh,
        'files':{str(p):sha(p) for p in paths.values()},'motion_dispatched':False}


def validate_reply(binding, reply, command):
    if not reply.get('pass'):return False
    if reply.get('binding_sha256')!=digest(binding):raise RuntimeError('Closure reply belongs to another request')
    if reply['hold_q']!=binding['hold_q'] or command['position'][-1]!=binding['hold_q']:
        raise RuntimeError('Closure checked target differs from dispatch target')
    return True
