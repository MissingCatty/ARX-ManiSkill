"""Read the existing calibrated wrist sensor and record real ARX teacher episodes."""
import json
from pathlib import Path
import time
import sys

import numpy as np


def observe(bridge, directory, trace=False):
    from PIL import Image
    from pxr import UsdGeom
    from scipy.spatial.transform import Rotation
    if trace:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts/curobo_arx'))
        from lilo_vla.trace import VERSION, stamp, identity, sha, canonical, rgb_identity
        capture_started = stamp()

    c = bridge.context
    camera = c.cameras.get('right_wrist')
    if camera is None or camera.frame_time is None:
        raise RuntimeError('Enable grasp_feedback_camera and wait for a valid wrist frame')
    rgb, depth = camera.get_rgba(), camera.get_depth()
    if rgb is None or depth is None or not np.any(rgb):
        raise RuntimeError('ARX wrist RGB-D is not ready')
    snapshot = bridge.snapshot()
    if trace:capture_finished = stamp()
    if abs(snapshot['sim_time']-camera.frame_time) > .21:
        raise RuntimeError('Wrist frame is stale; do not pair it with a newer action')
    ee = np.asarray(snapshot['link_world']['right_eef_link'])
    target = np.asarray(snapshot['objects']['test_object']['world'])
    names = snapshot['joint_names']; q = snapshot['joint_positions']
    state = np.r_[ee[:3,3]-target[:3,3],
        Rotation.from_matrix(target[:3,:3].T@ee[:3,:3]).as_rotvec(),
        [q[names.index(n)] for n in ('right_joint27','right_joint28')]]
    view = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(c.stage.GetPrimAtPath(camera.prim_path))).T
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(directory/'wrist.png')
    np.save(directory/'depth_m.npy', np.asarray(depth,dtype=np.float32))
    data = {'snapshot':snapshot, 'state':state.tolist(), 'camera_role':'right_wrist',
        'frame_time':camera.frame_time, 'capture_sim_time':snapshot['sim_time'],
        'camera_world_optical':(view@np.diag([1,-1,-1,1])).tolist(),
        'intrinsics':camera.get_intrinsics_matrix().tolist(),
        'rgb_path':str(directory/'wrist.png'), 'depth_path':str(directory/'depth_m.npy'),
        'state_convention':'LiLo release: world-position difference, object-relative axis-angle, two measured finger positions',
        'wall_time':time.time()}
    if trace:
        with Image.open(directory/'wrist.png') as saved:
            pixels = rgb_identity(saved)
        data['trace'] = dict(version=VERSION, observation_id=identity(),
            capture_started=capture_started, capture_finished=capture_finished,
            png_sha256=sha((directory/'wrist.png').read_bytes()), rgb=pixels,
            snapshot_sha256=sha(canonical(snapshot)), frame_sim_time=camera.frame_time,
            state_sim_time=snapshot['sim_time'], saved=stamp())
    (directory/'observation.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


def sample(bridge):
    recording = getattr(bridge,'lilo_recording',None)
    if recording is None or recording.get('error'):
        return
    camera = bridge.context.cameras.get('right_wrist')
    if camera is None or camera.frame_time is None or camera.frame_time == recording.get('last_frame'):
        return
    data = observe(bridge, recording['directory']/f"{recording['samples']:06d}")
    active = bridge.active
    data['stage'] = active['request'].get('label') if active else None
    data['teacher_command'] = active['request'] if active else None
    with (recording['directory']/'samples.jsonl').open('a') as stream:
        stream.write(json.dumps(data)+'\n')
    recording['samples'] += 1
    recording['last_frame'] = camera.frame_time


def handle(bridge, request):
    action = request['action']
    if action == 'grasp_lilo_status':
        return {'available':True,'camera_role':'right_wrist',
                'trace_version':'arx_policy_trace_s02_v1',
                'trajectory_trace_version':getattr(bridge,'trajectory_trace_version',None),
                'recording':getattr(bridge,'lilo_recording',None) is not None}
    if action == 'grasp_lilo_observation':
        return observe(bridge, request['directory'], trace=request.get('trace',False))
    if action == 'grasp_lilo_record_start':
        if getattr(bridge,'lilo_recording',None) is not None:
            raise RuntimeError('Teacher recording already active')
        directory = Path(request['directory']); directory.mkdir(parents=True,exist_ok=False)
        bridge.lilo_recording = {'directory':directory,'samples':0,'last_frame':None}
        if not getattr(bridge,'lilo_hook_installed',False):
            original = bridge.before_step
            def before_step(poll_requests=True):
                if poll_requests:
                    try:
                        sample(bridge)
                    except Exception as exc:
                        recording = bridge.lilo_recording
                        if recording is not None:
                            recording['error'] = str(exc)
                            (recording['directory']/'recording_error.json').write_text(json.dumps({'error':str(exc)}))
                return original(poll_requests)
            bridge.before_step = before_step
            bridge.lilo_hook_installed = True
        return {'recording':True,'directory':str(directory)}
    if action == 'grasp_lilo_record_stop':
        recording = getattr(bridge,'lilo_recording',None)
        if recording is None:
            raise RuntimeError('No teacher recording is active')
        sample(bridge)
        result = {'samples':recording['samples'],'directory':str(recording['directory']),
                  'error':recording.get('error'),
                  'source':'Actual ARX wrist frames and measured state; no synthesized frames/actions'}
        (recording['directory']/'recording.json').write_text(json.dumps(result,indent=2)+'\n')
        bridge.lilo_recording = None
        return result
    raise ValueError(f'Unknown LiLo observation action: {action}')
