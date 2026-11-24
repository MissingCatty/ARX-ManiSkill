"""Restore the historical successful Rubik's-cube task, then export ARX models."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/curobo_arx'))
from client import Client
from build_model import build

OUT = ROOT/'runs/arx_dynamic_pick'
SOURCE = ROOT/'runs/weight_comparison_20261015/01_rubiks_cube_w1/01_rubiks_cube'


def main():
    OUT.mkdir(exist_ok=True)
    client = Client(OUT/'mailbox')
    deadline = time.monotonic()+1200
    while not (client.root/'ready.json').exists():
        if time.monotonic()>deadline: raise TimeoutError('Dynamic ARX startup timeout')
        time.sleep(.5)
    asset = json.loads((SOURCE/'active_asset.json').read_text())
    original = json.loads((SOURCE/'capture_snapshot.json').read_text())
    scenario = {'target':'test_object','asset':asset['name'],
        'placement':{'position':[4.048,2.08,.679+asset['size_m'][2]/2+.012]}}
    (OUT/'task_contract.json').write_text(json.dumps({
        'user_request':'ARX living-room grasp with constant-velocity crossing obstacle, online cuRobo, original-view video',
        'main_autoresearch':'paused_by_user; checkpoint retained; no main experiments',
        'new_scene_intervention_authorized':'One NEW kinematic collision cuboid moving at fixed world velocity',
        'robot_configuration':'unchanged; original USD/URDF, collision spheres, limits, drives, gains, gripper, calibration',
        'original_environment_physics':'unchanged; original room/table/native task initial setup',
        'grasp':'archived successful ZeroGrasp Rubik cube selection; no new NN success claim',
        'source':str(SOURCE),'scenario':scenario,
        'recording':'Original user WebRTC viewpoint, no titles/text/audio',
        'validation':'Actual Isaac execution, dynamic contacts and measured target hold; planning-only evidence reported separately',
        'storage_reserve_bytes':1073741824,'started_unix':time.time(),
    },indent=2))
    result = client.request('dynamic_setup',scenario=scenario,timeout=180)
    # Preserve full scene/model properties BEFORE adding the explicitly authorized obstacle.
    (OUT/'initial_setup.json').write_text(json.dumps(result,indent=2))
    snap = result['snapshot']
    base_err = np.max(np.abs(np.array(snap['link_world']['body4'])-original['link_world']['body4']))
    target_err = np.linalg.norm(np.array(snap['objects']['test_object']['world'])[:3,3]-np.array(original['objects']['test_object']['world'])[:3,3])
    assert base_err<.001 and target_err<.003, (base_err,target_err)
    view = json.loads((ROOT/'runs/arx_recovery_20261021_2359/restored_viewport.json').read_text())
    client.request('grasp_viewport_state',restore=view)
    (OUT/'viewport.json').write_text(json.dumps(client.request('grasp_viewport_state'),indent=2))
    (OUT/'snapshot.json').write_text(json.dumps(snap,indent=2))
    (OUT/'selected_grasp.json').write_text((SOURCE/'selected_grasp.json').read_text())
    robot = ROOT.parent/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd'
    build(robot,snap,OUT/'model',fine_gripper=True)
    nominal = copy.deepcopy(snap)
    for name in ('right_joint27','right_joint28'):
        nominal['joint_positions'][nominal['joint_names'].index(name)]=.022
    build(robot,nominal,OUT/'model_closed',fine_gripper=True,gripper_pitch=.004)
    client.request('export_scene',timeout=180)
    exported = json.loads((client.root/'scene_meshes.json').read_text())
    meshes = OUT/'meshes'; meshes.mkdir(exist_ok=True)
    scene = {'mesh':{}}; sources={}
    target = snap['objects']['test_object']['path']
    for i,m in enumerate(exported['meshes']):
        if m['path'].startswith('/World/Robot/') or m['path']==target or m['path'].startswith(target+'/'):
            continue
        path = meshes/f'{i}.obj'
        trimesh.Trimesh(vertices=m['vertices'],faces=m['faces'],process=False).export(path)
        key=f'collider_{i}'; sources[key]=m['path']
        scene['mesh'][key]={'file_path':str(path),'pose':[0,0,0,1,0,0,0]}
    (OUT/'scene.yml').write_text(yaml.safe_dump(scene))
    (OUT/'scene_sources.json').write_text(json.dumps(sources,indent=2))
    from curobo.kinematics import Kinematics,KinematicsCfg
    from curobo.types import JointState
    import torch
    from scipy.spatial.transform import Rotation
    kin=Kinematics(KinematicsCfg.from_robot_yaml_file(str(OUT/'model/arx.yml')))
    q=torch.tensor([[snap['joint_positions'][snap['joint_names'].index(n)] for n in kin.joint_names]],device='cuda')
    state=kin.compute_kinematics(JointState.from_position(q,joint_names=kin.joint_names))
    observed=np.linalg.inv(snap['link_world']['body4'])@np.array(snap['link_world']['right_eef_link'])
    poserr=float(np.linalg.norm(state.tool_poses.position.cpu().numpy().reshape(3)-observed[:3,3]))
    quat=state.tool_poses.quaternion.cpu().numpy().reshape(4)
    roterr=float(Rotation.from_matrix(Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_matrix()@observed[:3,:3].T).magnitude())
    assert poserr<.002 and roterr<.01
    report={'ready':True,'base_matrix_error':float(base_err),'target_pose_error_m':float(target_err),
        'fk_position_error_m':poserr,'fk_rotation_error_rad':roterr,'scene_colliders':len(sources),
        'robot_usd_sha256':hashlib.sha256(robot.read_bytes()).hexdigest(),'configuration_changed':False}
    (OUT/'setup_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':main()
