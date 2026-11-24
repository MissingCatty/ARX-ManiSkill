"""Export current ARX scene, derive its cuRobo model and verify measured FK."""
import argparse,json,sys,os
from gpu_policy import COMPUTE_GPU,compute_env

os.environ.update(compute_env(os.environ,os.environ.get('CUDA_VISIBLE_DEVICES',os.environ.get('ARX_PLANNER_GPU',COMPUTE_GPU))))
from pathlib import Path
import numpy as np
import torch,trimesh,yaml
from client import Client
from build_model import build
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=root/'runs/arx_curobo');parser.add_argument('--reuse-scene',action='store_true');parser.add_argument('--fine-gripper',action='store_true');parser.add_argument('--model-source',type=Path,help='Reuse frozen model/model_closed geometry from an existing ARX archive');args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);client=Client(root/'runs/arx_curobo/mailbox');mailbox=client.root
snap=client.request('snapshot');(out/'snapshot.json').write_text(json.dumps(snap,indent=2))
if args.model_source:
    from model_archive import relocate
    relocate(args.model_source/'model',out/'model',snap)
    relocate(args.model_source/'model_closed',out/'model_closed',snap,measured_gripper=False)
else:
    build(root.parent/'asset/robots/arx_x5_lift/1.0.0/usd/robot.usd',snap,out/'model',fine_gripper=args.fine_gripper)
if not args.reuse_scene:
    client.request('export_scene')
scene=json.loads((mailbox/'scene_meshes.json').read_text());scene_cfg={'mesh':{}};sources={}
(out/'meshes').mkdir(exist_ok=True)
for i,m in enumerate(scene['meshes']):
    # Self-collision is disabled in this scene; export external geometry only.
    if '/World/Robot/' in m['path']:continue
    path=out/'meshes'/f'{i}.obj'
    trimesh.Trimesh(vertices=m['vertices'],faces=m['faces'],process=False).export(path)
    sources[f'collider_{i}']=m['path']
    scene_cfg['mesh'][f'collider_{i}']={'file_path':str(path),'pose':[0,0,0,1,0,0,0]}
(out/'scene.yml').write_text(yaml.safe_dump(scene_cfg))
(out/'scene_sources.json').write_text(json.dumps(sources,indent=2))
robot=Kinematics(KinematicsCfg.from_robot_yaml_file(str(out/'model/arx.yml')))
measured=dict(zip(snap['joint_names'],snap['joint_positions']))
q=torch.tensor([[measured[n] for n in robot.joint_names]],device='cuda',dtype=torch.float32)
state=robot.compute_kinematics(JointState.from_position(q,joint_names=robot.joint_names))
actual=np.linalg.inv(snap['link_world']['body4'])@np.asarray(snap['link_world']['right_eef_link'])
position=state.tool_poses.position.flatten().detach().cpu().numpy()
from scipy.spatial.transform import Rotation
quat=state.tool_poses.quaternion.flatten().detach().cpu().numpy()
rotation=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
report={'position_error_m':float(np.linalg.norm(position-actual[:3,3])),
        'rotation_error_rad':float(Rotation.from_matrix(rotation@actual[:3,:3].T).magnitude()),
        'fk_position':position.tolist(),'measured_position':actual[:3,3].tolist(),'joint_names':robot.joint_names,
        'joint_positions':q.flatten().tolist(),'scene_colliders':len(scene_cfg['mesh'])}
print(json.dumps(report,indent=2),flush=True)
(out/'fk_report.json').write_text(json.dumps(report,indent=2))
assert report['position_error_m']<0.002 and report['rotation_error_rad']<0.01,'ARX model FK mismatch'
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(out/'model/arx.yml'),scene_model=str(out/'scene.yml'),collision_activation_distance=0.0,n_meshes=len(scene_cfg['mesh'])+1))
# Match arx-sim/config/runtime.yaml: articulation self-collision is disabled.
checker.self_collision_cost=None
limits=torch.tensor([snap['joint_limits'][snap['joint_names'].index(n)] for n in robot.joint_names],device=q.device)
q_checked=torch.maximum(torch.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
report['start_limit_clamp_max_rad']=float((q_checked-q).abs().max())
valid=checker.validate(q_checked[:,None,:]);dworld,dself=checker.get_scene_self_collision_distance_from_joints(q_checked[:,None,:])
report.update(self_collision_enabled=False,start_valid=bool(valid.all()),world_penetration_cost=float(dworld.max()),self_penetration_cost=float(dself.max()))
print(json.dumps(report,indent=2),flush=True)
(out/'fk_report.json').write_text(json.dumps(report,indent=2))
