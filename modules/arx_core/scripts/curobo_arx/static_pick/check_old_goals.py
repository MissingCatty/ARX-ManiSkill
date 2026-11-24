import json,sys
from pathlib import Path
import numpy as np,torch
from curobo.motion_planner import MotionPlanner
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick';out=P/sys.argv[1]
def read(p):return json.loads(p.read_text())
snap=read(out/'snapshot.json');b=np.array(snap['link_world']['body4']);sel=read(P/'selected_grasp.json');parts=read(out/'rack.json')['parts']
k=Kinematics(KinematicsCfg.from_robot_yaml_file(str(out/'model/arx.yml')))
names=['q_pre','q_grasp','q_lift'];q=torch.tensor([sel[n] for n in names],device='cuda');s=k.compute_kinematics(JointState.from_position(q,joint_names=k.joint_names)).robot_spheres.reshape(len(q),-1,4).cpu().numpy();w=s[...,:3]@b[:3,:3].T+b[:3,3];results={}
for part in parts:
 d=np.abs(w-np.array(part['center']))-np.array(part['dims'])/2
 clear=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)-s[...,3]
 results[part['name']]={n:float(clear[i].min()) for i,n in enumerate(names)}
report={'scope':'Model-only original successful joint goals vs registered rack; open gripper model, signed sphere/box separation, negative means overlap. Does not prove all alternative grasps infeasible.','minimum_clearance_m':results}
(out/'old_goal_clearance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
