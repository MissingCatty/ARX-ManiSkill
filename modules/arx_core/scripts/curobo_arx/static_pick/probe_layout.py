import json,sys
from pathlib import Path
import numpy as np,torch
from curobo.motion_planner import MotionPlanner
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick'
b=np.array(json.loads((P/'snapshot.json').read_text())['link_world']['body4'])
k=Kinematics(KinematicsCfg.from_robot_yaml_file(str(P/'model/arx.yml')))
q=torch.tensor(json.loads((P/'attempt_00/static_pregrasp_guide.json').read_text())['position'],device='cuda')
s=k.compute_kinematics(JointState.from_position(q,joint_names=k.joint_names)).robot_spheres.reshape(len(q),-1,4).cpu().numpy()
w=s[...,:3]@b[:3,:3].T+b[:3,3]
parts=[{'name':'crossbar','center':[3.98,2.24,.80],'dims':[.64,.045,.045]}, {'name':'left_post','center':[3.68,2.24,.4],'dims':[.045,.045,.8]}, {'name':'right_post','center':[4.28,2.24,.4],'dims':[.045,.045,.8]}, {'name':'left_foot','center':[3.68,2.26,.015],'dims':[.12,.13,.03]}, {'name':'right_foot','center':[4.28,2.26,.015],'dims':[.12,.13,.03]}]
for part in parts:
 d=np.abs(w-np.array(part['center']))-np.array(part['dims'])/2
 clear=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)-s[...,3]
 print(part['name'],'min',clear.min(),'initial',clear[0].min(),'endpoint',clear[-1].min())
print(json.dumps(parts))
