"""Model-only layout design; no simulator commands or altered robot configuration."""
import json
from pathlib import Path
import numpy as np,torch
from curobo.motion_planner import MotionPlanner
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick';out=P/'pole_layout_design';out.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
b=np.array(read(P/'snapshot.json')['link_world']['body4']);sel=read(P/'selected_grasp.json');path=np.array(read(P/'attempt_00/static_pregrasp_guide.json')['position']);k=Kinematics(KinematicsCfg.from_robot_yaml_file(str(P/'model/arx.yml')))
q=np.concatenate([path,np.array([sel[n] for n in ['q_pre','q_grasp','q_lift']])]);s=k.compute_kinematics(JointState.from_position(torch.tensor(q,device='cuda',dtype=torch.float32),joint_names=k.joint_names)).robot_spheres.reshape(len(q),-1,4).cpu().numpy();w=s[...,:3]@b[:3,:3].T+b[:3,3]
def clearance(x,y,h,r):
 d=np.stack([np.linalg.norm(w[...,:2]-[x,y],axis=-1)-r,np.abs(w[...,2]-h/2)-h/2],-1)
 return (np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)-s[...,3]).min(-1)
rows=[]
for h in [.8,.9,1.]:
 for x in np.arange(3.65,4.301,.01):
  for y in np.arange(2.22,2.461,.01):
   d=clearance(x,y,h,.0175); safe=float(min(d[0],*d[-3:]));blocked=float(d[:len(path)].min())
   if safe>=.015 and blocked<=-.008:rows.append({'center':[float(x),float(y),h/2],'height':h,'radius':.0175,'endpoint_min_clearance_m':safe,'old_path_min_clearance_m':blocked,'old_path_first_collision_index':int(np.where(d[:len(path)]<0)[0][0])})
(out/'eligible_layouts.json').write_text(json.dumps(rows,indent=2))
assert rows,'No layout meets declared endpoint clearance and path obstruction checks'
chosen=max(rows,key=lambda a:(a['endpoint_min_clearance_m'], -abs(a['height']-.9)))
(out/'chosen.json').write_text(json.dumps(chosen,indent=2));print(json.dumps(chosen),flush=True)
