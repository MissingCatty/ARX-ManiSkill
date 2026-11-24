"""Independent analytic sphere/cylinder clearance of planned and measured pregrasp."""
import sys,json
from pathlib import Path
import numpy as np,torch
from curobo.motion_planner import MotionPlanner
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick';out=P/sys.argv[1]
def read(p):return json.loads(p.read_text())
spec=read(out/'pole.json');b=np.array(read(out/'snapshot.json')['link_world']['body4']);kin=Kinematics(KinematicsCfg.from_robot_yaml_file(str(out/'model/arx.yml')))
def clearance(q):
 q=torch.tensor(np.asarray(q),device='cuda',dtype=torch.float32);result=[]
 for chunk in q.split(64):
  s=kin.compute_kinematics(JointState.from_position(chunk,joint_names=kin.joint_names)).robot_spheres.reshape(len(chunk),-1,4).cpu().numpy();w=s[...,:3]@b[:3,:3].T+b[:3,3];pieces=[]
  for center,r,h in [(spec['center'],spec['radius'],spec['height']),([*spec['center'][:2],.0125],.06,.025)]:
   d=np.stack([np.linalg.norm(w[...,:2]-center[:2],axis=-1)-r,np.abs(w[...,2]-center[2])-h/2],-1)
   pieces.append((np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)-s[...,3]).min(-1))
  result.extend(np.minimum(*pieces).tolist())
 return np.array(result)
cmd=read(out/'pregrasp_command.json');q=np.array(cmd['position']);times=np.array(cmd['time_from_start']);dense_t=np.linspace(times[0],times[-1],10*(len(q)-1)+1);dense=np.stack([np.interp(dense_t,times,q[:,i]) for i in range(q.shape[1])],-1)
measured=np.array([r['position'] for r in read(out/'pregrasp_execution.json')['records']]);straight=np.linspace(q[0],q[-1],1001)
report={'scope':'Analytic signed clearance between unchanged ARX collision spheres and exact new pole cylinders; planning-model clearance, separate from measured PhysX contacts','command_min_clearance_m':float(clearance(dense).min()),'measured_state_model_min_clearance_m':float(clearance(measured).min()),'same_endpoint_linear_joint_path_min_clearance_m':float(clearance(straight).min()),'linear_joint_path_scope':'Counterfactual straight joint interpolation only, not an executed or separately optimized obstacle-free plan','samples':{'command_dense':len(dense),'measured':len(measured),'linear':len(straight)}}
(out/'path_clearance_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
