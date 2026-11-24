"""ARX stop-and-go dynamic collision avoidance with a cuRobo optimized path.

Original robot/physics and full hard model are retained. The controller optimizes
when to enter an obstructed path, using fresh obstacle states at every render.
"""
import argparse,copy,json,os,sys,time,uuid
from pathlib import Path
import numpy as np
import torch,yaml
from scipy.spatial.transform import Rotation
from curobo.motion_planner import MotionPlanner
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
from curobo.types import JointState,Pose
import curobo._src.runtime as runtime
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/"runs/arx_dynamic_pick";DATA=TASK.resolve().parent;runtime.cache_dir=str(DATA/"curobo-cache")
sys.path.insert(0,str(ROOT/"scripts/curobo_arx"))
from client import Client
from gripper_update import WarmedGripperUpdate
from settling import assess_settling

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2))
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--name",required=True);args=parser.parse_args();out=TASK/args.name;out.mkdir(exist_ok=False)
 c=Client(TASK/"mailbox");snap=c.request("snapshot");base=np.array(snap["link_world"]["body4"]);inv=np.linalg.inv(base);names=snap["right_arm"];selected=read(TASK/"selected_grasp.json");target=snap["objects"]["test_object"]["path"];initial_z=np.array(snap["objects"]["test_object"]["world"])[2,3]
 report={"success":False,"status":"warming_up","method":"cuRobo full optimized reference + online time scheduling (stop-and-go)","native_MPC_used":False,"physical_obstacles":1,"robot_physics_configuration_changed":False,"stages":{}}
 write(out/"result.json",report);write(out/"initial_snapshot.json",snap);write(out/"viewport_initial.json",c.request("grasp_viewport_state"))
 scene=yaml.safe_load((TASK/"scene.yml").read_text());sources=read(TASK/"scene_sources.json");limits=torch.tensor([snap["joint_limits"][snap["joint_names"].index(n)] for n in names],device="cuda")
 def checker(model,margin):
  world=copy.deepcopy(scene)
  if margin:
   for k,m in world["mesh"].items():
    if "breakfast_table" in sources[k]:m["pose"][:3]=(np.array(m["pose"][:3])+inv[:3,2]*margin).tolist()
  check=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),scene_model=world,n_meshes=len(scene["mesh"])+1,collision_activation_distance=0.));check.self_collision_cost=None;return check
 open_check=checker(TASK/"model/arx.yml",0.);closed_check=checker(TASK/"model_closed/arx.yml",.008);update=WarmedGripperUpdate(TASK/"model_closed/arx.yml",closed_check.kinematics)
 pre=read(TASK/"attempt_00/static_pregrasp_guide.json");source=Path(read(TASK/"task_contract.json")["source"]);approach=read(source/"approach_command.json");lift=read(source/"lift_command.json");qp,qg,ql=[torch.tensor(selected[k],device="cuda") for k in ("q_pre","q_grasp","q_lift")]
 def measured(s):
  q=torch.tensor([s["joint_positions"][s["joint_names"].index(n)] for n in names],device="cuda");return torch.maximum(torch.minimum(q,limits[:,1]-1e-4),limits[:,0]+1e-4)
 def spheres(q):
  s=open_check.kinematics.compute_kinematics(JointState.from_position(q.contiguous(),joint_names=names)).robot_spheres
  s=s.reshape(q.shape[0],-1,4)
  return s[...,:3]@torch.tensor(base[:3,:3],device="cuda",dtype=torch.float32).T+torch.tensor(base[:3,3],device="cuda",dtype=torch.float32),s[...,3]
 def distance(world,radius,centers,dims):
  delta=abs(world-centers[:,None])-torch.tensor(dims,device="cuda",dtype=torch.float32)/2
  return torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)-radius
 qpre=torch.tensor(pre["position"],device="cuda");world,rad=spheres(qpre)
 corridor_center=torch.tensor([4.5,2.17,.88],device="cuda").repeat(len(qpre),1);corridor=distance(world,rad,corridor_center,[3.,.12,.18]).amin(-1).cpu().numpy()
 unsafe=np.flatnonzero(corridor<.03);gate=max(0,int(unsafe[0])-2) if len(unsafe) else len(qpre)//2
 assert corridor[gate]>.015,"No safe initial waiting configuration"
 qgate=qpre[gate].clone();write(out/"waiting_configuration.json",{"reference_index":gate,"corridor_clearance_m":float(corridor[gate]),"q_wait":qgate.tolist(),"all_reference_corridor_distances_m":corridor.tolist(),"scope":"Conservative moving-box swept corridor for selecting a hold pose; actual box geometry unchanged"})
 def smooth_segment(q,t):
  q=np.asarray(q);t=np.asarray(t)-t[0]
  if len(q)<2 or t[-1]<1e-8:return np.repeat(q[:1],2,axis=0),np.array([0.,1.]),{"stationary":True}
  duration=max(2.,2*float(t[-1]));n=max(81,int(np.ceil(duration/.025))+1);times=np.linspace(0,duration,n);u=times/duration;phase=(10*u**3-15*u**4+6*u**5)*t[-1]
  result=np.stack([np.interp(phase,t,q[:,i]) for i in range(q.shape[1])],axis=1);step=times[1]-times[0];fd=result.copy();scale=1.;maxima={}
  for order,bound in [(1,1.5),(2,5.),(3,100.)]:
   fd=np.diff(fd,axis=0)/step;peak=float(abs(fd).max()) if len(fd) else 0.;maxima[str(order)]=peak;scale=max(scale,(peak/bound)**(1/order))
  times*=scale
  return result,times,{"time_scale":scale,"duration_s":float(times[-1]),"command_fd_maxima":{k:v/scale**int(k) for k,v in maxima.items()},"source":"Smooth timing of unchanged fully optimized cuRobo reference geometry"}
 q0,t0,r0=smooth_segment(pre["position"][:gate+1],pre["time_from_start"][:gate+1]);q1,t1,r1=smooth_segment(pre["position"][gate:],pre["time_from_start"][gate:]);write(out/"retiming.json",{"entry":r0,"remaining_pregrasp":r1})
 def validate(q,check):
  q=torch.as_tensor(q,device="cuda",dtype=torch.float32);assert torch.isfinite(q).all() and ((q>=limits[:,0]-1e-6)&(q<=limits[:,1]+1e-6)).all()
  dense=torch.nn.functional.interpolate(q.T[None],size=5*(len(q)-1)+1,mode="linear",align_corners=True).transpose(1,2).reshape(-1,len(names)).contiguous()
  assert all(check.validate(part[:,None]).all() for part in dense.split(64)),"Reference failed unchanged full-model checking"
 for q,check in [(q0,open_check),(q1,open_check),(approach["position"],open_check),(lift["position"],closed_check)]:validate(q,check)
 # Remaining robot timeline used to optimize admission time, not a timed replay
 # of the moving obstacle. Every iteration uses its fresh observed position.
 qa=np.array(approach["position"]);ta=np.array(approach["time_from_start"]);qL=np.array(lift["position"]);tL=np.array(lift["time_from_start"])
 future_q=np.concatenate([q1,qa[1:],np.repeat(selected["q_grasp"][None] if isinstance(selected["q_grasp"],np.ndarray) else np.array(selected["q_grasp"])[None],15,axis=0),qL[1:],np.repeat(np.array(selected["q_lift"])[None],15,axis=0)])
 future_t=np.r_[t1,t1[-1]+ta[1:],t1[-1]+ta[-1]+np.linspace(.2,3,15),t1[-1]+ta[-1]+3+tL[1:],t1[-1]+ta[-1]+3+tL[-1]+np.linspace(.2,3,15)]
 fw,fr=spheres(torch.tensor(future_q,device="cuda",dtype=torch.float32));ft=torch.tensor(future_t,device="cuda",dtype=torch.float32)
 dims=np.array([.1,.12,.18]);velocity=np.array([.04,0.,0.]);cross=np.array([3.96,2.17,.88]);layouts=[]
 for ct in np.arange(4.,31.,1.):
  start=cross-velocity*ct;centers=torch.tensor(start,device="cuda",dtype=torch.float32)+(float(t0[-1])+ft[:,None]+.4)*torch.tensor(velocity,device="cuda",dtype=torch.float32)
  clear=float(distance(fw,fr,centers,dims).min());layouts.append({"cross_time_s":float(ct),"future_without_wait_clearance_m":clear})
 eligible=[v for v in layouts if v["future_without_wait_clearance_m"]<-.03];assert eligible,"No genuinely blocked crossing layout found"
 chosen=min(eligible,key=lambda v:abs(v["cross_time_s"]-(t0[-1]+t1[-1])));ct=chosen["cross_time_s"];start=cross-velocity*ct;write(out/"crossing_registration.json",{"chosen":chosen,"all_model_only_timing_checks":layouts,"velocity_m_s":velocity.tolist(),"physical_dims_m":dims.tolist(),"initial_center":start.tolist(),"old_robot_target_table_poses_unchanged":True})
 report.update(status="ready",reference= str(source));write(out/"result.json",report);print("READY",json.dumps({"gate":gate,"gate_clearance":float(corridor[gate]),"entry_duration":float(t0[-1]),"remaining_duration":float(t1[-1]),"cross_time":ct}),flush=True)
 def goal_world(q):
  state=open_check.kinematics.compute_kinematics(JointState.from_position(torch.tensor(q,device="cuda",dtype=torch.float32).reshape(1,6),joint_names=names));pos=state.tool_poses.position.reshape(3).cpu().numpy();quat=state.tool_poses.quaternion.reshape(4).cpu().numpy();m=np.eye(4);m[:3,3]=pos;m[:3,:3]=Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_matrix();return base@m
 def forbidden(e):
  if e["num_contacts"]<=0:return False
  actors=e["a"]+e["b"]
  if "/World/DynamicPickObstacle" in actors:return True
  if not any(n in actors for n in ("/right_link","/right_eef_link")):return False
  if "/World/Robot/" in e["a"] and "/World/Robot/" in e["b"]:return False
  return not (target in actors and any(n in actors for n in ("right_link27","right_link28")))
 def execute(label,q,t,goal,gripper=None):
  command={"joint_names":names,"position":np.asarray(q).tolist(),"time_from_start":np.asarray(t).tolist(),"position_tolerance_rad":.003}
  if gripper is not None:command["gripper_target"]=gripper
  write(out/(label+"_command.json"),command);print("EXECUTE",label,flush=True);result=c.request("trajectory",timeout=180,**command);write(out/(label+"_execution.json"),result);assess=assess_settling(result,goal);report["stages"][label]=assess;write(out/(label+"_settling.json"),assess)
  assert assess["accepted"],label+" failed original convergence gate";assert not any(forbidden(e) for e in result["contacts"]),label+" forbidden measured contact";return result
 recording=False;streaming=False;logs=[]
 try:
  info=c.request("dynamic_obstacle",center=start.tolist(),dims=dims.tolist(),velocity=velocity.tolist());write(out/"obstacle_spec.json",info)
  c.request("grasp_video_start",directory="/workspace/COOLAS-ARX/runs/arx_dynamic_pick/"+args.name+"/frames");recording=True
  execute("safe_wait_pose",q0,t0,goal_world(qgate.tolist()))
  c.request("dynamic_stream_start",trace_path="/workspace/COOLAS-ARX/runs/arx_dynamic_pick/"+args.name+"/waiting_state.jsonl");streaming=True
  begin=read(c.root/"stream_state.json")["sim_time"];last=None;deadline=time.monotonic()+180;had_blocked=False;clear_count=0
  while True:
   s=read(c.root/"stream_state.json")
   if time.monotonic()>deadline:raise TimeoutError("Online wait wall budget exceeded")
   if time.time()-s["wall_time"]>5:raise RuntimeError("ARX live state stopped")
   if s["sim_time"]==last:time.sleep(.005);continue
   last=s["sim_time"];elapsed=last-begin
   if elapsed>60:raise TimeoutError("Online wait simulation budget exceeded")
   current=torch.tensor(s["moving_obstacle"]["center"],device="cuda",dtype=torch.float32);v=torch.tensor(velocity,device="cuda",dtype=torch.float32)
   clearance=min(float(distance(fw,fr,current[None]+(ft[:,None]+delay)*v,dims).min()) for delay in (0.,.3,.6))
   q=measured(s);assert open_check.validate(q[None,None].contiguous()).all(),"Measured wait pose invalid in original static model"
   if (c.root/"stream_events.json").exists():assert not any(forbidden(e) for e in read(c.root/"stream_events.json")["recent"]),"Measured unwanted contact while waiting"
   blocked=clearance<.025;had_blocked=had_blocked or blocked;clear_count=0 if blocked else clear_count+1
   command={"id":uuid.uuid4().hex,"joint_names":names,"position":[qgate.tolist(),qgate.tolist()],"time_from_start":[0.,.2],"sim_time":last};tmp=c.root/"stream_command.tmp";tmp.write_text(json.dumps(command));tmp.replace(c.root/"stream_command.json")
   row={"sim_time":last,"elapsed":elapsed,"wall_time":s["wall_time"],"future_path_clearance_m":clearance,"blocked":blocked,"obstacle_center":s["moving_obstacle"]["center"],"measured_joint_error_rad":float((q-qgate).abs().max())};logs.append(row)
   with (out/"online_admission.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
   if len(logs)%10==0:print("ONLINE",round(elapsed,2),"blocked",blocked,"clearance",round(clearance,4),flush=True)
   if clear_count>=3:break
  stop=c.request("dynamic_stream_stop");streaming=False;write(out/"waiting_stop.json",stop);assert had_blocked,"Grasp was never blocked: do not claim a dynamic avoidance demonstration"
  report["online_wait_s"]=logs[-1]["elapsed"];report["online_checks"]=len(logs);report["avoidance_exercised"]=had_blocked
  execute("pregrasp",q1,t1,selected["pregrasp_world"])
  execute("approach",qa,ta,selected["grasp_world"])
  q=measured(c.request("snapshot"));close=execute("close",torch.stack([q,qg]).cpu().numpy(),[0.,3.],selected["grasp_world"],.1)
  fingers={n for e in close["contacts"] if e["num_contacts"]>0 and target in e["a"]+e["b"] for n in ("right_link27","right_link28") if n in e["a"]+e["b"]};assert len(fingers)==2,"Bilateral closure failed"
  closed=c.request("snapshot");write(out/"closed_snapshot.json",closed);update.apply([closed_check.kinematics],dict(zip(closed["joint_names"],closed["joint_positions"])));validate(qL,closed_check)
  execute("lift",qL,tL,selected["lift_world"],.1);q=measured(c.request("snapshot"));hold=execute("hold",torch.stack([q,ql]).cpu().numpy(),[0.,3.],selected["lift_world"],.1)
  rises=[np.asarray(r["objects"]["test_object"])[2,3]-initial_z for r in hold["records"]];contacts=c.request("dynamic_contacts")["contacts"];write(out/"all_contacts.json",contacts);bad=[e for e in contacts if forbidden(e)]
  report.update(status="completed",success=bool(min(rises)>.06 and not bad),bilateral_closure=True,hold_min_lift_m=float(min(rises)),forbidden_contact_events=len(bad),obstacle_contact_events=sum(e["num_contacts"]>0 and "/World/DynamicPickObstacle" in e["a"]+e["b"] for e in contacts));assert report["success"],"Physical grasp/avoidance success gate failed"
 except Exception as exc:report.update(status="failed",error=str(exc));raise
 finally:
  if streaming:
   try:write(out/"waiting_stop.json",c.request("dynamic_stream_stop"))
   except Exception as exc:report["stream_stop_error"]=str(exc)
  if recording:
   try:report["recording"]=c.request("grasp_video_stop")
   except Exception as exc:report["recording_error"]=str(exc)
  try:c.request("dynamic_obstacle_stop")
  except Exception as exc:report["obstacle_stop_error"]=str(exc)
  write(out/"result.json",report);print("RESULT",json.dumps(report),flush=True)
if __name__=="__main__":main()
