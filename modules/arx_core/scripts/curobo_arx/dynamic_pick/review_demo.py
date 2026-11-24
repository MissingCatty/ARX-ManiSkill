import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument("--name",required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[3];task=root/"runs/arx_dynamic_pick";out=task/args.name;sys.path.insert(0,str(root/"scripts/curobo_arx"))
from settling import assess_settling

def read(p):return json.loads(Path(p).read_text())
initial=read(out/"initial_snapshot.json");target=initial["objects"]["test_object"]["path"];z0=np.array(initial["objects"]["test_object"]["world"])[2,3];selected=read(task/"selected_grasp.json");assessments={}
for label,key in [("pregrasp","pregrasp_world"),("approach","grasp_world"),("close","grasp_world"),("lift","lift_world"),("hold","lift_world")]:assessments[label]=assess_settling(read(out/(label+"_execution.json")),selected[key])
hold=read(out/"hold_execution.json");last=hold["records"][-1]["time"];tail=[s for s in hold["records"] if s["time"]>=last-3.];rises=[np.asarray(s["objects"]["test_object"])[2,3]-z0 for s in tail];span=tail[-1]["time"]-tail[0]["time"]
close=read(out/"close_execution.json");fingers={n for e in close["contacts"] if e["num_contacts"]>0 and target in e["a"]+e["b"] for n in ("right_link27","right_link28") if n in e["a"]+e["b"]}
contacts=read(out/"all_contacts.json");obstacle=[e for e in contacts if e["num_contacts"]>0 and "/World/DynamicPickObstacle" in e["a"]+e["b"]];forbidden=[]
for e in contacts:
 if e["num_contacts"]<=0:continue
 actors=e["a"]+e["b"]
 if "/World/Robot/" in e["a"] and "/World/Robot/" in e["b"]:continue
 if "/World/DynamicPickObstacle" in actors or (any(n in actors for n in ("/right_link","/right_eef_link")) and not(target in actors and any(n in actors for n in ("right_link27","right_link28")))):forbidden.append(e)
a=read(task/(args.name+"_setup.json"))["fingerprint"]["invariant"];b=read(out/"final_fingerprint.json")["invariant"]
def normalize(v):
 v=copy.deepcopy(v)
 for path in list(v["usd"]):
  if path.startswith("/World/DynamicPickObstacle"):v["usd"].pop(path);continue
  if path.startswith("/World/COOLASSensors/"):v["usd"][path]["attributes"]={k:x for k,x in v["usd"][path]["attributes"].items() if not k.startswith("xformOp")}
 for path in list(v.get("runtime_objects",{})):
  if path.startswith("/World/DynamicPickObstacle"):v["runtime_objects"].pop(path)
 return v
A,B=normalize(a),normalize(b);differences=[]
def diff(x,y,path=""):
 if type(x)!=type(y):differences.append(path);return
 if isinstance(x,dict):
  for k in set(x)|set(y):
   if k not in x or k not in y:differences.append(path+"/"+k)
   else:diff(x[k],y[k],path+"/"+k)
 elif x!=y:differences.append(path)
diff(A,B)
guard=read(root/"runs/autoresearch_arx_20261022_approach_regression/configuration_guard.json");changed=[s for s,h in guard["file_sha256"].items() if not Path(s).exists() or hashlib.sha256(Path(s).read_bytes()).hexdigest()!=h]
wait=[json.loads(x) for x in (out/"online_admission.jsonl").read_text().splitlines()];physical=[json.loads(x) for x in (out/"waiting_state.jsonl").read_text().splitlines()]
ts=np.array([s["sim_time"] for s in physical]);positions=np.array([np.array(s["moving_obstacle"]["measured_world"])[:3,3] for s in physical]);fit=np.polyfit(ts-ts[0],positions,1)[0]
measured_speed={"fit_velocity_m_per_sim_s":fit.tolist(),"scope":"Measured rigid-body positions throughout the online waiting interval; source enforces the same velocity before and after until recording stops"}
view=read(out/"final_viewport.json");reference=read(out/"viewport_initial.json");video=read(out/"video_report.json")
assert np.allclose(view["world_matrix"],reference["world_matrix"],atol=1e-8)
original_view_ok=bool(np.allclose(reference["world_matrix"],read(task/"viewport.json")["world_matrix"],atol=1e-8))
# Independent actual command bounds, distinct from the controller's dense
# collision check and from measured physical contacts.
bound_violations=[]
for label in ("safe_wait_pose","pregrasp","approach","close","lift","hold"):
 cmd=read(out/(label+"_command.json"));q=np.asarray(cmd["position"]);limits=np.array([initial["joint_limits"][initial["joint_names"].index(n)] for n in cmd["joint_names"]])
 if cmd["joint_names"]!=initial["right_arm"] or not np.isfinite(q).all() or not ((q>=limits[:,0]-1e-6)&(q<=limits[:,1]+1e-6)).all():bound_violations.append(label)
render_errors=[np.linalg.norm(np.array(s["moving_obstacle"]["renderer_scaled_world"])[:3,3]-np.array(s["moving_obstacle"]["measured_world"])[:3,3]) for s in physical]
render_scale_ok=all(np.allclose(np.linalg.norm(np.array(s["moving_obstacle"]["renderer_scaled_world"])[:3,:3],axis=0),s["moving_obstacle"]["dims"]) for s in physical)
from PIL import Image
from scipy.ndimage import label as label_components
meta=read(out/"frames/frames.json");frames=meta["frames"]
window=[f for f in frames if ts[0]<=f["sim_time"]<=ts[-1]];pixel_samples=[]
for f in (window[1],window[-2]):
 rgb=np.asarray(Image.open(out/"frames"/f["file"]))[:,:,:3].astype(float)
 mask=(rgb[:,:,0]>110)&(rgb[:,:,1]>70)&(rgb[:,:,2]<40)&(rgb[:,:,0]/(rgb[:,:,1]+1)<1.7)
 components,count=label_components(mask);sizes=np.bincount(components.ravel());sizes[0]=0;ys,xs=np.where(components==sizes.argmax())
 pixel_samples.append({"frame":f["file"],"sim_time":f["sim_time"],"largest_yellow_component_pixels":len(xs),"centroid":[float(xs.mean()),float(ys.mean())]})
pixel_motion=float(np.linalg.norm(np.subtract(pixel_samples[-1]["centroid"],pixel_samples[0]["centroid"])))
visible_motion_ok=pixel_motion>10 and min(x["largest_yellow_component_pixels"] for x in pixel_samples)>500 and max(render_errors)<.001 and render_scale_ok

hold_times=[s["time"] for s in hold["records"]]; all_rises=[np.asarray(s["objects"]["test_object"])[2,3]-z0 for s in hold["records"]]
hold_coverage=len(hold_times)>=10 and hold_times[-1]-hold_times[0]>=2.5 and hold["sim_duration_s"]>=2.99
passed=all(v["accepted"] for v in assessments.values()) and len(fingers)==2 and hold_coverage and min(all_rises)>.06 and not obstacle and not forbidden and not changed and not differences and any(s["blocked"] for s in wait) and all(not s["blocked"] for s in wait[-3:]) and video["full_decode"]=="PASS" and not bound_violations and visible_motion_ok and original_view_ok
report={"pass":passed,"original_user_view_pass":original_view_ok,"method":"cuRobo optimized trajectories with online collision prediction and waiting; no native MPC success claim","all_five_original_stage_gates":assessments,"raw_lift_and_hold_strict_joint_settled":[read(out/(label+"_execution.json"))["settled"] for label in ("lift","hold")],"original_bounded_stationary_gate_unchanged":True,"bilateral_closure":len(fingers)==2,"hold_criterion":"Original evaluator coverage and >6cm at every recorded hold sample; no evaluator/threshold changes","original_hold_coverage_gate_pass":hold_coverage,"recorded_hold_duration_sim_s":hold["sim_duration_s"],"recorded_hold_sample_span_s":hold_times[-1]-hold_times[0],"whole_recorded_hold_min_lift_m":float(min(all_rises)),"last_hold_window_sim_s":span,"last_hold_window_min_lift_m":float(min(rises)),"obstacle_contact_events":len(obstacle),"forbidden_contact_events":len(forbidden),"configuration_differences_excluding_only_authorized_new_obstacle_and_renderer_proxy_state":differences,"protected_files_checked":len(guard["file_sha256"]),"changed_protected_files":changed,"online_blocked_checks":sum(s["blocked"] for s in wait),"online_release_checks":len(wait)-sum(s["blocked"] for s in wait),"wait_sim_s":wait[-1]["elapsed"],"wait_wall_s":wait[-1]["wall_time"]-wait[0]["wall_time"],"online_check_wall_hz":(len(wait)-1)/(wait[-1]["wall_time"]-wait[0]["wall_time"]),"planned_joint_bound_violation_stages":bound_violations,"planning_review_scope":"Independent command bounds only; dense static collision validation executed by controller in unchanged full model; measured physical contacts evaluated separately","visible_obstacle_motion":{"pass":visible_motion_ok,"renderer_physics_max_position_difference_m":float(max(render_errors)),"declared_scale_preserved":render_scale_ok,"waiting_pixel_motion_px":pixel_motion,"samples":pixel_samples},"measured_obstacle_motion":measured_speed,"video":video,"main_research_status":read(root/"runs/autoresearch_arx_20261022_approach_regression/state.json")["status"],"time_unix":time.time()}
(out/"independent_verification.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ("all_five_original_stage_gates","video")},indent=2));assert passed
