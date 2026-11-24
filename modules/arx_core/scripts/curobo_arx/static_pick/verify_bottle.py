import sys,json
from pathlib import Path
import numpy as np,torch,yaml,trimesh
from curobo.motion_planner import MotionPlanner
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick';out=P/sys.argv[1]
def read(p):return json.loads(p.read_text())
snap=read(out/'snapshot.json');b=np.array(snap['link_world']['body4']);scene=yaml.safe_load((out/'scene.yml').read_text());sources=read(out/'scene_sources.json');spec=read(out/'pole.json');checks=[]
for part,center,r,h in [('shaft',spec['center'],spec['radius'],spec['height']),('base',[*spec['center'][:2],.0125],.06,.025)]:
 keys=[k for k,v in sources.items() if v=='/World/StaticPickPole/'+part];assert len(keys)==1
 m=trimesh.load(scene['mesh'][keys[0]]['file_path']);v=np.asarray(m.vertices)@b[:3,:3].T+b[:3,3]
 half=np.array([r/np.cos(np.pi/128),r/np.cos(np.pi/128),h/2]);err=max(abs(v.min(0)-(np.array(center)-half)).max(),abs(v.max(0)-(np.array(center)+half)).max());assert err<1e-5
 checks.append({'part':part,'bounds_error_m':float(err),'circumscription_extra_radius_m':float(r/np.cos(np.pi/128)-r)})
assert set(read(P/'scene_sources.json').values()).issubset(set(sources.values()))
scene['mesh']={k:v for k,v in scene['mesh'].items() if not sources[k].startswith(snap['objects']['test_object']['path'])}
check=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(out/'model/arx.yml'),scene_model=scene,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.));check.self_collision_cost=None
source=Path(read(out/'registration.json')['historical_source']);sel=read(source/'selected_grasp.json');path=read(source/'pregrasp_command.json')['position'];q=torch.tensor(path+[sel[n] for n in ['q_pre','q_grasp','q_lift']],device='cuda');valid=torch.cat([check.validate(part[:,None]).flatten() for part in q.split(64)])
report={'scope':'Full-model check, old path never physically executed into pole','pole_export_checks':checks,'old_pregrasp_samples':len(path),'old_pregrasp_invalid_samples':int((~valid[:len(path)]).sum()),'initial_valid':bool(valid[0]),'old_goal_validity':dict(zip(['pregrasp','grasp','lift'],valid[-3:].tolist())),'original_success_path_now_blocked':bool((~valid[:len(path)]).any()),'all_original_colliders_retained':True}
(out/'baseline_path_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
assert report['initial_valid']
