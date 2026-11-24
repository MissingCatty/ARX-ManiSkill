import sys,json
from pathlib import Path
import numpy as np,torch,yaml,trimesh
from curobo.motion_planner import MotionPlanner
from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
R=Path(__file__).resolve().parents[3];P=R/'runs/arx_dynamic_pick';out=P/sys.argv[1]
def read(p):return json.loads(p.read_text())
snap=read(out/'snapshot.json');b=np.array(snap['link_world']['body4']);scene=yaml.safe_load((out/'scene.yml').read_text());sources=read(out/'scene_sources.json');parts=read(out/'rack.json')['parts'];bounds=[]
for part in parts:
 keys=[k for k,v in sources.items() if v=='/World/StaticPickRack/'+part['name']];assert len(keys)==1,(part,keys)
 m=trimesh.load(scene['mesh'][keys[0]]['file_path']);v=np.asarray(m.vertices)@b[:3,:3].T+b[:3,3]
 err=max(abs(v.min(0)-(np.array(part['center'])-np.array(part['dims'])/2)).max(),abs(v.max(0)-(np.array(part['center'])+np.array(part['dims'])/2)).max());bounds.append({'part':part['name'],'bounds_error_m':float(err)});assert err<1e-5,bounds
scene['mesh']={k:v for k,v in scene['mesh'].items() if not sources[k].startswith(snap['objects']['test_object']['path'])}
assert set(read(P/'scene_sources.json').values()).issubset(set(sources.values())), 'Original scene collider missing'
check=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(out/'model/arx.yml'),scene_model=scene,n_meshes=len(scene['mesh'])+1,collision_activation_distance=0.));check.self_collision_cost=None
q=torch.tensor(read(P/'attempt_00/static_pregrasp_guide.json')['position'],device='cuda');valid=torch.cat([check.validate(part[:,None]).flatten() for part in q.split(64)])
report={'scope':'model-only counterfactual, old path never physically executed into rack','rack_export_bounds':bounds,'old_pregrasp_samples':len(q),'old_pregrasp_invalid_samples':int((~valid).sum()),'initial_valid':bool(valid[0]),'original_success_path_now_blocked':bool((~valid).any()),'all_original_colliders_retained':True}
(out/'baseline_path_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);assert report['original_success_path_now_blocked'] and report['initial_valid']
