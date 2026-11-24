"""Validate diverse assets and construct a balanced, fixed-head 200-scene dataset."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from asset_catalog import HERE, PROJECT, MS, CATALOG, read, save, sha, transformed

from arx_project.scenes import current_dataset
from arx_project.assets import CATALOG as SHARED_CATALOG
DEFAULT=current_dataset()
SELECTED=SHARED_CATALOG
LEGACY=PROJECT/'experiments/head_perception_scenes'
sys.path.insert(0,str(LEGACY))
import scene_builder as scenes
LAYOUTS=scenes.LAYOUTS

def now():return datetime.now().astimezone().isoformat()

def runtime(db=None):
 if db is None:db=load_object_catalog(current_dataset()/'object_catalog.json')['models']
 os.environ.setdefault('MS_ASSET_DIR',str(MS/'assets'))
 sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(PROJECT/'experiments/pointcloud_benchmark')]
 import arx_runtime_imports
 arx_runtime_imports.install()
 import environment,baseline,run,sapien,torch
 import gymnasium as gym
 import asset_environment
 from head_camera_config import head_configuration
 from camera_imaging import camera_profiles,sensor_and_model_frames,lens_maps
 environment.INITIAL_JOINTS={**environment.INITIAL_JOINTS,'joint4':.452}
 override,record=head_configuration(environment.CURRENT_CONTROLLERS)
 environment.CURRENT_CONTROLLERS=deepcopy(environment.CURRENT_CONTROLLERS)
 environment.CURRENT_CONTROLLERS['camera']['definitions']['head']=record['camera_definition']
 baseline.DB=db
 keeper=[]
 def make(model,scene):
  env=gym.make('ARXPickCatalog-v1',model_id=model,object_catalog=db,perception_camera='head',scene_spec=scene,
    num_envs=1,obs_mode='rgb+depth+segmentation',control_mode='pd_joint_pos',reward_mode='none',
    render_mode='rgb_array',sim_backend='cpu',render_backend='sapien_cuda',sensor_configs=override).unwrapped
  if not keeper:keeper.append(sapien.render.SapienRenderer(env._render_device))
  return env
 scenes.semantic_class=lambda model:db[model]['semantic_class']
 return environment,baseline,run,sapien,torch,make,camera_profiles()['head'],sensor_and_model_frames,lens_maps

def normalize_rest(record,q):
 result=deepcopy(record);r=Rotation.from_quat([*q[1:],q[0]]).as_matrix();t=np.eye(4);t[:3,:3]=r
 for item in result['visuals']+result['collisions']:item['transform']=(t@np.asarray(item['transform'])).tolist()
 points=np.concatenate([transformed(item) for item in result['visuals']]);lo,hi=points.min(0),points.max(0)
 result.update(bbox={'min':lo.tolist(),'max':hi.tolist()},size_m=(hi-lo).tolist(),canonical_rest_quaternion_wxyz=q,
  rest_orientation_provenance='Observed stable tabletop orientation in the original ARX physics; applied equally to visual and collision geometry.')
 return result

def preflight():
 if SELECTED.exists():raise RuntimeError('Selected catalog already exists; do not overwrite')
 prepared=read(CATALOG);db={r['model_id']:r for rows in prepared['candidate_models'].values() for r in rows}
 environment,baseline,run,sapien,torch,make,profile,convert,maps=runtime(db)
 selected={};reports=[];start=time.perf_counter();seen=set()
 for category in prepared['categories']:
  successes=[]
  for record in prepared['candidate_models'][category]:
   model=record['model_id'];env=None
   folder=HERE/'assets/preflight'/model
   if (folder/'verification.json').exists() and (folder/'normalized_record.json').exists():
    witness=read(folder/'verification.json');canonical=read(folder/'normalized_record.json')
    assert witness['passed'] and canonical['source_geometry_sha256']==record['source_geometry_sha256']
    selected[model]=canonical;db[model]=canonical;successes.append(model);seen.add(record['source_geometry_sha256']);reports.append(witness)
    if len(successes)==2:break
    continue
   try:
    for file,digest in record['source_hashes'].items():assert sha(file)==digest,file
    if record['source_geometry_sha256'] in seen:raise RuntimeError('Repeated source geometry')
    spec=baseline.actor_spec('target',model,[-.35,.05],0.)
    env=make(model,{'layout':'asset_preflight','actors':[spec]});env.reset(seed=20261122)
    baseline.settle(env)
    body=env.obj._bodies[0];mass=float(body.mass);assert .00001<=mass<=3.,('Unsupported mass',mass)
    shapes=body.get_collision_shapes();assert len(shapes)>0
    q=env.obj.pose.raw_pose[0].cpu().numpy()[3:].tolist()
    # Two yaw placements test reproducible resting behavior without arm motion.
    poses=[]
    for yaw in [0.,1.1]:
     rotation=Rotation.from_euler('z',yaw)*Rotation.from_quat([*q[1:],q[0]])
     points=baseline.corners(record)@rotation.as_matrix().T
     p=np.r_[np.array([-.35,.05])-(points.min(0)[:2]+points.max(0)[:2])/2,-points.min(0)[2]+.002]
     xyzw=rotation.as_quat();env.obj.set_pose(sapien.Pose(p,xyzw[[3,0,1,2]]))
     body.linear_velocity=np.zeros(3);body.angular_velocity=np.zeros(3);baseline.settle(env)
     actual=env.obj.pose.raw_pose[0].cpu().numpy();poses.append(actual.tolist())
     error=(rotation.inv()*Rotation.from_quat(actual[[4,5,6,3]])).magnitude()
     assert error<.06,('Unstable rest orientation',float(error))
    canonical=normalize_rest(record,q);db[model]=canonical
    obs=env.get_obs();camera_state=run.snapshot(env,obs)
    camera=np.asarray(camera_state['camera_world_optical']);intrinsics=np.asarray(camera_state['intrinsics'])
    placement_counts={}
    for layout in ['S1_open','S2_edge','S5_occluded']:
     probe_entry={'id':'preflight_'+model,'model_id':model,'layout':layout}
     placement_counts[layout]=len(scenes.target_candidates(probe_entry,baseline,camera,intrinsics,np.random.default_rng(20261122)))
     assert placement_counts[layout]>0,('No complete head-view placement',layout)
    selected[model]=canonical;successes.append(model);seen.add(record['source_geometry_sha256'])
    folder.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(folder/'head.jpg',quality=88)
    data=obs['sensor_data']['head'];mask=data['segmentation'][0,:,:,0].cpu().numpy()==env.obj._objs[0].per_scene_id
    frame=Image.fromarray(data['rgb'][0].cpu().numpy());ys,xs=np.where(mask)
    if len(xs):frame=frame.crop((max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(640,int(xs.max())+13),min(480,int(ys.max())+13)))
    frame.thumbnail((150,150));canvas=Image.new('RGB',(160,160),'white');canvas.paste(frame,((160-frame.width)//2,(160-frame.height)//2));canvas.save(folder/'thumb.jpg',quality=88)
    from scene_tools import export_geometry
    collision=export_geometry(env)['target']
    witness={'passed':True,'model_id':model,'semantic_class':category,'mass_kg':mass,'native_convex_parts':len(shapes),
      'rest_quaternion_wxyz':q,'tested_poses':poses,'head_view_placement_counts':placement_counts,'collider_export':collision,'physical_grasps':0,
      'linear_speed_m_s':float(np.linalg.norm(body.linear_velocity)),
      'angular_speed_rad_s':float(np.linalg.norm(body.angular_velocity))}
    save(folder/'normalized_record.json',canonical);save(folder/'verification.json',witness);reports.append(witness)
    print('PREFLIGHT_PASS',len(selected),model,mass,len(shapes),flush=True)
   except (RuntimeError,AssertionError,ValueError) as error:
    reports.append({'passed':False,'model_id':model,'semantic_class':category,'reason':str(error),'physical_grasps':0})
    print('PREFLIGHT_REJECT',model,str(error),flush=True)
   finally:
    if env is not None:env.close()
   save(HERE/'assets/preflight_progress.json',{'at':now(),'selected':len(selected),'completed_categories':len({r['semantic_class'] for r in selected.values()}),
     'reports':reports,'physical_grasps':0,'elapsed_seconds':time.perf_counter()-start})
   if len(successes)==2:break
  if len(successes)!=2:raise RuntimeError('Fewer than two physically validated assets in category '+category)
 save(SELECTED,{'at':now(),'models':selected,'categories':prepared['categories'],'source_catalog_sha256':sha(CATALOG),
  'preflight_reports':reports,'physical_grasps':0,'source_hashes':{str(HERE/n):sha(HERE/n) for n in ['asset_catalog.py','asset_environment.py','pipeline.py']}})
 print('SELECTED_READY',len(selected),'40 categories',flush=True)

def distractors(entry,reg,baseline,rng):
 db=baseline.DB;lo,hi=reg['distractor_counts'][entry['layout']]
 count=int(rng.integers(lo,hi+1)) if hi else 0
 if not count:return []
 classes=[c for c in reg['semantic_classes'] if c!=entry['semantic_class']];rng.shuffle(classes)
 available=[str(rng.choice([m for m in reg['target_models'] if db[m]['semantic_class']==c])) for c in classes]
 if entry['layout']=='S5_occluded':
  target_size=np.ptp(baseline.corners(db[entry['model_id']]),axis=0)
  tall=[m for m in available if db[m]['size_m'][2]>=max(.075,.8*target_size[2])]
  if not tall:tall=[m for m in available if db[m]['size_m'][2]>=.075]
  if not tall:raise RuntimeError('No eligible occluding object')
  tall.sort(key=lambda m:db[m]['size_m'][2]/np.sqrt(np.prod(db[m]['size_m'][:2])),reverse=True)
  first=tall[0];available.remove(first);return [first,*available[:count-1]]
 return available[:count]

def prepare(root):
 assert not root.exists();selected=load_object_catalog(SELECTED);db=selected['models'];categories=selected['categories'];assert len(db)==80 and len(categories)==40
 rows=[]
 for layout_index,layout in enumerate(LAYOUTS):
  rng=np.random.default_rng(2026112201+layout_index);order=list(categories);rng.shuffle(order)
  for i,category in enumerate(order):
   models=sorted(model for model,row in db.items() if row['semantic_class']==category)
   model=models[(layout_index+categories.index(category))%2];index=len(rows)
   rows.append({'id':f'D{index+1:04d}_{layout}_{category}','model_id':model,'semantic_class':category,'family':category,
     'target_text':db[model]['target_text'],'layout':layout,'seed':2026112220+index*100003})
 root.mkdir();(root/'cases').mkdir();(root/'preflight').mkdir();(root/'gallery').mkdir();(root/'source').mkdir()
 for name in ['asset_catalog.py','asset_environment.py','pipeline.py']:(root/'source'/name).write_bytes((HERE/name).read_bytes())
 save(root/'object_catalog.json',selected)
 reg={'id':'diverse_head200_20261122_01','at':now(),'schedule':rows,'target_models':list(db),
  'semantic_classes':categories,'models_per_class':2,'scenes_per_class':5,'layouts':dict.fromkeys(LAYOUTS,40),
  'target_quotas_by_layout':{l:dict(Counter(e['model_id'] for e in rows if e['layout']==l)) for l in LAYOUTS},
  'source_catalog_sha256':sha(SELECTED),'object_catalog_sha256':sha(root/'object_catalog.json'),
  'distractor_counts':{'S1_open':[0,0],'S2_edge':[0,0],'S3_sparse':[2,3],'S4_clutter':[4,6],'S5_occluded':[2,4]},
  'camera_profile':read(PROJECT/'configs/head_camera_mount.json'),'lift_joint_m':.452,'physical_grasp_budget':0,
  'admission':{'maximum_variants':80,'minimum_target_pixels':64,'valid_depth_fraction':.98,'occlusion_S5':[.20,.40],
    'occlusion_others_max':.15,'all_actors_fully_in_head_frame':True,'IK_or_grasp_outcome_selection':False,'SAM3_selection':False},
  'user_authorization':'Begin local RoboCasa and AI2THOR integration; target 40 classes, 2 models each and five balanced layouts.',
  'source_hashes':{str(p):sha(p) for p in [HERE/'asset_catalog.py',HERE/'asset_environment.py',HERE/'pipeline.py',
    LEGACY/'scene_builder.py',MS/'arx_pipeline/environment.py',MS/'arx_pipeline/baseline.py',MS/'arx_pipeline/scene_tools.py',
    PROJECT/'configs/head_camera_mount.json']}}
 save(root/'registration.json',reg);save(root/'progress.json',{'phase':'registered','completed':0,'registered':200,'physical_grasps':0})

def build(root,resume=False):
 reg=read(root/'registration.json');selected=load_object_catalog(root/'object_catalog.json');db=selected['models']
 for path,digest in reg['source_hashes'].items():assert sha(path)==digest,path
 assert sha(root/'object_catalog.json')==reg['object_catalog_sha256']
 environment,baseline,run,sapien,torch,make,profile,convert,get_maps=runtime(db);maps=get_maps(profile)
 model=reg['target_models'][0];spec=baseline.actor_spec('target',model,[-.35,.05],0.)
 probe=make(model,{'layout':'probe','actors':[spec]})
 try:
  probe.reset(seed=0);baseline.settle(probe);state=run.snapshot(probe,probe.get_obs())
  camera,k=np.asarray(state['camera_world_optical']),np.asarray(state['intrinsics']);save(root/'preflight/head_camera.json',state)
 finally:probe.close()
 completed=[];started=time.perf_counter()
 try:
  for entry in reg['schedule']:
   if (root/'STOP').exists():raise RuntimeError('User STOP')
   existing=root/'cases'/entry['id']/'admission.json'
   if existing.exists():
    assert resume,'Existing scene requires explicit resume';record=read(existing);assert record['passed'];completed.append(record['entry']);continue
   rng=np.random.default_rng(entry['seed']);candidates=scenes.target_candidates(entry,baseline,camera,k,rng)
   models=distractors(entry,reg,baseline,rng);env=None;bound=None;failures=[]
   print('BUILD',len(completed)+1,entry['id'],'placements',len(candidates),flush=True)
   try:
    for variant,target in enumerate(candidates[:80]):
     try:
      scene=scenes.layout_scene(entry,target,models,baseline,camera,k,rng)
      if env is None:env=make(entry['model_id'],scene);env.reset(seed=entry['seed'])
      scenes.set_scene(env,scene,baseline,sapien,torch)
      if entry['layout']=='S5_occluded':
       if not scenes.adjust_blocker(env,scene,baseline,camera,k,rng):raise RuntimeError('No geometric 20-40% occlusion')
       scenes.set_scene(env,scene,baseline,sapien,torch)
      quality,obs,mask=scenes.head_quality(env,baseline)
      if not scenes.accepted(quality,entry['layout']):raise RuntimeError('Head admission: '+str(quality))
      for item in scene['actors']:
       actual={**item,'pose':env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()}
       if scenes.project(actual,baseline,camera,k) is None:raise RuntimeError('Settled actor outside frame: '+item['name'])
      bound=scenes.capture(root,entry,env,scene,quality,obs,mask,run,profile,convert,maps)
      save(root/'cases'/entry['id']/'construction_attempts.json',{'accepted_variant':variant,'rejected':failures});break
     except (RuntimeError,AssertionError) as error:
      if any(token in str(error).lower() for token in ['cuda','memory','cache integrity']):raise
      failures.append({'variant':variant,'reason':str(error)})
    if bound is None:raise RuntimeError('No admitted layout '+entry['id']+str(failures[-2:]))
   finally:
    if env is not None:env.close()
   completed.append(bound);save(root/'progress.json',{'phase':'building','completed':len(completed),'registered':200,
    'last_case':entry['id'],'by_layout':dict(Counter(e['layout'] for e in completed)),
    'elapsed_seconds':time.perf_counter()-started,'physical_grasps':0,'at':now()})
   print('ACCEPTED',len(completed),entry['id'],quality['visible_pixels'],quality['occlusion_fraction'],flush=True)
  scenes.galleries(root,completed)
  save(root/'manifest.json',{'registration_sha256':sha(root/'registration.json'),'schedule':completed,'completed':200,'physical_grasps':0})
  save(root/'progress.json',{'phase':'geometry_complete','completed':200,'registered':200,'by_layout':dict.fromkeys(LAYOUTS,40),'physical_grasps':0,'at':now()})
 except BaseException:
  save(root/'halt.json',{'at':now(),'completed':len(completed),'error':traceback.format_exc()});raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','prepare','build']);parser.add_argument('--root',type=Path,default=DEFAULT)
 parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 if args.stage=='preflight':preflight()
 elif args.stage=='prepare':prepare(args.root.resolve())
 else:build(args.root.resolve(),args.resume)

if __name__=='__main__':main()
