"""Continue unbuilt layouts using wide/combined physical occluders."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

import pipeline as frozen
from asset_catalog import HERE,PROJECT,read,save,sha

original_distractors=frozen.distractors

def wide_distractors(entry,reg,baseline,rng):
 original=original_distractors(entry,reg,baseline,rng)
 if entry['layout']!='S5_occluded':return original
 rows=[]
 for model in reg['target_models']:
  r=baseline.DB[model]
  if r['semantic_class']==entry['semantic_class']:continue
  size=np.asarray(r['size_m'])
  # Upright silhouette area, rather than slenderness, matters for wide targets.
  score=float(size[2]*max(size[:2]))*float(rng.uniform(.9,1.1))
  rows.append((score,model))
 chosen=[];classes=set()
 for _,model in sorted(rows,reverse=True):
  category=baseline.DB[model]['semantic_class']
  if category in classes:continue
  chosen.append(model);classes.add(category)
  if len(chosen)==len(original):break
 return chosen

def combined_occlusion(env,scene,baseline,camera,k,rng):
 import sapien
 scenes=frozen.scenes
 for item in scene['actors']:
  item['pose']=env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
 target=next(a for a in scene['actors'] if a['name']=='target')
 lo,hi=baseline.bounds(target);center=(lo+hi)/2
 direction=camera[:2,3]-center[:2];forward=direction/np.linalg.norm(direction);side=np.array([-forward[1],forward[0]])
 target_projection=scenes.project(target,baseline,camera,k)
 if target_projection is None:return False
 for blocker in scene['actors'][1:]:
  others=[a for a in scene['actors'] if a['name']!=blocker['name']]
  size=np.asarray(baseline.DB[blocker['model_id']]['size_m'])
  base_yaw=float(np.arctan2(forward[1],forward[0]))
  candidates=[]
  for yaw in [base_yaw,base_yaw+np.pi/2]:
   for along in np.linspace(.06,.50,12):
    for lateral in np.linspace(-.32,.32,65):
     spec=baseline.actor_spec(blocker['name'],blocker['model_id'],center[:2]+along*forward+lateral*side,yaw)
     if not scenes.nonoverlap(spec,others,baseline,.008):continue
     projection=scenes.project(spec,baseline,camera,k)
     if projection is None:continue
     overlap=np.maximum(0.,np.minimum(projection['hi'],target_projection['hi'])-np.maximum(projection['lo'],target_projection['lo']))
     fraction=float(np.prod(overlap)/target_projection['area'])
     if .04<=fraction<=.90:candidates.append((abs(fraction-.30),spec))
  candidates.sort(key=lambda pair:pair[0])
  best=None
  # The scene keeps the best sub-threshold occlusion before moving another actor.
  for _,spec in candidates[:100]:
   actor=env.objects[spec['name']];p=spec['pose'];actor.set_pose(sapien.Pose(p[:3],p[3:]))
   actor._bodies[0].linear_velocity=np.zeros(3);actor._bodies[0].angular_velocity=np.zeros(3)
   quality,_,_=scenes.head_quality(env,baseline)
   fraction=quality['occlusion_fraction']
   if scenes.accepted(quality,'S5_occluded'):
    blocker['pose']=spec['pose'];return True
   if fraction<.20 and (best is None or fraction>best[0]):best=(fraction,deepcopy(spec))
  if best is not None:blocker['pose']=best[1]['pose']
  p=blocker['pose'];actor=env.objects[blocker['name']];actor.set_pose(sapien.Pose(p[:3],p[3:]))
  actor._bodies[0].linear_velocity=np.zeros(3);actor._bodies[0].angular_velocity=np.zeros(3)
 return False

def main():
 root=frozen.DEFAULT;revision=root/'construction_revisions/wide_occluders_01';revision.mkdir(parents=True,exist_ok=True)
 receipt=revision/'registration.json'
 if not receipt.exists():
  existing={str(p):sha(p) for case in (root/'cases').iterdir() if (case/'admission.json').exists() for p in case.rglob('*') if p.is_file()}
  save(receipt,{'at':datetime.now().astimezone().isoformat(),'source':str(Path(__file__)),'source_sha256':sha(Path(__file__)),
    'original_registration_sha256':sha(root/'registration.json'),'existing_case_file_hashes':existing,
    'change':'For unbuilt S5 scenes prefer broad silhouettes and search combined occlusion from multiple physical distractors. Existing target models, layouts, camera, physics and 20-40% occlusion gate unchanged.',
    'physical_grasps':0,'new_scene_slots':0})
  if (root/'halt.json').exists():(root/'halt.json').rename(revision/'previous_halt.json')
 spec=read(receipt);assert spec['source_sha256']==sha(Path(__file__))
 frozen.distractors=wide_distractors;frozen.scenes.adjust_blocker=combined_occlusion
 started=time.perf_counter()
 save(HERE/'campaign_progress.json',{'stage':'building_remaining_occlusion_layouts','physical_grasps':0,'completed_before_resume':165})
 try:
  frozen.build(root,resume=True)
  for path,digest in spec['existing_case_file_hashes'].items():assert sha(path)==digest,path
  import verify_dataset
  verify_dataset.verify(root)
  active={'dataset':str(root),'object_catalog':str(root/'object_catalog.json'),'environment_id':'ARXPickCatalog-v1',
    'adapter':str(HERE/'asset_environment.py'),'runtime_factory':str(HERE/'pipeline.py')+':runtime',
    'scenes':200,'categories':40,'models':80,'state':'verified','grasp_tests_authorized':False}
  save(HERE/'active_dataset.json',active);save(PROJECT/'configs/scene_dataset.json',active)
  mainline=read(PROJECT/'configs/mainline.json');mainline['scene_dataset_config']='configs/scene_dataset.json';save(PROJECT/'configs/mainline.json',mainline)
  save(HERE/'campaign_progress.json',{'stage':'complete','passed':True,'completed':200,'categories':40,'models':80,
    'existing_165_cases_unchanged':True,'continuation_wall_seconds':time.perf_counter()-started,'physical_grasps':0})
 except BaseException:
  save(revision/'halt.json',{'error':traceback.format_exc(),'physical_grasps':0});raise

if __name__=='__main__':main()
