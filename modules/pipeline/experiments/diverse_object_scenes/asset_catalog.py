"""Source-preserving rigid-object recipes for RoboCasa and AI2THOR."""
from pathlib import Path
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
PROJECT=HERE.parents[1]
MS=PROJECT.parent/'maniskill'
ROBO=MS/'assets/data/scene_datasets/robocasa_dataset/assets/objects/objaverse'
THOR=MS/'assets/data/scene_datasets/ai2thor/ai2thorhab-uncompressed'
CATALOG=HERE/'assets/catalog.json'
ROBO_CLASSES=['apple','banana','orange','pear','peach','lemon','mango','avocado',
 'bell_pepper','carrot','corn','cucumber','eggplant','mushroom','onion','potato','tomato','broccoli',
 'bread','baguette','croissant','donut','canned_food','boxed_food','bottled_water','milk','yogurt',
 'mug','bowl','soap_dispenser','bar_soap','spatula','scissors','teapot','candle','sponge']
THOR_CLASSES={'book':'Book','remote_control':'RemoteControl','cell_phone':'CellPhone','keychain':'KeyChain'}
TEXT={'bell_pepper':'bell pepper','canned_food':'canned food','boxed_food':'food box',
 'bottled_water':'water bottle','soap_dispenser':'soap dispenser','bar_soap':'bar of soap',
 'remote_control':'remote control','cell_phone':'cell phone','alarm_clock':'alarm clock'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from arx_project.layout import read_json as read
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.pending');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def quat(q):
 q=np.array(q,dtype=float);return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
def pose(element):
 m=np.eye(4);m[:3,3]=np.fromstring(element.get('pos','0 0 0'),sep=' ')
 if 'quat' in element.attrib:m[:3,:3]=quat(np.fromstring(element.get('quat'),sep=' '))
 if any(k in element.attrib for k in ['euler','axisangle','xyaxes','zaxis']):raise ValueError('Unsupported source orientation')
 return m
def loaded(path):
 obj=trimesh.load(str(path),process=False)
 return obj.to_geometry() if isinstance(obj,trimesh.Scene) else obj
def transformed(recipe):
 mesh=loaded(recipe['file']);v=np.asarray(mesh.vertices)*recipe['scale'];t=np.asarray(recipe['transform'])
 return v@t[:3,:3].T+t[:3,3]
def finish(row):
 points=np.concatenate([transformed(r) for r in row['visuals']])
 lo,hi=points.min(0),points.max(0);size=hi-lo
 # The source metric scale is preserved; large furniture and sub-mm surfaces are excluded.
 if not np.isfinite(points).all() or size.max()>.32 or size.min()<.002 or np.sort(size)[1]<.015 or size[2]>.24:
  raise ValueError('Outside tabletop asset dimensions: '+str(size.tolist()))
 row.update(bbox={'min':lo.tolist(),'max':hi.tolist()},scales=[1.],size_m=size.tolist(),
  target_text=TEXT.get(row['semantic_class'],row['semantic_class'].replace('_',' ')))
 row['source_hashes']={str(p):sha(p) for p in row.pop('source_files')}
 row['raw_visual_sha256']=hashlib.sha256(''.join(sorted(sha(r['file']) for r in row['visuals'])).encode()).hexdigest()
 row['raw_collision_sha256']=hashlib.sha256(''.join(sorted(sha(r['file']) for r in row['collisions'])).encode()).hexdigest()
 return row

def geometry_signature(row):
 triangles=[]
 for item in row['visuals']:
  mesh=loaded(item['file']);vertices=transformed(item);triangles.append(vertices[np.asarray(mesh.faces)])
 triangles=np.concatenate(triangles)
 edges=np.stack([np.linalg.norm(triangles[:,i]-triangles[:,(i+1)%3],axis=1) for i in range(3)],axis=1)
 # Material, translation, rotation and uniform scale do not create a new shape.
 edges=np.sort(edges,axis=1);edges/=np.max(edges)
 unique=np.unique(np.rint(edges*100000).astype(np.int64),axis=0)
 return hashlib.sha256(unique.tobytes()).hexdigest()

def robocasa(path,category):
 xml=ET.parse(path).getroot();assets={m.get('name'):m for m in xml.findall('./asset/mesh')}
 visuals=[];collisions=[];source_files={path}
 def visit(body,parent):
  world=parent@pose(body)
  for g in body.findall('geom'):
   if g.get('type')=='box' and g.get('contype','1')!='0':
    size=np.fromstring(g.get('size'),sep=' ');name=hashlib.sha256(size.tobytes()).hexdigest()[:20]
    file=HERE/'assets/primitive_collisions'/(name+'.ply');file.parent.mkdir(exist_ok=True)
    if not file.exists():trimesh.creation.box(extents=2*size).export(str(file))
    source_files.add(file)
    collisions.append({'file':str(file),'scale':[1.,1.,1.],'transform':(world@pose(g)).tolist(),
      'density':float(g.get('density','1000')),'friction':float(g.get('friction','.5 0 0').split()[0]),'multiple':False,
      'source_primitive':'box','source_half_size':size.tolist()})
    continue
   if g.get('type')!='mesh':raise ValueError('Unsupported source geom: '+str(g.get('type')))
   a=assets[g.get('mesh')];file=(path.parent/a.get('file')).resolve();source_files.add(file)
   # MuJoCo specifies inverse refquat after source scale and refpos subtraction.
   ref=np.eye(4);ref[:3,:3]=quat(np.fromstring(a.get('refquat','1 0 0 0'),sep=' ')).T
   ref[:3,3]=-ref[:3,:3]@np.fromstring(a.get('refpos','0 0 0'),sep=' ')
   scale=np.fromstring(a.get('scale','1 1 1'),sep=' ');scale=np.repeat(scale,3) if len(scale)==1 else scale
   item={'file':str(file),'scale':scale.tolist(),'transform':(world@pose(g)@ref).tolist()}
   if g.get('contype','1')=='0' and g.get('conaffinity','1')=='0':visuals.append(item)
   else:
    item.update(density=float(g.get('density','1000')),friction=float(g.get('friction','.5 0 0').split()[0]),multiple=False)
    collisions.append(item)
  for child in body.findall('body'):visit(child,world)
 for body in xml.findall('./worldbody/body'):visit(body,np.eye(4))
 if not visuals or not collisions:raise ValueError('Missing source visual/collision')
 for p in path.parent.joinpath('visual').iterdir():
  if p.is_file():source_files.add(p)
 return finish({'model_id':'robocasa__'+category+'__'+path.parent.name,'source':'robocasa',
  'source_definition':str(path),'semantic_class':category,'visuals':visuals,'collisions':collisions,
  'source_files':source_files,'physics_provenance':'Source MJCF density and sliding friction; restitution 0 in PhysX.',
  'collision_provenance':'Original separate convex OBJ pieces; no whole-object convex hull.'})

def ai2thor(path,category):
 config=read(path);visual=(path.parent/config['render_asset']).resolve();mesh=loaded(visual)
 rotation=Rotation.from_euler('x',90,degrees=True).as_matrix();transform=np.eye(4);transform[:3,:3]=rotation
 # Reject unsuitable sizes before invoking a missing decomposition.
 points=np.asarray(mesh.vertices)@rotation.T;size=np.ptp(points,axis=0)
 if size.max()>.32 or size.min()<.002 or np.sort(size)[1]<.015 or size[2]>.24:raise ValueError('Outside tabletop dimensions: '+str(size))
 collision=Path(str(visual)+'.coacd.ply');provenance='Existing ManiSkill CoACD decomposition.'
 source_files={path,visual}
 if not collision.exists():
  collision=HERE/'assets/ai2thor_collisions'/(visual.stem+'.ply')
  if not collision.exists():
   import coacd
   coacd.set_log_level('error')
   parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32)),
      threshold=.04,max_convex_hull=32,preprocess_mode='auto',resolution=1000,mcts_nodes=20,mcts_iterations=100,seed=20261122)
   if not parts:raise ValueError('Empty CoACD decomposition')
   collision.parent.mkdir(exist_ok=True)
   trimesh.util.concatenate([trimesh.Trimesh(v,f,process=False) for v,f in parts]).export(str(collision))
  provenance='Offline CoACD: threshold .04, max 32 hulls, resolution 1000, mcts_nodes 20, mcts_iterations 100, seed 20261122.'
 source_files.add(collision)
 recipe={'scale':[1.,1.,1.],'transform':transform.tolist()}
 return finish({'model_id':'ai2thor__'+category+'__'+visual.stem,'source':'ai2thor','source_definition':str(path),
  'semantic_class':category,'visuals':[{'file':str(visual),**recipe}],
  'collisions':[{'file':str(collision),**recipe,'density':100.,'friction':.5,'multiple':True}],
  'source_files':source_files,'physics_provenance':'No source mass/friction: fixed simulation assumptions density 100 kg/m3, sliding friction .5, restitution 0.',
  'collision_provenance':provenance})

def prepare():
 assert len(ROBO_CLASSES)+len(THOR_CLASSES)==40
 if CATALOG.exists():raise RuntimeError('Prepared catalog exists; preserve it')
 candidates={};rejected=[];seen=set()
 mapping=read(MS/'assets/data/scene_datasets/ai2thor/ai2thor-hab/configs/object_semantic_id_mapping.json')
 for category in ROBO_CLASSES+list(THOR_CLASSES):
  candidate_limit=4 if category in ROBO_CLASSES else 2
  cache=HERE/'assets/candidate_classes'/(category+'.json')
  candidates[category]=[]
  if cache.exists():
   rows=read(cache)
   for row in rows:
    for file,digest in row['source_hashes'].items():assert sha(file)==digest,file
    row['source_geometry_sha256']=row.get('source_geometry_sha256') or geometry_signature(row)
    if row['source_geometry_sha256'] in seen:continue
    seen.add(row['source_geometry_sha256']);candidates[category].append(row)
   save(cache,candidates[category])
   if len(candidates[category])>=candidate_limit:continue
  paths=sorted((ROBO/category).glob('*/model.xml')) if category in ROBO_CLASSES else [p for p in sorted((THOR/'configs/objects').glob('*.object_config.json')) if read(p).get('semantic_id')==mapping[THOR_CLASSES[category]]]
  cached_sources={r['source_definition'] for r in candidates[category]}
  for path in paths:
   if str(path) in cached_sources:continue
   try:
    row=robocasa(path,category) if category in ROBO_CLASSES else ai2thor(path,category)
    row['source_geometry_sha256']=geometry_signature(row)
    if row['source_geometry_sha256'] in seen:raise ValueError('Duplicate geometry independent of material/rigid pose/uniform scale')
    seen.add(row['source_geometry_sha256']);candidates[category].append(row)
    print('ASSET',category,len(candidates[category]),row['model_id'],flush=True)
    if len(candidates[category])>=candidate_limit:break
   except (ValueError,RuntimeError,OSError) as error:rejected.append({'category':category,'source':str(path),'reason':str(error)})
  save(HERE/'assets/preparation_progress.json',{'completed_categories':len(candidates),'candidates':{k:len(v) for k,v in candidates.items()},'rejected':rejected})
  save(cache,candidates[category])
  if len(candidates[category])<2:raise RuntimeError('Fewer than two candidate models: '+category)
 save(CATALOG,{'categories':ROBO_CLASSES+list(THOR_CLASSES),'candidate_models':candidates,'rejected':rejected,
  'user_authorization':'2026-11-22: begin asset integration and 40-category, 2-model, five-layout 200-scene construction.',
  'physical_grasps':0,'source_hashes':{str(Path(__file__)):sha(Path(__file__))}})
 print('CATALOG_READY',sum(map(len,candidates.values())),flush=True)

if __name__=='__main__':prepare()
