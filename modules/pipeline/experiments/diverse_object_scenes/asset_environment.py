"""An independent ARX table environment with dataset-neutral object recipes."""
import numpy as np
import sapien

from asset_catalog import MS, read, CATALOG
import environment as original
from mani_skill.utils.registration import register_env
from arx_project.assets import AssetStore, resource_path
from mani_skill.utils.building.ground import build_ground

def candidate_db():
 store=AssetStore()
 return {name:store.model(name) for name in store.data['models']}

def make_builder(scene,record):
 builder=scene.create_actor_builder()
 for item in record['visuals']:
  builder.add_visual_from_file(item['file'],scale=item['scale'],pose=sapien.Pose(np.asarray(item['transform'])))
 for item in record['collisions']:
  material=sapien.physx.PhysxMaterial(item['friction'],item['friction'],0.)
  method=builder.add_multiple_convex_collisions_from_file if item['multiple'] else builder.add_convex_collision_from_file
  method(filename=item['file'],scale=item['scale'],pose=sapien.Pose(np.asarray(item['transform'])),
    material=material,density=item['density'])
 return builder

@register_env('ARXPickCatalog-v1',max_episode_steps=3000)
class ARXPickCatalog(original.ARXPickYCB):
 def __init__(self,*args,object_catalog=None,**kwargs):
  self.catalog=object_catalog if object_catalog is not None else candidate_db()
  super().__init__(*args,**kwargs)

 def _load_scene(self,options):
  builder=self.scene.create_actor_builder()
  height=original.TABLE_HEIGHT
  builder.add_box_collision(pose=sapien.Pose([0,0,height/2]),half_size=[2.418/2,1.209/2,height/2])
  table_file=resource_path('workspace_table')
  builder.add_visual_from_file(str(table_file),scale=[1.75,1.75,1.75*height/original.ORIGINAL_TABLE_HEIGHT],
    pose=sapien.Pose(q=[np.sqrt(.5),0,0,np.sqrt(.5)]))
  builder.initial_pose=sapien.Pose([-.12,0,-height],[np.sqrt(.5),0,0,np.sqrt(.5)])
  self.table=builder.build_kinematic(name='table-workspace')
  self.ground=build_ground(self.scene,altitude=-height)
  self.objects,self.object_models,self.object_info={},{},{}
  specs=self.scene_spec['actors'] if self.scene_spec else [{'name':'target','model_id':self.model_id}]
  for spec in specs:
   row=self.catalog[spec['model_id']];builder=make_builder(self.scene,row);p=spec.get('pose',[-.4,0,.05,1,0,0,0])
   builder.initial_pose=sapien.Pose(p[:3],p[3:]);name=spec['name']
   self.objects[name]=builder.build(name=name);self.object_models[name]=spec['model_id'];self.object_info[name]=row
  self.obj=self.objects['target'];self.object_bbox=self.catalog[self.model_id]['bbox'];self.object_scale=1.
