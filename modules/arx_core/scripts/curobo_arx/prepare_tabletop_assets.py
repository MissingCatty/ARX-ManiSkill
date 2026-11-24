"""Normalize eight cached tabletop USD assets without changing source geometry."""
from pathlib import Path
import json,os
from pxr import Usd,UsdGeom,UsdPhysics,Gf
root=Path(__file__).resolve().parents[2];out=root.parent/'asset/tabletop_objects';out.mkdir(exist_ok=True)
entries=json.loads((root/'runs/arx_grasp/asset_inventory.json').read_text())
names=['rubiks_cube','005_tomato_soup_can','raisin_box','butter','yogurt_cup','bbq_sauce_bottle','lemon_01','remote_control'];catalog=[]
for name in names:
 choices=[e for e in entries if e['name']==name and e['rigid_bodies']]
 e=choices[0];original=Path(e['path']);src=root.parent/'asset/tabletop_cache'/str(original).split('arena_robolab_official_cache/')[1]
 original_stage=Usd.Stage.Open(str(src));dest=out/(name+'.usda');stage=Usd.Stage.CreateNew(str(dest)) if not dest.exists() else Usd.Stage.Open(str(dest));stage.GetRootLayer().Clear()
 UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
 obj=UsdGeom.Xform.Define(stage,'/Object');stage.SetDefaultPrim(obj.GetPrim());model=UsdGeom.Xform.Define(stage,'/Object/Model')
 # Preserve source root transforms under a separate normalization transform.
 child=stage.DefinePrim('/Object/Model/Source');child.GetReferences().AddReference(os.path.relpath(src,out))
 for prim in list(stage.Traverse()):
  if prim.IsInstance():prim.SetInstanceable(False)
 for prim in list(stage.Traverse()):
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
 unit=UsdGeom.GetStageMetersPerUnit(original_stage)
 model.AddTranslateOp().Set(Gf.Vec3d(0))
 if UsdGeom.GetStageUpAxis(original_stage)==UsdGeom.Tokens.y:model.AddRotateXOp().Set(90)
 model.AddScaleOp().Set(Gf.Vec3f(unit))
 bounds=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy]).ComputeWorldBound(obj.GetPrim()).ComputeAlignedRange();center=bounds.GetMidpoint();size=list(bounds.GetSize())
 model.GetOrderedXformOps()[0].Set(-center)
 UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim()).CreateRigidBodyEnabledAttr(True)
 # Retain an explicit source mass where one exists, otherwise use a documented 50 g test mass.
 masses=[UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in original_stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
 mass=next((float(m) for m in masses if m and m>0),.05)
 UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
 stage.GetRootLayer().Save();catalog.append({'name':name,'path':str(dest),'container_path':'/workspace/asset/tabletop_objects/'+dest.name,'size_m':size,'mass_kg':mass,'source':str(src)})
(root/'runs/arx_grasp/tabletop_catalog.json').write_text(json.dumps(catalog,indent=2));print(json.dumps(catalog,indent=2))
