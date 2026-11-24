"""Import declared local USD/mesh assets into a new immutable training pack.

Mesh conversion requires explicit units, mass, inertia, friction, restitution
and collision representation. Existing assets and historical scenes are never
modified. Incomplete entries are retained in import_receipt.json as unusable.
"""
import argparse
from pathlib import Path
import numpy as np
from .common import digest,read,sha,write


def import_assets(manifest,out):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    import trimesh
    manifest=Path(manifest).resolve();out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    catalog,receipts=[],[]
    for record in read(manifest):
        name=record['name'];receipt={'name':name,'available':False}
        try:
            if not name.replace('_','').isalnum():raise ValueError('Unsafe asset identifier')
            path=(manifest.parent/record['path']).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            if not record.get('source') or not record.get('usage_conditions'):raise ValueError('Source and usage conditions required')
            if path.suffix.lower() in ('.usd','.usda','.usdc'):
                stage=Usd.Stage.Open(str(path))
                if not stage.GetDefaultPrim() or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9:
                    raise ValueError('USD requires a default prim and explicitly registered metre units')
                if not record.get('size_m'):raise ValueError('USD dimensions must be declared')
                target=path
            else:
                physics=record['physics']
                scale=float(record['meters_per_unit'])
                if not 0<scale<100:raise ValueError('Invalid mesh units')
                for key in ('mass_kg','diagonal_inertia_kg_m2','center_of_mass_m','principal_axes_wxyz',
                            'static_friction','dynamic_friction','restitution','collision_approximation'):
                    if key not in physics:raise ValueError('Explicit new-asset physical setting required: '+key)
                mesh=trimesh.load(path,force='mesh',process=False)
                vertices=np.asarray(mesh.vertices)*scale
                if not len(vertices) or not np.isfinite(vertices).all():raise ValueError('Invalid/empty mesh')
                target=out/(name+'.usda')
                stage=Usd.Stage.CreateNew(str(target));UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
                root=UsdGeom.Xform.Define(stage,'/Object').GetPrim();stage.SetDefaultPrim(root)
                UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
                mass=UsdPhysics.MassAPI.Apply(root)
                mass.CreateMassAttr(float(physics['mass_kg']))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*physics['diagonal_inertia_kg_m2']))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(*physics['center_of_mass_m']))
                q=physics['principal_axes_wxyz'];mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
                shape=UsdGeom.Mesh.Define(stage,'/Object/Geometry')
                shape.CreatePointsAttr(vertices.tolist());shape.CreateFaceVertexCountsAttr([3]*len(mesh.faces));shape.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist())
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr(physics['collision_approximation'])
                material=UsdShade.Material.Define(stage,'/Object/PhysicsMaterial')
                api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                api.CreateStaticFrictionAttr(float(physics['static_friction']));api.CreateDynamicFrictionAttr(float(physics['dynamic_friction']));api.CreateRestitutionAttr(float(physics['restitution']))
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material,materialPurpose='physics')
                stage.GetRootLayer().Save()
                record={**record,'size_m':np.ptp(vertices,axis=0).tolist(),'mass_kg':float(physics['mass_kg'])}
            catalog.append({**record,'path':str(target),'container_path':str(target),'registered_original_sha256':sha(path)})
            receipt.update(available=True,usd_path=str(target),sha256=sha(target))
        except (OSError,KeyError,ValueError,RuntimeError) as exc:
            receipt['reason']=str(exc)
            catalog.append({**record,'path':str((manifest.parent/record['path']).resolve()),
                            'import_failure':str(exc)})
        receipts.append(receipt)
    write(out/'catalog.json',catalog)
    write(out/'import_receipt.json',{'manifest_sha256':sha(manifest),'records':receipts,'near_shape_deduplication':'Not performed; composed exact geometry grouping occurs in production prepare'})
    return catalog


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();print(len(import_assets(a.manifest,a.out)))
