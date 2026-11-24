"""Import the downloaded YCB 16k tennis-ball mesh as a rigid ARX test prop.

Source geometry and UVs retain native metre scale. Physics uses a convex hull
and the 58 g mass reported for the tennis ball in the YCB ICAR paper, Table I.
This does not model felt, shell deformation, or measured contact parameters.
"""
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def main():
    root = Path(__file__).resolve().parents[2]
    source = root.parent / "asset/ycb_downloads/056_tennis_ball/google_16k/textured.obj"
    destination = root.parent / "asset/tabletop_objects/ycb_056_tennis_ball.usda"
    catalog_path = root / "runs/arx_grasp/tabletop_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    name = "ycb_056_tennis_ball"
    if destination.exists() or any(a["name"] == name for a in catalog):
        raise ValueError("Tennis ball already imported")
    data = trimesh.load(source, force="mesh", process=False)
    size = data.bounds[1] - data.bounds[0]
    assert np.all((size > .05) & (size < .08)), ("Unexpected native dimensions", size)
    points = data.vertices - data.bounds.mean(axis=0)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    obj = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(obj.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(.058)
    mesh = UsdGeom.Mesh.Define(stage, "/Object/Geometry")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(data.faces))
    mesh.CreateFaceVertexIndicesAttr(data.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(data.vertex_normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(*(-size / 2)), Gf.Vec3f(*(size / 2))])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                          UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(data.visual.uv, dtype=np.float32)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    material = UsdShade.Material.Define(stage, "/Object/Material")
    shader = UsdShade.Shader.Define(stage, "/Object/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
    texture = UsdShade.Shader.Define(stage, "/Object/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(os.path.relpath(source.parent / "texture_map.png", destination.parent)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    st = UsdShade.Shader.Define(stage, "/Object/Material/UV")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    entry = {"name": name, "path": str(destination),
             "container_path": "/workspace/asset/tabletop_objects/" + destination.name,
             "size_m": size.tolist(), "mass_kg": .058, "source": str(source),
             "source_page": "https://ycb-benchmarks.s3.amazonaws.com/index.html",
             "mass_source": "https://www.ri.cmu.edu/pub_files/2015/7/ICAR-FINAL.pdf",
             "mass_policy": "YCB paper tennis-ball mass, 58 g", "license": "CC BY 4.0",
             "physics_model": "Rigid convex hull of native-scale mesh; default USD contact material"}
    catalog.append(entry)
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
