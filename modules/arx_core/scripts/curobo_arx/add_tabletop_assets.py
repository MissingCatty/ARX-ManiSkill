"""Append cached rigid USD objects, preserving authored mass or density.

Does not change geometry or existing catalogue entries. Restart the ARX
diagnostic simulation after adding entries so rigid-body views bind at startup.
"""
import argparse
import json
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def add(names):
    root = Path(__file__).resolve().parents[2]
    cache = root.parent / "asset/tabletop_cache"
    output = root.parent / "asset/tabletop_objects"
    catalog_path = root / "runs/arx_grasp/tabletop_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    existing = {a["name"] for a in catalog}
    additions = []
    for name in names:
        if name in existing:
            raise ValueError(f"Asset already registered: {name}")
        sources = list(cache.rglob(name + ".usd"))
        if len(sources) != 1:
            raise ValueError(f"Expected one source for {name}, found {sources}")
        source = sources[0]
        original = Usd.Stage.Open(str(source))
        assert any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in original.Traverse()), name
        masses = [float(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0)
                  for p in original.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
        densities = [float(UsdPhysics.MassAPI(p).GetDensityAttr().Get() or 0)
                     for p in original.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
        mass = next((m for m in masses if m > 0), None)
        if mass is None and not any(d > 0 for d in densities):
            raise ValueError(f"No authored mass or positive density for {name}")
        destination = output / (name + ".usda")
        if destination.exists():
            raise ValueError(f"Refusing to overwrite {destination}")
        stage = Usd.Stage.CreateNew(str(destination))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        obj = UsdGeom.Xform.Define(stage, "/Object")
        stage.SetDefaultPrim(obj.GetPrim())
        model = UsdGeom.Xform.Define(stage, "/Object/Model")
        stage.DefinePrim("/Object/Model/Source").GetReferences().AddReference(os.path.relpath(source, output))
        for prim in list(stage.Traverse()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        model.AddTranslateOp().Set(Gf.Vec3d(0))
        if UsdGeom.GetStageUpAxis(original) == UsdGeom.Tokens.y:
            model.AddRotateXOp().Set(90)
        model.AddScaleOp().Set(Gf.Vec3f(UsdGeom.GetStageMetersPerUnit(original)))
        bound = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]).ComputeWorldBound(obj.GetPrim()).ComputeAlignedRange()
        size = list(bound.GetSize())
        model.GetOrderedXformOps()[0].Set(-bound.GetMidpoint())
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim()).CreateRigidBodyEnabledAttr(True)
        if mass is not None:
            UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(mass)
        # Density-authored descendants remain under the new rigid body; PhysX
        # calculates mass from them. Do not replace them with a guessed mass.
        stage.GetRootLayer().Save()
        entry = {"name": name, "path": str(destination),
                 "container_path": "/workspace/asset/tabletop_objects/" + destination.name,
                 "size_m": size, "mass_kg": mass, "source": str(source),
                 "mass_policy": "source_mass" if mass is not None else "source_density_auto_mass",
                 "source_densities_kg_m3": [d for d in densities if d > 0]}
        additions.append(entry)
        print(json.dumps(entry), flush=True)
    catalog_path.write_text(json.dumps(catalog + additions, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    add(parser.parse_args().names)
