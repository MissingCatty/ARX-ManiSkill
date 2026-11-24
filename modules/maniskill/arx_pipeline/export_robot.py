"""Export COOLAS-ARX USD to a SAPIEN URDF without editing source assets.

Run with ../curobo/.venv/bin/python (USD bindings are installed there).
Joint frames, inertials and mesh vertices come from USD; drive gains and finger
collision subdivision come from the current COOLAS-ARX runtime configuration.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
ARX = ROOT / "arx_core"
OUT = ROOT / "maniskill/assets/arx"
SOURCE = ARX / "asset/robots/arx_x5_lift/1.0.0/usd/robot.usd"
PREFIX = "/World/scene_0/controllable__lift2__exported_robot"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(pos, quat):
    m = np.eye(4)
    q = np.array([*quat.GetImaginary(), quat.GetReal()])
    # USD uses a zero quaternion as the unauthored principal-axes sentinel.
    m[:3, :3] = Rotation.from_quat(q).as_matrix() if np.linalg.norm(q) > 0 else np.eye(3)
    m[:3, 3] = pos
    return m


def origin(element, t):
    ET.SubElement(element, "origin", xyz=" ".join(map(str, t[:3, 3])),
                  rpy=" ".join(map(str, Rotation.from_matrix(t[:3, :3]).as_euler("xyz"))))


def triangles(polygons):
    return [[p[0], p[i], p[i + 1]] for p in polygons for i in range(1, len(p) - 1)]


def split_faces(points, faces, settings):
    # Same centroid-based overlapping cells as arx_sim/inprocess_app.py.
    axes = ["xyz".index(a) for a in settings["split_axes"]]
    bounds = [(points[:, a].min(), points[:, a].max()) for a in axes]
    counts = [settings["split_segments"]["xyz"[a]] for a in axes]
    centroids = np.array([points[f].mean(0) for f in faces])[:, axes]
    for cell in itertools.product(*[range(n) for n in counts]):
        selected = np.ones(len(faces), dtype=bool)
        for j, (index, n, (lo, hi)) in enumerate(zip(cell, counts, bounds)):
            step = (hi - lo) / n
            selected &= (centroids[:, j] >= lo + index * step - settings["split_overlap_m"])
            selected &= (centroids[:, j] <= lo + (index + 1) * step + settings["split_overlap_m"])
        group = [f for f, keep in zip(faces, selected) if keep]
        if group and len(set(itertools.chain.from_iterable(group))) >= 4:
            yield group


def main():
    if (OUT / "robot.urdf").exists():
        raise FileExistsError("Export already exists; retain it and inspect its manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    stage = Usd.Stage.Open(str(SOURCE))
    cache = UsdGeom.XformCache()
    world = lambda p: np.asarray(cache.GetLocalToWorldTransform(p)).T
    controllers = yaml.safe_load((ARX / "arx-sim/config/controllers.yaml").read_text())["controllers"]
    runtime = yaml.safe_load((ARX / "arx-sim/config/runtime.yaml").read_text())["simulation"]
    robot = ET.Element("robot", name="coolas_arx_x5_lift")
    links = {}
    manifest = {"source": str(SOURCE), "source_sha256": sha(SOURCE), "links": {}, "joints": {},
                "controllers": controllers, "runtime": runtime, "exporter_sha256": sha(__file__),
                "scope": "New ManiSkill task; source robot preserved, engine equivalence not asserted"}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.GetName().startswith("base_footprint"):
            links[p.GetName()] = p
    for name, p in links.items():
        link = ET.SubElement(robot, "link", name=name)
        inertia = UsdPhysics.MassAPI(p)
        mass = inertia.GetMassAttr().Get()
        if mass and mass > 0 and np.isfinite(inertia.GetCenterOfMassAttr().Get()).all():
            item = ET.SubElement(link, "inertial")
            origin(item, matrix(inertia.GetCenterOfMassAttr().Get(), inertia.GetPrincipalAxesAttr().Get()))
            ET.SubElement(item, "mass", value=str(mass))
            d = inertia.GetDiagonalInertiaAttr().Get()
            ET.SubElement(item, "inertia", ixx=str(d[0]), iyy=str(d[1]), izz=str(d[2]), ixy="0", ixz="0", iyz="0")
        rec = {"mass": float(mass or 0), "visuals": [], "collisions": [],
               "disable_gravity": bool(p.GetAttribute("physxRigidBody:disableGravity").Get())}
        for mesh_prim in Usd.PrimRange(p):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(mesh_prim)
            points = np.array(m.GetPointsAttr().Get(), dtype=float)
            if not len(points):
                continue
            indices = list(m.GetFaceVertexIndicesAttr().Get())
            faces, at = [], 0
            for count in m.GetFaceVertexCountsAttr().Get():
                faces.append(indices[at:at + count]); at += count
            transform = np.linalg.inv(world(p)) @ world(mesh_prim)
            vertices = points @ transform[:3, :3].T + transform[:3, 3]
            is_collision = mesh_prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(mesh_prim).GetCollisionEnabledAttr().Get()
            is_visual = "/visuals/" in str(mesh_prim.GetPath())
            for tag, groups in [("visual", [faces] if is_visual else []),
                                ("collision", list(split_faces(points, faces, controllers["gripper_contact"]["replacement_colliders"]))
                                 if is_collision and name in ["right_link27", "right_link28"] else [faces] if is_collision else [])]:
                for part, group in enumerate(groups):
                    filename = f"meshes/{name}_{mesh_prim.GetName()}_{tag}_{part}.stl"
                    trimesh.Trimesh(vertices=vertices, faces=triangles(group), process=False).export(OUT / filename)
                    obj = ET.SubElement(link, tag)
                    geometry = ET.SubElement(obj, "geometry")
                    ET.SubElement(geometry, "mesh", filename=filename)
                    if tag == "visual":
                        material = ET.SubElement(obj, "material", name=f"{name}_color")
                        color = m.GetDisplayColorAttr().Get()
                        rgb = list(color[0]) if color else ([.12, .13, .15] if "link" in name else [.65, .67, .70])
                        ET.SubElement(material, "color", rgba=" ".join(map(str, [*rgb, 1])))
                    rec["visuals" if tag == "visual" else "collisions"].append(filename)
        manifest["links"][name] = rec
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 or not b1 or b0[0].name not in links or b1[0].name not in links:
            continue
        parent, child, name = b0[0].name, b1[0].name, p.GetName()
        kind = "revolute" if p.IsA(UsdPhysics.RevoluteJoint) else "prismatic" if p.IsA(UsdPhysics.PrismaticJoint) else "fixed"
        # A massless intermediate link retains both original USD joint frames.
        virtual = name + "_frame"
        ET.SubElement(robot, "link", name=virtual)
        joint = ET.SubElement(robot, "joint", name=name, type=kind)
        ET.SubElement(joint, "parent", link=parent); ET.SubElement(joint, "child", link=virtual)
        origin(joint, matrix(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()))
        if kind != "fixed":
            axis = np.eye(3)["XYZ".index(p.GetAttribute("physics:axis").Get())]
            ET.SubElement(joint, "axis", xyz=" ".join(map(str, axis)))
            section = "lift" if name == "joint4" else "gripper" if name in controllers["gripper"]["joints"] else "left_gripper" if name in controllers["left_gripper"]["joints"] else "right_arm" if name.startswith("right_") else "left_arm"
            cfg = controllers[section]
            unit = math.pi / 180 if kind == "revolute" else 1
            limits = {"lower": float(p.GetAttribute("physics:lowerLimit").Get()) * unit,
                      "upper": float(p.GetAttribute("physics:upperLimit").Get()) * unit,
                      "velocity": float(p.GetAttribute("physxJoint:maxJointVelocity").Get()) * unit,
                      "effort": cfg["max_effort"]}
            ET.SubElement(joint, "limit", **{k: str(v) for k, v in limits.items()})
            drive = "angular" if kind == "revolute" else "linear"
            manifest["joints"][name] = {"section": section, "limits": limits, "stiffness": cfg["drive_stiffness"],
                                       "damping": cfg["drive_damping"], "drive_mode": p.GetAttribute(f"drive:{drive}:physics:type").Get()}
        fixed = ET.SubElement(robot, "joint", name=name + "_offset", type="fixed")
        ET.SubElement(fixed, "parent", link=virtual); ET.SubElement(fixed, "child", link=child)
        origin(fixed, np.linalg.inv(matrix(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())))
    ET.indent(robot)
    ET.ElementTree(robot).write(OUT / "robot.urdf", encoding="unicode")
    camera = UsdGeom.Camera(stage.GetPrimAtPath(PREFIX + "/head_camera_link/Camera"))
    manifest["head_camera"] = {"width": 640, "height": 480,
        "fx": 640 * camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get(),
        "fy": 480 * camera.GetFocalLengthAttr().Get() / camera.GetVerticalApertureAttr().Get(),
        "near": float(camera.GetClippingRangeAttr().Get()[0]),
        "far": float(camera.GetClippingRangeAttr().Get()[1]),
        "local_gl": (np.linalg.inv(world(links["head_camera_link"])) @ world(camera.GetPrim())).tolist()}
    manifest["files"] = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(OUT), "links": len(links), "joints": list(manifest["joints"]),
                      "finger_collision_parts": {k: len(manifest["links"][k]["collisions"]) for k in ["right_link27", "right_link28"]}}))


if __name__ == "__main__":
    main()
