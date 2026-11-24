"""Prepare deterministic common-object assets and calibrated ARX observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
RUN = Path(__file__).parent / "runs/five_methods_50_20261122_01"
ASSETS = WORKSPACE / "maniskill/assets/data/scene_datasets/robocasa_dataset/assets/objects/objaverse"
CATEGORIES = {
    "mug": ("马克杯", ["mug", "coffee cup"]),
    "water_bottle": ("水瓶", ["water bottle", "bottle"]),
    "canned_food": ("食品罐头", ["food can", "can"]),
    "boxed_food": ("盒装食品", ["food box", "box"]),
    "milk": ("牛奶包装", ["milk carton", "milk bottle", "milk container"]),
    "ketchup": ("番茄酱瓶", ["ketchup bottle", "bottle"]),
    "bowl": ("碗", ["bowl"]),
    "apple": ("苹果", ["apple"]),
    "spoon": ("勺子", ["spoon"]),
    "scissors": ("剪刀", ["scissors"]),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def pose(element, degrees=False):
    out = np.eye(4)
    out[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    if element.get("quat"):
        q = np.fromstring(element.get("quat"), sep=" ")
        out[:3, :3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    elif element.get("euler"):
        out[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("euler"), sep=" "), degrees=degrees).as_matrix()
    elif any(element.get(n) for n in ["axisangle", "xyaxes", "zaxis"]):
        raise ValueError("Unsupported pose representation")
    return out


def convert_asset(source, target):
    xml_path = source / "model.xml"
    root = ET.parse(xml_path).getroot()
    meshes = {m.get("name"):m for m in root.findall("./asset/mesh")}
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") == "degree"
    scene = trimesh.Scene()
    geometry = []

    def visit(node, parent):
        transform = parent @ pose(node, degrees) if node.tag == "body" else parent
        for geom in node.findall("geom"):
            if geom.get("group") != "1":
                continue
            spec = meshes[geom.get("mesh")]
            mesh_path = source / spec.get("file")
            reference = np.eye(4)
            if spec.get("refquat"):
                q = np.fromstring(spec.get("refquat"), sep=" ")
                reference[:3, :3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            reference[:3, 3] = np.fromstring(spec.get("refpos", "0 0 0"), sep=" ")
            scale = np.eye(4)
            scale[np.arange(3), np.arange(3)] = np.fromstring(spec.get("scale", "1 1 1"), sep=" ")
            # Reference-frame compensation and native physical scale are baked
            # before the geom/body pose. The original assets remain untouched.
            matrix = transform @ pose(geom, degrees) @ scale @ np.linalg.inv(reference)
            loaded = trimesh.load(mesh_path, force="scene", process=False)
            for name in loaded.graph.nodes_geometry:
                local, key = loaded.graph[name]
                part = loaded.geometry[key].copy()
                part.apply_transform(matrix @ local)
                geometry.append(part)
                scene.add_geometry(part, node_name=f"part_{len(geometry)}")
        for child in node.findall("body"):
            visit(child, transform)

    visit(root.find("worldbody"), np.eye(4))
    if not geometry:
        raise RuntimeError(f"No visible meshes in {xml_path}")
    joined = trimesh.util.concatenate(geometry)
    assert np.isfinite(joined.vertices).all() and joined.area > 0
    target.mkdir(parents=True, exist_ok=True)
    scene.export(target / "visual.glb")
    joined.export(target / "reference_local.ply")
    hashes = {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*'))
              if p.is_file() and p.suffix.lower() in ('.xml','.obj','.mtl','.png','.jpg','.jpeg') and 'collision' not in p.parts}
    return {"source_directory": str(source), "source_files":hashes,
            "visual_path":str(target / "visual.glb"), "reference_path":str(target / "reference_local.ply"),
            "visual_sha256":sha(target / "visual.glb"), "reference_sha256":sha(target / "reference_local.ply"),
            "bounds_m":joined.bounds.tolist(), "vertices":len(joined.vertices), "faces":len(joined.faces),
            "appearance_policy":"native OBJ/MTL textures", "xml_reference_transform_applied":True}


def prepare_assets():
    manifest_path = RUN / "scene_selection.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for case in manifest['cases'] + [manifest['warmup']]:
            assert sha(case['visual_path']) == case['visual_sha256']
        print('Existing 50-scene selection retained', flush=True)
        return manifest
    cases = []
    for category, (label, prompts) in CATEGORIES.items():
        models = sorted([p for p in (ASSETS/category).iterdir() if (p/'model.xml').exists()],
                        key=lambda p:int(p.name.rsplit('_',1)[1]))[:5]
        assert len(models) == 5
        for style_index, source in enumerate(models):
            case_id = f'C{len(cases)+1:03d}'
            record = convert_asset(source, RUN/'assets'/case_id)
            cases.append({"id":case_id,"category":category,"category_cn":label,"style_id":source.name,
                          "style_index":style_index,"prompts":prompts, **record})
            print('Asset',case_id,category,source.name,flush=True)
    warmup_source = sorted((ASSETS/'plate').glob('plate_*'), key=lambda p:int(p.name.rsplit('_',1)[1]))[0]
    warmup = {"id":"warmup","category":"plate","style_id":warmup_source.name,"prompts":["plate"],
              **convert_asset(warmup_source, RUN/'assets/warmup')}
    manifest = {"scene_count":50,"category_count":10,"styles_per_category":5,"seed":20261122,
                "selection":"first five available numeric model IDs in each predeclared category; not selected on reconstruction outputs",
                "scene_type":"static single-object tabletop observation with real distinct asset styles",
                "scope":"style variation robustness; pretraining membership unknown, not a proven unseen-data split",
                "robot":"ARX","camera":"head, D405 calibration with downward 45-degree mounting override",
                "target_xy_m":[-.48,.20],"yaw_rad":0.,"physical_grasp_trials":0,
                "cases":cases,"warmup":warmup}
    write(manifest_path,manifest)
    return manifest


def capture(manifest):
    os.environ.setdefault('MS_ASSET_DIR',str(WORKSPACE/'maniskill/assets'))
    sys.path.insert(0,str(WORKSPACE/'maniskill/arx_pipeline'))
    import sapien
    import gymnasium as gym
    import torch
    from environment import ARXPickYCB, ARX, META, INITIAL_JOINTS, BASE_POSE, CURRENT_CONTROLLERS, TABLE_HEIGHT, ORIGINAL_TABLE_HEIGHT, table_source, build_ground
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh
    from head_camera_config import head_configuration
    from camera_imaging import camera_profiles, intrinsic_matrix, lens_maps, sensor_and_model_frames

    @register_env('ARXPointCloudObservation-v1',max_episode_steps=2)
    class Observation(ARXPickYCB):
        def _load_scene(self,options):
            builder=self.scene.create_actor_builder()
            builder.add_box_collision(pose=sapien.Pose([0,0,TABLE_HEIGHT/2]),half_size=[2.418/2,1.209/2,TABLE_HEIGHT/2])
            builder.add_visual_from_file(str(Path(table_source.__file__).parent/'assets/table.glb'),scale=[1.75,1.75,1.75*TABLE_HEIGHT/ORIGINAL_TABLE_HEIGHT],pose=sapien.Pose(q=[np.sqrt(.5),0,0,np.sqrt(.5)]))
            builder.initial_pose=sapien.Pose([-.12,0,-TABLE_HEIGHT],[np.sqrt(.5),0,0,np.sqrt(.5)])
            self.table=builder.build_kinematic(name='table-workspace')
            self.ground=build_ground(self.scene,altitude=-TABLE_HEIGHT)
            self.targets={}
            for i,case in enumerate(manifest['cases']+[manifest['warmup']]):
                builder=self.scene.create_actor_builder()
                builder.add_visual_from_file(case['visual_path'])
                builder.initial_pose=sapien.Pose([10+i,10,-10])
                self.targets[case['id']]=builder.build_kinematic(name=case['id'])
            self.obj=self.targets['C001']
        def _initialize_episode(self,env_idx,options):
            names=[j.name for j in self.agent.robot.active_joints]
            limits=self.agent.robot.qlimits.cpu().numpy()[0]
            self.initial_q=np.clip(np.array([INITIAL_JOINTS[n] for n in names],dtype=np.float32),limits[:,0],limits[:,1])
            self.agent.reset(self.initial_q)
            self.agent.robot.set_pose(sapien.Pose(BASE_POSE))
        def _get_obs_extra(self,info):
            return {}

    override,camera_record=head_configuration(CURRENT_CONTROLLERS)
    env=gym.make('ARXPointCloudObservation-v1',num_envs=1,obs_mode='rgb+depth+segmentation',control_mode='pd_joint_pos',
                 reward_mode='none',render_mode='rgb_array',sim_backend='cpu',render_backend='sapien_cuda',sensor_configs=override).unwrapped
    profile=camera_profiles()['head'];maps=lens_maps(profile)
    receipts=[]
    try:
        env.reset(seed=20261122)
        for case in [manifest['warmup']]+manifest['cases']:
            target=env.targets[case['id']]
            mesh=get_actor_visual_mesh(target._objs[0])
            # The exact triangles actually sent to the renderer define scoring
            # geometry; collision hulls are never substituted as shape truth.
            bounds=mesh.bounds
            center=bounds.mean(0)
            xyz=[manifest['target_xy_m'][0]-center[0],manifest['target_xy_m'][1]-center[1],.002-bounds[0,2]]
            target.set_pose(sapien.Pose(xyz))
            started=time.perf_counter();obs=env.get_obs();capture_s=time.perf_counter()-started
            data=obs['sensor_data']['head'];params=obs['sensor_param']['head']
            rgb=data['rgb'].cpu().numpy()[0];depth=data['depth'].cpu().numpy()[0,:,:,0].astype(np.float32)*.001
            seg=data['segmentation'].cpu().numpy()[0,:,:,0];gt=seg==target._objs[0].per_scene_id
            frames=sensor_and_model_frames(rgb,depth,profile,maps)
            # Map evaluation-only segmentation through the same nearest-neighbor
            # image grids. It is never used by the segmentation/model workers.
            import cv2
            raw_gt=cv2.remap(gt.astype(np.uint8),maps[0][...,0],maps[0][...,1],cv2.INTER_NEAREST)
            gt=cv2.remap(raw_gt,maps[1][...,0],maps[1][...,1],cv2.INTER_NEAREST).astype(bool)
            out=RUN/'observations'/case['id'];truth=RUN/'ground_truth'/case['id'];out.mkdir(parents=True,exist_ok=True);truth.mkdir(parents=True,exist_ok=True)
            Image.fromarray(frames['raw_rgb']).save(out/'raw_rgb.png');Image.fromarray(frames['raw_depth']).save(out/'raw_depth.png')
            Image.fromarray(frames['rgb']).save(out/'rgb.png');np.save(out/'depth_m.npy',frames['depth_m'])
            Image.fromarray(gt.astype(np.uint8)*255).save(truth/'mask.png')
            world=target.pose.to_transformation_matrix().cpu().numpy()[0]
            mesh.apply_transform(world);mesh.export(truth/'surface_world.ply')
            k=params['intrinsic_cv'].cpu().numpy()[0];assert np.max(np.abs(k-intrinsic_matrix(profile)))<1e-4
            state={'id':case['id'],'camera_role':'head','intrinsics':k.tolist(),
                   'camera_world_optical':(params['cam2world_gl'].cpu().numpy()[0]@np.diag([1,-1,-1,1])).tolist(),
                   'qpos':env.agent.robot.qpos.cpu().numpy()[0].tolist(),
                   'raw_depth_scale_m':profile['raw_depth_scale_m'],'depth_file_unit':'meters',
                   'model_input_distortion_model':'none','model_input_distortion_coeffs':[0.]*5,
                   'real_camera_profile':profile,'mounting':camera_record,'capture_s':capture_s,
                   'scene_policy':'static kinematic objects, no robot actions or physics stepping; appearance/geometry benchmark only'}
            write(out/'camera.json',state)
            write(truth/'object_pose.json',{'object_world':world.tolist()})
            receipt={'id':case['id'],'visible_target_pixels':int(gt.sum()),
                     'touches_border':bool(gt[0].any() or gt[-1].any() or gt[:,0].any() or gt[:,-1].any()),
                     'input_hashes':{f:sha(out/f) for f in ['rgb.png','depth_m.npy','camera.json']},
                     'ground_truth_sha256':sha(truth/'surface_world.ply'),'capture_s':capture_s}
            receipts.append(receipt);write(RUN/'capture_progress.json',{'captured':len(receipts),'receipts':receipts,'model_trials':0,'physical_trials':0})
            print('Captured',case['id'],'pixels',receipt['visible_target_pixels'],'border',receipt['touches_border'],flush=True)
            target.set_pose(sapien.Pose([10,10,-10]))
        write(RUN/'capture_complete.json',{'receipts':receipts,'formal_observations':50,'warmup_observations':1,'physical_trials':0})
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--capture',action='store_true');args=parser.parse_args()
    manifest=prepare_assets()
    if args.capture:capture(manifest)
