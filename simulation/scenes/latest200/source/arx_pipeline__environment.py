"""ARX dual-arm robot in the existing ManiSkill table/YCB environment."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import sapien
import torch
import yaml
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import scene_builder as table_source
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.types import SimConfig, SceneConfig

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/arx"
META = json.loads((ASSETS / "manifest.json").read_text())
ARX_ROOT = ROOT.parent / "COOLAS-ARX"
INITIAL_SOURCE = ARX_ROOT / "scripts/curobo_arx/static_pick/reference/historical_reference/initial_setup.json"
CONTROLLERS_SOURCE = ARX_ROOT / "arx-sim/config/controllers.yaml"
RUNTIME_SOURCE = ARX_ROOT / "arx-sim/config/runtime.yaml"
INITIAL_STATE = json.loads(INITIAL_SOURCE.read_text())
INITIAL_JOINTS = dict(zip(INITIAL_STATE["joint_names"], INITIAL_STATE["joint_positions"]))
CURRENT_CONTROLLERS = yaml.safe_load(CONTROLLERS_SOURCE.read_text())["controllers"]
CURRENT_RUNTIME = yaml.safe_load(RUNTIME_SOURCE.read_text())["simulation"]
HEAD = CURRENT_CONTROLLERS["camera"]["definitions"]["head"]
PERCEPTION_CAMERA = "head"
ALIGNMENT = {
    "initial_state_source": str(INITIAL_SOURCE),
    "initial_state_policy": "Current static_pick/reproduce_setup.py restores this recorded grasp state; generic server home is a separate state",
    "camera_source": str(CONTROLLERS_SOURCE),
    "camera_role": PERCEPTION_CAMERA,
    "camera_serial": HEAD["serial"],
    "camera_mount": CURRENT_CONTROLLERS["camera"]["roles"]["head"],
    "camera_definition": HEAD,
    "depth_scale": CURRENT_CONTROLLERS["camera"]["depth_scale"],
    "camera_tick_hz": CURRENT_RUNTIME["camera_tick_hz"],
    "scope": "Match articulated joint initial state and calibrated camera; new ManiSkill world placement and renderer are recorded separately",
}


def alignment_for_camera(role):
    spec = CURRENT_CONTROLLERS["camera"]["definitions"][role]
    return {**ALIGNMENT, "camera_role": role, "camera_serial": spec["serial"],
            "camera_mount": CURRENT_CONTROLLERS["camera"]["roles"][role], "camera_definition": spec}
ARM = [f"right_joint{i}" for i in range(21, 27)]
FINGERS = ["right_joint27", "right_joint28"]
ORIGINAL_TABLE_HEIGHT = .9196429
TABLE_HEIGHT = .50
BASE_POSE = [-1.05, .25, -TABLE_HEIGHT]
SCENE_CUBOIDS = {
    "table": {"center": [-.12, 0., -TABLE_HEIGHT/2], "dims": [1.209, 2.418, TABLE_HEIGHT]},
    "ground": {"center": [0., 0., -TABLE_HEIGHT-.05], "dims": [6., 6., .1]},
}
RECORDING_VIEW = {
    "name": "robot_left_front",
    "eye_local": [1.20, .80, 1.20],
    "target_local": [.60, -.20, .60],
    "width": 1920, "height": 1080, "vertical_fov_rad": .80,
    "coordinates": "Relative to fixed base_link: +X forward, +Y left, +Z up",
}


@register_agent()
class ARX(BaseAgent):
    uid = "coolas_arx_x5_lift"
    urdf_path = str(ASSETS / "robot.urdf")
    fix_root_link = True
    disable_self_collisions = not META["runtime"]["enable_self_collisions"]
    # Each collision tag is already one original convex piece.
    load_multiple_collisions = False
    urdf_config = dict(
        _materials={"finger": META["controllers"]["gripper_contact"]["material"]},
        link={n: {"material": "finger"} for n in ["right_link27", "right_link28", "left_link17", "left_link18"]},
    )

    def _load_articulation(self, initial_pose=None):
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid
        loader.fix_root_link = True
        loader.disable_self_collisions = self.disable_self_collisions
        loader.load_multiple_collisions_from_file = False
        sapien_utils.apply_urdf_config(loader, sapien_utils.parse_urdf_config(self.urdf_config))
        builder = loader.parse(self.urdf_path)["articulation_builders"][0]
        # URDF requires intermediate links to express both USD joint frames.
        # SAPIEN supports pose_in_child directly: collapse those representation
        # links before building so they introduce no artificial mass/inertia.
        real = []
        for link in builder.link_builders:
            if link.name not in META["links"]:
                continue
            if link.parent is not None:
                frame = link.parent
                assert frame.name not in META["links"]
                offset = link.joint_record
                assert offset.joint_type == "fixed"
                record = deepcopy(frame.joint_record)
                record.pose_in_child = offset.pose_in_child * offset.pose_in_parent.inv() * record.pose_in_child
                link.joint_record = record
                link.parent = frame.parent
            real.append(link)
        builder.link_builders = real
        for index, link in enumerate(real):
            link.index = index
        builder.initial_pose = initial_pose
        self.robot = builder.build()
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    @property
    def _controller_configs(self):
        names = [j.name for j in self.robot.active_joints]
        specs = [META["joints"][n] for n in names]
        return {"pd_joint_pos": PDJointPosControllerConfig(
            joint_names=names, lower=None, upper=None,
            stiffness=[s["stiffness"] for s in specs], damping=[s["damping"] for s in specs],
            force_limit=[s["limits"]["effort"] for s in specs],
            drive_mode=[s["drive_mode"] for s in specs], normalize_action=False)}

    def _after_loading_articulation(self):
        self.tcp = self.robot.links_map["right_eef_link"]
        for link in self.robot.get_links():
            if link.name in META["links"]:
                for obj in link._objs:
                    obj.disable_gravity = META["links"][link.name]["disable_gravity"]
        for joint in self.robot.active_joints:
            for obj in joint._objs:
                obj.max_velocity = META["joints"][joint.name]["limits"]["velocity"]

    @property
    def _sensor_configs(self):
        # USD/OpenGL camera (-Z forward, +Y up) -> SAPIEN (+X forward, +Z up).
        gl_from_ros = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        # IsaacRgbdCamera uses the color camera for both RGB and aligned Z-depth.
        # inprocess_app._apply_camera_definition sets the same 0.05..10 m range.
        cameras = []
        for role in ["head", "right_wrist"]:
            spec = CURRENT_CONTROLLERS["camera"]["definitions"][role]
            mount, _ = CURRENT_CONTROLLERS["camera"]["roles"][role].split("/", 1)
            pose = np.asarray(spec["xform_rows"]).T @ gl_from_ros
            k = spec["color_intrinsics"]
            cameras.append(CameraConfig(role, sapien.Pose(pose), k["width"], k["height"],
                near=.05, far=10., intrinsic=np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]),
                mount=self.robot.links_map[mount], shader_pack="default"))
        return cameras


@register_env("ARXPickYCB-v1", max_episode_steps=3000)
class ARXPickYCB(BaseEnv):
    SUPPORTED_ROBOTS = [ARX.uid]

    def __init__(self, *args, model_id="005_tomato_soup_can", perception_camera=PERCEPTION_CAMERA, scene_spec=None, **kwargs):
        self.model_id = model_id
        self.scene_spec = scene_spec
        if perception_camera not in ("head", "right_wrist"):
            raise ValueError("Unsupported calibrated perception camera: " + perception_camera)
        self.perception_camera = perception_camera
        self.camera_alignment = alignment_for_camera(perception_camera)
        super().__init__(*args, robot_uids=ARX.uid, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=120, control_freq=40, scene_config=SceneConfig(
            solver_position_iterations=16, solver_velocity_iterations=4))

    @property
    def _default_sensor_configs(self):
        return []  # Perception comes from the calibrated robot-mounted head.

    @property
    def _default_human_render_camera_configs(self):
        eye = np.asarray(BASE_POSE) + RECORDING_VIEW["eye_local"]
        target = np.asarray(BASE_POSE) + RECORDING_VIEW["target_local"]
        return CameraConfig("overview", sapien_utils.look_at(eye, target),
                            RECORDING_VIEW["width"], RECORDING_VIEW["height"],
                            RECORDING_VIEW["vertical_fov_rad"], .05, 50., shader_pack="default")

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(BASE_POSE))

    def _load_scene(self, options):
        # User-authorized table height adjustment. Keep original XY footprint,
        # material, tabletop world height, and collision-box representation.
        # Scale only the table visual's vertical extent to match its collider.
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(pose=sapien.Pose([0, 0, TABLE_HEIGHT/2]),
                                  half_size=[2.418/2, 1.209/2, TABLE_HEIGHT/2])
        table_file = Path(table_source.__file__).parent / "assets/table.glb"
        builder.add_visual_from_file(str(table_file), scale=[1.75, 1.75, 1.75*TABLE_HEIGHT/ORIGINAL_TABLE_HEIGHT],
                                     pose=sapien.Pose(q=[np.sqrt(.5), 0, 0, np.sqrt(.5)]))
        builder.initial_pose = sapien.Pose([-.12, 0, -TABLE_HEIGHT], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
        self.table = builder.build_kinematic(name="table-workspace")
        self.ground = build_ground(self.scene, altitude=-TABLE_HEIGHT)
        db = json.loads((ROOT / "assets/data/assets/mani_skill2_ycb/info_pick_v0.json").read_text())
        self.objects, self.object_models, self.object_info = {}, {}, {}
        specs = self.scene_spec["actors"] if self.scene_spec else [{"name":"target", "model_id":self.model_id}]
        for spec in specs:
            builder = actors.get_actor_builder(self.scene, id="ycb:" + spec["model_id"])
            pose = spec.get("pose", [-.4, 0, .05, 1, 0, 0, 0])
            builder.initial_pose = sapien.Pose(pose[:3], pose[3:])
            self.objects[spec["name"]] = builder.build(name=spec["name"])
            self.object_models[spec["name"]] = spec["model_id"]
            self.object_info[spec["name"]] = db[spec["model_id"]]
        self.obj = self.objects["target"]
        info = db[self.model_id]
        self.object_bbox = info["bbox"]
        self.object_scale = info.get("scales", [1.])[0]

    def _initialize_episode(self, env_idx, options):
        if self.num_envs != 1:
            raise ValueError("This registered adapter uses one CPU physics environment")
        names = [j.name for j in self.agent.robot.active_joints]
        limits = self.agent.robot.qlimits.cpu().numpy()[0]
        # Match the source reset's clipping to unchanged URDF joint limits.
        self.initial_q = np.clip(np.array([INITIAL_JOINTS[n] for n in names], dtype=np.float32),
                                 limits[:, 0], limits[:, 1])
        assert ((self.initial_q >= limits[:, 0] - 1e-6) & (self.initial_q <= limits[:, 1] + 1e-6)).all()
        self.agent.reset(self.initial_q)
        self.agent.robot.set_pose(sapien.Pose(BASE_POSE))
        if self.scene_spec:
            for spec in self.scene_spec["actors"]:
                p = spec["pose"]
                self.objects[spec["name"]].set_pose(sapien.Pose(p[:3], p[3:]))
            self.spawn = deepcopy(self.scene_spec)
            return
        xy = self._episode_rng.uniform(-.035, .035, 2) + np.array([-.4, 0.])
        yaw = float(self._episode_rng.uniform(-np.pi, np.pi))
        z = -self.object_bbox["min"][2] * self.object_scale + .002
        self.obj.set_pose(sapien.Pose([*xy, z], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
        self.spawn = {"xy": xy.tolist(), "yaw": yaw, "z": z}

    def evaluate(self):
        # Final success additionally requires recorded bilateral contact and a
        # continuous >=3 s hold, evaluated by the batch runner.
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def _after_simulation_step(self):
        callback = getattr(self, "trace_callback", None)
        if callback is not None:
            callback()

    def _get_obs_extra(self, info):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose, "object_pose": self.obj.pose.raw_pose}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)


def make(model_id="005_tomato_soup_can", perception_camera=PERCEPTION_CAMERA, scene_spec=None):
    import gymnasium as gym
    return gym.make("ARXPickYCB-v1", model_id=model_id, perception_camera=perception_camera, scene_spec=scene_spec, num_envs=1,
                    obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                    reward_mode="none", render_mode="rgb_array", sim_backend="cpu",
                    render_backend="sapien_cuda").unwrapped
