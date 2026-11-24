from __future__ import annotations

import math
from typing import Any

import numpy as np

from arx_sim.geometry import matrix_to_quaternion_wxyz, pose_matrix


def _world_pose(prim: Any) -> np.ndarray:
    positions, orientations = prim.get_world_poses()
    return pose_matrix(positions.numpy()[0], orientations.numpy()[0])


class Ros2Publishers:
    """Publish DualMap-compatible data with Isaac Sim's bundled ROS 2 stack."""

    def __init__(self, config: dict[str, Any], camera_role: str = "head") -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_msgs.msg import TFMessage

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("coolas_isaac_io")
        self._types = {
            "clock": Clock,
            "image": Image,
            "camera_info": CameraInfo,
            "odom": Odometry,
            "tf": TFMessage,
        }
        ros = config["ros2"]
        self._topics = ros["topics"]
        self._frames = ros["frames"]
        self._interval = 1.0 / float(ros.get("publish_hz", 30.0))
        self._last_sensor_time = -float("inf")
        self._camera_role = camera_role
        self._static_sent = False
        histories = {"keep_last": HistoryPolicy.KEEP_LAST, "keep_all": HistoryPolicy.KEEP_ALL}
        reliabilities = {"best_effort": ReliabilityPolicy.BEST_EFFORT, "reliable": ReliabilityPolicy.RELIABLE}
        durabilities = {"volatile": DurabilityPolicy.VOLATILE, "transient_local": DurabilityPolicy.TRANSIENT_LOCAL}

        def qos(name: str) -> QoSProfile:
            values = ros["qos"][name]
            try:
                return QoSProfile(
                    history=histories[str(values["history"])],
                    depth=int(values["depth"]),
                    reliability=reliabilities[str(values["reliability"])],
                    durability=durabilities[str(values["durability"])],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid ros2.qos.{name} configuration: {values}") from exc

        sensor_qos = qos("sensor")
        reliable_qos = qos("reliable")
        static_qos = qos("static")
        self._clock = self._node.create_publisher(Clock, self._topics["clock"], reliable_qos)
        self._rgb = self._node.create_publisher(Image, self._topics["rgb"], sensor_qos)
        self._depth = self._node.create_publisher(Image, self._topics["depth"], sensor_qos)
        self._info = self._node.create_publisher(CameraInfo, self._topics["camera_info"], sensor_qos)
        self._odom = self._node.create_publisher(Odometry, self._topics["odometry"], sensor_qos)
        self._tf = self._node.create_publisher(TFMessage, self._topics["tf"], reliable_qos)
        self._tf_static = self._node.create_publisher(TFMessage, self._topics["tf_static"], static_qos)
        self._depth_scale = float(config["controllers"]["camera"]["depth_scale"])

    @staticmethod
    def _stamp(seconds: float) -> Any:
        from builtin_interfaces.msg import Time

        whole = math.floor(seconds)
        nanosec = int(round((seconds - whole) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        return Time(sec=int(whole), nanosec=nanosec)

    def _transform(self, parent: str, child: str, matrix: np.ndarray, stamp: Any) -> Any:
        from geometry_msgs.msg import TransformStamped

        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        translation = matrix[:3, 3]
        quaternion = matrix_to_quaternion_wxyz(matrix[:3, :3])
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, translation)
        msg.transform.rotation.w = float(quaternion[0])
        msg.transform.rotation.x = float(quaternion[1])
        msg.transform.rotation.y = float(quaternion[2])
        msg.transform.rotation.z = float(quaternion[3])
        return msg

    def publish(self, context: Any) -> None:
        snapshot = context.snapshot()
        sim_time = float(snapshot.sim_time)
        stamp = self._stamp(sim_time)
        clock = self._types["clock"]()
        clock.clock = stamp
        self._clock.publish(clock)
        camera = context.cameras[self._camera_role]
        frame_time = camera.frame_time
        if frame_time is None or frame_time - self._last_sensor_time + 1e-9 < self._interval:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            return
        self._last_sensor_time = frame_time
        stamp = self._stamp(float(frame_time))

        rgb = camera.get_rgba(device="cpu")
        depth = camera.get_depth(device="cpu")
        if rgb is not None and depth is not None:
            rgb = np.asarray(rgb)[:, :, :3]
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8, copy=False)
            depth = np.asarray(depth, dtype=np.float32)
            raw = np.zeros(depth.shape, dtype=np.uint16)
            valid = np.isfinite(depth) & (depth > 0)
            raw[valid] = np.clip(np.rint(depth[valid] / self._depth_scale), 1, 65535).astype(np.uint16)
            optical = self._frames["camera_optical"]
            rgb_msg = self._types["image"]()
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = optical
            rgb_msg.height, rgb_msg.width = rgb.shape[:2]
            rgb_msg.encoding = "rgb8"
            rgb_msg.is_bigendian = False
            rgb_msg.step = int(rgb_msg.width * 3)
            rgb_msg.data = rgb.tobytes()
            depth_msg = self._types["image"]()
            depth_msg.header = rgb_msg.header
            depth_msg.height, depth_msg.width = raw.shape
            depth_msg.encoding = "16UC1"
            depth_msg.is_bigendian = False
            depth_msg.step = int(depth_msg.width * 2)
            depth_msg.data = raw.tobytes()
            intrinsic = camera.get_intrinsics_matrix()
            info = self._types["camera_info"]()
            info.header = rgb_msg.header
            info.height, info.width = raw.shape
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            info.k = intrinsic.reshape(-1).tolist()
            info.r = np.eye(3).reshape(-1).tolist()
            info.p = [intrinsic[0, 0], 0.0, intrinsic[0, 2], 0.0,
                      0.0, intrinsic[1, 1], intrinsic[1, 2], 0.0,
                      0.0, 0.0, 1.0, 0.0]
            self._rgb.publish(rgb_msg)
            self._depth.publish(depth_msg)
            self._info.publish(info)

        base_world = _world_pose(context.base_prim)
        lift_world = _world_pose(context.lift_prim)
        arm_world = _world_pose(context.arm_base_prim)
        ee_world = _world_pose(context.ee_prim)
        camera_world = _world_pose(context.camera_prims[self._camera_role])
        odom = self._types["odom"]()
        odom.header.stamp = stamp
        odom.header.frame_id = self._frames["world"]
        odom.child_frame_id = self._frames["base"]
        q = matrix_to_quaternion_wxyz(base_world[:3, :3])
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, base_world[:3, 3])
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = map(float, q)
        odom.twist.twist.linear.x = float(snapshot.chassis_velocity[0])
        odom.twist.twist.linear.y = float(snapshot.chassis_velocity[1])
        odom.twist.twist.angular.z = float(snapshot.chassis_velocity[2])
        self._odom.publish(odom)
        dynamic = [
            self._transform(self._frames["world"], self._frames["base"], base_world, stamp),
            self._transform(self._frames["base"], self._frames["lift"], np.linalg.inv(base_world) @ lift_world, stamp),
            self._transform(self._frames["lift"], self._frames["arm_base"], np.linalg.inv(lift_world) @ arm_world, stamp),
            self._transform(self._frames["arm_base"], self._frames["ee"], np.linalg.inv(arm_world) @ ee_world, stamp),
            self._transform(
                self._frames["base"],
                self._frames.get("camera_mount", self._frames["right_camera"]),
                np.linalg.inv(base_world) @ camera_world,
                stamp,
            ),
        ]
        self._tf.publish(self._types["tf"](transforms=dynamic))
        if not self._static_sent:
            tcp_world = _world_pose(context.tcp_prim)
            tcp_local = np.linalg.inv(ee_world) @ tcp_world
            # USD camera: +X right, +Y up, -Z forward. ROS optical:
            # +X right, +Y down, +Z forward.
            optical_local = np.diag([1.0, -1.0, -1.0, 1.0])
            static = [
                self._transform(self._frames["ee"], self._frames["tcp"], tcp_local, stamp),
                self._transform(
                    self._frames.get("camera_mount", self._frames["right_camera"]),
                    self._frames["camera_optical"],
                    optical_local,
                    stamp,
                ),
            ]
            self._tf_static.publish(self._types["tf"](transforms=static))
            self._static_sent = True
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
