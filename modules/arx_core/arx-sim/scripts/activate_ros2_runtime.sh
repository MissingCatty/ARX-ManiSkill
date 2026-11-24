#!/usr/bin/env bash
# Source Isaac Sim's bundled ROS 2 runtime into a non-Isaac Python process.
# This does not import Isaac/Kit or replace packages in the caller's environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file must be sourced, not executed." >&2
  exit 2
fi

_arx_sim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_isaac_root="${ISAAC_SIM_ROOT:-${_arx_sim_root}/../isaacsim}"
_ros_distro="${ARX_SIM_ROS_DISTRO:-jazzy}"
_isaac_ros_root="${_isaac_root}/exts/isaacsim.ros2.core/${_ros_distro}"

if [[ ! -d "${_isaac_ros_root}/rclpy" || ! -d "${_isaac_ros_root}/lib" ]]; then
  echo "Isaac ROS 2 runtime not found: ${_isaac_ros_root}" >&2
  return 2
fi

export ROS_DISTRO="${_ros_distro}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTHONPATH="${_isaac_ros_root}/rclpy${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${_isaac_ros_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

unset _arx_sim_root _isaac_root _ros_distro _isaac_ros_root
